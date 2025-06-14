import argparse
import torch
import os
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

from PIL import Image

import requests
from PIL import Image
from io import BytesIO

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig


from visionllmv2.util.misc import nested_tensor_from_tensor_list
from visionllmv2.utils import disable_torch_init
from visionllmv2.mm_utils import expand2square, dynamic_preprocess, KeywordsStoppingCriteria
from visionllmv2.conversation import conv_templates, SeparatorStyle
from visionllmv2.constant import IGNORE_INDEX, DEFAULT_TOKENS
from visionllmv2.datasets.llava_data import tokenizer_image_token
from visionllmv2.model.modeling_visionllmv2 import VisionLLMv2Model

IMAGE_TOKEN_INDEX = -200

# def insert_ids(output_ids, insert_positions, emb_ids):
#         """
#         Args:
#         output_ids: torch.Tensor, original ids
#         insert_positions: torch.Tensor, positions for inserting ids
#         emb_ids: torch.Tensor, the ids to be inserted

#         Returns:
#         new_output_ids: torch.Tensor, output_ids
#         """
#         device = output_ids.device

#         # calculate new length
#         new_length = output_ids.size(0) + len(insert_positions) * emb_ids.size(0)
#         # creat a new output_ids 
#         new_output_ids = torch.zeros(new_length, dtype=torch.long).to(device)

#         output_index = 0
#         new_output_index = 0
#         for i in range(output_ids.size(0)):
#             new_output_ids[new_output_index] = output_ids[output_index]
#             new_output_index += 1

#             # if current pos need insert ids
#             if output_index in insert_positions:
#                 for emb_id in emb_ids:
#                     new_output_ids[new_output_index] = emb_id
#                     new_output_index += 1

#             output_index += 1
        
#         # the last pos whether need insert ids
#         if output_index in insert_positions:
#             for emb_id in emb_ids:
#                 new_output_ids[new_output_index] = emb_id
#         return new_output_ids

def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def eval_model(args):
    # Model
    disable_torch_init()
    model_name = os.path.expanduser(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, trust_remote_code=True)
    model = VisionLLMv2Model.from_pretrained(model_name, low_cpu_mem_usage=False, torch_dtype=torch.bfloat16).cuda()
    model.get_llm().config.use_cache = True
    # init special token ids
    model.init_special_token_ids(tokenizer)
    if model.config.vis_encoder_config.model_type == 'intern_vit_6b' and model.config.llm_config.model_type == 'internlm2': # InternVL
        model.imp_token_id = tokenizer.convert_tokens_to_ids('<img>')  
        IM_PATCH_TOKEN = '<img>'
    else:
        IM_PATCH_TOKEN = DEFAULT_TOKENS['imp']

    # get image
    image = load_image(args.image_file)
    processor = CLIPImageProcessor.from_pretrained(args.model_name, torch_dtype=torch.bfloat16)
    if args.image_aspect_ratio == 'anyres':
        image = dynamic_preprocess(image, image_size=args.image_size, max_num=args.image_max_tile) # list[pil_img]
        image = [processor.preprocess(x, return_tensors='pt')['pixel_values'][0] for x in image]
        image = torch.stack(image)  # [1 + n_tile, 3, h, w]
        image_token_len = int((args.image_size // 14) ** 2)
        if args.use_pixelshuffle:
            image_token_len = image_token_len // 4
        image_token_len = image_token_len * len(image)
    elif args.image_aspect_ratio == 'pad':
        image = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
        image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        image_token_len = int((args.image_size // 14) ** 2)
        if args.use_pixelshuffle:
            image_token_len = image_token_len // 4
    else:  # resize
        image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        image_token_len = int((args.image_size // 14) ** 2)
        if args.use_pixelshuffle:
            image_token_len = image_token_len // 4
    if args.image_aspect_ratio == 'anyres':
        image_tensor = [image.cuda().to(torch.bfloat16)]   # 1 x [n_split, 3, h, w]
    else:
        image_tensor = image.unsqueeze(0).cuda().to(torch.bfloat16)  # [1, 3, h, w]
    
    # place <image>  before the question.
    qs = args.query
    qs = DEFAULT_TOKENS['img'] + '\n' + qs
    conv_mode = args.conv_mode
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    # get input ids
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda() # [1, L]
    # replace with 'imp' tokens
    use_im_start_end = args.use_im_start_end
    replace_token = IM_PATCH_TOKEN * image_token_len
    if use_im_start_end:
        replace_token = DEFAULT_TOKENS['boi'] + replace_token + DEFAULT_TOKENS['eoi']
    replace_token_ids = tokenizer([replace_token], return_tensors="pt").input_ids[0][1:].cuda() # [L,], remove start token
    index = input_ids[0].argmin()  # find the index of IMAGE_TOKEN_INDEX
    new_input_ids = torch.cat([input_ids[0, :index], replace_token_ids, input_ids[0, index+1:]], dim=0).unsqueeze(0)
    input_ids = new_input_ids

    # stop criterion, this is needed for internlm2
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    # generate
    model.eval()
    with torch.inference_mode():
        outputs = model.generate(
            input_ids,
            images=image_tensor,
            do_sample=True,
            temperature=0.7,
            max_new_tokens=1024,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
            output_hidden_states=True,
            output_scores=True,
            return_dict_in_generate=True,
        )
    output_hidden_states = torch.cat([out[-1] for out in outputs.hidden_states[1:]], dim=1)  # [B, out_len-1, C], -1 for the end_token
    output_ids = outputs.sequences   # [B, L]

    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=False)[0]
    outputs = outputs.strip('</s>').strip()
    print(outputs)

    # # ------------------------------------------------------------------
    # # if use_gdino
    # output_ids = output_ids[:, input_token_len:]  # [B, out_len]
    # transform = T.Compose([
    #     T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
    #     T.Resize(size=800, max_size=1333, interpolation=InterpolationMode.BICUBIC),
    #     T.ToTensor(),
    #     T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    # ])
    
    # emb_ids = torch.tensor([x for x in range(model.emb_token_id, model.emb_token_id + model.num_embs)], dtype=torch.long).to(output_ids.device)  
    # new_output_ids = []
    # for cur_output_ids in output_ids:  # inference, bs=1
    #     cur_new_output_ids = cur_output_ids
    #     emb_start_pos_det = torch.where(cur_output_ids==model.det_tool_id)[0]
    #     emb_start_pos_seg = torch.where(cur_output_ids==model.seg_tool_id)[0]
    #     emb_start_pos_grd = torch.where(cur_output_ids==model.grd_tool_id)[0]
    #     emb_start_pos = torch.cat([emb_start_pos_det, emb_start_pos_seg, emb_start_pos_grd], dim=0)
    #     emb_start_pos = torch.sort(emb_start_pos)[0]
    #     cur_new_output_ids = insert_ids(cur_new_output_ids, emb_start_pos, emb_ids)
    #     new_output_ids.append(cur_new_output_ids)
    # output_ids = torch.stack(new_output_ids, dim=0)
    
    # if model.det_tool_id in output_ids:
    #     assert model.use_gdino
    #     image_aug = transform(load_image(args.image_file)).cuda().to(torch.bfloat16)  # [3, h, w], after aug
    #     img_h, img_w = image_aug.shape[-2:]
    #     img_meta = {'img_shape': (img_h, img_w)}
    #     img_metas = [img_meta]
    #     images_aug = nested_tensor_from_tensor_list([image_aug], size_divisibility=32)
    #     pixel_values, pixel_mask = images_aug.tensors, ~images_aug.mask  # [bs, 3, h, w], [bs, h, w]
    #     pixel_mask = pixel_values[:, 0, :, :] != 0  # valid is 1
    #     # select the corresponding [EMB] hidden states as text_query
    #     batch_size, seq_len, hidden_size = output_hidden_states.shape
    #     emb_select = (output_ids[:, :-1] >= model.emb_token_id) & (output_ids[:, :-1] <= model.emb_token_id + model.num_embs - 1)  # [bs, seq_len]
    #     # if have [EMB] tokens
    #     if emb_select.sum() != 0:
    #         num_patches = emb_select.sum(-1) // model.num_embs  # [bs,]
    #         max_num_patches = num_patches.max()
    #         text_query = torch.zeros((batch_size, max_num_patches, model.num_embs, hidden_size), dtype=output_hidden_states.dtype, device=output_hidden_states.device) # [bs, max_num_patches, num_embs, c]
    #         text_query_masks = torch.zeros(batch_size, max_num_patches, dtype=torch.bool, device=output_hidden_states.device)       # [bs, max_num_patches], valid is 1
    #         for batch_idx in range(batch_size):
    #             if num_patches[batch_idx] != 0:
    #                 text_query_i = output_hidden_states[batch_idx, emb_select[batch_idx], :].reshape(-1, model.num_embs, hidden_size)  # [num_patch_i*num_embs, c] -> [num_patch_i, num_embs, c]
    #                 text_query[batch_idx, :num_patches[batch_idx]] = text_query_i
    #                 text_query_masks[batch_idx, :num_patches[batch_idx]] = 1
    #         with torch.inference_mode():
    #             gdino_outputs = model.gdino(pixel_values, pixel_mask=pixel_mask, text_query=text_query, text_query_masks=text_query_masks)
    #         gdino_logits = gdino_outputs.logits[:, :, :max_num_patches]  # remove padding logits
    #         gdino_outputs.logits = gdino_logits
    #         from visionllmv2.model.modeling_visionllmv2 import debug_predictions
    #         debug_predictions(pixel_values.float(), gdino_outputs, img_metas)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="facebook/opt-350m")
    parser.add_argument("--image-file", type=str, required=True)
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--conv-mode", type=str, default='vicuna_v1')
    parser.add_argument('--image_aspect_ratio', type=str, default='anyres')
    parser.add_argument("--use_im_start_end", type=bool, default=False)
    parser.add_argument("--image_size", type=int, default=336)
    parser.add_argument("--image_max_tile", type=int, default=4)
    parser.add_argument("--use_pixelshuffle", type=bool, default=False)
    args = parser.parse_args()

    eval_model(args)