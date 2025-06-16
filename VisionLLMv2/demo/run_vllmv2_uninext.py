import argparse
import torch
import os
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

import re
import cv2
import requests
from PIL import Image
from io import BytesIO
import numpy as np

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig

# visionllmv2
from visionllmv2.util.misc import nested_tensor_from_tensor_list
from visionllmv2.utils import disable_torch_init
from visionllmv2.mm_utils import expand2square, dynamic_preprocess, KeywordsStoppingCriteria
from visionllmv2.conversation import conv_templates, SeparatorStyle
from visionllmv2.constant import IGNORE_INDEX, DEFAULT_TOKENS
from visionllmv2.datasets.llava_data import tokenizer_image_token
from visionllmv2.model.modeling_visionllmv2 import VisionLLMv2Model

# uninext
from detectron2.config import get_cfg
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.projects.uninext import add_uninext_config
from detectron2.structures import BoxMode

from .colormap import colormap
from .utils import rle_to_tensor_for_video, rle_decode
from .uninext_predictor import UNINEXTImagePredictor, UNINEXTVideoPredictor, UNINEXTVideoSOTPredictor

IMAGE_TOKEN_INDEX = -200

def extract_task(string):
    # <task> ... (..) </task>, 提取 () 中间的内容
    pattern = r'<task>.*?\((.*?)\)</task>' 
    # 使用正则表达式搜索匹配项
    match = re.search(pattern, string)
    # 如果找到匹配项，返回括号内的内容，否则返回空字符串
    if match:
        return match.group(1)
    else:
        return ""

def extract_refs(text: str) -> list[str]:
    # 提取每个 <ref> ... </ref> 中间的内容
    return re.findall(r'<ref>(.*?)</ref>', text, re.DOTALL)

# UNINEXT config
def setup_cfg(args):
    cfg = get_cfg()
    add_uninext_config(cfg)
    cfg.merge_from_file(args.config_file)
    # cfg.freeze()
    return cfg


def load_image(image_file):
    
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def eval_model(args):
    assert args.image_file or args.image_folder, "Please Specify the image file or image folder."
    if args.image_folder:
        image_files = sorted(os.listdir(args.image_folder))
        image_files = [os.path.join(args.image_folder, file) for file in image_files]
        image_file = image_files[0]
    else:
        image_file = args.image_file

    # Load VisionLLMv2 Model
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
    image = load_image(image_file)
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
            repetition_penalty=1.1,
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

    if '<task>' not in outputs:
        return


    # --------------------------------------------------------------------------------------------------
    # If use UNINEXT
    if "r50" in args.uninext_weights: 
        uninext_type = "r50"
    elif "convnext_large" in args.uninext_weights:
        uninext_type = "convnext_large"
    else:
        uninext_type = "vit_huge"

    task = extract_task(outputs)
    if task in ['od', 'is', 'rec', 'res', 'vis', 'vos', 'rvos', 'sot', 'mot', 'mots']: 
        test_categories = extract_refs(outputs) # list[dict]
        test_categories = [{"isthing": 1, "id": i + 1, "name": cat.strip()} for i, cat in enumerate(test_categories)]  # list[dict]

        # set uninext config file
        if task in ['od', 'is', 'rec', 'res']:  # image task
            args.config_file = f"UNINEXT/projects/UNINEXT/configs/image_joint_{uninext_type}.yaml"
        else:  # video task
            args.config_file = f"UNINEXT/projects/UNINEXT/configs/video_joint_{uninext_type}_demo.yaml"
        cfg = setup_cfg(args)
        cfg.MODEL.WEIGHTS = args.uninext_weights
        # maybe modified for different tasks
        if task in ['vis', 'rvos']:
            cfg.INPUT.MIN_SIZE_TEST = 720
            cfg.MODEL.USE_IOU_BRANCH = False
        elif task in ['vos', 'sot']:
            cfg.INPUT.MIN_SIZE_TEST = 720
            cfg.SOT.ONLINE_UPDATE = True
        elif task in ['mots', 'mot']:
            cfg.MODEL.USE_IOU_BRANCH = False
            cfg.TRACK.INIT_SCORE_THR = 0.4
            cfg.TRACK.OBJ_SCORE_THR = 0.3
        cfg.freeze()

        # run uninext
        if task in ['od', 'is', 'rec', 'res']:  # image task
            input_image = cv2.imread(image_file)
            predictor = UNINEXTImagePredictor(cfg)
            predictions = predictor(input_image, task='detection', test_categories=test_categories)['instances']  # d2 Instance, have been postprocessed to original size
            # visualization
            os.makedirs('uninext_outputs', exist_ok=True)
            visualize_image_predictions(image_file, predictions, test_categories, show_box=True, show_mask=True)
            print(f"Saving results to uninext_outputs/{os.path.basename(image_file)}")
        elif task in ['vis', 'rvos']:
            input_images = []
            for image_file in image_files:
                input_image = cv2.imread(image_file)
                input_images.append(input_image)
            predictor = UNINEXTVideoPredictor(cfg)
            prediction = predictor(input_images, task=task, test_categories=test_categories)  # dict
            output_path = os.path.join('uninext_outputs/', image_files[0].split('/')[-2])
            os.makedirs(output_path, exist_ok=True) 
            visualize_vis_predictions(image_files, prediction, test_categories, output_path)
            print(f"Saving results to {output_path}")
        elif task in ['vos', 'sot']:
            input_images = []
            for image_file in image_files:
                input_image = cv2.imread(image_file)
                input_images.append(input_image)
            # read ref annotation
            if args.ref_mask:
                mask_anno = Image.open(args.ref_mask).convert('L')  
                mask_anno = np.array(mask_anno) / 255.0  # normalize to [0, 1]
            elif args.ref_box:
                width, height = Image.open(image_files[0]).size
                mask_anno = create_mask_from_box(args.ref_box, height, width)  # binary [0, 1]
            else:
                print(f"Not provide the ref annotation for {task}. [END]")
                return 
            predictor = UNINEXTVideoSOTPredictor(cfg)
            predictions = predictor(input_images, task=task, mask_anno=mask_anno)  # list[np.array]
            predictions.insert(0, mask_anno)  # insert first frame mask anno
            output_path = os.path.join('uninext_outputs/', image_files[0].split('/')[-2])
            os.makedirs(output_path, exist_ok=True) 
            visualize_vos_predictions(image_files, predictions, output_path)
            print(f"Saving results to {output_path}")
        elif task in ['mot', 'mots']:
            input_images = []
            for image_file in image_files:
                input_image = cv2.imread(image_file)
                input_images.append(input_image)
            predictor = UNINEXTVideoPredictor(cfg)
            predictions = predictor(input_images, task=task, test_categories=test_categories)  # defaultdict(list)
            output_path = os.path.join('uninext_outputs/', image_files[0].split('/')[-2])
            os.makedirs(output_path, exist_ok=True) 
            visualize_mot_predictions(image_files, predictions, test_categories, output_path)
            print(f"Saving results to {output_path}")

    else:
        print(f"{task} is not supported by UNINEXT. [END]")


def create_mask_from_box(box, height, width):
    """
    根据边界框生成二值遮罩
    
    参数:
    box (list): 边界框坐标 [x1, y1, x2, y2]
    height (int): 图片高度
    width (int): 图片宽度
    
    返回:
    np.ndarray: 二值遮罩，类型为np.uint8，背景为0，框内区域为255
    """
    # 创建全零数组（背景）
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # 提取边界框坐标
    x1, y1, x2, y2 = box
    
    # 确保坐标在有效范围内
    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(width, int(x2))
    y2 = min(height, int(y2))
    
    # 检查框是否有效
    if x1 < x2 and y1 < y2:
        # 将框内区域设为255（白色）
        mask[y1:y2, x1:x2] = 1
    
    return mask



def visualize_image_predictions(image_path, predictions, test_categories, show_box=True, show_mask=True):
    # image: cv2 image of [H, W, 3], BGR format
    # predictions: d2 Instances
    # test_categories: list[dict]
    image = cv2.imread(image_path)    # [H, W, 3], BGR format
    pred_scores = predictions.scores  # [N,]
    pred_classes = predictions.pred_classes
    pred_boxes = predictions.pred_boxes.tensor
    pred_masks = predictions.pred_masks
    choose = pred_scores > 0.5
    pred_scores = pred_scores[choose].cpu().numpy()
    pred_classes = pred_classes[choose].cpu().numpy()
    pred_boxes = pred_boxes[choose].cpu().numpy()
    pred_boxes = BoxMode.convert(pred_boxes, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
    pred_masks = pred_masks[choose].cpu().numpy().astype(np.float32)

    # visualize
    color_list = colormap().tolist()
    save_image = image.astype(np.float32)
    for inst_idx, (class_idx, box, mask) in enumerate(zip(pred_classes, pred_boxes, pred_masks)):
        color = color_list[inst_idx%79]
        if show_box:
            x1, y1, w, h = box
            cv2.rectangle(save_image, (int(x1), int(y1)), (int(x1+w), int(y1+h)), color, thickness=2)
            cv2.putText(save_image, test_categories[class_idx]['name'], (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color=color, thickness=2)
        if show_mask:
            color_mask = np.array(color) * mask[:, :, None] * 0.5
            save_image += color_mask
        save_path = f"uninext_outputs/{os.path.basename(image_path)}"
        cv2.imwrite(save_path, save_image)
    
    
def visualize_vis_predictions(image_paths, prediction, test_categories, output_path):
    # image_paths (list[str])
    # prediction (dict): 'pred_scores', 'pred_labels', 'pred_masks'
    pred_scores = torch.tensor(prediction['pred_scores'])          # [n_obj,]
    pred_classes = torch.tensor(prediction['pred_labels']).long()  # [n_obj,]
    pred_masks = prediction['pred_masks']  # list[list[rle]]
    pred_masks = rle_to_tensor_for_video(pred_masks)               # [n_obj, n_frame, ori_h, ori_w]
    choose = pred_scores > 0.5
    pred_scores = pred_scores[choose].cpu().numpy()
    pred_classes = pred_classes[choose].cpu().numpy()
    pred_masks = pred_masks[choose].cpu().numpy().astype(np.float32)

    # visualize
    color_list = colormap().tolist()
    n_obj, n_frame, _, _ = pred_masks.shape
    for frame_idx in range(n_frame):
        cur_pred_masks = pred_masks[:, frame_idx, :, :]  # [n_obj, ori_h, ori_w]
        image = cv2.imread(image_paths[frame_idx])
        save_image = image.astype(np.float32)
        for inst_idx, (class_idx, mask) in enumerate(zip(pred_classes, cur_pred_masks)):
            color = color_list[inst_idx%79]
            color_mask = np.array(color) * mask[:, :, None] * 0.5
            save_image += color_mask
            # bbox and text
            if np.any(mask):  # visible
                box = bounding_box(mask)  # [x, y, w, h]
                x1, y1, w, h = box
                cv2.rectangle(save_image, (int(x1), int(y1)), (int(x1+w), int(y1+h)), color, thickness=2)
                cv2.putText(save_image, f"{inst_idx} " + test_categories[class_idx]['name'], (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color=color, thickness=2)
        save_path = os.path.join(output_path, os.path.basename(image_paths[frame_idx]))
        cv2.imwrite(save_path, save_image)

def visualize_vos_predictions(image_paths, predictions, output_path):
    # image_paths: list[str]
    # predictions: list[array], size of original [H, W], type uint8
    n_frame = len(image_paths)
    for frame_idx in range(n_frame):
        image = cv2.imread(image_paths[frame_idx])
        save_image = image.astype(np.float32)
        mask = predictions[frame_idx]
        if np.any(mask):
            box = bounding_box(mask)  # [x, y, w, h]
            x1, y1, w, h = box
            cv2.rectangle(save_image, (int(x1), int(y1)), (int(x1+w), int(y1+h)), color=(0,0,255), thickness=2)
            color_mask = np.array((0, 0, 255)) * mask[:, :, None] * 0.5
            save_image += color_mask
        save_path = os.path.join(output_path, os.path.basename(image_paths[frame_idx]))
        cv2.imwrite(save_path, save_image)

def visualize_mot_predictions(image_paths, predictions, test_categories, output_path):
    # image_paths: list[str]
    # predictions: defaultdict(list), 'bbox_result', 'track_result'. for mots=True
    # set mots=True inside UNINEXT_VID_DEMO
    track_results = predictions['track_result']  # list[dict]
    # length of n_frame
    # dict for each inst_id, 'bbox', 'label' 'segm'

    # visualize
    color_list = colormap().tolist()
    n_frame = len(image_paths)
    for frame_idx in range(n_frame):
        # image
        image = cv2.imread(image_paths[frame_idx])
        height, width = image.shape[:2]
        save_image = image.astype(np.float32)
        # prediction
        track_result = track_results[frame_idx]  # dict
        for inst_idx in track_result.keys():
            color = color_list[inst_idx%79]
            label = track_result[inst_idx]['label']
            bbox = track_result[inst_idx]['bbox'][:4]  # [5,], last one is score
            mask = rle_decode(track_result[inst_idx]['segm'], height, width)  # np.uint8, in [0, 1]
            # plot
            x1, y1, x2, y2 = bbox
            w, h = x2 - x1, y2 - y1
            cv2.rectangle(save_image, (int(x1), int(y1)), (int(x1+w), int(y1+h)), color, thickness=2)
            cv2.putText(save_image, f"{inst_idx} " + test_categories[label]['name'], (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color=color, thickness=2)
            color_mask = np.array(color) * mask[:, :, None] * 0.5
            save_image += color_mask
        save_path = os.path.join(output_path, os.path.basename(image_paths[frame_idx]))
        cv2.imwrite(save_path, save_image)






    

def bounding_box(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]
    return [int(x1), int(y1), int(x2-x1), int(y2-y1)] # (x1, y1, w, h) 


def parse_box(s):
    # 去除方括号并分割字符串
    s = s.strip('[]')
    parts = s.split(',')
    # 将各部分转换为浮点数
    try:
        box = [float(part.strip()) for part in parts]
    except ValueError:
        raise argparse.ArgumentTypeError("Box values must be numbers")
    # 验证是否为4个值
    if len(box) != 4:
        raise argparse.ArgumentTypeError("Box must contain 4 values: x1, y1, x2, y2")
    return box


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # visionllmv2
    parser.add_argument("--model-name", type=str, default="facebook/opt-350m")
    parser.add_argument("--image-file", type=str)    # for image task
    parser.add_argument("--image-folder", type=str)  # for video task
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--conv-mode", type=str, default='vicuna_v1')
    parser.add_argument('--image_aspect_ratio', type=str, default='anyres')
    parser.add_argument("--use_im_start_end", type=bool, default=False)
    parser.add_argument("--image_size", type=int, default=336)
    parser.add_argument("--image_max_tile", type=int, default=4)
    parser.add_argument("--use_pixelshuffle", type=bool, default=False)
    # uninext
    parser.add_argument("--uninext_weights", type=str, default="checkpoints/uninext/video_joint_convnext_large.pth")
    parser.add_argument("--ref_mask", type=str, help='path to the mask anno for VOS')    # vos, first frame mask anno
    parser.add_argument("--ref_box", type=parse_box, help='Bounding box in the format [x1, y1, x2, y2]')
    args = parser.parse_args()

    eval_model(args)

