import sys
import argparse

sys.path.append(".")
import token
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers import CLIPVisionConfig, CLIPVisionModel, CLIPImageProcessor
import torch
import json
from visionllmv2.model.configuration_visionllmv2 import VisionLLMv2Config
from visionllmv2.model.modeling_visionllmv2 import VisionLLMv2Model

def parse_args():
    parser = argparse.ArgumentParser(description='Convert InternVL Model to VisionLLMv2 format.')
    parser.add_argument('--pretrained_internvl2_path', type=str, help='pretrained model path')
    parser.add_argument('--output_path', type=str, help='output model path')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # Load the original model
    internvl_path = (
        "/mnt/petrelfs/share_data/wangwenhai/internvl/release/InternVL-Chat-V1-5/"
    )
    vllmv2_path = "/mnt/petrelfs/share_data/wujiannan/workspace/VisionLLMv2/final_dirs/llava-7b-joint-stage3/"
    # config = json.load(open(path + "config.json"))
    # print('a' * 20)
    vllm2_config = AutoConfig.from_pretrained(vllmv2_path, trust_remote_code=True)
    # print('b' * 20)
    # vllm2 = VisionLLMv2WithInternVL(config)
    vllm2 = (
        VisionLLMv2Model.from_pretrained(
            vllmv2_path, torch_dtype=torch.bfloat16, trust_remote_code=True
        )
        .eval()
    )
    # If you have an 80G A100 GPU, you can put the entire model on a single GPU.
    internvl = (
        AutoModel.from_pretrained(
            internvl_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        .eval()
    )
    internvl_tokenizer = AutoTokenizer.from_pretrained(
        internvl_path, trust_remote_code=True
    )
    # print(model)

    # import pdb; pdb.set_trace()
    vllm2.llm = internvl.language_model
    # vllm2.mlp1 = internvl.mlp1
    vllm2.vl_bridge = internvl.mlp1
    vllm2.vis_encoder = internvl.vision_model
    img_processor = CLIPImageProcessor.from_pretrained(internvl_path)
    # vllm2.config.gdino_config = vllm2_config.gdino_config
    # vllm2.config.downsample_ratio = internvl.downsample_ratio
    vllm2.config.use_pixelshuffle = True
    # "use_region_encoder": true,
    # "use_unipose": true,
    vllm2.config.use_region_encoder = False
    vllm2.config.use_unipose = False
    vllm2.config.use_gdino = False
    vllm2.config.use_llm_lora = False
    # vllm2.config.use_internvl = True
    vllm2.config.vis_output_layer = -1
    vllm2.config.l_hidden_size = internvl.config.llm_config.hidden_size
    vllm2.config.v_hidden_size = internvl.config.vision_config.hidden_size
    vllm2.config.vl_bridge_type = "internvl_mlp"
    vllm2.config.vis_encoder_config = internvl.config.vision_config
    vllm2.config.llm_config = internvl.config.llm_config
    vllm2.tokenizer = internvl_tokenizer
    vllm2.emb_embeddings = None
    vllm2.gdino = None
    vllm2.unipose = None
    vllm2.region_encoder = None
    vllm2.llm.save_pretrained("internvl/internlm/")
    vllm2.vis_encoder.save_pretrained("internvl/intern_vit/")
    img_processor.save_pretrained("internvl/intern_vit/")
    vllm2.tokenizer.save_pretrained("internvl/internlm/")
    vllm2.save_pretrained("internvl_ckpt/")
    vllm2.tokenizer.save_pretrained("internvl_ckpt/")
    img_processor.save_pretrained("internvl_ckpt/")
    print("Done")
    # Load the modified model
    # model = transformers.VisionLLMv2.from_pretrained("tatsu/llama-visionllm-v2-base")
    # model.save_pretrained("tatsu/llama-visionllm-v2-base")