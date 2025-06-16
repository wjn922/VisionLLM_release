import os
import sys
import argparse

sys.path.append(".")
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers import CLIPVisionConfig, CLIPVisionModel, CLIPImageProcessor
import torch
import json
from visionllmv2.model.configuration_visionllmv2 import VisionLLMv2Config
from visionllmv2.model.modeling_visionllmv2 import VisionLLMv2Model


def parse_args():
    parser = argparse.ArgumentParser(description='Convert pretrained vllmv2 work dirs to cleaned vllmv2 weights (only vllm part).')
    parser.add_argument('--pretrained_vllmv2_path', type=str, help='pretrained vllmv2 model path.')
    parser.add_argument('--output_vllmv2_path', type=str, help='output vllmv2 model path')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    pretrained_vllmv2_path = args.pretrained_vllmv2_path
    output_vllmv2_path = args.output_vllmv2_path

    # load pretrained vllmv2 model
    config = VisionLLMv2Config.from_pretrained(pretrained_vllmv2_path)
    model = VisionLLMv2Model.from_pretrained(pretrained_vllmv2_path, torch_dtype=torch.bfloat16).eval()

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_vllmv2_path, use_fast=False, trust_remote_code=True
    )
    model.tokenizer = tokenizer

    preprocessor = CLIPImageProcessor.from_pretrained(pretrained_vllmv2_path)
    
    # delete tools
    # gdino
    config.use_gdino = False
    config.gdino_config = None
    model.use_gdino = False
    model.gdino = None
    # unipose
    config.use_unipose = False
    config.unipose_config = None
    model.use_unipose = False
    model.unipose = None
    # sd
    config.use_sd = False
    config.sd_config = None
    model.use_sd = False
    model.sd = None
    # ip2p
    config.use_ip2p = False
    config.ip2p_config = None
    model.use_ip2p = False
    model.ip2p = None

    # import ipdb; ipdb.set_trace()
    model.config = config
    
    # save
    # config.save_pretrained(output_vllmv2_path)
    model.save_pretrained(output_vllmv2_path)
    model.tokenizer.save_pretrained(output_vllmv2_path)
    preprocessor.save_pretrained(output_vllmv2_path)

    print(f"Model has been saved to {output_vllmv2_path}")