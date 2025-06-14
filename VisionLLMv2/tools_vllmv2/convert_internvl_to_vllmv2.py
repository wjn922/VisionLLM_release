import sys
import argparse
import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers import CLIPImageProcessor

# 导入自定义模型配置和模型类
sys.path.append(".")
from visionllmv2.model.configuration_visionllmv2 import VisionLLMv2Config
from visionllmv2.model.modeling_visionllmv2 import VisionLLMv2Model

def parse_args():
    parser = argparse.ArgumentParser(description='Convert InternVL Model to VisionLLMv2 format.')
    parser.add_argument('--pretrained_internvl2_path', type=str, required=True, 
                        help='Path to pretrained InternVL model')
    parser.add_argument('--pretrained_vllmv2_path', type=str, required=True, 
                        help='Path to pretrained VisionLLMv2 model')
    parser.add_argument('--output_path', type=str, required=True, 
                        help='Output directory for converted model')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 加载预训练模型
    print(f"Loading InternVL model from {args.pretrained_internvl2_path}")
    internvl = AutoModel.from_pretrained(
        args.pretrained_internvl2_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    ).eval()
    
    internvl_tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_internvl2_path, trust_remote_code=True
    )
    
    print(f"Loading VisionLLMv2 model from {args.pretrained_vllmv2_path}")
    vllm2_config = AutoConfig.from_pretrained(args.pretrained_vllmv2_path, trust_remote_code=True)
    vllm2 = VisionLLMv2Model.from_pretrained(
        args.pretrained_vllmv2_path, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).eval()

    import ipdb; ipdb.set_trace()
    
    # 模型组件替换
    vllm2.llm = internvl.language_model
    vllm2.vl_bridge = internvl.mlp1
    vllm2.vis_encoder = internvl.vision_model
    
    # 加载图像处理器
    img_processor = CLIPImageProcessor.from_pretrained(args.pretrained_internvl2_path)
    
    # 配置参数设置
    vllm2.config.use_pixelshuffle = True
    vllm2.config.use_region_encoder = False
    vllm2.config.use_unipose = False
    vllm2.config.use_gdino = False
    vllm2.config.use_llm_lora = False
    vllm2.config.vis_output_layer = -1
    vllm2.config.l_hidden_size = internvl.config.llm_config.hidden_size
    vllm2.config.v_hidden_size = internvl.config.vision_config.hidden_size
    vllm2.config.vl_bridge_type = "internvl_mlp"
    vllm2.config.vis_encoder_config = internvl.config.vision_config
    vllm2.config.llm_config = internvl.config.llm_config
    
    # 移除不需要的组件
    vllm2.tokenizer = internvl_tokenizer
    vllm2.emb_embeddings = None
    vllm2.gdino = None
    vllm2.unipose = None
    vllm2.region_encoder = None
    
    # 创建输出目录
    output_path = args.output_path
    # internlm_path = f"{output_path}/internlm"
    # intern_vit_path = f"{output_path}/intern_vit"
    
    # 保存模型组件
    print(f"Saving model components to {output_path}")
    # vllm2.llm.save_pretrained(internlm_path)
    # vllm2.vis_encoder.save_pretrained(intern_vit_path)
    # img_processor.save_pretrained(intern_vit_path)
    # vllm2.tokenizer.save_pretrained(internlm_path)
    
    # 保存完整模型
    vllm2.save_pretrained(output_path)
    vllm2.tokenizer.save_pretrained(output_path)
    img_processor.save_pretrained(output_path)
    
    print("Conversion completed successfully!")

if __name__ == "__main__":
    main()