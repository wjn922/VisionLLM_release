import os
import argparse
from PIL import Image
import numpy as np

def process_youtubevos_mask(input_path, obj_id, output_path):
    """
    处理YouTube-VOS数据集中的掩码图片
    
    参数:
    input_path (str): 输入掩码图片路径
    obj_id (int): 要提取的目标ID
    output_path (str): 输出二值掩码图片路径
    """
    try:
        # 读取掩码图片
        mask_img = Image.open(input_path).convert('P')
        mask_array = np.array(mask_img)
        
        # 创建二值掩码 (0和1)
        binary_mask = np.zeros_like(mask_array, dtype=np.uint8)
        binary_mask[mask_array == obj_id] = 1
        
        # 保存二值掩码图
        binary_img = Image.fromarray(binary_mask * 255, mode='L')
        binary_img.save(output_path)
        
        print(f"成功处理掩码图片，目标ID: {obj_id}")
        print(f"输入路径: {input_path}")
        print(f"输出路径: {output_path}")
        
    except Exception as e:
        print(f"处理掩码时出错: {e}")

def main():
    parser = argparse.ArgumentParser(description='处理YouTube-VOS掩码图片')
    parser.add_argument('--input', required=True, help='输入掩码图片路径')
    parser.add_argument('--obj_id', type=int, required=True, help='要提取的目标ID')
    parser.add_argument('--output', required=True, help='输出二值掩码图片路径')
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    process_youtubevos_mask(args.input, args.obj_id, args.output)

if __name__ == "__main__":
    main()    