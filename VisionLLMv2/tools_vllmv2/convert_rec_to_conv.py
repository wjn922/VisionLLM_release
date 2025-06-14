import json
import random
import os
import argparse
from tqdm import tqdm

from prompts_rec import QUESTIONS, ANSWERS

def refcoco_to_llava(refcoco_json_path, output_jsonl_path, num_samples=None, max_turns=3):
    """
    将RefCOCO格式的JSON数据转换为LLAVA对话格式的JSONL数据
    
    Args:
        refcoco_json_path: RefCOCO格式的JSON文件路径
        output_jsonl_path: 输出的JSONL文件路径
        num_samples: 要采样的图片数量，None表示使用所有图片
        max_turns: 每轮对话的最大轮数
    """
    # 加载RefCOCO数据
    with open(refcoco_json_path, 'r', encoding='utf-8') as f:
        refcoco_data = json.load(f)
    
    # 获取所有图片数据
    images = refcoco_data.get('images', [])
    
    # 采样图片
    if num_samples is not None and num_samples < len(images):
        images = random.sample(images, num_samples)
    
    # 转换数据
    success = 0
    with open(output_jsonl_path, 'w', encoding='utf-8') as f:
        for idx, img_data in tqdm(enumerate(images), total=len(images)):
            img_file_name = img_data.get('file_name')
            expressions = img_data.get('expressions', [])
            
            if not img_file_name or not expressions:
                print(f"image {img_file_name} has no valid data, skip.")
                continue
            
            # 创建对话数据
            conversation = {
                "id": success,
                "image": img_file_name,
                "conversations": []
            }
            
            # 生成多轮对话
            turns = random.randint(1, max_turns)
            for turn in range(turns):
                # 随机选择一个表达式
                selected_expression = random.choice(expressions)
                
                # 随机选择问题和答案模板
                question_template = random.choice(QUESTIONS)
                answer_template = random.choice(ANSWERS)
                
                # 替换模板中的占位符，只在第一轮问题前添加<image>标签
                if turn == 0:
                    question = "<image>\n" + question_template.replace("<expression>", selected_expression)
                else:
                    question = question_template.replace("<expression>", selected_expression)
                
                # 答案中的表达式放在<ref>标签中
                ref_expression = f"<ref>{selected_expression}</ref>"
                answer = answer_template.replace("<expression>", ref_expression)
                
                # 添加到对话中
                conversation["conversations"].append({"from": "human", "value": question})
                conversation["conversations"].append({"from": "gpt", "value": answer})
            
            # 写入JSONL文件
            f.write(json.dumps(conversation) + '\n')
            success += 1
    
    print(f"转换完成! 已将 {success} 张图片转换为LLAVA对话格式并保存到 {output_jsonl_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert RefCOCO format to LLAVA conversation format')
    parser.add_argument('--input_path', '-i', required=True, help='Path to RefCOCO format JSON file')
    parser.add_argument('--output_path', '-o', required=True, help='Path to output JSONL file')
    parser.add_argument('--samples', '-s', type=int, default=1000, help='Number of samples to convert (default: 1000)')
    parser.add_argument('--max-turns', type=int, default=3, help='Maximum number of conversation turns per image (default: 3)')
    
    args = parser.parse_args()
    
    refcoco_to_llava(
        refcoco_json_path=args.input_path,
        output_jsonl_path=args.output_path,
        num_samples=args.samples,
        max_turns=args.max_turns
    )
