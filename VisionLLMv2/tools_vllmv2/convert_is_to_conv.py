# Convert the COCO annotations to LLaVA style conversations.

import json
import random
import os
import argparse
from pycocotools.coco import COCO
from tqdm import tqdm

from prompts_is import QUESTIONS, ANSWERS

def select_categories(all_category_names, gt_category_names):
    """根据概率选择类别"""
    rand = random.random()
    if rand < 0.2:  # 20%概率选择所有COCO类别
        selected_categories = all_category_names
    elif rand < 0.6:  # 40%概率选择所有GT类别
        selected_categories = gt_category_names
    else:  # 40%概率选择随机数量的GT类别
        num_categories = random.randint(1, len(gt_category_names))
        selected_categories = random.sample(gt_category_names, num_categories)
    
    # 打乱类别顺序
    random.shuffle(selected_categories)
    return selected_categories

def coco_to_llava(coco_json_path, output_jsonl_path, num_samples=None, max_turns=3):
    """
    将COCO格式的JSON数据转换为LLAVA对话格式的JSONL数据
    
    Args:
        coco_json_path: COCO格式的JSON文件路径
        output_jsonl_path: 输出的JSONL文件路径
        num_samples: 要采样的图片数量，None表示使用所有图片
        max_turns: 每轮对话的最大轮数
    """
    # 加载COCO数据
    coco = COCO(coco_json_path)
    
    # 获取所有类别
    all_categories = coco.loadCats(coco.getCatIds())
    all_category_names = [cat['name'] for cat in all_categories]
    
    # 获取所有图片
    all_img_ids = coco.getImgIds()
    
    # 采样图片
    if num_samples is not None and num_samples < len(all_img_ids):
        img_ids = random.sample(all_img_ids, num_samples)
    else:
        img_ids = all_img_ids
    
    # 转换数据
    success = 0
    with open(output_jsonl_path, 'w', encoding='utf-8') as f:
        for idx, img_id in tqdm(enumerate(img_ids), total=len(img_ids)):
            img_info = coco.loadImgs(img_id)[0]
            
            # 获取该图片的GT类别
            ann_ids = coco.getAnnIds(imgIds=img_id)
            annotations = coco.loadAnns(ann_ids)
            gt_category_ids = set(ann['category_id'] for ann in annotations)
            gt_categories = coco.loadCats(gt_category_ids)
            gt_category_names = [cat['name'] for cat in gt_categories]

            if len(gt_category_names) == 0:
                print(f"image {img_info['file_name']} has no gt, skip.")
                continue
            
            # 创建对话数据
            conversation = {
                "id": success,
                "image": img_info['file_name'],
                "conversations": []
            }
            
            # 生成多轮对话
            turns = random.randint(1, max_turns)
            for turn in range(turns):
                # 每轮独立选择类别
                selected_categories = select_categories(all_category_names, gt_category_names)
                
                # 格式化类别
                class_str = ", ".join(selected_categories)
                ref_class_str = ", ".join([f"<ref>{cat}</ref>" for cat in selected_categories])
                
                # 随机选择问题和答案模板
                question_template = random.choice(QUESTIONS)
                answer_template = random.choice(ANSWERS)
                
                # 替换模板中的占位符，只在第一轮问题前添加<image>标签
                if turn == 0:
                    question = "<image>\n" + question_template.replace("<class>", class_str)
                else:
                    question = question_template.replace("<class>", class_str)
                
                answer = answer_template.replace("<class>", ref_class_str)
                
                # 添加到对话中
                conversation["conversations"].append({"from": "human", "value": question})
                conversation["conversations"].append({"from": "gpt", "value": answer})
            
            # 写入JSONL文件
            f.write(json.dumps(conversation) + '\n')
            success += 1
    
    print(f"转换完成! 已将 {success} 张图片转换为LLAVA对话格式并保存到 {output_jsonl_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert COCO format to LLAVA conversation format')
    parser.add_argument('--input_path', '-i', required=True, help='Path to COCO format JSON file')
    parser.add_argument('--output_path', '-o', required=True, help='Path to output JSONL file')
    parser.add_argument('--samples', '-s', type=int, default=1000, help='Number of samples to convert (default: 1000)')
    parser.add_argument('--max-turns', type=int, default=3, help='Maximum number of conversation turns per image (default: 3)')
    
    args = parser.parse_args()
    
    coco_to_llava(
        coco_json_path=args.input_path,
        output_jsonl_path=args.output_path,
        num_samples=args.samples,
        max_turns=args.max_turns
    )
