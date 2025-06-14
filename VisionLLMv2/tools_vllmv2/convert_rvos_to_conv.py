import json
import random
import os
import argparse
from pycocotools.ytvos import YTVOS
from tqdm import tqdm

from prompts_rvos import QUESTIONS, ANSWERS

def select_expression(video_info):
    """随机选择视频中的一个表达式"""
    expressions = video_info.get('expressions', [])
    if not expressions:
        return None
    return random.choice(expressions)

def rvos_to_llava(rvos_json_path, output_jsonl_path, num_samples=None, max_turns=3):
    """
    将RVOS格式的JSON数据转换为LLAVA对话格式的JSONL数据
    
    Args:
        rvos_json_path: RVOS格式的JSON文件路径
        output_jsonl_path: 输出的JSONL文件路径
        num_samples: 要采样的视频数量，None表示使用所有视频
        max_turns: 每轮对话的最大轮数
    """
    # 加载RVOS数据（使用YTVOS类因为格式已转换）
    rvos = YTVOS(rvos_json_path)
    
    # 获取所有视频
    all_video_ids = rvos.getVidIds()
    
    # 采样视频，支持重复采样
    if num_samples is not None:
        if num_samples <= len(all_video_ids):
            # 样本数小于等于总视频数，不重复采样
            video_ids = random.sample(all_video_ids, num_samples)
        else:
            # 样本数大于总视频数，允许重复采样
            video_ids = []
            # 先添加所有视频ID
            video_ids.extend(all_video_ids)
            # 再随机添加剩余需要的视频ID
            remaining = num_samples - len(all_video_ids)
            video_ids.extend(random.choices(all_video_ids, k=remaining))
    else:
        video_ids = all_video_ids
    
    # 转换数据
    success = 0
    with open(output_jsonl_path, 'w', encoding='utf-8') as f:
        for idx, video_id in tqdm(enumerate(video_ids), total=len(video_ids)):
            video_info = rvos.loadVids(video_id)[0]
            
            # 获取该视频的所有帧
            frames = video_info['file_names']
            
            # 以0.5的概率选择第一帧，以0.5的概率选择任意一帧
            if random.random() < 0.5:
                selected_frame = frames[0]
            else:
                selected_frame = random.choice(frames)
            
            # 获取该视频的表达式
            expressions = video_info.get('expressions', [])
            
            if len(expressions) == 0:
                print(f"video {video_info['file_names'][0]} has no expressions, skip.")
                continue
            
            # 创建对话数据
            conversation = {
                "id": success,
                "image": selected_frame,
                "conversations": []
            }
            
            # 生成多轮对话
            turns = random.randint(1, max_turns)
            for turn in range(turns):
                # 随机选择一个表达式
                expression = select_expression(video_info)
                if not expression:
                    continue
                
                # 随机选择问题和答案模板
                question_template = random.choice(QUESTIONS)
                answer_template = random.choice(ANSWERS)
                
                # 替换模板中的占位符，只在第一轮问题前添加<image>标签
                if turn == 0:
                    question = "<image>\n" + question_template.replace("<expression>", expression)
                else:
                    question = question_template.replace("<expression>", expression)
                
                ref_expression = f"<ref>{expression}</ref>"
                answer = answer_template.replace("<expression>", ref_expression)
                
                # 添加到对话中
                conversation["conversations"].append({"from": "human", "value": question})
                conversation["conversations"].append({"from": "gpt", "value": answer})
            
            # 写入JSONL文件
            f.write(json.dumps(conversation) + '\n')
            success += 1
    
    print(f"转换完成! 已将 {success} 个视频转换为LLAVA对话格式并保存到 {output_jsonl_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert RVOS format to LLAVA conversation format')
    parser.add_argument('--input_path', '-i', required=True, help='Path to RVOS format JSON file')
    parser.add_argument('--output_path', '-o', required=True, help='Path to output JSONL file')
    parser.add_argument('--samples', '-s', type=int, default=1000, help='Number of samples to convert (default: 1000)')
    parser.add_argument('--max-turns', type=int, default=3, help='Maximum number of conversation turns per video (default: 3)')
    
    args = parser.parse_args()
    
    rvos_to_llava(
        rvos_json_path=args.input_path,
        output_jsonl_path=args.output_path,
        num_samples=args.samples,
        max_turns=args.max_turns
    )
