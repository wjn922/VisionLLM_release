import json
import random
import os
import argparse
from pycocotools.ytvos import YTVOS
from tqdm import tqdm

from prompts_vos import QUESTIONS, ANSWERS

def vos_to_llava(vos_json_path, output_jsonl_path, num_samples=None, max_turns=3):
    """
    将VOS格式的JSON数据转换为LLAVA对话格式的JSONL数据
    
    Args:
        vos_json_path: VOS格式的JSON文件路径
        output_jsonl_path: 输出的JSONL文件路径
        num_samples: 要采样的视频数量，None表示使用所有视频
        max_turns: 每轮对话的最大轮数
    """
    # 加载VOS数据
    vos = YTVOS(vos_json_path)
    
    # 获取所有视频
    all_video_ids = vos.getVidIds()
    
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
            video_info = vos.loadVids(video_id)[0]
            
            # 获取该视频的所有帧
            frames = video_info['file_names']
            
            # 以0.5的概率选择第一帧，以0.5的概率选择任意一帧
            if random.random() < 0.5:
                selected_frame = frames[0]
            else:
                selected_frame = random.choice(frames)
            
            # 获取该视频的GT类别
            ann_ids = vos.getAnnIds(vidIds=video_id)
            annotations = vos.loadAnns(ann_ids)
            
            # 如果没有标注，跳过当前视频
            if len(annotations) == 0:
                print(f"video {video_info['file_names'][0]} has no annotations, skip.")
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
                # 随机选择问题和答案模板
                question_template = random.choice(QUESTIONS)
                answer_template = random.choice(ANSWERS)
                
                # 替换模板中的占位符，只在第一轮问题前添加<image>标签
                if turn == 0:
                    question = "<image>\n" + question_template
                else:
                    question = question_template
                
                answer = answer_template
                
                # 添加到对话中
                conversation["conversations"].append({"from": "human", "value": question})
                conversation["conversations"].append({"from": "gpt", "value": answer})
            
            # 写入JSONL文件
            f.write(json.dumps(conversation) + '\n')
            success += 1
    
    print(f"转换完成! 已将 {success} 个视频转换为LLAVA对话格式并保存到 {output_jsonl_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert VOS format to LLAVA conversation format')
    parser.add_argument('--input_path', '-i', required=True, help='Path to VOS format JSON file')
    parser.add_argument('--output_path', '-o', required=True, help='Path to output JSONL file')
    parser.add_argument('--samples', '-s', type=int, default=1000, help='Number of samples to convert (default: 1000)')
    parser.add_argument('--max-turns', type=int, default=3, help='Maximum number of conversation turns per video (default: 3)')
    
    args = parser.parse_args()
    
    vos_to_llava(
        vos_json_path=args.input_path,
        output_jsonl_path=args.output_path,
        num_samples=args.samples,
        max_turns=args.max_turns
    )