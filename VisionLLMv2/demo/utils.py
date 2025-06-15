import torch
import numpy as np
from typing import List, Dict, Union, Tuple
from pycocotools import mask as mask_utils

def rle_decode(rle: Union[Dict, str], height: int = None, width: int = None) -> np.ndarray:
    """
    使用 pycocotools 解码 RLE 掩码
    
    Args:
        rle: RLE 编码对象（字典格式）或压缩的 RLE 字符串
        height: 可选，掩码高度（当 rle 是压缩字符串时需要）
        width: 可选，掩码宽度（当 rle 是压缩字符串时需要）
        
    Returns:
        二维二进制掩码数组，1表示前景，0表示背景
    """
    if isinstance(rle, str):
        if height is None or width is None:
            raise ValueError("当 RLE 是字符串格式时，必须提供 height 和 width")
        rle = mask_utils.frPyObjects(rle, height, width)
    
    # 解码 RLE
    mask = mask_utils.decode(rle)
    
    # 如果是多通道掩码，只取第一个通道
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    
    return mask

def rle_to_tensor_for_video(pred_masks: List[List[Union[Dict, str]]], 
                 height: int = None, 
                 width: int = None) -> torch.Tensor:
    """
    将 list[list[rle]] 格式的掩码转换为 PyTorch 张量
    
    Args:
        pred_masks: 嵌套列表结构，形状为 [n_obj, n_frame, rle]
        height: 可选，掩码高度（当 RLE 是字符串格式时需要）
        width: 可选，掩码宽度（当 RLE 是字符串格式时需要）
        
    Returns:
        三维 PyTorch 张量，形状为 [n_obj, n_frame, height, width]
    """
    n_obj = len(pred_masks)
    if n_obj == 0:
        raise ValueError("pred_masks 不能为空")
    
    n_frame = len(pred_masks[0])
    
    # 检测 RLE 格式（字典或字符串）
    sample_rle = next((rle for obj_masks in pred_masks for rle in obj_masks if rle is not None), None)
    if sample_rle is None:
        raise ValueError("pred_masks 中不包含有效的 RLE 数据")
    
    # 如果 RLE 是字符串格式，需要用户提供 height 和 width
    if isinstance(sample_rle, str) and (height is None or width is None):
        raise ValueError("当 RLE 是字符串格式时，必须提供 height 和 width")
    
    # 如果 RLE 是字典格式，尝试从第一个 RLE 获取尺寸
    if isinstance(sample_rle, dict):
        if 'size' in sample_rle:
            height, width = sample_rle['size']
        else:
            raise ValueError("RLE 字典中未找到 'size' 字段，请提供 height 和 width")
    
    # 初始化张量
    masks_tensor = torch.zeros((n_obj, n_frame, height, width), dtype=torch.bool)
    
    # 解码每个 RLE 掩码并填充到张量
    for obj_idx, obj_masks in enumerate(pred_masks):
        for frame_idx, rle in enumerate(obj_masks):
            if rle is None:
                # 处理空掩码情况
                continue
            
            # 解码 RLE 为 numpy 数组
            mask_np = rle_decode(rle, height, width)
            
            # 转换为 PyTorch 张量并填充到对应位置
            masks_tensor[obj_idx, frame_idx] = torch.from_numpy(mask_np).bool()
    
    return masks_tensor