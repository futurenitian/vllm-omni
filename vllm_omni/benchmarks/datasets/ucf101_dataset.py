import os
import json
from typing import List, Dict, Any
from datasets import load_dataset
from transformers import PreTrainedTokenizerBase
from vllm.benchmarks.datasets import SampleRequest, process_video

logger = logging.getLogger(__name__)

def get_ucf101_samples(args, tokenizer) -> List[SampleRequest]:
    """
    获取UCF101数据集样本
    参数结构与random-mm保持一致
    """
    # 解析参数（使用与random-mm类似的命名）
    split = getattr(args, 'ucf101_split', 'train')
    subset_size = getattr(args, 'ucf101_subset_size', 100)
    text_template = getattr(args, 'ucf101_text_template', 
                          'Describe the action in this video: {}')
    
    # 加载HF数据集
    print(f"Loading UCF101 dataset (split: {split}, subset: {subset_size})...")
    
    try:
        dataset = load_dataset(
            "r-shar/ucf101",
            split=split,
            cache_dir=args.dataset_path,  # 使用相同的dataset_path作为缓存
        )
        
        # 限制样本数（简单处理：取前N个）
        if subset_size and subset_size < len(dataset):
            dataset = dataset.select(range(subset_size))
        
        # 生成样本请求
        input_requests = []
        for i in range(args.num_prompts):
            # 循环使用数据集
            idx = i % len(dataset)
            sample = dataset[idx]
            
            # 提取视频路径（简化处理）
            video_path = extract_video_path(sample)
            label = extract_label(sample)
            
            # 创建多模态内容
            if video_path and os.path.exists(video_path):
                # 读取视频文件
                with open(video_path, 'rb') as f:
                    video_bytes = f.read()
                
                # 使用现有的process_video函数
                video_content = process_video({'bytes': video_bytes})
                text_content = {
                    "type": "text",
                    "text": text_template.format(label)
                }
                
                # 组合内容
                multimodal_content = [video_content, text_content]
                content_json = json.dumps(multimodal_content)
                
                # 估算token长度
                prompt_len = len(tokenizer.encode(content_json))
                
                # 创建请求（与random-mm保持相同结构）
                request = SampleRequest(
                    request_id=f"{args.request_id_prefix}_{i:06d}",
                    inputs=content_json,
                    prompt_len=prompt_len,
                    output_len=args.random_output_len  # 使用相同的output_len参数
                )
                
                input_requests.append(request)
            else:
                print(f"Warning: Video file not found for sample {idx}")
        
        print(f"Generated {len(input_requests)} UCF101 requests")
        return input_requests
        
    except Exception as e:
        raise RuntimeError(f"Failed to load UCF101 dataset: {e}")


def extract_video_path(sample: Dict[str, Any]) -> str:
    """从样本中提取视频路径（简化版）"""
    # UCF101数据集的常见格式
    if isinstance(sample.get('video'), dict):
        return sample['video'].get('path', '')
    elif isinstance(sample.get('video'), str):
        return sample['video']
    elif isinstance(sample.get('file'), str):
        return sample['file']
    return ''


def extract_label(sample: Dict[str, Any]) -> str:
    """从样本中提取标签"""
    label_idx = sample.get('label', 0)
    # 简化：直接使用数字标签
    return f"action_{label_idx:03d}"