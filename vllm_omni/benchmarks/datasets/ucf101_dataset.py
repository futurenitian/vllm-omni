import os
import json
from typing import List, Dict, Any
from datasets import load_dataset
from transformers import PreTrainedTokenizerBase
from vllm.benchmarks.datasets import SampleRequest, process_video
import logging

logger = logging.getLogger(__name__)

def get_ucf101_samples(args, tokenizer) -> List[SampleRequest]:
    split = getattr(args, 'ucf101_split', 'train')
    subset_size = getattr(args, 'ucf101_subset_size', 100)
    text_template = getattr(args, 'ucf101_text_template', 
                          'Describe the action in this video: {}')
    
    dataset_root = "/data2/fwl/vllm-omni-benchmark/vllm-omni/vllm_omni/benchmarks/datasets"
    print(f"Loading UCF101 dataset from {dataset_root} (split: {split}, subset: {subset_size})...")
    
    try:
        dataset = load_dataset(
            "videofolder",
            data_dir=dataset_root,
            split=split,
            cache_dir=None
        )
    
        if subset_size and subset_size < len(dataset):
            dataset = dataset.select(range(subset_size))  
        
        input_requests = []
        for i in range(args.num_prompts):
            idx = i % len(dataset)
            sample = dataset[idx]
            
            video_bytes = extract_video_bytes(sample, dataset_root)
            label = extract_label(sample)
            
            if video_bytes:
                video_content = process_video({'bytes': video_bytes})
                text_content = {
                    "type": "text",
                    "text": text_template.format(label)
                }
                
                multimodal_content = [video_content, text_content]
                content_json = json.dumps(multimodal_content)
                
                prompt_len = len(tokenizer.encode(content_json))

                request = SampleRequest(
                    request_id=f"{args.request_id_prefix}_{i:06d}",
                    inputs=content_json,
                    prompt_len=prompt_len,
                    output_len=args.random_output_len 
                )
                
                input_requests.append(request)
            else:
                print(f"Warning: No video bytes found for sample {idx}")
        
        print(f"Generated {len(input_requests)} UCF101 requests")
        return input_requests
        
    except Exception as e:
        raise RuntimeError(f"Failed to load UCF101 dataset from {dataset_root}: {e}")

def extract_video_bytes(sample: Dict[str, Any], dataset_root: str) -> bytes:
    """
    从样本中提取视频字节（适配本地绝对路径的数据集）
    :param sample: 数据集样本
    :param dataset_root: 数据集根路径（/data2/fwl/.../datasets）
    :return: 视频字节流
    """
    # 情况1：样本中直接有视频字节
    if isinstance(sample.get('video'), bytes):
        return sample['video']
    elif isinstance(sample.get('video'), dict) and 'bytes' in sample['video']:
        return sample['video']['bytes']
    
    # 情况2：样本中只有相对路径，拼接绝对路径后读取
    video_rel_path = sample.get('video_path') or sample.get('file')
    if video_rel_path:
        video_abs_path = os.path.join(dataset_root, video_rel_path)
        if os.path.exists(video_abs_path):
            with open(video_abs_path, 'rb') as f:
                return f.read()
    
    # 情况3：直接从绝对路径读取（若样本中存的是绝对路径）
    video_abs_path = sample.get('video_abs_path')
    if video_abs_path and os.path.exists(video_abs_path):
        with open(video_abs_path, 'rb') as f:
            return f.read()
    
    return b''

def extract_label(sample: Dict[str, Any]) -> str:
    label_idx = sample.get('label', 0)
    return f"action_{label_idx:03d}"