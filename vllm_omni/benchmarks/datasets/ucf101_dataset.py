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
    

    print(f"Loading UCF101 dataset (split: {split}, subset: {subset_size})...")
    
    try:
        dataset = load_dataset(
            "r-shar/ucf101",
            split=split,
            cache_dir=args.dataset_path 
        )
        
        if subset_size and subset_size < len(dataset):
            dataset = dataset.select(range(subset_size))
        
        input_requests = []
        for i in range(args.num_prompts):
            idx = i % len(dataset)
            sample = dataset[idx]
            
            video_path = extract_video_path(sample)
            label = extract_label(sample)
            
            if video_path and os.path.exists(video_path):
                with open(video_path, 'rb') as f:
                    video_bytes = f.read()
                
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
                print(f"Warning: Video file not found for sample {idx}")
        
        print(f"Generated {len(input_requests)} UCF101 requests")
        return input_requests
        
    except Exception as e:
        raise RuntimeError(f"Failed to load UCF101 dataset: {e}")


def extract_video_path(sample: Dict[str, Any]) -> str:
    if isinstance(sample.get('video'), dict):
        return sample['video'].get('path', '')
    elif isinstance(sample.get('video'), str):
        return sample['video']
    elif isinstance(sample.get('file'), str):
        return sample['file']
    return ''


def extract_label(sample: Dict[str, Any]) -> str:
    label_idx = sample.get('label', 0)
    return f"action_{label_idx:03d}"