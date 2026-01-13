

import os
import sys
import time
import json
import vllm
import traceback
from dataclasses import dataclass
from typing import Literal
import aiohttp
import numpy as np
from tqdm.asyncio import tqdm
from transformers import PreTrainedTokenizerBase
from vllm.benchmarks.datasets import SampleRequest

from vllm.benchmarks.lib.endpoint_request_func import (_update_payload_common,
                                                       RequestFuncInput,_validate_api_url,_get_chat_content,
                                                       StreamedResponseHandler,_update_headers_common)


vllm.benchmarks.lib.endpoint_request_func.async_request_openai_chat_completions = async_request_openai_chat_completions
from vllm.benchmarks.datasets import get_samples as get_samples_old
def get_samples(args, tokenizer):
    from vllm_omni.benchmarks.datasets.random_multi_modal_dataset import OmniRandomMultiModalDataset
    if args.dataset_name == "random-mm":
        if args.backend not in [
            "openai-chat"]:
            raise ValueError(
                "Multi-modal content (images) is only supported on "
                "'openai-chat' backend."
            )
        dataset = OmniRandomMultiModalDataset(
            random_seed=args.seed, dataset_path=args.dataset_path
        )
        input_requests = dataset.sample(
            tokenizer=tokenizer,
            num_requests=args.num_prompts,
            prefix_len=args.random_prefix_len,
            range_ratio=args.random_range_ratio,
            input_len=args.random_input_len,
            output_len=args.random_output_len,
            base_items_per_request=args.random_mm_base_items_per_request,
            limit_mm_per_prompt=args.random_mm_limit_mm_per_prompt,
            num_mm_items_range_ratio=args.random_mm_num_mm_items_range_ratio,
            bucket_config=args.random_mm_bucket_config,
            request_id_prefix=args.request_id_prefix,
            no_oversample=args.no_oversample,
        )
        return input_requests
    else:
        return get_samples_old(args, tokenizer)
vllm.benchmarks.datasets.get_samples = get_samples

from vllm.benchmarks.lib.endpoint_request_func import RequestFuncOutput as RequestFuncOutput_old

@dataclass
class RequestFuncOutput(RequestFuncOutput_old):
    audio_ttft: float = 0.0

vllm.benchmarks.lib.endpoint_request_func.RequestFuncOutput = RequestFuncOutput

async def async_request_openai_chat_completions(
    request_func_input: RequestFuncInput,
    session: aiohttp.ClientSession,
    pbar: tqdm | None = None,
    mm_position: Literal["first", "last"] = "last",
) -> RequestFuncOutput:
    api_url = request_func_input.api_url
    _validate_api_url(api_url, "OpenAI Chat Completions API", "chat/completions")

    content = _get_chat_content(request_func_input, mm_position=mm_position)

    payload = {
        "model": request_func_input.model_name
        if request_func_input.model_name
        else request_func_input.model,
        "messages": [
            {"role": "user", "content": content},
        ],
        "temperature": 0.0,
        "max_completion_tokens": request_func_input.output_len,
        "stream": True,
        "stream_options": {
            "include_usage": True,
        },
    }
    _update_payload_common(payload, request_func_input)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
    }
    _update_headers_common(headers, request_func_input)

    output = RequestFuncOutput()
    output.prompt_len = request_func_input.prompt_len

    generated_text = ""
    ttft = 0.0
    st = time.perf_counter()
    output.start_time = st
    most_recent_timestamp = st
    try:
        async with session.post(url=api_url, json=payload, headers=headers) as response:
            if response.status == 200:
                handler = StreamedResponseHandler()
                async for chunk_bytes in response.content.iter_any():
                    chunk_bytes = chunk_bytes.strip()
                    if not chunk_bytes:
                        continue

                    messages = handler.add_chunk(chunk_bytes)
                    for message in messages:
                        # NOTE: SSE comments (often used as pings) start with
                        # a colon. These are not JSON data payload and should
                        # be skipped.
                        if message.startswith(":"):
                            continue

                        chunk = message.removeprefix("data: ")

                        if chunk != "[DONE]":
                            timestamp = time.perf_counter()
                            data = json.loads(chunk)

                            if choices := data.get("choices"):
                                content = choices[0]["delta"].get("content")
                                # First token
                                if ttft == 0.0:
                                    ttft = timestamp - st
                                    output.ttft = ttft

                                # Decoding phase
                                else:
                                    modality = data.get("modality")
                                    if modality == "text":
                                        output.itl.append(timestamp - most_recent_timestamp)
                                    elif modality == "audio":
                                        output.audio_ttft = timestamp - most_recent_timestamp

                                generated_text += content or ""
                            elif usage := data.get("usage"):
                                output.output_tokens = usage.get("completion_tokens")

                            most_recent_timestamp = timestamp

                output.generated_text = generated_text
                output.success = True
                output.latency = most_recent_timestamp - st
            else:
                output.error = response.reason or ""
                output.success = False
    except Exception:
        output.success = False
        exc_info = sys.exc_info()
        output.error = "".join(traceback.format_exception(*exc_info))

    if pbar:
        pbar.update(1)
    return output


from vllm.benchmarks.serve import BenchmarkMetrics as BenchmarkMetrics_old
@dataclass
class BenchmarkMetrics(BenchmarkMetrics_old):
    mean_audio_ttft_ms: float
    median_audio_ttft_ms: float
    std_audio_ttft_ms: float
vllm.benchmarks.serve.BenchmarkMetrics = BenchmarkMetrics


from vllm.benchmarks.serve import calculate_metrics as calculate_metrics_old
def calculate_metrics(
    input_requests: list[SampleRequest],
    outputs: list[RequestFuncOutput],
    dur_s: float,
    tokenizer: PreTrainedTokenizerBase,
    selected_percentiles: list[float],
    goodput_config_dict: dict[str, float],
):
    audio_ttfts = []
    result = calculate_metrics_old(input_requests, outputs, dur_s, tokenizer, selected_percentiles, goodput_config_dict)
    for i in range(len(outputs)):
        if outputs[i] is not None and outputs[i].success:
            audio_ttfts.append(outputs[i].audio_ttft)
    mean_ttft_ms = np.mean(audio_ttfts or 0) * 1000
    std_ttft_ms = np.std(audio_ttfts or 0) * 1000
    median_ttft_ms = np.median(audio_ttfts or 0) * 1000
    percentiles_ttft_ms = [(p, np.percentile(audio_ttfts or 0, p) * 1000)
                           for p in selected_percentiles]

    print("{s:{c}^{n}}".format(s=' Supplemental result ', n=50, c='='))
    print("{s:{c}^{n}}".format(s="Time to audio First Token", n=50, c='-'))
    print("{:<40} {:<10.2f}".format(
        f"Mean Audio TTFT  (ms):",mean_ttft_ms))
    print("{:<40} {:<10.2f}".format(
        f"Median Audio TTFT (ms):",median_ttft_ms))
    result.mean_audio_ttft_ms = mean_ttft_ms
    result.median_audio_ttft_ms = median_ttft_ms
    result.std_audio_ttft_ms = std_ttft_ms
    for p, value in percentiles_ttft_ms:
        p_word = str(int(p)) if int(p) == p else str(p)
        print("{:<40} {:<10.2f}".format(f"P{p_word} Audio TTFT (ms):",
                                        value))
        result[f"p{p_word}_audio_ttft_ms"] = value
    print("=" * 50)
    return result
vllm.benchmarks.serve.calculate_metrics = calculate_metrics

