from vllm_omni.benchmarks.datasets import OmniRandomMultiModalDataset
from vllm.benchmarks.datasets import get_samples

import os
import sys
import time
import json
import traceback
from dataclasses import dataclass
from typing import Literal
import aiohttp
from tqdm.asyncio import tqdm
from vllm.benchmarks.lib.endpoint_request_func import (async_request_openai_completions,async_request_openai_audio,
                                                       async_request_openai_embeddings, RequestFunc,_update_payload_common,
                                                       RequestFuncInput,_validate_api_url,_get_chat_content,
                                                       StreamedResponseHandler,_update_headers_common)


AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)

@dataclass
class RequestFuncOutput:
    generated_text: str = ""
    success: bool = False
    latency: float = 0.0
    output_tokens: int = 0
    ttft: float = 0.0  # Time to first token
    itl: list[float] = field(default_factory=list)  # list of inter-token latencies
    tpot: float = 0.0  # avg next-token latencies
    prompt_len: int = 0
    error: str = ""
    start_time: float = 0.0
    audio_ttft: float = 0.0


async def async_request_openai_chat_completions(
    request_func_input: RequestFuncInput,
    session: aiohttp.ClientSession,
    pbar: tqdm | None = None,
    mm_position: Literal["first", "last"] = "last",
) -> MixRequestFuncOutput:
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

    output = MixRequestFuncOutput()
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

# TODO: Add more request functions for different API protocols.
ASYNC_REQUEST_FUNCS: dict[str, RequestFunc] = {
    "vllm": async_request_openai_completions,
    "openai": async_request_openai_completions,
    "openai-chat": async_request_openai_chat_completions,
    "openai-audio": async_request_openai_audio,
    "openai-embeddings": async_request_openai_embeddings,
}

OPENAI_COMPATIBLE_BACKENDS = [
    k for k, v in ASYNC_REQUEST_FUNCS.items()
    if v in (async_request_openai_completions,
             async_request_openai_chat_completions)]


def get_samples(args, tokenizer):
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
        return get_samples(args, tokenizer)