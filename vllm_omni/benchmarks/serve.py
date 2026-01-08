# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
r"""Benchmark online serving throughput.

On the server side, run one of the following commands
to launch the vLLM OpenAI API server:
    vllm-omni serve <your_model> <engine arguments>

On the client side, run:
    vllm-omni bench serve \
        --backend <backend or endpoint type. Default 'openai'> \
        --label <benchmark result label. Default using backend> \
        --model <your_model> \
        --dataset-name <dataset_name. Default 'random'> \
        --request-rate <request_rate. Default inf> \
        --num-prompts <num_prompts. Default 1000>
"""
import argparse
import asyncio
import gc
import importlib.util
import json
import os
import random
import shutil
import sys
from datetime import datetime
from typing import Any, Literal, Optional, List, Iterable
from transformers import PreTrainedTokenizerBase
import numpy as np

from vllm_omni.benchmarks.lib.endpoint_request_func import (
    OPENAI_COMPATIBLE_BACKENDS,ASYNC_REQUEST_FUNCS, MixRequestFuncOutput)

from vllm.benchmarks.serve import (BenchmarkMetrics,EmbedBenchmarkMetrics,calculate_metrics,
                                   _get_current_request_rate, calculate_metrics_for_embeddings,
                                   TaskType,get_request,check_goodput_args,get_first_model_from_server,
                                   save_to_pytorch_benchmark_format,add_cli_args)
from vllm_omni.benchmarks.datasets import get_omni_samples
from vllm.benchmarks.datasets import SampleRequest
from vllm.transformers_utils.tokenizer import get_tokenizer
from vllm.utils.network_utils import join_host_port
from vllm.utils.gc_utils import freeze_gc_heap


MILLISECONDS_TO_SECONDS_CONVERSION = 1000

TERM_PLOTLIB_AVAILABLE = ((importlib.util.find_spec("termplotlib") is not None)
                          and (shutil.which("gnuplot") is not None))

async def patched_metrics(
    outputs: list[MixRequestFuncOutput],
    dur_s: float):
    audio_completed = 0
    result = {}
    for i in range(len(outputs)):
        if outputs[i].success:
            audio_completed += outputs[i].output_audio_num
    result.audio_throughput = audio_completed / dur_s
    return result

async def patched_benchmark(
    task_type: TaskType,
    endpoint_type: str,
    api_url: str,
    base_url: str,
    model_id: str,
    model_name: str,
    tokenizer: PreTrainedTokenizerBase,
    input_requests: list[SampleRequest],
    logprobs: Optional[int],
    request_rate: float,
    burstiness: float,
    disable_tqdm: bool,
    num_warmups: int,
    profile: bool,
    selected_percentile_metrics: list[str],
    selected_percentiles: list[float],
    ignore_eos: bool,
    goodput_config_dict: dict[str, float],
    max_concurrency: Optional[int],
    lora_modules: Optional[Iterable[str]],
    extra_headers: Optional[dict],
    extra_body: Optional[dict],
    ramp_up_strategy: Optional[Literal["linear", "exponential"]] = None,
    ramp_up_start_rps: Optional[int] = None,
    ramp_up_end_rps: Optional[int] = None,
    ready_check_timeout_sec: int = 600,
):
    converted_outputs: List[Any] = []
    original_gather = asyncio.gather
    original_async_funcs = {}
    try:
        import vllm.benchmarks.serve as benchmark_module
        if hasattr(benchmark_module, 'ASYNC_REQUEST_FUNCS'):
            original_async_funcs['benchmark_module'] = benchmark_module.ASYNC_REQUEST_FUNCS
            benchmark_module.ASYNC_REQUEST_FUNCS = ASYNC_REQUEST_FUNCS
            for name, mod in sys.modules.items():
                if hasattr(mod, 'ASYNC_REQUEST_FUNCS') and mod.__name__ == benchmark_module.__name__:
                    mod.ASYNC_REQUEST_FUNCS = ASYNC_REQUEST_FUNCS

    except ImportError as e:
        print(f"import error: {e}")
        raise

    async def intercepted_gather(*tasks, **gather_kwargs):
        original_results = await original_gather(*tasks, **gather_kwargs)
        converted_outputs.extend(original_results)
        return original_results
    asyncio.gather = intercepted_gather

    try:
        from vllm.benchmarks.serve import benchmark as original_benchmark
        original_result = await original_benchmark(task_type=task_type,endpoint_type=endpoint_type,api_url=api_url,base_url=base_url,
                                                   model_id=model_id,model_name=model_name,tokenizer=tokenizer,
                                                   input_requests=input_requests,logprobs=logprobs,
                                                   request_rate=request_rate,burstiness=burstiness,
                                                   disable_tqdm=disable_tqdm,profile=profile,
                                                   num_warmups=num_warmups,
    selected_percentile_metrics=selected_percentile_metrics,selected_percentiles=selected_percentiles,
                                                   ignore_eos=ignore_eos,goodput_config_dict=goodput_config_dict,
                                                   max_concurrency=max_concurrency,
    lora_modules=lora_modules,extra_headers=extra_headers,extra_body=extra_body,
                                                   ramp_up_strategy=ramp_up_strategy,ramp_up_start_rps=ramp_up_start_rps,
    ramp_up_end_rps=ramp_up_end_rps,ready_check_timeout_sec=ready_check_timeout_sec)
        return original_result, converted_outputs

    finally:
        asyncio.gather = original_gather
        if 'benchmark_module' in original_async_funcs:
            import vllm.benchmarks.serve as benchmark_module
            benchmark_module.ASYNC_REQUEST_FUNCS = original_async_funcs['benchmark_module']
            for name, mod in sys.modules.items():
                if hasattr(mod, 'ASYNC_REQUEST_FUNCS') and mod.__name__ == benchmark_module.__name__:
                    mod.ASYNC_REQUEST_FUNCS = original_async_funcs['benchmark_module']

def main(args: argparse.Namespace) -> dict[str, Any]:
    return asyncio.run(main_async(args))

async def main_async(args: argparse.Namespace) -> dict[str, Any]:
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Validate ramp-up arguments
    if args.ramp_up_strategy is not None:
        if args.request_rate != float("inf"):
            raise ValueError(
                "When using ramp-up, do not specify --request-rate. "
                "The request rate will be controlled by ramp-up parameters. "
                "Please remove the --request-rate argument."
            )
        if args.ramp_up_start_rps is None or args.ramp_up_end_rps is None:
            raise ValueError(
                "When using --ramp-up-strategy, both --ramp-up-start-rps and "
                "--ramp-up-end-rps must be specified"
            )
        if args.ramp_up_start_rps < 0 or args.ramp_up_end_rps < 0:
            raise ValueError("Ramp-up start and end RPS must be non-negative")
        if args.ramp_up_start_rps > args.ramp_up_end_rps:
            raise ValueError("Ramp-up start RPS must be less than end RPS")
        if args.ramp_up_strategy == "exponential" and args.ramp_up_start_rps == 0:
            raise ValueError("For exponential ramp-up, the start RPS cannot be 0.")

    label = args.label

    if args.base_url is not None:
        api_url = f"{args.base_url}{args.endpoint}"
        base_url = f"{args.base_url}"
    else:
        host_port = join_host_port(args.host, args.port)
        api_url = f"http://{host_port}{args.endpoint}"
        base_url = f"http://{host_port}"

    # Headers
    headers = None
    if args.header:
        headers = {}
        for item in args.header:
            if "=" in item:
                kvstring = item.split("=", 1)
                headers[kvstring[0].strip()] = kvstring[1].strip()
            else:
                raise ValueError("Invalid header format. Please use KEY=VALUE format.")

    # Fetch model from server if not specified
    if args.model is None:
        print("Model not specified, fetching first model from server...")
        model_name, model_id = await get_first_model_from_server(base_url, headers)
        print(f"First model name: {model_name}, first model id: {model_id}")
    else:
        model_name = args.served_model_name
        model_id = args.model

    tokenizer_id = args.tokenizer if args.tokenizer is not None else model_id
    tokenizer_mode = args.tokenizer_mode

    tokenizer = get_tokenizer(
        tokenizer_id,
        tokenizer_mode=tokenizer_mode,
        trust_remote_code=args.trust_remote_code,
    )

    if args.dataset_name is None:
        raise ValueError(
            "Please specify '--dataset-name' and the corresponding "
            "'--dataset-path' if required."
        )

    # Map general --input-len and --output-len to all dataset-specific arguments
    if args.input_len is not None:
        args.random_input_len = args.input_len
        args.sonnet_input_len = args.input_len

    if args.output_len is not None:
        args.random_output_len = args.output_len
        args.sonnet_output_len = args.output_len
        args.sharegpt_output_len = args.output_len
        args.custom_output_len = args.output_len
        args.hf_output_len = args.output_len
        args.spec_bench_output_len = args.output_len
        args.prefix_repetition_output_len = args.output_len

    # when using random datasets, default to ignoring EOS
    # so generation runs to the requested length
    if (
            args.dataset_name in ("random", "random-mm")
            and args.backend in OPENAI_COMPATIBLE_BACKENDS
    ):
        args.ignore_eos = True

    # Load the dataset.
    input_requests = get_omni_samples(args, tokenizer)
    goodput_config_dict = check_goodput_args(args)

    backend = args.backend
    task_type = (
        TaskType.POOLING
        if "embeddings" in backend or "rerank" in backend
        else TaskType.GENERATION
    )

    # Collect the sampling parameters.
    if task_type == TaskType.GENERATION:
        sampling_params = {
            k: v
            for k, v in {
            "top_p": args.top_p,
            "top_k": args.top_k,
            "min_p": args.min_p,
            "temperature": args.temperature,
            "frequency_penalty": args.frequency_penalty,
            "presence_penalty": args.presence_penalty,
            "repetition_penalty": args.repetition_penalty,
        }.items()
            if v is not None
        }

        # Sampling parameters are only supported by openai-compatible backend.
        if sampling_params and args.backend not in OPENAI_COMPATIBLE_BACKENDS:
            raise ValueError(
                "Sampling parameters are only supported by openai-compatible backends."
            )

        if "temperature" not in sampling_params:
            sampling_params["temperature"] = 0.0  # Default to greedy decoding.

        default_percentile_metrics = "ttft,tpot,itl"
    else:
        sampling_params = {}
        default_percentile_metrics = "e2el"

    extra_body = args.extra_body or {}
    extra_body = {**sampling_params, **extra_body}

    percentile_metrics: str = args.percentile_metrics or default_percentile_metrics

    # Avoid GC processing "static" data - reduce pause times.
    freeze_gc_heap()

    benchmark_result, outputs = await patched_benchmark(
        task_type=task_type,
        endpoint_type=backend,
        api_url=api_url,
        base_url=base_url,
        model_id=model_id,
        model_name=model_name,
        tokenizer=tokenizer,
        input_requests=input_requests,
        logprobs=args.logprobs,
        request_rate=args.request_rate,
        burstiness=args.burstiness,
        disable_tqdm=args.disable_tqdm,
        num_warmups=args.num_warmups,
        profile=args.profile,
        selected_percentile_metrics=percentile_metrics.split(","),
        selected_percentiles=[float(p) for p in args.metric_percentiles.split(",")],
        ignore_eos=args.ignore_eos,
        goodput_config_dict=goodput_config_dict,
        max_concurrency=args.max_concurrency,
        lora_modules=args.lora_modules,
        extra_headers=headers,
        extra_body=extra_body,
        ramp_up_strategy=args.ramp_up_strategy,
        ramp_up_start_rps=args.ramp_up_start_rps,
        ramp_up_end_rps=args.ramp_up_end_rps,
        ready_check_timeout_sec=args.ready_check_timeout_sec,
    )
    patched_result = patched_metrics(outputs, benchmark_result['duration'])
    print("{s:{c}^{n}}".format(s=' Supplemental result ', n=50, c='='))
    print("{:<40} {:<10.2f}".format("Audio throughput (num/s):",
                                    patched_result['audio_throughput']))
    print("=" * 50)
    # Save config and results to json
    result_json: dict[str, Any] = {}

    # Setup
    current_dt = datetime.now().strftime("%Y%m%d-%H%M%S")
    result_json["date"] = current_dt
    result_json["endpoint_type"] = args.backend  # for backward compatibility
    result_json["backend"] = args.backend
    result_json["label"] = label
    result_json["model_id"] = model_id
    result_json["tokenizer_id"] = tokenizer_id
    result_json["num_prompts"] = args.num_prompts

    # Metadata
    if args.metadata:
        for item in args.metadata:
            if "=" in item:
                kvstring = item.split("=", 1)
                result_json[kvstring[0].strip()] = kvstring[1].strip()
            else:
                raise ValueError(
                    "Invalid metadata format. Please use KEY=VALUE format."
                )

    # Traffic
    result_json["request_rate"] = (
        args.request_rate if args.request_rate < float("inf") else "inf"
    )
    result_json["burstiness"] = args.burstiness
    result_json["max_concurrency"] = args.max_concurrency

    if args.ramp_up_strategy is not None:
        result_json["ramp_up_strategy"] = args.ramp_up_strategy
        result_json["ramp_up_start_rps"] = args.ramp_up_start_rps
        result_json["ramp_up_end_rps"] = args.ramp_up_end_rps

    # Merge with benchmark result
    result_json = {**result_json, **benchmark_result, **patched_result}

    if not args.save_detailed:
        # Remove fields with too many data points
        for field in [
            "input_lens",
            "output_lens",
            "ttfts",
            "itls",
            "generated_texts",
            "errors",
        ]:
            if field in result_json:
                del result_json[field]
            if field in benchmark_result:
                del benchmark_result[field]

        # Save to file
    if args.save_result or args.append_result:
        base_model_id = model_id.split("/")[-1]
        max_concurrency_str = (
            f"-concurrency{args.max_concurrency}"
            if args.max_concurrency is not None
            else ""
        )
        label = label or args.backend
        if args.ramp_up_strategy is not None:
            file_name = f"{label}-ramp-up-{args.ramp_up_strategy}-{args.ramp_up_start_rps}qps-{args.ramp_up_end_rps}qps{max_concurrency_str}-{base_model_id}-{current_dt}.json"  # noqa
        else:
            file_name = f"{label}-{args.request_rate}qps{max_concurrency_str}-{base_model_id}-{current_dt}.json"  # noqa
        if args.result_filename:
            file_name = args.result_filename
        if args.result_dir:
            os.makedirs(args.result_dir, exist_ok=True)
            file_name = os.path.join(args.result_dir, file_name)
        with open(
            file_name, mode="a+" if args.append_result else "w", encoding="utf-8"
        ) as outfile:
            # Append a newline.
            if args.append_result and outfile.tell() != 0:
                outfile.write("\n")
            json.dump(result_json, outfile)
        save_to_pytorch_benchmark_format(args, result_json, file_name)

    return result_json
