import argparse
import asyncio
import importlib.util
import shutil
import sys
from typing import Any, Literal

from vllm_omni.benchmarks.lib.endpoint_request_func import (
    OPENAI_COMPATIBLE_BACKENDS,ASYNC_REQUEST_FUNCS, MixRequestFuncOutput)


MILLISECONDS_TO_SECONDS_CONVERSION = 1000

TERM_PLOTLIB_AVAILABLE = ((importlib.util.find_spec("termplotlib") is not None)
                          and (shutil.which("gnuplot") is not None))

def main(args: argparse.Namespace) -> dict[str, Any]:
    original_gather = asyncio.gather
    original_functions = {}
    FUNCTIONS_TO_REPLACE = {
        'ASYNC_REQUEST_FUNCS': ASYNC_REQUEST_FUNCS,
        'get_sample': get_sample,
        'caculate_metrics': caculate_metrics
    }
    try:
        import vllm.benchmarks.serve as benchmark_module
        # Replace the ASYNC_REQUEST_FUNCS
        for func_name, new_func in FUNCTIONS_TO_REPLACE.items():
            if hasattr(benchmark_module, func_name):
                original_functions[func_name] = getattr(benchmark_module, func_name)
                setattr(benchmark_module, func_name, new_func)

        for name, mod in sys.modules.items():
            if mod.__name__ == benchmark_module.__name__:
                for func_name, new_func in FUNCTIONS_TO_REPLACE.items():
                    if hasattr(mod, func_name):
                        setattr(mod, func_name, new_func)

    except ImportError as e:
        print(f"import error: {e}")
        raise

    try:
        from vllm.benchmarks.serve import main_async as original_main
        original_result = asyncio.run(original_main(args))
        return original_result

    finally:
        asyncio.gather = original_gather
        for func_name, original_func in original_functions.items():
            if hasattr(benchmark_module, func_name):
                setattr(benchmark_module, func_name, original_func)

        for name, mod in sys.modules.items():
            if mod.__name__ == benchmark_module.__name__:
                for func_name, original_func in original_functions.items():
                    if hasattr(mod, func_name):
                        setattr(mod, func_name, original_func)
