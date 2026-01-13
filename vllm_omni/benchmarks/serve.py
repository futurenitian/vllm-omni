import argparse
import asyncio
from typing import Any, Literal
from vllm_omni.benchmarks.patch import patch
from vllm.benchmarks.serve import main_async

def main(args: argparse.Namespace) -> dict[str, Any]:
    return asyncio.run(main_async(args))
