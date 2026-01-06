# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
E2E Online tests for Qwen3-Omni model with video input and audio output.
"""

import os
import time
from pathlib import Path
import openai
import pytest
import concurrent.futures
from tests.conftest import OmniRunner
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

models = ["Qwen/Qwen3-Omni-30B-A3B-Instruct"]

# CI stage config for 2*H100-80G GPUs
stage_configs = [str(Path(__file__).parent.parent / "stage_configs" / "qwen3_omni_ci.yaml")]

# Create parameter combinations for model and stage config
test_params = [(model, stage_config) for model in models for stage_config in stage_configs]

def client(omni_server):
    """OpenAI client for the running vLLM-Omni server."""
    return openai.OpenAI(
        base_url=f"http://{omni_server.host}:{omni_server.port}/v1",
        api_key="EMPTY",
    )

def get_system_prompt():
    return {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": (
                    "You are Qwen, a virtual human developed by the Qwen Team, "
                    "Alibaba Group, capable of perceiving auditory and visual inputs, "
                    "as well as generating text and speech."
                ),
            }
        ],
    }


@pytest.mark.full
@pytest.mark.H100_2
@pytest.mark.parametrize("test_config", test_params)
def test_text_to_text_001(omni_runner: type[OmniRunner], test_config) -> None:
    """Test processing text, generating text output via OpenAI API."""
    model, stage_config_path = test_config
    with omni_runner(model, seed=42, stage_configs_path=stage_config_path, stage_init_timeout=300) as runner:
        # Prepare inputs
        question = "What is the capital of China?"

        start_time = time.perf_counter()
        outputs = runner.generate(
            prompts=question,
        )

        #Find and verify text output (thinker stage)
        text_output = None
        output_count = 0
        for stage_output in outputs:
            print(stage_output)
            # if stage_output.final_output_type == "text":
            #     text_output = stage_output
            #     output_count += 1
            #     break
        #
        #
        #
        # # Verify only output text
        # assert len(chat_completion.choices) == 1, "The generated content includes more than just text."
        #
        # # Verify text output success
        # text_choice = chat_completion.choices[0]
        # assert text_choice.message.content is not None, "No text output is generated"
        # assert chat_completion.usage.completion_tokens == 10, "The output length differs from the requested max_tokens."
        #
        # # Verify E2E
        # print(f"the request e2e is: {time.perf_counter() - start_time}")
        # # TODO: Verify the E2E latency after confirmation baseline.