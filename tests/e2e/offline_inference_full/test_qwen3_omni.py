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


@pytest.mark.full
@pytest.mark.H100_2
@pytest.mark.parametrize("test_config", test_params)
def test_text_to_text_001(test_config) -> None:
    """Test processing text, generating text output via OpenAI API."""
    model, stage_config_path = test_config
    with OmniRunner(model, seed=42, stage_configs_path=stage_config_path, stage_init_timeout=300) as runner:
        # Prepare inputs
        question = "What is the capital of China?"
        start_time = time.perf_counter()
        outputs = runner.generate_multimodal(
            prompts=question,
            modalities=["text"]
        )

        #Find and verify text output (thinker stage)
        stage_outputs = list()
        for stage_output in outputs:
            stage_outputs.append(stage_output)

        # Verify E2E
        print(f"the request e2e is: {time.perf_counter() - start_time}")
        # TODO: Verify the E2E latency after confirmation baseline.

        # Verify only output text
        assert len(stage_outputs) == 1, "The generated content includes more than just text."

        # Verify text output success
        assert stage_outputs[0].final_output_type == "text", "No text output is generated"
        assert "Beijing" in stage_outputs[0].request_output[0].outputs[0].text, "The generated text is incorrect."


