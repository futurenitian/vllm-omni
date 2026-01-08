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
from tests.conftest import (
    dummy_messages_from_mix_data,
    generate_synthetic_audio,
    generate_synthetic_video,
    generate_synthetic_image,
    cosine_similarity_text,
    convert_audio_to_text
)
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

models = ["/data2/models/Qwen3-Omni-30B-A3B-Instruct"]

# CI stage config for 2*H100-80G GPUs
stage_configs = [str(Path(__file__).parent.parent / "stage_configs" / "qwen3_omni_ci.yaml")]

# Create parameter combinations for model and stage config
test_params = [(model, stage_config) for model in models for stage_config in stage_configs]

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
def test_mix_to_text_audio_001(test_config) -> None:
    """Test processing video+image+audio, generating text+audio output via OpenAI API."""
    model, stage_config_path = test_config
    num_concurrent_requests = 5
    max_tokens = 10

    with OmniRunner(model, seed=42, stage_configs_path=stage_config_path, stage_init_timeout=300) as runner:
        video_data_urls = []
        for _ in range(2):
            video_base64 = generate_synthetic_video(width=1280, height=720, num_frames=300)
            video_data_urls.append(f"data:video/mp4;base64,{video_base64}")

        image_data_urls = []
        for _ in range(2):
            image_base64 = generate_synthetic_image(width=16, height=16)
            image_data_urls.append(f"data:image/jpeg;base64,{image_base64}")

        audio_data_urls = []
        for _ in range(2):
            audio_base64 = generate_synthetic_audio(duration=10, num_channels=2)
            audio_data_urls.append(f"data:audio/wav;base64,{audio_base64}")

        messages = dummy_messages_from_mix_data(
            system_prompt=get_system_prompt(),
            video_data_url=video_data_urls,
            image_data_url=image_data_urls,
            audio_data_url=audio_data_urls,
            content_text=f"Describe this content briefly in {max_tokens} tokens."
        )

        e2e_list = list()
        chat_completions = list()
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_concurrent_requests) as executor:
            futures = [
                executor.submit(
                    runner.generate_multimodal,
                    prompts=messages,
                    modalities=["text", "audio"],
                    max_tokens=max_tokens,
                    stop=None
                )
                for _ in range(num_concurrent_requests)
            ]
            start_time = time.perf_counter()
            for future in concurrent.futures.as_completed(futures):
                # generate_multimodal返回Generator，转换为list处理
                completion = list(future.result())
                chat_completions.append(completion)
                current_e2e = time.perf_counter() - start_time
                e2e_list.append(current_e2e)

        # Verify all requests succeeded
        assert len(chat_completions) == num_concurrent_requests, "Not all requests succeeded."

        # Calculate and print E2E metrics
        avg_e2e = sum(e2e_list) / len(e2e_list)
        throughput = num_concurrent_requests / sum(e2e_list) if sum(e2e_list) > 0 else 0
        print(f"the avg e2e is: {avg_e2e}")
        print(f"the throughput is: {throughput} requests/second")

        # Verify outputs for each request
        for stage_outputs in chat_completions:
            # Verify output count (text + audio)
            assert len(stage_outputs) >= 2, "Missing text/audio output"

            # Find and verify text output
            text_output = next((s for s in stage_outputs if s.final_output_type == "text"), None)
            assert text_output is not None, "No text output is generated"
            text_content = text_output.request_output[0].outputs[0].text
            assert text_content is not None and text_content.strip(), "Empty text output"
            assert len(text_content.split()) <= max_tokens, "The output length differs from the requested max_tokens."

            # Find and verify audio output
            audio_output = next((s for s in stage_outputs if s.final_output_type == "audio"), None)
            assert audio_output is not None, "No audio output is generated"
            audio_data = audio_output.request_output[0].outputs[0].audio.data
            assert audio_data is not None, "Empty audio data"
            assert audio_output.request_output[0].outputs[0].audio.expires_at > time.time(), "The generated audio has expired."

            # Verify audio-text similarity
            audio_text = convert_audio_to_text(audio_data)
            similarity = cosine_similarity_text(audio_text, text_content)
            assert similarity > 0.7, f"The audio content is not same as the text (similarity: {similarity:.2f})"

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