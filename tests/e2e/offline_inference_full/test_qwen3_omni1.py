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
    """
    Test processing video+image+audio+text input, generating text+audio output (5 concurrent requests).
    Requirements:
    - Stage 0: TP=2, max_batch_size=5, gpu=0.6, device=0,1
    - Stage 1: TP=1, max_batch_size=5, gpu=0.3, device=1
    - Stage 2: max_batch_size=1, gpu=0.15, device=0
    - Multimodal specs: video(1280*720,2,300) | image(16*16,2) | audio(10s,2ch,2)
    - Output: text(audio+max_token=10, length=67) + audio
    - Verify: audio/text generation, E2E latency/throughput, audio-text similarity
    """
    # Core test parameters (strictly match requirements)
    model, stage_config_path = test_config
    num_concurrent_requests = 5
    max_tokens = 10
    text_length = 67

    # ======================== 1. Generate multimodal test data (strict specs) ========================
    # Video: 1280*720 resolution, 2 videos, 300 frames each
    video_data = []
    for _ in range(2):
        video_base64 = generate_synthetic_video(width=1280, height=720, num_frames=300)
        video_data.append(f"data:video/mp4;base64,{video_base64}")

    # Image: 16*16 resolution, 2 images
    image_data = []
    for _ in range(2):
        image_base64 = generate_synthetic_image(width=16, height=16)
        image_data.append(f"data:image/jpeg;base64,{image_base64}")

    # Audio: 10s duration, 2 channels, 2 audio files
    audio_data = []
    for _ in range(2):
        audio_base64 = generate_synthetic_audio(duration=10, num_channels=2)
        audio_data.append(f"data:audio/wav;base64,{audio_base64}")

    # ======================== 2. Construct Qwen3-Omni compatible prompt ========================
    # Extract system prompt text and build user prompt (fixed 67 chars)
    system_prompt_text = get_system_prompt()["content"][0]["text"]
    user_text = f"Describe this content briefly in {max_tokens} tokens."
    user_text = user_text.ljust(text_length)[:text_length]  # Ensure exact 67 characters

    # Qwen3-Omni multimodal placeholders (match data count)
    audio_placeholder = "<|audio_bos|><|audio_pad|><|audio_eos|>" * 2
    image_placeholder = "<|vision_bos|><|image_pad|><|vision_eos|>" * 2
    video_placeholder = "<|vision_bos|><|video_pad|><|vision_eos|>" * 2
    media_placeholders = audio_placeholder + image_placeholder + video_placeholder

    # Build full prompt (pure text format required by OmniRunner)
    prompt = (
        f"<|im_start|>system\n{system_prompt_text}<|im_end|>\n"
        f"<|im_start|>user\n{media_placeholders}{user_text}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    # ======================== 3. Initialize OmniRunner and run concurrent tests ========================
    with OmniRunner(
        model,
        seed=42,
        stage_configs_path=stage_config_path,
        stage_init_timeout=300
    ) as runner:
        e2e_latencies = []
        completions = []

        # Thread-safe single request handler
        def run_single_request() -> tuple[list, float]:
            """Execute single multimodal request and return results + E2E latency."""
            start_time = time.perf_counter()
            # Core call: strictly match the provided generate_multimodal template
            outputs = runner.generate_multimodal(
                prompts=prompt,          # Pure text prompt (match "question" in template)
                audios=audio_data,       # Audio data list (match "audio" in template)
                images=image_data,       # Image data list (match "image" in template)
                videos=video_data,       # Video data list (match "video" in template)
                modalities=["text", "audio"]  # Output modalities (match "modalities" in template)
            )
            # Convert generator to list for verification
            result = list(outputs)
            e2e_latency = time.perf_counter() - start_time
            return result, e2e_latency

        # Execute 5 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_concurrent_requests) as executor:
            futures = [executor.submit(run_single_request) for _ in range(num_concurrent_requests)]
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(futures):
                try:
                    completion, latency = future.result()
                    completions.append(completion)
                    e2e_latencies.append(latency)
                except Exception as e:
                    pytest.fail(f"Request failed with error: {str(e)}")

        # ======================== 4. Core verification logic ========================
        # Verify all 5 requests completed successfully
        assert len(completions) == num_concurrent_requests, \
            f"Expected {num_concurrent_requests} completions, got {len(completions)}"

        # Calculate and print key metrics (E2E latency & throughput)
        avg_e2e = sum(e2e_latencies) / len(e2e_latencies) if e2e_latencies else 0
        total_e2e = sum(e2e_latencies)
        throughput = num_concurrent_requests / total_e2e if total_e2e > 0 else 0
        
        print(f"[Metrics] Average E2E latency: {avg_e2e:.4f} seconds")
        print(f"[Metrics] Throughput: {throughput:.4f} requests/second")

        # Verify each completion's outputs
        for idx, stage_outputs in enumerate(completions):
            request_id = idx + 1
            print(f"\nVerifying request #{request_id}...")

            # 1. Verify output count (at least text + audio)
            assert len(stage_outputs) >= 2, \
                f"Request #{request_id}: Missing text/audio output (got {len(stage_outputs)} outputs)"

            # 2. Verify text output (success + length constraints)
            text_output = next((s for s in stage_outputs if s.final_output_type == "text"), None)
            assert text_output is not None, f"Request #{request_id}: No text output generated"
            
            text_content = text_output.request_output[0].outputs[0].text
            assert text_content is not None and text_content.strip(), \
                f"Request #{request_id}: Empty text output"
            assert len(text_content.split()) <= max_tokens, \
                f"Request #{request_id}: Text exceeds max_tokens (max: {max_tokens}, got: {len(text_content.split())})"
            assert len(text_content) <= text_length, \
                f"Request #{request_id}: Text exceeds length limit (max: {text_length}, got: {len(text_content)})"

            # 3. Verify audio output (success + validity)
            audio_output = next((s for s in stage_outputs if s.final_output_type == "audio"), None)
            assert audio_output is not None, f"Request #{request_id}: No audio output generated"
            
            audio_content = audio_output.request_output[0].outputs[0].audio.data
            assert audio_content is not None, f"Request #{request_id}: Empty audio data"
            assert audio_output.request_output[0].outputs[0].audio.expires_at > time.time(), \
                f"Request #{request_id}: Audio output expired"

            # 4. Verify audio-text similarity (content consistency)
            audio_text = convert_audio_to_text(audio_content)
            similarity = cosine_similarity_text(audio_text, text_content)
            assert similarity > 0.7, \
                f"Request #{request_id}: Audio-text similarity too low (got: {similarity:.2f}, min: 0.7)"
            print(f"Request #{request_id}: Audio-text similarity: {similarity:.2f} (pass)")

    # Small delay to avoid premature stage shutdown warnings
    time.sleep(2)
    
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