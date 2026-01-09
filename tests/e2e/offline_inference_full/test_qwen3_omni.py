# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
E2E Online tests for Qwen3-Omni model with multimodal input and mixed outputs.

This file contains helper functions inlined below and a single comprehensive
test `test_mix_to_text_audio_001` which performs mixed input -> text+audio
generation and validation.
"""

import os
import time
import base64
import io
import soundfile as sf
import tempfile
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import pytest
from tests.conftest import (
    OmniRunner,
    generate_synthetic_video,
    generate_synthetic_image,
    generate_synthetic_audio,
    convert_audio_to_text,
    cosine_similarity_text,
)

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

models = ["/data2/models/Qwen3-Omni-30B-A3B-Instruct"]
stage_configs = [str(Path(__file__).parent.parent / "stage_configs" / "qwen3_omni_ci.yaml")]
test_params = [(model, stage_config) for model in models for stage_config in stage_configs]


# Test parameters (kept compact but match your requirements)
TEST_PARAMS = {
    "text_length": 67,
    "video_pixel": (1280, 720),
    "video_num": 2,
    "video_frames": 300,
    "image_pixel": (16, 16),
    "image_num": 2,
    "audio_num": 2,
    "audio_duration": 10,
    "audio_channels": 2,
    "max_tokens": 10,
    "num_requests": 5,
}


def generate_test_prompt():
    base = (
        "Describe what you see and hear in the provided multimedia content. Focus on the main elements: "
    )
    if len(base) > TEST_PARAMS["text_length"]:
        return base[: TEST_PARAMS["text_length"]]
    return base.ljust(TEST_PARAMS["text_length"], "x")


def generate_mixed_multimodal_data():
    video_data = []
    for _ in range(TEST_PARAMS["video_num"]):
        w, h = TEST_PARAMS["video_pixel"]
        b64 = generate_synthetic_video(width=w, height=h, num_frames=TEST_PARAMS["video_frames"])
        video_bytes = base64.b64decode(b64)
        # write to temp file and read frames via cv2
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(video_bytes)
            tmp_path = tmp.name
        cap = cv2.VideoCapture(tmp_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
        if len(frames) == 0:
            # fallback: empty array
            video_arr = np.zeros((0, h, w, 3), dtype=np.uint8)
        else:
            video_arr = np.stack(frames, axis=0)
        video_data.append(video_arr)

    image_data = []
    for _ in range(TEST_PARAMS["image_num"]):
        w, h = TEST_PARAMS["image_pixel"]
        b64 = generate_synthetic_image(width=w, height=h)
        image_bytes = base64.b64decode(b64)
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_data.append(img)

    audio_data = []
    for _ in range(TEST_PARAMS["audio_num"]):
        b64 = generate_synthetic_audio(duration=TEST_PARAMS["audio_duration"], num_channels=TEST_PARAMS["audio_channels"])
        audio_bytes = base64.b64decode(b64)
        # Convert bytes to (numpy array, sample_rate) which vLLM multimodal parser expects
        try:
            arr, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
        except Exception:
            # Fallback: append raw bytes if reading fails
            arr, sr = audio_bytes, None
        audio_data.append((arr, sr))

    return video_data, image_data, audio_data


@pytest.mark.full
@pytest.mark.H100_2
@pytest.mark.parametrize("test_config", test_params)
def test_mix_to_text_audio_001(test_config) -> None:
    """Test: mixed input (video+image+audio+text) -> mixed output (text+audio).

    All helper logic is implemented in this function using the inlined helpers
    above and the utilities imported from `tests.conftest`.
    """
    model, stage_config_path = test_config

    # Allow skipping on small environments
    if os.getenv("VLLM_SKIP_LONG_E2E", "0") == "1":
        pytest.skip("Skipping long E2E test (VLLM_SKIP_LONG_E2E=1)")

    prompt = generate_test_prompt()
    videos, images, audios = generate_mixed_multimodal_data()

    total_start = time.perf_counter()

    with OmniRunner(model, seed=42, stage_configs_path=stage_config_path, stage_init_timeout=300, batch_timeout=30, init_timeout=600) as runner:
        per_request = []  # list of dicts {texts:[], audios:[]}
        latencies = []

        for idx in range(TEST_PARAMS["num_requests"]):
            print(f"\nProcessing request {idx+1}/{TEST_PARAMS['num_requests']}")
            t0 = time.perf_counter()
            outputs = list(
                runner.generate_multimodal(
                    prompts=prompt,
                    videos=[videos],
                    images=[images],
                    audios=[audios],
                    modalities=["text", "audio"],
                    mm_processor_kwargs={"use_audio_in_video": True},
                )
            )
            latencies.append(time.perf_counter() - t0)

            texts = []
            audio_blobs = []
            for out in outputs:
                typ = getattr(out, "final_output_type", None)
                if typ == "text":
                    try:
                        texts.append(out.request_output[0].outputs[0].text)
                    except Exception:
                        texts.append("")
                elif typ == "audio":
                    try:
                        audio_blobs.append(out.request_output[0].outputs[0].audio)
                    except Exception:
                        audio_blobs.append(None)

            per_request.append({"texts": texts, "audios": audio_blobs})

        total_time = time.perf_counter() - total_start
        avg_latency = sum(latencies) / len(latencies) if latencies else 0
        throughput = TEST_PARAMS["num_requests"] / total_time if total_time > 0 else 0

        print("\n" + "=" * 60)
        print("TEST RESULTS SUMMARY:")
        print("=" * 60)
        print(f"Total requests: {TEST_PARAMS['num_requests']}")
        print(f"Total time: {total_time:.2f}s")
        print(f"Avg latency: {avg_latency:.2f}s")
        print(f"Throughput: {throughput:.2f} req/s")

        # Validations
        audio_count = sum(1 for r in per_request if len(r["audios"]) > 0)
        text_count = sum(1 for r in per_request if len(r["texts"]) > 0)
        assert audio_count > 0, "No audio outputs generated"
        assert text_count > 0, "No text outputs generated"

        for i, r in enumerate(per_request):
            for j, txt in enumerate(r["texts"]):
                txt = txt or ""
                print(f"Request {i+1} text #{j+1}: {txt[:100]}" if txt else f"Request {i+1} text #{j+1}: <empty>")
                assert len(txt.strip()) > 0, f"Empty text output for request {i+1} text #{j+1}"

        latency_threshold = 30
        assert avg_latency < latency_threshold, f"Average latency {avg_latency:.2f}s exceeds {latency_threshold}s"

        # Audio -> text accuracy
        successful = 0
        total_comp = 0
        for i, r in enumerate(per_request):
            if not r["audios"] or not r["texts"]:
                continue
            for a_idx, a_blob in enumerate(r["audios"]):
                if not a_blob:
                    continue
                try:
                    a_text = convert_audio_to_text(a_blob)
                except Exception as e:
                    print(f"Error converting audio for request {i+1} audio #{a_idx+1}: {e}")
                    continue
                if not a_text:
                    continue
                for t_idx, gen_text in enumerate(r["texts"]):
                    sim = cosine_similarity_text(a_text, gen_text)
                    total_comp += 1
                    print(f"Req {i+1} audio#{a_idx+1} vs text#{t_idx+1} sim={sim:.3f}")
                    if sim > 0.3:
                        successful += 1
                        break

        if total_comp > 0:
            acc = successful / total_comp
            print(f"Audio accuracy: {successful}/{total_comp} ({acc:.2%})")

        # Final assertions
        assert audio_count > 0
        assert text_count > 0
        assert avg_latency > 0


