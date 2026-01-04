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
from tests.conftest import (OmniServer, dummy_messages_from_mix_data, modify_stage_config, convert_audio_to_text,
                            cosine_similarity_text, generate_synthetic_audio, generate_synthetic_image)

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
def test_text_to_text_001(test_config: tuple[str, str]) -> None:
    """Test processing text, generating text output via OpenAI API."""
    model, stage_config_path = test_config
    with OmniServer(model, ["--stage-configs-path", stage_config_path, "--stage-init-timeout", "90"]) as server:
        messages = dummy_messages_from_mix_data(
            system_prompt=get_system_prompt(),
        )

        # Test single completion
        api_client = client(server)
        start_time = time.perf_counter()
        chat_completion = api_client.chat.completions.create(
            model=server.model, messages=messages, max_tokens=10, stop=None, modalities=["text"]
        )
        # Verify only output text
        assert len(chat_completion.choices) == 1, "The generated content includes more than just text."

        # Verify text output success
        text_choice = chat_completion.choices[0]
        assert text_choice.message.content is not None, "No text output is generated"
        assert chat_completion.usage.completion_tokens == 10, "The output length differs from the requested max_tokens."

        # Verify E2E
        print(f"the request e2e is: {time.perf_counter() - start_time}")
        # TODO: Verify the E2E latency after confirmation baseline.


@pytest.mark.full
@pytest.mark.H100_3
@pytest.mark.parametrize("test_config", test_params)
def test_text_to_audio_001(test_config: tuple[str, str]) -> None:
    """Test processing text, generating audio output via OpenAI API."""

    model, stage_config_path = test_config
    stage_config_path = modify_stage_config(stage_config_path, {
        0: {"engine_args.gpu_memory_utilization": 0.95, "engine_args.tensor_parallel_size": 1, "runtime.devices": "2"}})
    with OmniServer(model, ["--stage-configs-path", stage_config_path, "--stage-init-timeout", "90"]) as server:
        messages = dummy_messages_from_mix_data(
            system_prompt=get_system_prompt(),
        )

        # Test single completion
        api_client = client(server)
        start_time = time.perf_counter()
        chat_completion = api_client.chat.completions.create(
            model=server.model, messages=messages, max_tokens=10, stop=None, modalities=["audio"]
        )

        # Verify only output audio
        assert len(chat_completion.choices) == 1, "The generated content includes more than just text."

        # Verify audio output success
        audio_message = chat_completion.choices[1].message
        assert audio_message.audio.data is not None, "No audio output is generated"
        assert audio_message.audio.expires_at > time.time(), "The generated audio has expired."

        # Verify E2E
        print(f"the request e2e is: {time.perf_counter() - start_time}")
        # TODO: Verify the E2E latency after confirmation baseline.


@pytest.mark.full
@pytest.mark.H100_2
@pytest.mark.parametrize("test_config", test_params)
def test_text_to_text_audio_001(test_config: tuple[str, str]) -> None:
    """Test processing text, generating audio output via OpenAI API."""

    model, stage_config_path = test_config
    num_concurrent_requests = 5
    stage_config_path = modify_stage_config(stage_config_path, {
        0: {"runtime.max_batch_size": num_concurrent_requests, "default_sampling_params.ignore_eos": True}, 1: {"runtime.max_batch_size": num_concurrent_requests}})
    with OmniServer(model, ["--stage-configs-path", stage_config_path, "--stage-init-timeout", "90"]) as server:
        messages = dummy_messages_from_mix_data(
            system_prompt=get_system_prompt(),
        )

        # Test single completion
        api_client = client(server)
        e2e_list = list()
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_concurrent_requests) as executor:
            # Submit multiple completion requests concurrently
            futures = [
                executor.submit(
                    api_client.chat.completions.create,
                    model=server.model,
                    messages=messages,
                    max_tokens=1000,
                    stop=None,
                )
                for _ in range(num_concurrent_requests)
            ]
            start_time = time.perf_counter()
            # Wait for all requests to complete and collect results
            chat_completions = list()
            for future in concurrent.futures.as_completed(futures):
                chat_completions.append(future.result())
                # Verify E2E
                current_e2e = time.perf_counter() - start_time
                print(f"the request e2e is: {current_e2e}")
                # TODO: Verify the E2E latency after confirmation baseline.
                e2e_list.append(current_e2e)

        print(f"the avg e2e is: {sum(e2e_list)/len(e2e_list)}")
        # Verify all completions succeeded
        assert len(chat_completions) == num_concurrent_requests, "Not all requests succeeded."
        for chat_completion in chat_completions:
            # Verify audio output success
            audio_message = chat_completion.choices[1].message
            audio_data = audio_message.audio.data
            assert audio_data is not None, "No audio output is generated"
            assert audio_message.audio.expires_at > time.time(), "The generated audio has expired."

            # Verify text output success
            text_choice = chat_completion.choices[0]
            text_content = text_choice.message.content
            assert text_choice.message.content is not None, "No text output is generated"
            assert chat_completion.usage.completion_tokens == 1000, "The output length differs from the requested max_tokens."

            # Verify text output same as audio output
            print(f"text content is: {text_content}")
            print(f"audio content is: {convert_audio_to_text(audio_data)}")
            assert cosine_similarity_text(convert_audio_to_text(audio_data),text_content) > 0.9, "The audio content is not same as the text"



@pytest.mark.full
@pytest.mark.H100_3
@pytest.mark.parametrize("test_config", test_params)
def test_audio_to_text_001(test_config: tuple[str, str]) -> None:
    """Test processing text, generating text output via OpenAI API."""
    model, stage_config_path = test_config
    stage_config_path = modify_stage_config(stage_config_path, {
        0: {"engine_args.gpu_memory_utilization": 0.95, "engine_args.tensor_parallel_size": 1, "runtime.devices": "2", "default_sampling_params.ignore_eos": True}})
    with OmniServer(model, ["--stage-configs-path", stage_config_path, "--stage-init-timeout", "90"]) as server:
        audio_data_url = f"data:audio/ogg;base64,{generate_synthetic_audio(1, 1)}"
        messages = dummy_messages_from_mix_data(
            audio_data_url=audio_data_url
        )
        # Test single completion
        api_client = client(server)
        start_time = time.perf_counter()
        chat_completion = api_client.chat.completions.create(
            model=server.model, messages=messages, max_tokens=1000, stop=None, modalities=["text"]
        )
        # Verify only output text
        assert len(chat_completion.choices) == 1, "The generated content includes more than just text."

        # Verify text output success
        text_choice = chat_completion.choices[0]
        assert text_choice.message.content is not None, "No text output is generated"
        assert chat_completion.usage.completion_tokens == 1000, "The output length differs from the requested max_tokens."

        # Verify E2E
        print(f"the request e2e is: {time.perf_counter() - start_time}")
        # TODO: Verify the E2E latency after confirmation baseline.



@pytest.mark.full
@pytest.mark.H100_2
@pytest.mark.parametrize("test_config", test_params)
def test_audio_to_text_audio_001(test_config: tuple[str, str]) -> None:
    """Test processing text, generating audio output via OpenAI API."""

    model, stage_config_path = test_config
    num_concurrent_requests = 5
    stage_config_path = modify_stage_config(stage_config_path, {
        0: {"runtime.max_batch_size": num_concurrent_requests, "default_sampling_params.ignore_eos": True}, 1: {"runtime.max_batch_size": num_concurrent_requests}})
    with OmniServer(model, ["--stage-configs-path", stage_config_path, "--stage-init-timeout", "90"]) as server:
        audio_data_url = []
        for _ in range(5):
            audio_data_url.append(f"data:audio/ogg;base64,{generate_synthetic_audio(1, 5)}")

        messages = dummy_messages_from_mix_data(
            audio_data_url=audio_data_url
        )

        # Test single completion
        api_client = client(server)
        e2e_list = list()
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_concurrent_requests) as executor:
            # Submit multiple completion requests concurrently
            futures = [
                executor.submit(
                    api_client.chat.completions.create,
                    model=server.model,
                    messages=messages,
                    max_tokens=10,
                    stop=None,
                )
                for _ in range(num_concurrent_requests)
            ]
            start_time = time.perf_counter()
            # Wait for all requests to complete and collect results
            chat_completions = list()
            for future in concurrent.futures.as_completed(futures):
                chat_completions.append(future.result())
                # Verify E2E
                current_e2e = time.perf_counter() - start_time
                print(f"the request e2e is: {current_e2e}")
                # TODO: Verify the E2E latency after confirmation baseline.
                e2e_list.append(current_e2e)

        print(f"the avg e2e is: {sum(e2e_list)/len(e2e_list)}")
        # Verify all completions succeeded
        assert len(chat_completions) == num_concurrent_requests, "Not all requests succeeded."
        for chat_completion in chat_completions:
            # Verify audio output success
            audio_message = chat_completion.choices[1].message
            audio_data = audio_message.audio.data
            assert audio_data is not None, "No audio output is generated"
            assert audio_message.audio.expires_at > time.time(), "The generated audio has expired."

            # Verify text output success
            text_choice = chat_completion.choices[0]
            text_content = text_choice.message.content
            assert text_choice.message.content is not None, "No text output is generated"
            assert chat_completion.usage.completion_tokens == 10, "The output length differs from the requested max_tokens."

            # Verify text output same as audio output
            print(f"text content is: {text_content}")
            print(f"audio content is: {convert_audio_to_text(audio_data)}")
            assert cosine_similarity_text(convert_audio_to_text(audio_data),text_content) > 0.9, "The audio content is not same as the text"



@pytest.mark.full
@pytest.mark.H100_3
@pytest.mark.parametrize("test_config", test_params)
def test_image_to_text_001(test_config: tuple[str, str]) -> None:
    """Test processing text, generating text output via OpenAI API."""
    model, stage_config_path = test_config
    stage_config_path = modify_stage_config(stage_config_path, {
        0: {"engine_args.gpu_memory_utilization": 0.95, "engine_args.tensor_parallel_size": 1, "runtime.devices": "2", "default_sampling_params.ignore_eos": True}})
    with OmniServer(model, ["--stage-configs-path", stage_config_path, "--stage-init-timeout", "90"]) as server:
        image_data_url = f"data:image/jpeg;base64,{generate_synthetic_image(16, 16)}"
        messages = dummy_messages_from_mix_data(
            image_data_url=image_data_url
        )
        # Test single completion
        api_client = client(server)
        start_time = time.perf_counter()
        chat_completion = api_client.chat.completions.create(
            model=server.model, messages=messages, max_tokens=1000, stop=None, modalities=["text"]
        )
        # Verify only output text
        assert len(chat_completion.choices) == 1, "The generated content includes more than just text."

        # Verify text output success
        text_choice = chat_completion.choices[0]
        assert text_choice.message.content is not None, "No text output is generated"
        assert chat_completion.usage.completion_tokens == 1000, "The output length differs from the requested max_tokens."

        # Verify E2E
        print(f"the request e2e is: {time.perf_counter() - start_time}")
        # TODO: Verify the E2E latency after confirmation baseline.


@pytest.mark.full
@pytest.mark.H100_2
@pytest.mark.parametrize("test_config", test_params)
def test_audio_to_text_audio_001(test_config: tuple[str, str]) -> None:
    """Test processing text, generating audio output via OpenAI API."""

    model, stage_config_path = test_config
    num_concurrent_requests = 5
    stage_config_path = modify_stage_config(stage_config_path, {
        0: {"runtime.max_batch_size": num_concurrent_requests, "default_sampling_params.ignore_eos": True}, 1: {"runtime.max_batch_size": num_concurrent_requests}})
    with OmniServer(model, ["--stage-configs-path", stage_config_path, "--stage-init-timeout", "90"]) as server:
        image_data_url = []
        for _ in range(4):
            image_data_url.append(f"data:image/jpeg;base64,{generate_synthetic_image(1280, 720)}")

        messages = dummy_messages_from_mix_data(
            image_data_url=image_data_url
        )

        # Test single completion
        api_client = client(server)
        e2e_list = list()
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_concurrent_requests) as executor:
            # Submit multiple completion requests concurrently
            futures = [
                executor.submit(
                    api_client.chat.completions.create,
                    model=server.model,
                    messages=messages,
                    max_tokens=10,
                    stop=None,
                )
                for _ in range(num_concurrent_requests)
            ]
            start_time = time.perf_counter()
            # Wait for all requests to complete and collect results
            chat_completions = list()
            for future in concurrent.futures.as_completed(futures):
                chat_completions.append(future.result())
                # Verify E2E
                current_e2e = time.perf_counter() - start_time
                print(f"the request e2e is: {current_e2e}")
                # TODO: Verify the E2E latency after confirmation baseline.
                e2e_list.append(current_e2e)

        print(f"the avg e2e is: {sum(e2e_list)/len(e2e_list)}")
        # Verify all completions succeeded
        assert len(chat_completions) == num_concurrent_requests, "Not all requests succeeded."
        for chat_completion in chat_completions:
            # Verify audio output success
            audio_message = chat_completion.choices[1].message
            audio_data = audio_message.audio.data
            assert audio_data is not None, "No audio output is generated"
            assert audio_message.audio.expires_at > time.time(), "The generated audio has expired."

            # Verify text output success
            text_choice = chat_completion.choices[0]
            text_content = text_choice.message.content
            assert text_choice.message.content is not None, "No text output is generated"
            assert chat_completion.usage.completion_tokens == 10, "The output length differs from the requested max_tokens."

            # Verify text output same as audio output
            print(f"text content is: {text_content}")
            print(f"audio content is: {convert_audio_to_text(audio_data)}")
            assert cosine_similarity_text(convert_audio_to_text(audio_data),text_content) > 0.9, "The audio content is not same as the text"

