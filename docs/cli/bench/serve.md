# vLLM-Omni Benchmark CLI Guide
The vllm bench command launches the vLLM-Omni benchmark to evaluate the performance of multimodal models.

## Notes
We currently only support using the "openai-chat-omni" backend.

## Basic Parameter Description
You can use `vllm bench serve --omni --help=all` to get descriptions of all parameters. The commonly used parameters are described below:
- `--omni`  
  Enable Omni (multimodal) mode, supporting multimodal inputs and outputs such as images, videos, and audio.

- `--backend`  
  Specify the backend adapter as openai-chat-omni, using OpenAI Chat compatible API behavior as the protocol. Currently only openai-chat-omni is supported.

- `--model`  
  The model identifier to load, filled according to the models supported by vLLM-Omni.

- `--endpoint`  
  The API endpoint exposed externally, to which clients send their requests.

- `--dataset-name`  
  The name of the dataset used; random-mm indicates generating random multimodal inputs (images, videos, audio).

- `--num-prompts`  
  The total number of requests to send, an integer.

- `--max-concurrency`  
  "Maximum number of concurrent requests. This can be used "
        "to help simulate an environment where a higher level component "
        "is enforcing a maximum number of concurrent requests. While the "
        "--request-rate argument controls the rate at which requests are "
        "initiated, this argument will control how many are actually allowed "
        "to execute at a time. This means that when used in combination, the "
        "actual request rate may be lower than specified with --request-rate, "
        "if the server is not processing requests fast enough to keep up."

- `--request-rate`  
  "Number of requests per second. If this is inf, "
        "then all the requests are sent at time 0. "
        "Otherwise, we use Poisson process or gamma distribution "
        "to synthesize the request arrival times."

- `--ignore-eos`  
  "Set ignore_eos flag when sending the benchmark request."

- `--metric-percentiles`  
  Comma-separated list of percentiles for selected metrics. "
        "To report 25-th, 50-th, and 75-th percentiles, use \"25,50,75\". "
        "Default value is \"99\"."
        "Use \"--percentile-metrics\" to select metrics.

- `--percentile-metrics`  
        "Comma-separated list of selected metrics to report percentils."
                    "This argument specifies the metrics to report percentiles."
                    'Allowed metric names are "ttft", "tpot", "itl", "e2el", "audio_ttfp", "audio_rtf". '

- `--save-result`  
Specify to save benchmark results to a json file

- `--save-detailed`  
"When saving the results, whether to include per request "
        "information such as response, error, ttfs, tpots, etc."

- `--result-dir`  
 "Specify directory to save benchmark json results."
        "If not specified, results are saved in the current directory."

- `--result-filename`  
"Specify the filename to save benchmark json results."
        "If not specified, results will be saved in "
        "{label}-{args.request_rate}qps-{base_model_id}-{current_dt}.json"

- `--extra-body`  
With the vLLM Omni OpenAI client, you can specify output modalities using the `extra_body` parameter:
    - Text only: `extra_body={"modalities": ["text"]}`
    - Text and audio: `extra_body={"modalities": ["text", "audio"]}`
    - Audio only: `extra_body={"modalities": ["audio"]}`

## Usage Examples

### Online Benchmark
<details class="admonition abstract" markdown="1">
<summary>Show more</summary>

First start serving your model:

```bash
vllm serve Qwen/Qwen2.5-Omni-7B --omni
```

Then run the benchmarking for sharegpt:

```bash
# download dataset
# wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
vllm bench serve \
  --omni \
  --endpoint /v1/chat/completions \
  --backend openai-chat-omni \
  --model Qwen/Qwen2.5-Omni-7B \
  --dataset-name sharegpt \
  --dataset-path <your data path>/ShareGPT_V3_unfiltered_cleaned_split.json \
  --num-prompts 10
```

Or run the benchmarking for random:

```bash
vllm bench serve \
  --omni \
  --endpoint /v1/chat/completions \
  --backend openai-chat-omni \
  --model Qwen/Qwen2.5-Omni-7B \
  --dataset-name random \
  --num-prompts 10 \
  --random-prefix-len 25 \
  --random-input-len 300 \
  --random-output-len 40 \
```

Parameter Description:
- `--random-prefix-len`  
  Number of fixed prefix tokens before the random context in a request.
  The total input length is the sum of random-prefix-len and a random
  context length sampled from [input_len * (1 - range_ratio),
  input_len * (1 + range_ratio)].

- `--random-input-len`  
  Number of input tokens per request

- `--random-output-len`  
  Number of output tokens per request

If successful, you will see the following output:

```text
============ Serving Benchmark Result ============
Successful requests:                     2
Failed requests:                         0
Benchmark duration (s):                  14.66
Request throughput (req/s):              0.14
Peak concurrent requests:                2.00
================== Text Result ===================
Total input tokens:                      36
Total generated tokens:                  68160
Output token throughput (tok/s):         4649.93
Peak output token throughput (tok/s):    105.00
Peak concurrent requests:                2.00
Total Token throughput (tok/s):          4652.39
---------------Time to First Token----------------
Mean TTFT (ms):                          122.65
Median TTFT (ms):                        122.65
P99 TTFT (ms):                           151.19
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          1.00
Median TPOT (ms):                        1.00
P99 TPOT (ms):                           1.75
---------------Inter-token Latency----------------
Mean ITL (ms):                           38.55
Median ITL (ms):                         28.86
P99 ITL (ms):                            132.25
================== Audio Result ==================
Total audio duration generated(s):       15.79
Total audio frames generated:            379050
Audio throughput(audio duration/s):      1.08
==================================================
```

</details>

### Multi-Modal Benchmark

<details class="admonition abstract" markdown="1">
<summary>Show more</summary>

Benchmark the performance of multi-modal requests in vLLM-Omni.

Generate synthetic image、video、audio inputs alongside random text prompts to stress-test vision models without external datasets.

Notes:

- Works only with online benchmark via the OpenAI backend (`--backend openai-chat-omni`) and endpoint `/v1/chat/completions`.

Start the server (example):

```bash
vllm serve Qwen/Qwen2.5-Omni-7B --omni
```

It is recommended to use the flag `--ignore-eos` to simulate real responses. You can set the size of the output via the arg `random-output-len`.

Then run the benchmarking script:
```bash
vllm bench serve \
  --omni \
  --backend openai-chat-omni \
  --model Qwen/Qwen2.5-Omni-7B \
  --endpoint /v1/chat/completions \
  --dataset-name random-mm \
  --num-prompts 100 \
  --max-concurrency 10 \
  --random-prefix-len 25 \
  --random-input-len 300 \
  --random-output-len 40 \
  --random-range-ratio 0.2 \
  --random-mm-base-items-per-request 2 \
  --random-mm-limit-mm-per-prompt '{"image": 3, "video": 1, "audio": 1}' \
  --random-mm-bucket-config '{(256, 256, 1): 0.7, (720, 1280, 1): 0.3}' \
  --random-mm-num-mm-items-range-ratio 0.5 \
  --request-rate inf \
  --ignore-eos
```

Parameter Description:  
- `--random-prefix-len`  
  Number of fixed prefix tokens before the random context in a request.
  The total input length is the sum of random-prefix-len and a random
  context length sampled from [input_len * (1 - range_ratio),
  input_len * (1 + range_ratio)].

- `--random-input-len`  
  Number of input tokens per request

- `--random-output-len`  
  Number of output tokens per request

- `--random-range-ratio`  
  Range ratio for sampling input/output length,
  used only for random sampling. Must be in the range [0, 1) to define
  a symmetric sampling range [length * (1 - range_ratio), length * (1 + range_ratio)].

- `--random-mm-base-items-per-request`  
  Base number of multimodal items per request for random-mm.
  Actual per-request count is sampled around this base using
  --random-mm-num-mm-items-range-ratio.

- `--random-mm-limit-mm-per-prompt`  
  Per-modality hard caps for items attached per request, e.g.
  '{"image": 3, "video": 1, "audio": 1}'. The sampled per-request item
  count is clamped to the sum of these limits. When a modality
  reaches its cap, its buckets are excluded and probabilities are
  renormalized.

- `--random-mm-num-mm-items-range-ratio`  
  Range ratio r in [0, 1] for sampling items per request.
  We sample uniformly from the closed integer range
  [floor(n*(1-r)), ceil(n*(1+r))]
  where n is the base items per request.
  r=0 keeps it fixed; r=1 allows 0 items. The maximum is clamped
  to the sum of per-modality limits from
  --random-mm-limit-mm-per-prompt. An error is raised if the computed min exceeds the max.

- `--random-mm-bucket-config`  
  The bucket config is a dictionary mapping a multimodal item
  sampling configuration to a probability.
  Currently allows for 3 modalities: audio, images and videos.
  A bucket key is a tuple of (height, width, num_frames)
  The value is the probability of sampling that specific item.
  Example:
  --random-mm-bucket-config
  "{(256, 256, 1): 0.5, (720, 1280, 16): 0.4, (0, 1, 5): 0.10}"
  First item: images with resolution 256x256 w.p. 0.5
  Second item: videos with resolution 720x1280 and 16 frames
  Third item: audios with 1s duration and 5 channels w.p. 0.1
  OBS.: If the probabilities do not sum to 1, they are normalized.

If successful, you will see the following output:

```text
============ Serving Benchmark Result ============
Successful requests:                     1
Failed requests:                         0
Request rate configured (RPS):           1.00
Benchmark duration (s):                  5.35
Request throughput (req/s):              0.19
Peak concurrent requests:                1.00
================== Text Result ===================
Total input tokens:                      10
Total generated tokens:                  3889
Output token throughput (tok/s):         727.13
Peak output token throughput (tok/s):    63.00
Peak concurrent requests:                1.00
Total Token throughput (tok/s):          729.00
---------------Time to First Token----------------
Mean TTFT (ms):                          161.25
Median TTFT (ms):                        161.25
P99 TTFT (ms):                           161.25
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          1.08
Median TPOT (ms):                        1.08
P99 TPOT (ms):                           1.08
---------------Inter-token Latency----------------
Mean ITL (ms):                           15.75
Median ITL (ms):                         14.23
P99 ITL (ms):                            68.19
----------------End-to-end Latency----------------
Mean E2EL (ms):                          4346.10
Median E2EL (ms):                        4346.10
P99 E2EL (ms):                           4346.10
================== Audio Result ==================
Total audio duration generated(s):       7.90
Total audio frames generated:            189525
Audio throughput(audio duration/s):      1.48
---------------Time to First Packet---------------
Mean AUDIO_TTFP (ms):                    2728.66
Median AUDIO_TTFP (ms):                  2728.66
P99 AUDIO_TTFP (ms):                     2728.66
-----------------Real Time Factor-----------------
Mean AUDIO_RTF (ms):                     0.00
Median AUDIO_RTF (ms):                   0.00
P99 AUDIO_RTF (ms):                      0.00
==================================================
```

Behavioral notes:

- If the requested base item count cannot be satisfied under the provided per-prompt limits, the tool raises an error rather than silently clamping.

How sampling works:

- Determine per-request item count k by sampling uniformly from the integer range defined by `--random-mm-base-items-per-request` and `--random-mm-num-mm-items-range-ratio`, then clamp k to at most the sum of per-modality limits.
- For each of the k items, sample a bucket (H, W, T) according to the normalized probabilities in `--random-mm-bucket-config`, while tracking how many items of each modality have been added.
- If a modality (e.g., image) reaches its limit from `--random-mm-limit-mm-per-prompt`, all buckets of that modality are excluded and the remaining bucket probabilities are renormalized before continuing.
This should be seen as an edge case, and if this behavior can be avoided by setting `--random-mm-limit-mm-per-prompt` to a large number. Note that this might result in errors due to engine config `--limit-mm-per-prompt`.
- The resulting request contains synthetic image data in `multi_modal_data` (OpenAI Chat format). When `random-mm` is used with the OpenAI Chat backend, prompts remain text and MM content is attached via `multi_modal_data`.

</details>
