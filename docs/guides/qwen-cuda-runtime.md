# Qwen on CUDA: safe runtime tuning

The pinned model is `onnx-community/Qwen2.5-1.5B-Instruct` revision
`6287331f475a3e20e8c879be8fd4bf3551ad9d34`, with the exact ONNX file
listed in `app/pilot/local-llm-spec.json`. The existing decoder keeps the KV
cache on the selected device with ONNX Runtime I/O Binding; it does not copy the
full cache back to the CPU on every token.

## Why CUDA Graph and shared KV buffers are not enabled

An inspection of the pinned 1,787,566,590-byte ONNX file found 3,094 graph
nodes, 56 `past_key_values.*` inputs and 56 `present.*` outputs. Every
`present.*` output is produced by `Concat` with the corresponding past cache.
The unoptimized graph has 28 `Softmax` nodes and no `Attention` or
`GroupQueryAttention` node. Thus the present cache grows with each token and
gets a new allocation. This is the ordinary dynamic cache interface, not an
in-place cache interface.

[ONNX Runtime's cache guide](https://onnxruntime.ai/docs/genai/howto/past-present-share-buffer.html)
describes fixed-capacity shared KV buffers as a distinct model/runtime setup.
[Its CUDA Graph requirements](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#using-cuda-graphs-preview)
require stable input and output shapes and addresses across replay. The pinned
graph and the present `LocalInstructModel` binding path do not meet those
requirements. Enabling either feature without a verified model re-export could
return incorrect tokens or fail at runtime. An ORT optimizer may change the
graph on a particular server; such a new graph must be inspected and validated
before changing this decision.

## CUDA decoding work avoided without changing tokens

The CUDA I/O binding path now keeps one immutable host-side attention mask and
position sequence and binds contiguous views of them for each forward pass.
The values and shapes passed to ONNX Runtime are unchanged. This removes a
small Python allocation on every generated token; ONNX Runtime still transfers
the selected inputs to the GPU, as described in its
[I/O Binding guide](https://onnxruntime.ai/docs/performance/tune-performance/iobinding.html).
When generation reaches `max_new_tokens`, the
last emitted token is no longer forwarded through the model, because its next
logits and KV outputs would never be read. Generation that ends with a model
stop token or a complete JSON object already skips that pass. The CPU path is
unchanged.

Tests verify exact input values, buffer reuse, answer equality, and the
expected number of inference passes. No NVIDIA device is available in the
development environment, so the end-to-end latency effect awaits a benchmark
on the deployment server.

## Measure prefill modes on the NVIDIA machine

On an actual CUDA session, the default is a 256-token prefill chunk with split
prefill enabled. Split prefill runs the final prompt token in a separate pass,
so the large vocabulary projection is requested only for that token. It also
adds an inference pass. CPU sessions retain the pinned 128-token chunk and do
not split prefill. A 512-token CUDA chunk remains available for comparison.

These CUDA defaults are enabled for evaluation on the target RTX 3070 or 4070;
their speed, memory use, and answer quality have **not** been measured there.
Changing prefill passes can change floating-point rounding and answers. The
application already selects CUDA automatically when ONNX Runtime offers it.
For tomorrow's validation, explicitly request CUDA so an unavailable or
silently substituted provider fails instead of producing a CPU result.
Install the pinned model and `onnxruntime-gpu` as described in
[local-ai.md](local-ai.md), then run from the repository root:

Windows PowerShell:

```powershell
$env:TRENDANALIZER_INFERENCE_PROVIDER = "cuda"
.\.venv\Scripts\python.exe -m scripts.check_cuda_runtime
.\.venv\Scripts\python.exe -m scripts.benchmark_qwen_prefill_cuda --compare-split --repeats 3 --warmups 1 --output qwen-prefill-benchmark.json
```

Linux:

```sh
export TRENDANALIZER_INFERENCE_PROVIDER=cuda
.venv/bin/python -m scripts.check_cuda_runtime
.venv/bin/python -m scripts.benchmark_qwen_prefill_cuda --compare-split --repeats 3 --warmups 1 --output qwen-prefill-benchmark.json
```

The benchmark rejects CPU fallback, measures prompt, prefill, and decode time,
separately reports computed and reused prefill tokens after warmup,
records the actual chunk and the token count of the logits-producing prefill
pass, checks schema and exact source quotations, and compares output text byte
for byte against the pinned `128/split0` run. It runs split off and on for each
chunk and reports both total and prefill timing. A mode is eligible only when
both cases pass quality checks and exactly match the baseline output. A failed
check makes the command exit with status 1. The two synthetic cases are a
deployment smoke check, not a precision/recall estimate. Check representative
real analyses for result quality, memory use, and end-to-end time before relying
on the new defaults. To return to the pinned 128-token, unsplit CUDA path, set
both overrides before starting the application:

```powershell
$env:TRENDANALIZER_QWEN_PREFILL_CHUNK = "128"
$env:TRENDANALIZER_QWEN_SPLIT_PREFILL = "0"
```

```sh
export TRENDANALIZER_QWEN_PREFILL_CHUNK=128
export TRENDANALIZER_QWEN_SPLIT_PREFILL=0
```

An invalid chunk value safely falls back to the pinned 128-token value.
