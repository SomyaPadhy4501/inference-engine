# Triton attention kernels + NF4 quantization: measured results

Custom Triton attention kernels (tiled prefill, split-KV decode) benchmarked against
PyTorch SDPA, plus an NF4 weight-memory measurement on Mistral-7B-Instruct-v0.2.

This repo started as a pair of Colab notebooks. The notebooks' saved A100 outputs are
preserved in `results/original-notebook-evidence.json` and in git history at `fa32edc`.
Everything below was re-measured from scratch; where the original numbers do not hold
up, that is stated plainly.

## Headline results (RTX 5070 Ti, 16GB, BF16)

**Decode — the custom kernel wins.** One query against a full KV cache, batch 8,
32 heads, dim 128. This is the memory-bound regime that dominates long-context
generation, and split-KV parallelization is what makes it fast at low batch.

| KV length | Triton | PyTorch SDPA | speedup vs SDPA |
|---:|---:|---:|---:|
| 1024 | 0.172 ms | 0.218 ms | 1.27x |
| 2048 | 0.331 ms | 0.425 ms | 1.28x |
| 4096 | 0.666 ms | 1.059 ms | **1.59x** |
| 8192 | 1.907 ms | 3.155 ms | 1.65x* |
| 16384 | 4.983 ms | 6.844 ms | 1.37x* |

**Prefill — the custom kernel loses.** Full-sequence attention, batch 2, 32 heads.

| Seq | naive | Triton | SDPA | naive/Triton | SDPA/Triton |
|---:|---:|---:|---:|---:|---:|
| 512 | 0.329 ms | 0.161 ms | 0.118 ms | 2.04x | 0.73x |
| 1024 | 2.753 ms | 0.581 ms | 0.444 ms | 4.74x | 0.76x |
| 2048 | 10.339 ms | 4.391 ms | 2.952 ms | 2.35x | 0.67x |
| 4096 | 46.421 ms | 16.332 ms | 12.111 ms | 2.84x | 0.74x |
| 8192 | OOM | 65.274 ms | 47.455 ms | n/a | 0.73x |

PyTorch SDPA is consistently ~1.35x faster than the hand-written prefill kernel. SDPA
dispatches to cuDNN/FlashAttention, which is more heavily tuned than anything written
here. The kernel does beat the naive baseline, but that baseline is a strawman.

**Weight memory — the NF4 claim holds.**

| | bytes | GB |
|---|---:|---:|
| FP16 | 14,483,464,448 | 14.48 |
| NF4 (double-quant off) | 4,450,703,616 | 4.45 |
| **Reduction** | | **3.25x** |

FP16 is counted exactly on the `meta` device (real module tree, no allocation), which
is what lets the comparison run on a GPU too small to hold FP16 Mistral-7B. NF4 is a
real GPU load with a verified forward pass, measured at 4.47 GB of CUDA allocation
against 4.45 GB of counted tensor storage. Tied weights are deduplicated by storage
pointer, and NF4 quantization state (absmax, code) is included.

\* Numbers marked with an asterisk varied across runs because a game was using the GPU
during measurement; 8192 decode measured 1.65x and 1.12x on two passes. The 4096 decode
figure reproduced at 1.59x and 1.60x and is the one to trust. Re-run on an idle GPU to
tighten the rest.

## Where the original resume numbers stand

| Original claim | Status |
|---|---|
| 4.65x at 8K context | **Does not reproduce as stated.** It is a non-causal attention microbenchmark against a naive materialized-softmax baseline on A100, not LLM inference. On this GPU the naive baseline OOMs at 8K/batch-2/32-head (needs ~25.7 GB), so the exact comparison cannot run at all. At reproducible shapes the naive ratio lands at 2.0–4.7x and swings with shape and GPU load. Against PyTorch SDPA the prefill kernel is *slower*. |
| 3.1x NF4 weight memory | **Holds.** Measured 3.25x, so the original claim was conservative. The notebook's 14.83/4.81 GB came from global CUDA allocations rather than isolated weight storage; the corrected measurement is cleaner and lands in the same place. |
| "Integrated vLLM" | **Overstated.** vLLM was called as a library. Its paged KV cache and continuous batching are vLLM's own; no custom kernel is wired into it, and the toy page allocator in the notebook is not connected to anything. |

Other notebook errors found: parent-process CUDA memory (0.3 GB) was reported as vLLM
worker memory; batch wall time divided by request count was labelled per-request
latency; a 8192x8192 BF16 matrix was called 512 MB when it is 128 MiB; the KV-cache
discussion used 32 heads and omitted the layer factor, but Mistral's GQA uses 8 KV
heads across 32 layers (2*32*8*128*2 = 131072 bytes/token at FP16); and a summary
claimed 1500+ tok/s that the displayed 530.4 tok/s does not support.

## Layout

```
inference_engine/attention.py   Triton kernels: tiled prefill, split-KV decode
benchmarks/attention.py         kernel benchmark vs naive and SDPA
benchmarks/weights.py           FP16 (meta) vs NF4 (real GPU) weight storage
benchmarks/vllm_batch.py        vLLM batched generation throughput
benchmarks/summary.py           print all results as tables
tests/                          correctness vs SDPA, storage accounting
results/                        saved measurements
reproduce_colab.ipynb           run the whole set on a cloud GPU
```

## Running it

Linux or WSL2. Windows Python cannot drive this environment, and vLLM's supported
GPU install is Linux only.

```bash
python3 -m venv .venv-linux && source .venv-linux/bin/activate
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements-gpu.txt

python -m unittest discover -s tests -v
python -m benchmarks.attention --batch 2 --heads 32 --output results/prefill.json
python -m benchmarks.attention --decode --batch 8 --heads 32 \
    --lengths 1024 2048 4096 8192 --output results/decode.json

REV=63a8b081895390a26e140280378bc85ec8bce07a
python -m benchmarks.weights --mode fp16 --revision $REV --output results/weights-fp16.json
python -m benchmarks.weights --mode nf4  --revision $REV --output results/weights-nf4.json
python -m benchmarks.summary
```

vLLM pins its own torch build, so install it in a separate environment:

```bash
python -m virtualenv .venv-vllm && ./.venv-vllm/bin/pip install vllm bitsandbytes
./.venv-vllm/bin/python -m benchmarks.vllm_batch --revision $REV --output results/vllm.json
```

Close other GPU workloads first. Timings are noise-sensitive, and a game or a browser
with hardware acceleration will move the numbers by tens of percent.

## Measurement protocol

The benchmark validates every custom output against SDPA before timing it, warms up
through Triton autotuning, times with CUDA events (excluding host dispatch), and saves
all samples alongside the median. Both baselines are reported: the naive
materialized-softmax path from the original notebook, and PyTorch SDPA. SDPA is an
auto-selected backend, not a pinned FlashAttention version.

The naive baseline holds the full N x N scores and probabilities at once; where that
does not fit, the run records a skip rather than silently shrinking the shape.

## Scope limits

The kernels handle non-causal attention with equal Q/K/V head counts, in FP16 or BF16.
Causal masking and GQA are not implemented, so these are not a drop-in Mistral
attention replacement — running them inside the model would need both, plus full-model
correctness checks. Decode assumes one query attending to all supplied KV positions.

`--dtype` defaults to BF16 on Ampere and newer and FP16 on older cards. FP16 exists so
the kernels run on a Turing T4, which has no BF16 tensor cores; T4 numbers are not
comparable to the BF16 results above.
