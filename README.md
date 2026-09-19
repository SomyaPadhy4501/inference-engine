# Triton attention kernels + NF4 quantization: measured results

Custom Triton attention kernels (tiled prefill, split-KV decode) benchmarked against
PyTorch SDPA, plus an NF4 weight-memory measurement on Mistral-7B-Instruct-v0.2.

This repo started as a pair of Colab notebooks. The notebooks' saved A100 outputs are
preserved in `results/original-notebook-evidence.json` and in git history at `fa32edc`.
Everything below was re-measured from scratch; where the original numbers do not hold
up, that is stated plainly.

## Headline results (RTX 5070 Ti, 16GB, BF16)

All figures are **min-of-N** over 50–100 samples. Interference from other GPU work only
ever makes a sample slower, so the fastest sample is the best estimate of true kernel
cost. This matters here: these runs were taken with a game on the GPU, and the median
ratios swung by 40% while the min ratios stayed flat to within 0.02x across independent
runs. `benchmarks/summary.py` prints both so the gap is visible.

**Decode — the custom kernel wins.** One query against a full KV cache, batch 8,
32 heads, dim 128. This is the memory-bound regime that dominates long-context
generation, and split-KV parallelization is what makes it fast at low batch.

| KV length | Triton | SDPA | speedup vs SDPA |
|---:|---:|---:|---:|
| 1024 | 0.169 ms | 0.216 ms | 1.28x |
| 2048 | 0.325 ms | 0.421 ms | 1.30x |
| 4096 | 0.640 ms | 0.830 ms | 1.30x |
| 8192 | 1.265 ms | 1.649 ms | 1.30x |
| 16384 | 2.713 ms | 3.575 ms | 1.32x |

Flat at **1.30x** across a 16x range of context lengths, reproduced across two
independent runs (at 8192: Triton 1.264/1.265 ms, SDPA 1.649/1.649 ms). The naive
baseline is only 1.05x slower than Triton here — decode never materializes a large
score matrix, so there is nothing for tiling to save. The win is over SDPA, whose
single-query path leaves the GPU underused.

**Prefill — the custom kernel loses.** Full-sequence attention, batch 2, 32 heads.

| Seq | naive | Triton | SDPA | naive/Triton | SDPA/Triton |
|---:|---:|---:|---:|---:|---:|
| 512 | 0.322 ms | 0.159 ms | 0.117 ms | 2.03x | 0.74x |
| 1024 | 1.369 ms | 0.576 ms | 0.439 ms | 2.38x | 0.76x |
| 2048 | 7.304 ms | 2.422 ms | 1.586 ms | 3.02x | 0.65x |
| 4096 | 40.579 ms | 11.679 ms | 8.181 ms | 3.47x | 0.70x |
| 8192 | OOM | 34.299 ms | 41.680 ms | n/a | 1.22x† |

PyTorch SDPA is consistently ~1.4x faster than the hand-written prefill kernel, which
dispatches to cuDNN/FlashAttention — more heavily tuned than anything written here. The
kernel does beat the naive baseline by 2.0–3.5x, but that baseline is a strawman.

† The 8192 row is an outlier and should not be quoted: Triton's samples there spanned
34–65 ms against SDPA's 42–47 ms, and it contradicts every other prefill length
including 8192 at batch 1 (0.64x). Free VRAM was down to ~8 GB during this run. Re-run
it on an idle GPU before believing it.

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

## Where the original resume numbers stand

| Original claim | Status |
|---|---|
| 4.65x at 8K context | **Does not reproduce as stated.** It is a non-causal attention microbenchmark against a naive materialized-softmax baseline on A100, not LLM inference. On this GPU the naive baseline OOMs at 8K/batch-2/32-head (needs ~25.7 GB), so the exact comparison cannot run at all. At reproducible shapes the naive ratio peaks at 3.5x. Against PyTorch SDPA the prefill kernel is *slower*. |
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
run_cloud.sh                    run the whole set on a cloud GPU
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

### On a cloud GPU

`run_cloud.sh` runs everything and adapts dtype and shapes to the GPU it finds:

```bash
git clone https://github.com/SomyaPadhy4501/inference-engine.git
cd inference-engine && bash run_cloud.sh
```

An L4 24GB is enough for BF16 and for FP16 Mistral-7B under vLLM. Reproducing the
original 8K/batch-2/32-head prefill comparison needs ~26GB for the naive baseline
alone, so it wants an A100 40GB; below that the script drops to batch 1 at 8K.

## Measurement protocol

The benchmark validates every custom output against SDPA before timing it, warms up
through Triton autotuning, times with CUDA events (excluding host dispatch), and saves
every sample alongside the median and the minimum. Both baselines are reported: the
naive materialized-softmax path from the original notebook, and PyTorch SDPA. SDPA is
an auto-selected backend, not a pinned FlashAttention version.

Quote the min-of-N ratio, not the median. Contention is one-sided — it can only add
time — so the median tracks how loaded the machine was, while the minimum converges on
the kernel's actual cost. A ratio that moves between the two statistics is a warning
that the run was noisy, which is exactly what the 8192 prefill row shows.

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
