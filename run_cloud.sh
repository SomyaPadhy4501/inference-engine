#!/usr/bin/env bash
# Run the full benchmark suite on a cloud GPU box (Lightning AI, RunPod, Colab, ...).
#
#   git clone https://github.com/SomyaPadhy4501/inference-engine.git
#   cd inference-engine && bash run_cloud.sh
#
# Picks dtype and shapes from the detected GPU, so it degrades gracefully on a small
# card instead of dying halfway through. Results land in results/ as JSON.
set -euo pipefail

REV=63a8b081895390a26e140280378bc85ec8bce07a
PY=${PY:-python}

echo "=== environment ==="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
$PY - <<'EOF'
import torch
major, minor = torch.cuda.get_device_capability()
vram = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f'{torch.cuda.get_device_name(0)}  sm_{major}{minor}  {vram:.1f} GB  torch {torch.__version__}')
with open('/tmp/gpu_facts', 'w') as f:
    f.write(f"DTYPE={'bf16' if major >= 8 else 'fp16'}\n")
    # FP16 Mistral-7B weights are 14.5 GB; leave room for the KV cache.
    f.write(f"VLLM_FP16={'1' if vram > 20 else '0'}\n")
    f.write(f"VLLM_4BIT={'1' if vram > 12 else '0'}\n")
    # The naive baseline materializes 3 * B * H * N^2 * 2 bytes at once.
    f.write(f"NAIVE_8K_BATCH={'2' if vram > 30 else '1'}\n")
EOF
source /tmp/gpu_facts
echo "dtype=$DTYPE  vllm_fp16=$VLLM_FP16  vllm_4bit=$VLLM_4BIT  naive_8k_batch=$NAIVE_8K_BATCH"

echo "=== dependencies ==="
$PY -m pip install -q transformers accelerate bitsandbytes

echo "=== correctness ==="
$PY -m unittest discover -s tests

echo "=== decode (the headline result) ==="
$PY -m benchmarks.attention --decode --batch 8 --heads 32 --dtype "$DTYPE" \
    --lengths 1024 2048 4096 8192 16384 --repeats 100 --warmup 30 \
    --output results/cloud-decode.json

echo "=== prefill, original notebook shapes ==="
$PY -m benchmarks.attention --batch 2 --heads 32 --dtype "$DTYPE" \
    --output results/cloud-prefill.json

echo "=== prefill at 8K with a baseline that fits ==="
$PY -m benchmarks.attention --batch "$NAIVE_8K_BATCH" --heads 32 --dtype "$DTYPE" \
    --lengths 8192 --output results/cloud-prefill-8k.json

echo "=== weight memory ==="
$PY -m benchmarks.weights --mode fp16 --revision $REV --output results/weights-fp16.json
$PY -m benchmarks.weights --mode nf4  --revision $REV --output results/weights-nf4.json

echo "=== vLLM batched generation ==="
if [ "$VLLM_FP16" = 1 ] || [ "$VLLM_4BIT" = 1 ]; then
    # vLLM pins its own torch build, so keep it out of the kernel environment.
    $PY -m pip install -q virtualenv
    $PY -m virtualenv -q .venv-vllm
    ./.venv-vllm/bin/pip install -q vllm bitsandbytes
    QUANT=$([ "$VLLM_FP16" = 1 ] && echo none || echo bitsandbytes)
    ./.venv-vllm/bin/python -m benchmarks.vllm_batch --revision $REV \
        --batch 16 --tokens 128 --quantization "$QUANT" --output results/vllm.json
else
    echo "skipped: not enough VRAM for a 7B model"
fi

echo "=== summary ==="
$PY -m benchmarks.summary
