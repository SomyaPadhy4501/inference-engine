"""vLLM batched generation throughput.

vLLM is used here as a library. Its continuous batching and paged KV cache are vLLM's
own; the Triton kernels in this repo are not wired into it.
"""
import argparse
import time
from benchmarks.common import metadata, save


def main():
    from vllm import LLM, SamplingParams
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', default='mistralai/Mistral-7B-Instruct-v0.2')
    p.add_argument('--revision', required=True)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--tokens', type=int, default=128)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--max-model-len', type=int, default=8192)
    p.add_argument('--gpu-memory-utilization', type=float, default=0.85)
    p.add_argument('--quantization', choices=['none', 'bitsandbytes'], default='none',
                   help='bitsandbytes 4-bit lets a 16GB GPU hold a 7B model')
    p.add_argument('--output', default='results/vllm.json')
    a = p.parse_args()
    if min(a.batch, a.tokens, a.repeats) <= 0:
        p.error('Batch, tokens, repeats must be positive')

    llm = LLM(model=a.model, revision=a.revision,
              dtype='float16' if a.quantization == 'none' else 'auto',
              quantization=None if a.quantization == 'none' else a.quantization,
              max_model_len=a.max_model_len,
              gpu_memory_utilization=a.gpu_memory_utilization,
              enable_prefix_caching=False)
    params = SamplingParams(temperature=0, max_tokens=a.tokens, ignore_eos=True)
    prompts = [f'Explain GPU inference optimization topic {i}:' for i in range(a.batch)]

    llm.generate(prompts, params, use_tqdm=False)  # warm up
    rows = []
    for _ in range(a.repeats):
        start = time.perf_counter()
        outputs = llm.generate(prompts, params, use_tqdm=False)
        elapsed = time.perf_counter() - start
        count = sum(len(o.outputs[0].token_ids) for o in outputs)
        rows.append(dict(batch_wall_seconds=elapsed, generated_tokens=count,
                         tokens_per_second=count / elapsed))
        print(rows[-1], flush=True)

    save(a.output, dict(
        environment=metadata(), configuration=vars(a), rows=rows,
        note='Aggregate batch throughput. Batch wall time over request count is not '
             'per-request latency, and the parent process does not see vLLM worker VRAM.'))


if __name__ == '__main__':
    main()
