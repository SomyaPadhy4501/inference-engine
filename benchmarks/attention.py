"""Attention kernel microbenchmark: Triton vs a naive baseline vs PyTorch SDPA.

This times one attention operation in isolation. It is not LLM generation, and a
speedup here does not translate one-for-one into end-to-end tokens per second.
"""
import argparse
import gc
import math
import statistics
import torch
import torch.nn.functional as F
from inference_engine.attention import flash_attention_triton, flash_decoding_triton
from benchmarks.common import metadata, save


def naive(q, k, v):
    """Materializes the full N x N score and probability matrices, as the original
    notebook's baseline did. This is the O(N^2) memory path Flash Attention avoids."""
    return torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1]), -1) @ v


def measure(fn, warmup, repeats):
    """Time with CUDA events so host-side dispatch is excluded from each sample."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    end = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    for i in range(repeats):
        start[i].record()
        fn()
        end[i].record()
    torch.cuda.synchronize()
    samples = [s.elapsed_time(e) for s, e in zip(start, end)]
    return dict(median_ms=statistics.median(samples), min_ms=min(samples),
                stdev_ms=statistics.stdev(samples) if repeats > 1 else 0.0,
                samples_ms=samples)


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--lengths', type=int, nargs='+', default=[512, 1024, 2048, 4096, 8192])
    p.add_argument('--batch', type=int, default=2)
    p.add_argument('--heads', type=int, default=32)
    p.add_argument('--dim', type=int, default=128)
    p.add_argument('--decode', action='store_true')
    p.add_argument('--repeats', type=int, default=50)
    p.add_argument('--warmup', type=int, default=20)
    p.add_argument('--dtype', choices=['bf16', 'fp16'], default=None,
                   help='defaults to bf16 on Ampere+, fp16 on older GPUs such as the T4')
    p.add_argument('--output', default='results/attention.json')
    a = p.parse_args()
    if min(a.lengths + [a.batch, a.heads, a.dim, a.repeats, a.warmup]) <= 0:
        p.error('All sizes and iteration counts must be positive')
    if a.dtype is None:
        a.dtype = 'bf16' if torch.cuda.get_device_capability()[0] >= 8 else 'fp16'
    dtype = torch.bfloat16 if a.dtype == 'bf16' else torch.float16

    torch.manual_seed(42)
    custom = flash_decoding_triton if a.decode else flash_attention_triton
    report = dict(environment=metadata(), configuration=vars(a),
                  scope='non-causal MHA kernel only; no Mistral, GQA, or model integration',
                  rows=[])

    for n in a.lengths:
        q = torch.randn(a.batch, a.heads, 1 if a.decode else n, a.dim,
                        device='cuda', dtype=dtype)
        k = torch.randn(a.batch, a.heads, n, a.dim, device='cuda', dtype=dtype)
        v = torch.randn_like(k)
        row = dict(length=n, timings={})

        # Validate the full custom output against SDPA before timing anything.
        ref = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        actual = custom(q, k, v)
        row['max_abs_error_vs_sdpa'] = (actual.float() - ref.float()).abs().max().item()
        torch.testing.assert_close(actual, ref, atol=0.01, rtol=0.02)
        del ref, actual

        for name, fn in [('naive', naive), ('triton', custom),
                         ('sdpa', F.scaled_dot_product_attention)]:
            # Naive holds scores and probabilities at once. Skip rather than crash.
            estimate = 3 * a.batch * a.heads * q.shape[2] * n * 2
            if name == 'naive' and estimate > torch.cuda.mem_get_info()[0] * 0.85:
                row['timings'][name] = dict(status='skipped_insufficient_vram',
                                            estimated_temp_bytes=estimate)
                continue
            try:
                result = measure(lambda: fn(q, k, v), a.warmup, a.repeats)
                torch.cuda.reset_peak_memory_stats()
                base = torch.cuda.memory_allocated()
                fn(q, k, v)
                torch.cuda.synchronize()
                result['peak_extra_allocated_bytes'] = torch.cuda.max_memory_allocated() - base
                row['timings'][name] = result
            except torch.OutOfMemoryError:
                row['timings'][name] = dict(status='out_of_memory')
                gc.collect()
                torch.cuda.empty_cache()

        # Report both ratios. Interference only ever adds time, so the minimum is the
        # robust estimate of true kernel cost; the median is what a loaded GPU felt like.
        for base in ('naive', 'sdpa'):
            if all('median_ms' in row['timings'][key] for key in (base, 'triton')):
                row[f'{base}_over_triton'] = (row['timings'][base]['median_ms']
                                              / row['timings']['triton']['median_ms'])
                row[f'{base}_over_triton_min'] = (row['timings'][base]['min_ms']
                                                  / row['timings']['triton']['min_ms'])
        report['rows'].append(row)
        save(a.output, report)
        print(f"n={n:>6} " + " ".join(
            f"{name}={t.get('min_ms', float('nan')):.3f}ms" for name, t in row['timings'].items())
            + f"  naive/triton={row.get('naive_over_triton_min', float('nan')):.2f}x"
              f"  sdpa/triton={row.get('sdpa_over_triton_min', float('nan')):.2f}x"
              " (min-of-N)", flush=True)
        del q, k, v
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
