"""Print a readable summary of whatever result JSON files exist in results/."""
import json
import sys
from pathlib import Path


def load(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError):
        return None


def attention_table(path):
    data = load(path)
    if not data or 'rows' not in data:
        return
    cfg, env = data['configuration'], data['environment']
    print(f"\n{path}")
    print(f"  {env['gpu']}  batch={cfg['batch']} heads={cfg['heads']} dim={cfg['dim']}"
          f" {'decode' if cfg['decode'] else 'prefill'}")
    print(f"  {'seq':>6} {'naive':>10} {'triton':>10} {'sdpa':>10} "
          f"{'naive/triton':>13} {'sdpa/triton':>12}")
    for row in data['rows']:
        def ms(name):
            t = row['timings'].get(name, {})
            return f"{t['median_ms']:.3f}" if 'median_ms' in t else 'n/a'
        def ratio(name):
            r = row.get(f'{name}_over_triton')
            return f"{r:.2f}x" if r else 'n/a'
        print(f"  {row['length']:>6} {ms('naive'):>10} {ms('triton'):>10} {ms('sdpa'):>10} "
              f"{ratio('naive'):>13} {ratio('sdpa'):>12}")


def weights_table(fp16_path, nf4_path):
    fp16, nf4 = load(fp16_path), load(nf4_path)
    if not fp16 or not nf4:
        return
    a = fp16['weight_storage_bytes']
    b = nf4['weight_storage_bytes']
    print(f"\nWeight storage ({nf4['configuration']['model']} @ "
          f"{nf4['configuration']['revision'][:12]})")
    print(f"  FP16 {a:>15,} bytes ({a / 1e9:.2f} GB)  [counted on meta]")
    print(f"  NF4  {b:>15,} bytes ({b / 1e9:.2f} GB)  [loaded on {nf4['devices'][0]}]")
    print(f"  Reduction: {a / b:.2f}x")
    delta = nf4.get('cuda_allocation_delta_bytes')
    if delta:
        print(f"  NF4 measured CUDA allocation delta: {delta / 1e9:.2f} GB")


def vllm_table(path):
    data = load(path)
    if not data or 'rows' not in data:
        return
    cfg = data['configuration']
    best = max(r['tokens_per_second'] for r in data['rows'])
    print(f"\nvLLM batched generation ({cfg['model']}, batch={cfg['batch']}, "
          f"{cfg['tokens']} tokens/request)")
    for i, r in enumerate(data['rows']):
        print(f"  run {i}: {r['generated_tokens']} tokens in "
              f"{r['batch_wall_seconds']:.2f}s = {r['tokens_per_second']:.1f} tok/s")
    print(f"  Best: {best:.1f} tok/s")


def main():
    results = Path(sys.argv[1] if len(sys.argv) > 1 else 'results')
    for path in sorted(results.glob('*.json')):
        attention_table(path)
    weights_table(results / 'weights-fp16.json', results / 'weights-nf4.json')
    vllm_table(results / 'vllm.json')
    print()


if __name__ == '__main__':
    main()
