"""Measure model weight storage: FP16 baseline vs NF4, in a fresh process.

FP16 uses the `meta` device, which builds the real module tree and counts the exact
bytes a load would allocate without reserving them. This is what lets the comparison
run on a GPU too small to hold FP16 Mistral-7B. NF4 is loaded for real on the GPU,
so its number is a measurement, not an estimate.
"""
import argparse
import torch
from benchmarks.common import metadata, save


def storage_bytes(model):
    """Sum unique tensor storages: parameters, buffers, and NF4 quantization state.

    Deduplicates by storage pointer so tied weights are not double counted. On `meta`
    there is no real storage, so fall back to the tensor's logical byte size.
    """
    seen = set()
    total = 0

    def visit(value):
        nonlocal total
        if isinstance(value, torch.Tensor):
            if value.device.type == 'meta':
                total += value.nelement() * value.element_size()
                return
            storage = value.untyped_storage()
            key = (str(value.device), storage.data_ptr())
            if key not in seen:
                seen.add(key)
                total += storage.nbytes()
        elif isinstance(value, dict):
            for item in value.values():
                visit(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item)

    for tensor in list(model.parameters()) + list(model.buffers()):
        visit(tensor)
        state = getattr(tensor, 'quant_state', None)
        if state is not None:
            visit(state.as_dict(packed=False))
    return total


def main():
    from transformers import AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['fp16', 'nf4'], required=True)
    p.add_argument('--model', default='mistralai/Mistral-7B-Instruct-v0.2')
    p.add_argument('--revision', required=True, help='Same immutable model commit for both modes')
    p.add_argument('--output', required=True)
    a = p.parse_args()
    environment = metadata()

    if a.mode == 'fp16':
        config = AutoConfig.from_pretrained(a.model, revision=a.revision)
        with torch.device('meta'):
            model = AutoModelForCausalLM.from_config(config, dtype=torch.float16)
        report = dict(devices=['meta'], cuda_allocation_delta_bytes=None,
                      note='Exact FP16 weight bytes from the real module tree on `meta`; '
                           'no GPU allocation was made, so this is a byte count, not measured VRAM.')
    else:
        torch.cuda.synchronize()
        base = torch.cuda.memory_allocated()
        model = AutoModelForCausalLM.from_pretrained(
            a.model, revision=a.revision, dtype=torch.float16, device_map={'': 'cuda'},
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type='nf4',
                bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=False),
        ).eval()
        torch.cuda.synchronize()
        devices = sorted({str(x.device) for x in model.parameters()})
        if devices != ['cuda:0']:
            raise RuntimeError(f'Unexpected placement {devices}; CPU offload is not permitted here')
        report = dict(devices=devices,
                      cuda_allocation_delta_bytes=torch.cuda.memory_allocated() - base,
                      note='Loaded on GPU. Double quantization is off, matching the original notebook.')
        # Confirm the quantized model actually runs; activations are not counted as weights.
        with torch.inference_mode():
            logits = model(torch.tensor([[1, 2, 3, 4]], device='cuda'), use_cache=False).logits
        report['forward_logits_finite'] = bool(torch.isfinite(logits).all())
        if not report['forward_logits_finite']:
            raise RuntimeError('Non-finite logits from the NF4 model')

    report = dict(environment=environment, configuration=vars(a),
                  weight_storage_bytes=storage_bytes(model), **report)
    save(a.output, report)
    print({k: v for k, v in report.items() if k != 'environment'})


if __name__ == '__main__':
    main()
