"""Triton attention kernels: tiled prefill (FlashAttention-style) and split-KV decode.

Scope: non-causal multi-head attention in FP16 or BF16 with equal Q/K/V head counts.
These are standalone kernels benchmarked against PyTorch, not a drop-in Mistral
attention replacement — causal masking and GQA are not implemented.

FP16 is supported so the kernels run on Turing (T4), whose tensor cores predate BF16.
"""
import torch
import triton
import triton.language as tl

# exp2 maps to a single hardware instruction; exp does not. Folding log2(e) into the
# score scale lets the online softmax use exp2 throughout. Kernels can only read
# globals declared as constexpr.
LOG2E = tl.constexpr(1.4426950408889634)


def _prefill_configs():
    configs = []
    for block_m, block_n, warps, stages in [
        (128, 128, 8, 3), (128, 64, 8, 4), (128, 64, 4, 4),
        (64, 128, 8, 3), (64, 64, 4, 4), (64, 32, 4, 5),
    ]:
        configs.append(triton.Config({'BLOCK_M': block_m, 'BLOCK_N': block_n},
                                     num_warps=warps, num_stages=stages))
    return configs


@triton.autotune(configs=_prefill_configs(), key=['N_CTX', 'HEAD_DIM'])
@triton.jit
def _prefill_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr,
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_kn, stride_kk,
    stride_vb, stride_vh, stride_vn, stride_vk,
    stride_ob, stride_oh, stride_om, stride_ok,
    NUM_HEADS,
    N_CTX: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """One program per (batch, head, query tile). Q stays in registers while the
    program streams K/V tiles and folds them into an online softmax."""
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    batch_idx = pid_bh // NUM_HEADS
    head_idx = pid_bh % NUM_HEADS

    q_base = Q_ptr + batch_idx * stride_qb + head_idx * stride_qh
    k_base = K_ptr + batch_idx * stride_kb + head_idx * stride_kh
    v_base = V_ptr + batch_idx * stride_vb + head_idx * stride_vh
    o_base = O_ptr + batch_idx * stride_ob + head_idx * stride_oh

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, HEAD_DIM)
    offs_n = tl.arange(0, BLOCK_N)

    q_ptrs = q_base + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk
    q = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)

    qk_scale: tl.constexpr = LOG2E / (HEAD_DIM ** 0.5)

    m_i = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # Full K/V tiles need no bounds check. Split the loop so the hot path stays branch
    # free and only the ragged tail pays for masking.
    full_n = (N_CTX // BLOCK_N) * BLOCK_N

    for start_n in range(0, full_n, BLOCK_N):
        cur_n = start_n + offs_n
        k = tl.load(k_base + cur_n[:, None] * stride_kn + offs_k[None, :] * stride_kk)
        scores = tl.dot(q, tl.trans(k)) * qk_scale

        m_new = tl.maximum(m_i, tl.max(scores, axis=1))
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(scores - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]

        v = tl.load(v_base + cur_n[:, None] * stride_vn + offs_k[None, :] * stride_vk)
        acc = tl.dot(p.to(v.dtype), v, acc)
        m_i = m_new

    if full_n < N_CTX:
        cur_n = full_n + offs_n
        n_mask = cur_n < N_CTX
        k = tl.load(k_base + cur_n[:, None] * stride_kn + offs_k[None, :] * stride_kk,
                    mask=n_mask[:, None], other=0.0)
        scores = tl.dot(q, tl.trans(k)) * qk_scale
        scores = tl.where(n_mask[None, :], scores, float('-inf'))

        m_new = tl.maximum(m_i, tl.max(scores, axis=1))
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(scores - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]

        v = tl.load(v_base + cur_n[:, None] * stride_vn + offs_k[None, :] * stride_vk,
                    mask=n_mask[:, None], other=0.0)
        acc = tl.dot(p.to(v.dtype), v, acc)

    acc = acc / l_i[:, None]
    o_ptrs = o_base + offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok
    tl.store(o_ptrs, acc.to(O_ptr.dtype.element_ty), mask=offs_m[:, None] < N_CTX)


def flash_attention_triton(Q, K, V):
    """Tiled non-causal attention over a full query sequence."""
    validate_inputs(Q, K, V, decode=False)
    B, H, N, d = Q.shape
    O = torch.empty_like(Q)
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_M']), B * H)
    _prefill_kernel[grid](
        Q, K, V, O,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        H, N_CTX=N, HEAD_DIM=d,
    )
    return O


@triton.jit
def _decode_stage1_kernel(
    Q_ptr, K_ptr, V_ptr, Partial_O_ptr, Partial_lse_ptr,
    stride_qb, stride_qh, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_pob, stride_poh, stride_pos, stride_pod,
    stride_plb, stride_plh, stride_pls,
    NUM_HEADS,
    SEQ_LEN: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
):
    """One program per (batch, head, KV split). Splitting the KV axis is what keeps
    the GPU busy at batch 1, where one query row alone cannot fill the SMs."""
    pid_split = tl.program_id(0)
    pid_bh = tl.program_id(1)
    batch_idx = pid_bh // NUM_HEADS
    head_idx = pid_bh % NUM_HEADS

    split_size = tl.cdiv(SEQ_LEN, NUM_SPLITS)
    start_n = pid_split * split_size
    end_n = tl.minimum(start_n + split_size, SEQ_LEN)

    offs_d = tl.arange(0, HEAD_DIM)
    q = tl.load(Q_ptr + batch_idx * stride_qb + head_idx * stride_qh + offs_d * stride_qd)

    qk_scale: tl.constexpr = LOG2E / (HEAD_DIM ** 0.5)

    m_i = float('-inf')
    l_i = 0.0
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)

    for block_start in range(start_n, end_n, BLOCK_N):
        offs_n = block_start + tl.arange(0, BLOCK_N)
        n_mask = offs_n < end_n

        k = tl.load(K_ptr + batch_idx * stride_kb + head_idx * stride_kh
                    + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd,
                    mask=n_mask[:, None], other=0.0)
        scores = tl.sum(q[None, :].to(tl.float32) * k.to(tl.float32), axis=1) * qk_scale
        scores = tl.where(n_mask, scores, float('-inf'))

        m_new = tl.maximum(m_i, tl.max(scores, axis=0))
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(scores - m_new)
        l_i = l_i * alpha + tl.sum(p, axis=0)
        acc = acc * alpha

        v = tl.load(V_ptr + batch_idx * stride_vb + head_idx * stride_vh
                    + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd,
                    mask=n_mask[:, None], other=0.0)
        acc += tl.sum(p[:, None] * v.to(tl.float32), axis=0)
        m_i = m_new

    # Store a normalized partial plus its log-sum-exp; stage 2 reweights from the LSE.
    acc = acc / tl.maximum(l_i, 1.0e-30)
    tl.store(Partial_O_ptr + batch_idx * stride_pob + head_idx * stride_poh
             + pid_split * stride_pos + offs_d * stride_pod, acc)
    tl.store(Partial_lse_ptr + batch_idx * stride_plb + head_idx * stride_plh
             + pid_split * stride_pls, tl.log2(l_i) + m_i)


@triton.jit
def _decode_stage2_kernel(
    Partial_O_ptr, Partial_lse_ptr, O_ptr,
    stride_pob, stride_poh, stride_pos, stride_pod,
    stride_plb, stride_plh, stride_pls,
    stride_ob, stride_oh, stride_od,
    NUM_HEADS,
    HEAD_DIM: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
):
    """Combine the per-split partials into one output, weighted by exp2 of their LSEs."""
    pid_bh = tl.program_id(0)
    batch_idx = pid_bh // NUM_HEADS
    head_idx = pid_bh % NUM_HEADS

    offs_d = tl.arange(0, HEAD_DIM)
    offs_s = tl.arange(0, NUM_SPLITS)

    lse = tl.load(Partial_lse_ptr + batch_idx * stride_plb
                  + head_idx * stride_plh + offs_s * stride_pls)
    weights = tl.exp2(lse - tl.max(lse, axis=0))
    weights = weights / tl.sum(weights, axis=0)

    partials = tl.load(Partial_O_ptr + batch_idx * stride_pob + head_idx * stride_poh
                       + offs_s[:, None] * stride_pos + offs_d[None, :] * stride_pod)
    acc = tl.sum(weights[:, None] * partials, axis=0)

    tl.store(O_ptr + batch_idx * stride_ob + head_idx * stride_oh + offs_d * stride_od,
             acc.to(O_ptr.dtype.element_ty))


def flash_decoding_triton(Q, K, V, num_splits=16):
    """Split-KV attention for a single query position against a full KV cache."""
    validate_inputs(Q, K, V, decode=True)
    if num_splits < 1 or num_splits & (num_splits - 1):
        raise ValueError('num_splits must be a positive power of two')
    B, H, _, d = Q.shape
    N = K.shape[2]

    q = Q.squeeze(2).contiguous()
    partial_O = torch.empty(B, H, num_splits, d, device=Q.device, dtype=torch.float32)
    partial_lse = torch.empty(B, H, num_splits, device=Q.device, dtype=torch.float32)

    _decode_stage1_kernel[(num_splits, B * H)](
        q, K, V, partial_O, partial_lse,
        q.stride(0), q.stride(1), q.stride(2),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        partial_O.stride(0), partial_O.stride(1), partial_O.stride(2), partial_O.stride(3),
        partial_lse.stride(0), partial_lse.stride(1), partial_lse.stride(2),
        H, SEQ_LEN=N, HEAD_DIM=d, BLOCK_N=128, NUM_SPLITS=num_splits,
    )

    O = torch.empty(B, H, d, device=Q.device, dtype=Q.dtype)
    _decode_stage2_kernel[(B * H,)](
        partial_O, partial_lse, O,
        partial_O.stride(0), partial_O.stride(1), partial_O.stride(2), partial_O.stride(3),
        partial_lse.stride(0), partial_lse.stride(1), partial_lse.stride(2),
        O.stride(0), O.stride(1), O.stride(2),
        H, HEAD_DIM=d, NUM_SPLITS=num_splits,
    )
    return O.unsqueeze(2)


def validate_inputs(q, k, v, decode):
    if any(x.ndim != 4 for x in (q, k, v)):
        raise ValueError('Expected [batch, heads, sequence, dimension] tensors')
    if any(x.device != q.device or x.dtype != q.dtype for x in (k, v)):
        raise ValueError('Inputs must share device and dtype')
    if not q.is_cuda or q.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError('These kernels support CUDA FP16 or BF16 only')
    if k.shape != v.shape or q.shape[:2] != k.shape[:2] or q.shape[-1] != k.shape[-1]:
        raise ValueError('Matching Q/K/V heads required; GQA is not implemented')
    if q.shape[-1] not in (32, 64, 128) or min(q.shape) < 1 or k.shape[2] < 1:
        raise ValueError('Nonempty tensors and head dimension 32, 64, or 128 required')
    if (decode and q.shape[2] != 1) or (not decode and q.shape[2] != k.shape[2]):
        raise ValueError('Decode requires one query; prefill requires equal sequence lengths')
