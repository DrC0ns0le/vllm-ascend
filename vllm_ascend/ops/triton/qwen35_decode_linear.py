# SPDX-License-Identifier: Apache-2.0
"""BF16/FP16 small-M linears for 910B. All tiling decisions precede graph replay.

M=1 can use a vector GEMV; M>1 uses a single Cube tile for all decode rows,
sharing each weight load across requests. Optional split-K writes disjoint FP32
partials and reduces once, without atomics, counters, or cross-program waits.
Weights stay in their original ND [N,K] layout; no per-bucket weight copies.
"""

import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.triton_utils import get_aicore_num, get_vectorcore_num, init_device_properties_triton

GEMV_N = 8
GEMV_K = 512
GEMM_M = 16
GEMM_N = 64
GEMM_K = 128
REDUCE_BLOCK = 512


@triton.jit
def _decode_gemv(
    X,
    W,
    B,
    Y,
    N: tl.constexpr,
    K: tl.constexpr,
    SPLIT_K: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    CORES: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    # 8 x 512 FP32 accumulator (16 KiB), with bounded single-buffer live
    # tiles. In particular, never materialize an N x K or M x N x K tile.
    tiles: tl.constexpr = triton.cdiv(N, BN)
    span: tl.constexpr = triton.cdiv(K, SPLIT_K * BK) * BK
    for work in range(tl.program_id(0), tiles * SPLIT_K, CORES):
        part = work // tiles
        ns = (work % tiles) * BN + tl.arange(0, BN)
        ks = tl.arange(0, BK)
        acc = tl.zeros((BN, BK), tl.float32)
        for block in range(triton.cdiv(span, BK)):
            k = part * span + block * BK + ks
            x = tl.load(X + k, mask=k < K, other=0).to(tl.float32)
            w = tl.load(W + ns[:, None] * K + k[None, :], (ns[:, None] < N) & (k[None, :] < K), other=0)
            acc += w.to(tl.float32) * x[None, :]
        value = tl.sum(acc, 1)
        if HAS_BIAS and SPLIT_K == 1:
            value += tl.load(B + ns, ns < N, other=0).to(tl.float32)
        tl.store(Y + part * N + ns, value, ns < N)


@triton.jit
def _decode_skinny_gemm(
    X,
    W,
    B,
    Y,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    SX: tl.constexpr,
    SPLIT_K: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    CORES: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    tiles: tl.constexpr = triton.cdiv(N, BN)
    span: tl.constexpr = triton.cdiv(K, SPLIT_K * BK) * BK
    ms = tl.arange(0, BM)
    ks = tl.arange(0, BK)
    for work in range(tl.program_id(0), tiles * SPLIT_K, CORES):
        part = work // tiles
        ns = (work % tiles) * BN + tl.arange(0, BN)
        acc = tl.zeros((BM, BN), tl.float32)
        for block in range(triton.cdiv(span, BK)):
            k = part * span + block * BK + ks
            x = tl.load(X + ms[:, None] * SX + k[None, :], (ms[:, None] < M) & (k[None, :] < K), other=0)
            # Contiguous loads along K, followed by an on-chip transpose.
            # All M rows consume this weight tile in the same dot operation.
            w = tl.load(W + ns[:, None] * K + k[None, :], (ns[:, None] < N) & (k[None, :] < K), other=0)
            acc = tl.dot(x, tl.trans(w), acc)
        if HAS_BIAS and SPLIT_K == 1:
            acc += tl.load(B + ns, ns < N, other=0).to(tl.float32)[None, :]
        tl.store(Y + part * M * N + ms[:, None] * N + ns[None, :], acc, (ms[:, None] < M) & (ns[None, :] < N))


@triton.jit
def _decode_split_k_reduce(
    P, B, Y, SIZE: tl.constexpr, N: tl.constexpr, SPLIT_K: tl.constexpr, HAS_BIAS: tl.constexpr, BLOCK: tl.constexpr
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.zeros((BLOCK,), tl.float32)
    for part in tl.static_range(SPLIT_K):
        value += tl.load(P + part * SIZE + offsets, offsets < SIZE, other=0)
    if HAS_BIAS:
        value += tl.load(B + offsets % N, offsets < SIZE, other=0).to(tl.float32)
    tl.store(Y + offsets, value, offsets < SIZE)


def decode_linear(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None, split_k: int = 1, force_cube: bool = False
) -> torch.Tensor:
    """Capture-safe launch, also callable directly for NPU comparison tests.

    Weight ND format is established by the layer at load time. This function
    does not convert weights, inspect device values, tune, or synchronize.
    """
    if x.ndim != 2 or weight.ndim != 2 or not 1 <= x.shape[0] <= 8:
        raise ValueError("decode_linear requires [M,K] with 1 <= M <= 8 and [N,K] weights")
    m, k = x.shape
    n = weight.shape[0]
    if (
        x.dtype not in (torch.bfloat16, torch.float16)
        or weight.dtype != x.dtype
        or weight.device != x.device
        or weight.shape[1] != k
        or x.stride(1) != 1
        or not weight.is_contiguous()
    ):
        raise ValueError("decode_linear requires BF16/FP16 ND weights and contiguous K dimensions on the same device")
    if split_k not in (1, 2, 4):
        raise ValueError("split_k must be 1, 2, or 4")
    if bias is not None and (
        bias.shape != (n,) or bias.dtype != x.dtype or bias.device != x.device or not bias.is_contiguous()
    ):
        raise ValueError("decode_linear bias must be contiguous [N] with the input dtype and device")
    output = torch.empty((m, n), dtype=x.dtype, device=x.device)
    partial = output if split_k == 1 else torch.empty((split_k, m, n), dtype=torch.float32, device=x.device)
    init_device_properties_triton()
    if m == 1 and not force_cube:
        cores = min(get_vectorcore_num(), triton.cdiv(n, GEMV_N) * split_k)
        _decode_gemv[(cores,)](
            x,
            weight,
            bias,
            partial,
            n,
            k,
            split_k,
            bias is not None,
            cores,
            GEMV_N,
            GEMV_K,
            num_stages=1,
            multibuffer=False,
            enable_fp_fusion=False,
        )
    else:
        cores = min(get_aicore_num(), triton.cdiv(n, GEMM_N) * split_k)
        _decode_skinny_gemm[(cores,)](
            x,
            weight,
            bias,
            partial,
            m,
            n,
            k,
            x.stride(0),
            split_k,
            bias is not None,
            cores,
            GEMM_M,
            GEMM_N,
            GEMM_K,
            num_stages=2,
        )
    if split_k > 1:
        _decode_split_k_reduce[(triton.cdiv(m * n, REDUCE_BLOCK),)](
            partial, bias, output, m * n, n, split_k, bias is not None, REDUCE_BLOCK
        )
    return output
