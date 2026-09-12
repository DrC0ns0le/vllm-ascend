# SPDX-License-Identifier: Apache-2.0
"""Fuse the tiny BA projection and plain-layout Qwen3.5 QKVZ preparation.

The large, aligned QKVZ matmul is unchanged. Each program loads one activation
row once for a B/A pair, accumulates in FP32, and writes gates in the input dtype
to contiguous B/A buffers. It also copies its share of QKVZ to contiguous QKV/Z
outputs. There is no intermediate BA tensor or later Z reshape materialization.
"""

import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num, init_device_properties_triton


@triton.jit
def _gdn_ba_prepare(
    X,
    W,
    BIAS,
    QKVZ,
    QKV,
    Z,
    B,
    A,
    M: tl.constexpr,
    K: tl.constexpr,
    H: tl.constexpr,
    QKV_SIZE: tl.constexpr,
    Z_SIZE: tl.constexpr,
    SX: tl.constexpr,
    SQ: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    CORES: tl.constexpr,
    BLOCK_K: tl.constexpr,
    COPY: tl.constexpr,
    BLOCK_COPY: tl.constexpr,
):
    ks = tl.arange(0, BLOCK_K)
    cp = tl.arange(0, BLOCK_COPY)
    for work in range(tl.program_id(0), M * H, CORES):
        row = work // H
        head = work % H
        x = tl.load(X + row * SX + ks, ks < K, other=0).to(tl.float32)
        wb = tl.load(W + head * K + ks, ks < K, other=0).to(tl.float32)
        wa = tl.load(W + (H + head) * K + ks, ks < K, other=0).to(tl.float32)
        bv = tl.sum(wb * x, 0)
        av = tl.sum(wa * x, 0)
        if HAS_BIAS:
            bv += tl.load(BIAS + head).to(tl.float32)
            av += tl.load(BIAS + H + head).to(tl.float32)
        tl.store(B + row * H + head, bv)
        tl.store(A + row * H + head, av)

        # COPY can be non-power-of-two (384 for 4B). Ownership is disjoint
        # across heads, including the QKV/Z boundary and the final head.
        index = head * COPY + cp
        valid = (cp < COPY) & (index < QKV_SIZE + Z_SIZE)
        value = tl.load(QKVZ + row * SQ + index, valid, other=0)
        tl.store(QKV + row * QKV_SIZE + index, value, valid & (index < QKV_SIZE))
        z_index = tl.maximum(index - QKV_SIZE, 0)
        tl.store(Z + row * Z_SIZE + z_index, value, valid & (index >= QKV_SIZE))


def gdn_ba_prepare(
    x: torch.Tensor, qkvz: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None, qkv_size: int, head_dim: int
):
    m, k = x.shape
    heads = weight.shape[0] // 2
    z_size = heads * head_dim
    if (
        not 1 <= m <= 8
        or (k, heads, head_dim, qkv_size) not in ((2048, 16, 128, 6144), (2560, 32, 128, 8192))
        or x.dtype not in (torch.bfloat16, torch.float16)
        or qkvz.dtype != x.dtype
        or weight.dtype != x.dtype
        or qkvz.shape != (m, qkv_size + z_size)
        or weight.shape != (2 * heads, k)
        or x.stride(1) != 1
        or qkvz.stride(1) != 1
        or not weight.is_contiguous()
        or weight.device != x.device
        or qkvz.device != x.device
    ):
        raise ValueError("gdn_ba_prepare requires Qwen3.5-2B/4B TP=1 BF16/FP16 plain-layout projections, M in [1,8]")
    if bias is not None and (
        bias.shape != (2 * heads,) or bias.dtype != x.dtype or bias.device != x.device or not bias.is_contiguous()
    ):
        raise ValueError("gdn_ba_prepare bias must be contiguous [2*heads] with the input dtype and device")
    qkv = torch.empty((m, qkv_size), device=x.device, dtype=x.dtype)
    z = torch.empty((m, heads, head_dim), device=x.device, dtype=x.dtype)
    b = torch.empty((m, heads), device=x.device, dtype=x.dtype)
    a = torch.empty_like(b)
    init_device_properties_triton()
    cores = min(get_vectorcore_num(), m * heads)
    # next_power_of_2 is a Python helper on older Triton-Ascend versions:
    # evaluate it here on ints, never on a tl.constexpr inside the JIT body.
    # Keep COPY distinct: the 4B layout owns 384 elements/head, not 512.
    copy = triton.cdiv(qkv_size + z_size, heads)
    block_copy = triton.next_power_of_2(copy)
    _gdn_ba_prepare[(cores,)](
        x,
        weight,
        bias,
        qkvz,
        qkv,
        z,
        b,
        a,
        m,
        k,
        heads,
        qkv_size,
        z_size,
        x.stride(0),
        qkvz.stride(0),
        bias is not None,
        cores,
        triton.next_power_of_2(k),
        copy,
        block_copy,
        num_stages=1,
        multibuffer=False,
        enable_fp_fusion=False,
    )
    return qkv, z, b, a
