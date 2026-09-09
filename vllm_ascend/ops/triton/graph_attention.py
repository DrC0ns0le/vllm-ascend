# SPDX-License-Identifier: Apache-2.0
"""Paged causal attention with device lengths and no replay task updates.

The fixed launch reads graph-owned query boundaries, KV lengths and block
tables. Query tiles are compacted on device; empty requests and token padding
consume no attention tasks. This targets the small-context 910B FULL route.
"""

import torch
from vllm.triton_utils import tl, triton

ATTENTION_PROGRAMS = 32
QUERY_TILE = 16
KEY_TILE = 64


@triton.jit
def _graph_attention_kernel(
    q,
    k,
    v,
    cu,
    lengths,
    blocks,
    output,
    H: tl.constexpr,
    HK: tl.constexpr,
    D: tl.constexpr,
    QT: tl.constexpr,
    QH: tl.constexpr,
    QD: tl.constexpr,
    KB: tl.constexpr,
    KT: tl.constexpr,
    KH: tl.constexpr,
    KD: tl.constexpr,
    VB: tl.constexpr,
    VT: tl.constexpr,
    VH: tl.constexpr,
    VD: tl.constexpr,
    OT: tl.constexpr,
    OH: tl.constexpr,
    OD: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    COLUMNS: tl.constexpr,
    ROWS: tl.constexpr,
    BR: tl.constexpr,
    BQ: tl.constexpr,
    BK: tl.constexpr,
    SCALE: tl.constexpr,
    WINDOW: tl.constexpr,
    PROGRAMS: tl.constexpr,
):
    rows = tl.arange(0, BR)
    starts = tl.load(cu + rows, rows < ROWS, other=0)
    ends = tl.load(cu + rows + 1, rows < ROWS, other=0)
    counts = tl.cdiv(ends - starts, BQ)
    cumulative = tl.cumsum(counts, 0)
    total = tl.sum(counts, 0)
    for task in range(tl.program_id(0), total * H, PROGRAMS):
        tile, head = task // H, task % H
        row = tl.sum(((tile >= cumulative) & (rows < ROWS)).to(tl.int32), 0)
        first_tile = tl.sum(tl.where(rows == row, cumulative - counts, 0), 0)
        begin, end = tl.load(cu + row), tl.load(cu + row + 1)
        qpos = (tile - first_tile) * BQ + tl.arange(0, BQ)
        qvalid = begin + qpos < end
        kv_length = tl.load(lengths + row)
        prefix = kv_length - (end - begin)
        dim = tl.arange(0, D)
        query = tl.load(q + (begin + qpos[:, None]) * QT + head * QH + dim[None, :] * QD, qvalid[:, None], other=0)
        acc = tl.zeros((BQ, D), dtype=tl.float32)
        maximum = tl.full((BQ,), -float("inf"), dtype=tl.float32)
        denominator = tl.zeros((BQ,), dtype=tl.float32)
        limit = tl.minimum(kv_length, prefix + (tile - first_tile + 1) * BQ)
        lower = 0
        if WINDOW > 0:
            lower = tl.maximum(0, prefix + (tile - first_tile) * BQ - WINDOW + 1) // BK * BK
        for base in range(lower, limit, BK):
            pos = base + tl.arange(0, BK)
            block = tl.load(blocks + row * COLUMNS + pos // BLOCK_SIZE, pos < limit, other=0)
            valid = (pos < limit) & (block >= 0)
            kh = head // (H // HK)
            key = tl.load(
                k + block[None, :] * KB + (pos[None, :] % BLOCK_SIZE) * KT + kh * KH + dim[:, None] * KD,
                valid[None, :],
                other=0,
            )
            scores = tl.dot(query, key) * SCALE
            allowed = qvalid[:, None] & valid[None, :] & (pos[None, :] <= prefix + qpos[:, None])
            if WINDOW > 0:
                allowed &= pos[None, :] > prefix + qpos[:, None] - WINDOW
            scores = tl.where(allowed, scores, -float("inf"))
            next_maximum = tl.maximum(maximum, tl.max(scores, 1))
            # Empty window tiles and inactive queries must not form inf-inf.
            safe_maximum = tl.where(next_maximum == -float("inf"), 0.0, next_maximum)
            correction = tl.exp(maximum - safe_maximum)
            probability = tl.exp(scores - safe_maximum[:, None])
            value = tl.load(
                v + block[:, None] * VB + (pos[:, None] % BLOCK_SIZE) * VT + kh * VH + dim[None, :] * VD,
                valid[:, None],
                other=0,
            )
            acc = acc * correction[:, None] + tl.dot(probability.to(v.dtype.element_ty), value)
            denominator = denominator * correction + tl.sum(probability, 1)
            maximum = next_maximum
        result = acc / tl.where(denominator > 0, denominator, 1.0)[:, None]
        tl.store(output + (begin + qpos[:, None]) * OT + head * OH + dim[None, :] * OD, result, qvalid[:, None])


def graph_paged_attention(query, key, value, metadata, output, *, num_heads, scale, sliding_window=None):
    if key.ndim != 4 or value.shape != key.shape or key.shape[-1] not in (64, 128, 256):
        raise ValueError("Device FULL attention requires [block, token, KV head, dim] caches with dim 64/128/256")
    if query.dtype not in (torch.float16, torch.bfloat16) or key.dtype != query.dtype or value.dtype != query.dtype:
        raise ValueError("Device FULL attention requires matching FP16/BF16 query and KV caches")
    if num_heads % key.shape[2]:
        raise ValueError("Device FULL attention requires an integral GQA head ratio")
    q = query.view(query.shape[0], num_heads, key.shape[-1])
    out = output.view_as(q)
    rows = metadata.block_tables.shape[0] - 1  # Exclude the token-padding sentinel.
    _graph_attention_kernel[(ATTENTION_PROGRAMS,)](
        q,
        key,
        value,
        metadata.query_start_loc,
        metadata.seq_lens_device,
        metadata.block_tables,
        out,
        H=num_heads,
        HK=key.shape[2],
        D=key.shape[-1],
        QT=q.stride(0),
        QH=q.stride(1),
        QD=q.stride(2),
        KB=key.stride(0),
        KT=key.stride(1),
        KH=key.stride(2),
        KD=key.stride(3),
        VB=value.stride(0),
        VT=value.stride(1),
        VH=value.stride(2),
        VD=value.stride(3),
        OT=out.stride(0),
        OH=out.stride(1),
        OD=out.stride(2),
        BLOCK_SIZE=key.shape[1],
        COLUMNS=metadata.block_tables.shape[1],
        ROWS=rows,
        BR=triton.next_power_of_2(rows),
        BQ=QUERY_TILE,
        BK=KEY_TILE,
        SCALE=scale,
        WINDOW=sliding_window or 0,
        PROGRAMS=ATTENTION_PROGRAMS,
        num_warps=4,
        num_stages=1,
        multibuffer=False,
    )
    return output
