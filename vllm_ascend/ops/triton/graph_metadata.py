# SPDX-License-Identifier: Apache-2.0
"""Refresh graph-owned paged-attention buffers in one device launch."""

import torch
from vllm.triton_utils import tl, triton

METADATA_BLOCK_SIZE = 256
ATTENTION_QUERY_TILE = 16
ATTENTION_WORK_BLOCK = 32


def allocate_attention_work(token_capacity, request_capacity, device):
    # Row zero holds the live tile count. Remaining rows hold (request,
    # query offset). ceil(T / tile) + N bounds the sum of per-request ceils.
    capacity = triton.cdiv(token_capacity, ATTENTION_QUERY_TILE) + request_capacity
    return torch.empty((capacity + 1, 2), dtype=torch.int32, device=device)


@triton.jit(do_not_specialize=["count", "actual"])
def _refresh_attention_metadata_kernel(
    query_starts,
    slots,
    blocks,
    target_query_starts,
    target_slots,
    target_blocks,
    lengths,
    target_lengths,
    work,
    count,
    actual,
    TOKENS: tl.constexpr,
    ROWS: tl.constexpr,
    COLUMNS: tl.constexpr,
    SOURCE_COLUMNS: tl.constexpr,
    QUERY_STRIDE: tl.constexpr,
    SLOT_STRIDE: tl.constexpr,
    BLOCK_ROW_STRIDE: tl.constexpr,
    BLOCK_COLUMN_STRIDE: tl.constexpr,
    LENGTH_STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
    WORK_CAPACITY: tl.constexpr,
    QUERY_TILE: tl.constexpr,
    WORK_BLOCK: tl.constexpr,
    BR: tl.constexpr,
):
    x = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    boundary = tl.load(query_starts + x * QUERY_STRIDE, x <= count, other=actual)
    # The final dummy sequence consumes token padding; all unused request
    # rows before it are empty. Its KV writes remain masked by slot -1.
    boundary = tl.where(x == ROWS, TOKENS, boundary)
    tl.store(target_query_starts + x, boundary, x <= ROWS)
    slot = tl.load(slots + x * SLOT_STRIDE, x < actual, other=-1)
    tl.store(target_slots + x, slot, x < TOKENS)
    row, column = x // COLUMNS, x % COLUMNS
    block = tl.load(
        blocks + row * BLOCK_ROW_STRIDE + column * BLOCK_COLUMN_STRIDE,
        (row < count) & (column < SOURCE_COLUMNS),
        other=0,
    )
    tl.store(target_blocks + x, block, x < ROWS * COLUMNS)
    length = tl.load(lengths + x * LENGTH_STRIDE, x < count, other=0)
    tl.store(target_lengths + x, length, x < ROWS)
    # Keep integer compaction in this vector-only metadata kernel. The
    # attention matmul kernel consumes scalar descriptors, without carrying
    # a scan/reduction tensor through its nested Cube/Vector loops.
    seq = tl.arange(0, BR)
    starts = tl.load(query_starts + seq * QUERY_STRIDE, seq < count, other=actual).to(tl.int32)
    ends = tl.load(query_starts + (seq + 1) * QUERY_STRIDE, seq < count, other=actual).to(tl.int32)
    counts = tl.cdiv(ends - starts, QUERY_TILE)
    cumulative = tl.cumsum(counts, 0)
    total = tl.sum(counts, 0)
    if tl.program_id(0) == 0:
        tl.store(work, total)
    task = tl.program_id(0) * WORK_BLOCK + tl.arange(0, WORK_BLOCK)
    owner = tl.sum(((task[:, None] >= cumulative[None, :]) & (seq[None, :] < count)).to(tl.int32), 1)
    first = tl.sum(tl.where(owner[:, None] == seq[None, :], (cumulative - counts)[None, :], 0), 1)
    tl.store(work + (task + 1) * 2, tl.where(task < total, owner, -1), task < WORK_CAPACITY)
    tl.store(work + (task + 1) * 2 + 1, (task - first) * QUERY_TILE, task < WORK_CAPACITY)


def refresh_attention_metadata(target, source, count, actual):
    rows, columns = target.block_tables.shape
    tokens = target.slot_mapping.shape[0]
    size = max(tokens, rows * columns, rows + 1)
    work_capacity = target.attention_work.shape[0] - 1
    lengths = getattr(source, "seq_lens_device", None)
    if lengths is None:
        lengths = source.seq_lens.to(source.query_start_loc.device)
    programs = max(triton.cdiv(size, METADATA_BLOCK_SIZE), triton.cdiv(work_capacity, ATTENTION_WORK_BLOCK))
    _refresh_attention_metadata_kernel[(programs,)](
        source.query_start_loc,
        source.slot_mapping,
        source.block_tables,
        target.query_start_loc,
        target.slot_mapping,
        target.block_tables,
        lengths,
        target.seq_lens_device,
        target.attention_work,
        count,
        actual,
        TOKENS=tokens,
        ROWS=rows,
        COLUMNS=columns,
        SOURCE_COLUMNS=source.block_tables.shape[1],
        QUERY_STRIDE=source.query_start_loc.stride(0),
        SLOT_STRIDE=source.slot_mapping.stride(0),
        BLOCK_ROW_STRIDE=source.block_tables.stride(0),
        BLOCK_COLUMN_STRIDE=source.block_tables.stride(1),
        LENGTH_STRIDE=lengths.stride(0),
        BLOCK=METADATA_BLOCK_SIZE,
        WORK_CAPACITY=work_capacity,
        QUERY_TILE=ATTENTION_QUERY_TILE,
        WORK_BLOCK=ATTENTION_WORK_BLOCK,
        BR=triton.next_power_of_2(rows - 1),
        num_warps=4,
    )
