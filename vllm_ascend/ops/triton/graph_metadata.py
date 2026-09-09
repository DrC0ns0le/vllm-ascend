# SPDX-License-Identifier: Apache-2.0
"""Refresh graph-owned paged-attention buffers in one device launch."""

from vllm.triton_utils import tl, triton

METADATA_BLOCK_SIZE = 256


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


def refresh_attention_metadata(target, source, count, actual):
    rows, columns = target.block_tables.shape
    tokens = target.slot_mapping.shape[0]
    size = max(tokens, rows * columns, rows + 1)
    lengths = getattr(source, "seq_lens_device", None)
    if lengths is None:
        lengths = source.seq_lens.to(source.query_start_loc.device)
    _refresh_attention_metadata_kernel[(triton.cdiv(size, METADATA_BLOCK_SIZE),)](
        source.query_start_loc,
        source.slot_mapping,
        source.block_tables,
        target.query_start_loc,
        target.slot_mapping,
        target.block_tables,
        lengths,
        target.seq_lens_device,
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
        num_warps=4,
    )
