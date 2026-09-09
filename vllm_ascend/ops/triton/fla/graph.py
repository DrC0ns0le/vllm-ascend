# SPDX-License-Identifier: Apache-2.0
"""Device-metadata baseline GDN for whole-model mixed graph capture.

All launch sizes depend on tensor capacity, never current request lengths.
The existing chunk kernels do the arithmetic; an empty sentinel sequence
makes excess tasks inert. Read/write anchors are explicit so the state
interface does not assume that future checkpoint caches update in place.
"""

import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.gdn_graph_metadata import GDN_GRAPH_HEAD_DIM, GDNFullGraphMetadata

from .chunk_delta_h import chunk_gated_delta_rule_fwd_h
from .chunk_o import chunk_fwd_o
from .chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd
from .cumsum import chunk_local_cumsum
from .l2norm import l2norm_fwd
from .single_token import single_token_gdn
from .solve_tril import solve_tril
from .wy_fast import recompute_w_u_fwd

CHUNK_SIZE = 64
SOLVE_BLOCK_SIZE = 1216
CUMSUM_WORKING_SET = 2**18
STATE_BLOCK_SIZE = 1024
METADATA_BLOCK_SIZE = 32


@triton.jit(do_not_specialize=["count"])
def _chunk_metadata_kernel(
    cu,
    state_indices,
    initial_flags,
    target_cu,
    read_indices,
    write_indices,
    target_flags,
    chunk_indices,
    solve_indices,
    cumsum_indices,
    offsets,
    count,
    N: tl.constexpr,
    CHUNK_CAPACITY: tl.constexpr,
    SOLVE_CAPACITY: tl.constexpr,
    CUMSUM_CAPACITY: tl.constexpr,
    CHUNK: tl.constexpr,
    SOLVE_CHUNK: tl.constexpr,
    CUMSUM_CHUNK: tl.constexpr,
    QUERY_STRIDE: tl.constexpr,
    STATE_STRIDE: tl.constexpr,
    FLAG_STRIDE: tl.constexpr,
    BN: tl.constexpr,
    BC: tl.constexpr,
    SKIP_SINGLE_TOKEN: tl.constexpr,
):
    seq = tl.arange(0, BN)
    terminal = tl.load(cu + count * QUERY_STRIDE)
    starts = tl.load(cu + seq * QUERY_STRIDE, seq < count, other=terminal)
    ends = tl.load(cu + (seq + 1) * QUERY_STRIDE, seq < count, other=terminal)
    if tl.program_id(0) == 0:
        slots = tl.load(state_indices + seq * STATE_STRIDE, seq < count, other=-1)
        initial = tl.load(initial_flags + seq * FLAG_STRIDE, seq < count, other=0)
        tl.store(target_cu + seq, starts, seq < N)
        tl.store(target_cu + N, terminal)
        tl.store(read_indices + seq, slots, seq < N)
        tl.store(write_indices + seq, slots, seq < N)
        tl.store(target_flags + seq, initial, seq < N)
    task = tl.program_id(0) * BC + tl.arange(0, BC)
    # Read the original boundaries for every table. No program consumes
    # another program's writes, so fusion needs no cross-core barrier.
    for table in tl.static_range(3):
        if table == 0:
            indices, capacity, size = chunk_indices, CHUNK_CAPACITY, CHUNK
        elif table == 1:
            indices, capacity, size = solve_indices, SOLVE_CAPACITY, SOLVE_CHUNK
        else:
            indices, capacity, size = cumsum_indices, CUMSUM_CAPACITY, CUMSUM_CHUNK
        counts = tl.cdiv(ends - starts, size)
        if SKIP_SINGLE_TOKEN:
            counts = tl.where(ends - starts > 1, counts, 0)
        cumulative = tl.cumsum(counts, 0)
        begins = cumulative - counts
        total = tl.sum(counts, 0)
        if table == 0:
            if tl.program_id(0) == 0:
                tl.store(offsets + seq, begins, seq < N)
                tl.store(offsets + N, total)
        owner = tl.sum(((task[:, None] >= cumulative[None, :]) & (seq[None, :] < N)).to(tl.int32), 1)
        first = tl.sum(tl.where(owner[:, None] == seq[None, :], begins[None, :], 0), 1)
        # Excess tasks point at the terminal empty sequence. Never duplicate
        # a real chunk, which would race even if the results were equal.
        valid = task < total
        tl.store(indices + task * 2, tl.where(valid, owner, N - 1), task < capacity)
        tl.store(indices + task * 2 + 1, tl.where(valid, task - first, 0), task < capacity)


def cumsum_block_size(num_heads):
    return triton.next_power_of_2(max(1, CUMSUM_WORKING_SET // (num_heads * CHUNK_SIZE)))


def allocate_graph_metadata(token_capacity, request_capacity, num_heads, device):
    # The sum of per-request ceil(length / chunk) is bounded by
    # ceil(total_tokens / chunk) + request_capacity - 1.
    def indices(chunk):
        capacity = triton.cdiv(token_capacity, chunk) + request_capacity
        return torch.empty((capacity, 2), dtype=torch.int32, device=device)

    rows = request_capacity + 1
    return GDNFullGraphMetadata(
        query_start_loc=torch.empty(rows + 1, dtype=torch.int64, device=device),
        state_read_indices=torch.empty(rows, dtype=torch.int32, device=device),
        state_write_indices=torch.empty(rows, dtype=torch.int32, device=device),
        has_initial_state=torch.empty(rows, dtype=torch.bool, device=device),
        chunk_indices=indices(CHUNK_SIZE),
        chunk_offsets=torch.empty(rows + 1, dtype=torch.int32, device=device),
        solve_indices=indices(SOLVE_BLOCK_SIZE),
        cumsum_indices=indices(cumsum_block_size(num_heads)),
    )


def update_graph_metadata(
    target, query_start_loc, state_indices, has_initial_state, num_heads, *, skip_single_token=False
):
    count = state_indices.shape[0]
    if count > target.request_capacity or query_start_loc.shape != (count + 1,):
        raise ValueError("GDN graph request capacity or query boundaries mismatch")
    if state_indices.shape not in ((count,), (count, 1)):
        raise ValueError("GDN graph state indices require one cache slot per request")
    if has_initial_state.shape != (count,):
        raise ValueError("GDN graph initial-state flags must match requests")
    # One launch refreshes every owned buffer. Request count is a runtime
    # scalar rather than a JIT specialization; shapes depend on capacity only.
    rows = target.state_read_indices.shape[0]
    capacity = max(target.chunk_indices.shape[0], target.solve_indices.shape[0], target.cumsum_indices.shape[0])
    _chunk_metadata_kernel[(triton.cdiv(capacity, METADATA_BLOCK_SIZE),)](
        query_start_loc,
        state_indices,
        has_initial_state,
        target.query_start_loc,
        target.state_read_indices,
        target.state_write_indices,
        target.has_initial_state,
        target.chunk_indices,
        target.solve_indices,
        target.cumsum_indices,
        target.chunk_offsets,
        count,
        N=rows,
        CHUNK_CAPACITY=target.chunk_indices.shape[0],
        SOLVE_CAPACITY=target.solve_indices.shape[0],
        CUMSUM_CAPACITY=target.cumsum_indices.shape[0],
        CHUNK=CHUNK_SIZE,
        SOLVE_CHUNK=SOLVE_BLOCK_SIZE,
        CUMSUM_CHUNK=cumsum_block_size(num_heads),
        QUERY_STRIDE=query_start_loc.stride(0),
        STATE_STRIDE=state_indices.stride(0),
        FLAG_STRIDE=has_initial_state.stride(0),
        BN=triton.next_power_of_2(rows),
        BC=METADATA_BLOCK_SIZE,
        SKIP_SINGLE_TOKEN=skip_single_token,
        num_warps=4,
    )


@triton.jit
def _state_transfer_kernel(
    cache,
    packed,
    indices,
    flags,
    cu,
    STRIDE_N: tl.constexpr,
    STRIDE_H: tl.constexpr,
    STRIDE_V: tl.constexpr,
    STRIDE_K: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    WRITE: tl.constexpr,
    BLOCK: tl.constexpr,
    MIN_LENGTH: tl.constexpr,
):
    row = tl.program_id(0)
    index = tl.load(indices + row)
    active = (index >= 0) & (tl.load(cu + row + 1) - tl.load(cu + row) >= MIN_LENGTH)
    if active:
        x = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        head, key, value = x // (K * V), (x // V) % K, x % V
        cache_ptr = cache + index * STRIDE_N + head * STRIDE_H + value * STRIDE_V + key * STRIDE_K
        packed_ptr = packed + row * H * K * V + x
        mask = x < H * K * V
        if WRITE:
            data = tl.load(packed_ptr, mask, other=0)
            tl.store(cache_ptr, data, mask)
        else:
            initial = tl.load(flags + row) != 0
            data = tl.load(cache_ptr, mask & initial, other=0)
            tl.store(packed_ptr, data, mask)


def transfer_state(cache, packed, metadata, *, write, skip_single_token=False):
    rows, heads, key_dim, value_dim = packed.shape
    _state_transfer_kernel[(rows, triton.cdiv(heads * key_dim * value_dim, STATE_BLOCK_SIZE))](
        cache,
        packed,
        metadata.state_write_indices if write else metadata.state_read_indices,
        metadata.has_initial_state,
        metadata.query_start_loc,
        STRIDE_N=cache.stride(0),
        STRIDE_H=cache.stride(1),
        STRIDE_V=cache.stride(2),
        STRIDE_K=cache.stride(3),
        H=heads,
        K=key_dim,
        V=value_dim,
        WRITE=write,
        BLOCK=STATE_BLOCK_SIZE,
        MIN_LENGTH=2 if skip_single_token else 1,
    )


def chunk_gated_delta_rule_graph(q, k, v, g, beta, state, metadata, *, output=None):
    if q.shape[-1] != GDN_GRAPH_HEAD_DIM or v.shape[-1] != GDN_GRAPH_HEAD_DIM:
        raise ValueError("The graph GDN baseline currently requires K=V=128")
    q, k = l2norm_fwd(q), l2norm_fwd(k)
    if output is None:
        output = torch.empty_like(v)
    # Both branches stay in the same captured graph. Device lengths select
    # disjoint rows, including fresh one-token requests with no prior state.
    single_token_gdn(q, k, v, g, beta, state, metadata, output)
    cu = metadata.query_start_loc
    g = chunk_local_cumsum(g, CHUNK_SIZE, cu_seqlens=cu, block_indices=metadata.cumsum_indices)
    A = chunk_scaled_dot_kkt_fwd(k, beta, g, cu_seqlens=cu, chunk_indices=metadata.chunk_indices)
    A = solve_tril(
        A,
        cu_seqlens=cu,
        chunk_indices_large_block=metadata.solve_indices,
        chunk_indices_bt=metadata.chunk_indices,
        output_dtype=k.dtype,
    )
    w, u = recompute_w_u_fwd(k, v, beta, g, A, cu_seqlens=cu, chunk_indices=metadata.chunk_indices)
    h, v_new, _ = chunk_gated_delta_rule_fwd_h(
        k,
        w,
        u,
        g,
        output_final_state=True,
        cu_seqlens=cu,
        chunk_indices=metadata.chunk_indices,
        chunk_offsets=metadata.chunk_offsets,
        skip_single_token=True,
        state_cache=state,
        state_metadata=metadata,
        token_major_g=True,
    )
    output = chunk_fwd_o(
        q,
        k,
        v_new,
        h,
        g,
        cu_seqlens=cu,
        chunk_offsets=metadata.chunk_offsets,
        output=output,
        skip_single_token=True,
        token_major_g=True,
    )
    return output
