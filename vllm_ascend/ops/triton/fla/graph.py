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
from .solve_tril import solve_tril
from .wy_fast import recompute_w_u_fwd

CHUNK_SIZE = 64
SOLVE_BLOCK_SIZE = 1216
CUMSUM_WORKING_SET = 2**18
STATE_BLOCK_SIZE = 1024
METADATA_BLOCK_SIZE = 32


@triton.jit
def _chunk_metadata_kernel(
    cu,
    indices,
    offsets,
    N: tl.constexpr,
    CAPACITY: tl.constexpr,
    CHUNK: tl.constexpr,
    BN: tl.constexpr,
    BC: tl.constexpr,
    STORE_OFFSETS: tl.constexpr,
):
    seq = tl.arange(0, BN)
    starts = tl.load(cu + seq, seq < N, other=0)
    ends = tl.load(cu + seq + 1, seq < N, other=0)
    counts = tl.cdiv(ends - starts, CHUNK)
    cumulative = tl.cumsum(counts, 0)
    begins = cumulative - counts
    total = tl.sum(counts, 0)
    if STORE_OFFSETS:
        if tl.program_id(0) == 0:
            tl.store(offsets + seq, begins, seq < N)
            tl.store(offsets + N, total)
    task = tl.program_id(0) * BC + tl.arange(0, BC)
    owner = tl.sum(((task[:, None] >= cumulative[None, :]) & (seq[None, :] < N)).to(tl.int32), 1)
    first = tl.sum(tl.where(owner[:, None] == seq[None, :], begins[None, :], 0), 1)
    # N-1 is the terminal empty sequence. Never duplicate a real chunk:
    # multiple writers to the same output would race even with equal values.
    valid = task < total
    tl.store(indices + task * 2, tl.where(valid, owner, N - 1), task < CAPACITY)
    tl.store(indices + task * 2 + 1, tl.where(valid, task - first, 0), task < CAPACITY)


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


def update_graph_metadata(target, query_start_loc, state_indices, has_initial_state, num_heads):
    count = state_indices.shape[0]
    if count > target.request_capacity or query_start_loc.shape != (count + 1,):
        raise ValueError("GDN graph request capacity or query boundaries mismatch")
    if has_initial_state.shape != (count,):
        raise ValueError("GDN graph initial-state flags must match requests")
    # copy_ with an expanded device scalar avoids fill_(device_scalar)'s
    # scalar extraction. Empty rows, including the sentinel, end at live T.
    target.query_start_loc.copy_(query_start_loc[-1:].expand_as(target.query_start_loc))
    target.query_start_loc[: count + 1].copy_(query_start_loc)
    target.state_read_indices.fill_(-1)
    target.state_write_indices.fill_(-1)
    target.has_initial_state.zero_()
    target.state_read_indices[:count].copy_(state_indices)
    target.state_write_indices[:count].copy_(state_indices)
    target.has_initial_state[:count].copy_(has_initial_state)
    rows = target.state_read_indices.shape[0]
    for indices, size in (
        (target.chunk_indices, CHUNK_SIZE),
        (target.solve_indices, SOLVE_BLOCK_SIZE),
        (target.cumsum_indices, cumsum_block_size(num_heads)),
    ):
        _chunk_metadata_kernel[(triton.cdiv(indices.shape[0], METADATA_BLOCK_SIZE),)](
            target.query_start_loc,
            indices,
            target.chunk_offsets,
            N=rows,
            CAPACITY=indices.shape[0],
            CHUNK=size,
            BN=triton.next_power_of_2(rows),
            BC=METADATA_BLOCK_SIZE,
            STORE_OFFSETS=size == CHUNK_SIZE,
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
):
    row = tl.program_id(0)
    index = tl.load(indices + row)
    active = (index >= 0) & (tl.load(cu + row + 1) > tl.load(cu + row))
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
            initial = tl.load(flags + row)
            data = tl.load(cache_ptr, mask & initial, other=0)
            tl.store(packed_ptr, data, mask)


def transfer_state(cache, packed, metadata, *, write):
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
    )


def chunk_gated_delta_rule_graph(q, k, v, g, beta, state, metadata, *, output=None):
    if q.shape[-1] != GDN_GRAPH_HEAD_DIM or v.shape[-1] != GDN_GRAPH_HEAD_DIM:
        raise ValueError("The graph GDN baseline currently requires K=V=128")
    q, k = l2norm_fwd(q), l2norm_fwd(k)
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
    initial = torch.empty(
        (metadata.state_read_indices.shape[0], v.shape[2], k.shape[-1], v.shape[-1]),
        dtype=state.dtype,
        device=state.device,
    )
    transfer_state(state, initial, metadata, write=False)
    h, v_new, final = chunk_gated_delta_rule_fwd_h(
        k,
        w,
        u,
        g,
        initial,
        output_final_state=True,
        cu_seqlens=cu,
        chunk_indices=metadata.chunk_indices,
        chunk_offsets=metadata.chunk_offsets,
    )
    output = chunk_fwd_o(q, k, v_new, h, g, cu_seqlens=cu, chunk_offsets=metadata.chunk_offsets, output=output)
    transfer_state(state, final, metadata, write=True)
    return output
