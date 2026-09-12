# SPDX-License-Identifier: Apache-2.0
"""Native dense FIA for fixed graph layouts with changing cached contexts."""

import torch
import torch_npu


def gather_native_cache(cache, block_tables, seq_lengths, capacity):
    """Read only allocated live pages; zero padding before attention matmuls."""
    block_size = cache.shape[1]
    positions = torch.arange(capacity, device=cache.device)
    columns = (positions // block_size).clamp(max=block_tables.shape[1] - 1)
    blocks = block_tables.index_select(1, columns).to(torch.int64)
    valid = positions[None, :] < seq_lengths[:, None]
    blocks = torch.where(valid, blocks, 0)
    slots = blocks * block_size + positions[None, :] % block_size
    dense = cache.flatten(0, 1).index_select(0, slots.reshape(-1))
    dense = dense.view(block_tables.shape[0], capacity, *cache.shape[2:])
    return torch.where(valid[:, :, None, None], dense, 0).transpose(1, 2).contiguous()


def native_cached_attention(impl, query, metadata, output):
    layout = metadata.native_full
    count, width = layout.shape.requests, layout.shape.width
    seq_lengths = layout.contexts + layout.lengths
    key = gather_native_cache(impl.key_cache, metadata.block_tables, seq_lengths, layout.capacity)
    value = gather_native_cache(impl.value_cache, metadata.block_tables, seq_lengths, layout.capacity)
    query = query.view(count, width, impl.num_heads, impl.head_size).transpose(1, 2).contiguous()
    mask = layout.mask
    if impl.sliding_window is not None:
        query_positions = layout.contexts[:, None] + torch.arange(width, device=query.device)[None, :]
        key_positions = torch.arange(layout.capacity, device=query.device)
        mask = mask | (key_positions[None, None, :] < query_positions[:, :, None] - impl.sliding_window).unsqueeze(1)
    attention, _ = torch_npu.npu_fused_infer_attention_score(
        query,
        key,
        value,
        num_heads=impl.num_heads,
        num_key_value_heads=impl.num_kv_heads,
        input_layout="BNSD",
        atten_mask=mask,
        sparse_mode=0,
        scale=impl.scale,
    )
    output.copy_(attention.transpose(1, 2).reshape_as(output))
    return output
