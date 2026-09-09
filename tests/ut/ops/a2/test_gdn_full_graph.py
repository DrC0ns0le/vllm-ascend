# SPDX-License-Identifier: Apache-2.0
"""Ascend Triton compile/run regressions for the packed FULL GDN route."""

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.ops.triton.fla.chunk_delta_h import chunk_gated_delta_rule_fwd_h
from vllm_ascend.ops.triton.fla.graph import allocate_graph_metadata, transfer_state, update_graph_metadata
from vllm_ascend.ops.triton.fla.single_token import single_token_gdn


@pytest.mark.parametrize("flag_dtype", [torch.bool, torch.int8])
def test_single_token_gdn_compiles_predicate_masks_and_preserves_inactive_rows(flag_dtype):
    torch.manual_seed(31)
    heads, key_heads, dim, capacity = 4, 2, 128, 8
    q = torch.nn.functional.normalize(torch.randn(1, capacity, key_heads, dim), dim=-1).bfloat16()
    k = torch.nn.functional.normalize(torch.randn_like(q.float()), dim=-1).bfloat16()
    v = torch.randn(1, capacity, heads, dim).bfloat16()
    g = -torch.rand(1, capacity, heads) * 0.05
    beta = torch.rand(1, capacity, heads)
    state = torch.randn(6, heads, dim, dim) * 0.05
    state[3] = float("nan")  # Fresh requests must not read stale cache data.
    expected_state = state.clone()
    expected_output = torch.full_like(v, float("nan"))
    for token, slot, fresh in ((0, 1, False), (1, 3, True)):
        h = torch.zeros_like(state[slot]) if fresh else state[slot].clone()
        kt = k[0, token].float().repeat_interleave(heads // key_heads, 0)
        qt = q[0, token].float().repeat_interleave(heads // key_heads, 0)
        h *= g[0, token].exp()[:, None, None]
        delta = (v[0, token].float() - torch.einsum("hvk,hk->hv", h, kt)) * beta[0, token, :, None]
        h += delta[:, :, None] * kt[:, None, :]
        expected_state[slot] = h
        expected_output[0, token] = torch.einsum("hvk,hk->hv", h, qt) / dim**0.5
    metadata = SimpleNamespace(
        query_start_loc=torch.tensor([0, 1, 2, 4, 4], device="npu"),
        state_read_indices=torch.tensor([1, 3, 4, -1], dtype=torch.int32, device="npu"),
        state_write_indices=torch.tensor([1, 3, 4, -1], dtype=torch.int32, device="npu"),
        has_initial_state=torch.tensor([2, 0, -1, 0], dtype=flag_dtype, device="npu"),
    )
    device_state = state.to("npu")
    output = torch.full_like(v, float("nan"), device="npu")
    single_token_gdn(*(x.to("npu") for x in (q, k, v, g, beta)), device_state, metadata, output)
    torch.npu.synchronize()
    torch.testing.assert_close(output.cpu(), expected_output, atol=1e-3, rtol=1e-2, equal_nan=True)
    torch.testing.assert_close(device_state.cpu(), expected_state, atol=1e-5, rtol=1e-4, equal_nan=True)


@pytest.mark.parametrize("cache_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("strided_cache", [False, True])
def test_chunk_recurrence_direct_cache_matches_staged_state(cache_dtype, strided_cache):
    torch.manual_seed(37)
    capacity, heads, dim = 128, 2, 128
    metadata = allocate_graph_metadata(capacity, 3, heads, "npu")
    update_graph_metadata(
        metadata,
        torch.tensor([0, 1, 4, 69], device="npu"),
        torch.tensor([3, 1, 4], dtype=torch.int32, device="npu"),
        torch.tensor([True, False, True], device="npu"),
        heads,
        skip_single_token=True,
    )
    cache = (torch.randn(6, heads, dim, dim) * 0.05).to(cache_dtype)
    cache[1] = float("nan")
    if strided_cache:
        cache = cache.transpose(-1, -2)
    direct = cache.to("npu")
    staged = direct.clone()
    k = (torch.randn(1, capacity, 1, dim) * 0.05).bfloat16().to("npu")
    w = (torch.randn(1, capacity, heads, dim) * 0.01).bfloat16().to("npu")
    u = torch.randn(1, capacity, heads, dim).bfloat16().to("npu")
    g = (-torch.rand(1, capacity, heads * 2) * 0.05).to("npu")[..., ::2]
    initial = torch.empty((4, heads, dim, dim), dtype=cache_dtype, device="npu")
    transfer_state(staged, initial, metadata, write=False, skip_single_token=True)
    kwargs = dict(
        output_final_state=True,
        cu_seqlens=metadata.query_start_loc,
        chunk_indices=metadata.chunk_indices,
        chunk_offsets=metadata.chunk_offsets,
        skip_single_token=True,
    )
    expected_h, expected_v, final = chunk_gated_delta_rule_fwd_h(k, w, u, g, initial, **kwargs)
    transfer_state(staged, final, metadata, write=True, skip_single_token=True)
    actual_h, actual_v, final_cache = chunk_gated_delta_rule_fwd_h(
        k, w, u, g, state_cache=direct, state_metadata=metadata, token_major_g=True, **kwargs
    )
    assert final_cache is direct
    torch.npu.synchronize()
    torch.testing.assert_close(direct.cpu(), staged.cpu(), atol=1e-3, rtol=1e-2, equal_nan=True)
    # Only three live chunks and tokens [1, 69) belong to the chunk route.
    torch.testing.assert_close(actual_h[:, :3].cpu(), expected_h[:, :3].cpu(), atol=1e-3, rtol=1e-2)
    torch.testing.assert_close(actual_v[:, 1:69].cpu(), expected_v[:, 1:69].cpu(), atol=1e-3, rtol=1e-2)
