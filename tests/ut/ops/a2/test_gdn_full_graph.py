# SPDX-License-Identifier: Apache-2.0
"""Ascend Triton compile/run regressions for the packed FULL GDN route."""

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.ops.triton.fla.chunk_delta_h import chunk_gated_delta_rule_fwd_h
from vllm_ascend.ops.triton.fla.chunk_o import chunk_fwd_o
from vllm_ascend.ops.triton.fla.chunk_state_output import chunk_state_output
from vllm_ascend.ops.triton.fla.graph import allocate_graph_metadata, transfer_state, update_graph_metadata
from vllm_ascend.ops.triton.fla.single_token import single_token_gdn


@pytest.mark.parametrize("flag_dtype", [torch.bool, torch.int8])
@pytest.mark.parametrize("compact", [False, True])
def test_single_token_gdn_compiles_predicate_masks_and_preserves_inactive_rows(flag_dtype, compact):
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
    if compact:
        metadata.single_token_work = torch.tensor([2, 0, 1], dtype=torch.int32, device="npu")
    output = torch.full_like(v, float("nan"), device="npu")
    single_token_gdn(*(x.to("npu") for x in (q, k, v, g, beta)), device_state, metadata, output)
    torch.npu.synchronize()
    torch.testing.assert_close(output.cpu(), expected_output, atol=1e-3, rtol=1e-2, equal_nan=True)
    torch.testing.assert_close(device_state.cpu(), expected_state, atol=1e-5, rtol=1e-4, equal_nan=True)


@pytest.mark.parametrize("cache_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("strided_cache", [False, True])
@pytest.mark.parametrize("graph_replay", [False, True])
@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("requests", [3, 64])
def test_chunk_recurrence_direct_cache_matches_staged_state(cache_dtype, strided_cache, graph_replay, fused, requests):
    torch.manual_seed(37)
    capacity, heads, dim = 256, 16, 128
    lengths = [1, 3, 65] + [2] * (requests - 3)
    boundaries = [0, *torch.tensor(lengths).cumsum(0).tolist()]
    slots = [3, 1, 4] + list(range(5, requests + 2))
    flags_initial = [True, False, True] + [True] * (requests - 3)
    live_chunks = sum((length + 63) // 64 for length in lengths if length > 1)
    terminal = boundaries[-1]
    metadata = allocate_graph_metadata(capacity, requests, heads, "npu")
    update_graph_metadata(
        metadata,
        torch.tensor(boundaries, device="npu"),
        torch.tensor(slots, dtype=torch.int32, device="npu"),
        torch.tensor(flags_initial, device="npu"),
        heads,
        skip_single_token=True,
    )
    cache = (torch.randn(requests + 3, heads, dim, dim) * 0.05).to(cache_dtype)
    cache[1] = float("nan")
    cache[1, :, :, : dim // 2] = float("inf")
    if strided_cache:
        cache = cache.transpose(-1, -2)
    direct = cache.to("npu")
    staged = direct.clone()
    k = (torch.randn(1, capacity, 1, dim) * 0.05).bfloat16().to("npu")
    q = torch.randn_like(k) * 0.05
    output = torch.full((1, capacity, heads, dim), float("nan"), dtype=k.dtype, device="npu")
    w = (torch.randn(1, capacity, heads, dim) * 0.01).bfloat16().to("npu")
    u = torch.randn(1, capacity, heads, dim).bfloat16().to("npu")
    g = (-torch.rand(1, capacity, heads * 2) * 0.05).to("npu")[..., ::2]
    initial = torch.empty((requests + 1, heads, dim, dim), dtype=cache_dtype, device="npu")
    kwargs = dict(
        output_final_state=True,
        cu_seqlens=metadata.query_start_loc,
        chunk_indices=metadata.chunk_indices,
        chunk_offsets=metadata.chunk_offsets,
        skip_single_token=True,
    )

    def run_direct():
        if fused:
            return chunk_state_output(q, k, w, u, g, direct, metadata, output), None, direct
        return chunk_gated_delta_rule_fwd_h(
            k, w, u, g, state_cache=direct, state_metadata=metadata, token_major_g=True, **kwargs
        )

    if graph_replay:
        # Compile before capture, then restore state modified by warmup/capture.
        run_direct()
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            actual_h, actual_v, final_cache = run_direct()
        direct.copy_(staged)

    # Fresh -> continuation -> fresh slot reuse, without replacing graph inputs.
    for flags in (flags_initial, [True] * requests, [True] + [row % 3 == 0 for row in range(1, requests)]):
        metadata.has_initial_state[:requests].copy_(torch.tensor(flags, device="npu"))
        for row, slot in enumerate(slots):
            if not flags[row]:
                for state in (direct, staged):
                    state[slot] = float("nan")
                    state[slot, :, :, : dim // 2] = float("inf")
        transfer_state(staged, initial, metadata, write=False, skip_single_token=True)
        expected_h, expected_v, final = chunk_gated_delta_rule_fwd_h(k, w, u, g, initial, **kwargs)
        expected_output = chunk_fwd_o(
            q,
            k,
            expected_v,
            expected_h,
            g,
            cu_seqlens=metadata.query_start_loc,
            chunk_offsets=metadata.chunk_offsets,
            skip_single_token=True,
            token_major_g=True,
        )
        transfer_state(staged, final, metadata, write=True, skip_single_token=True)
        if graph_replay:
            graph.replay()
        else:
            actual_h, actual_v, final_cache = run_direct()
        assert final_cache is direct
        torch.npu.synchronize()
        torch.testing.assert_close(direct.cpu(), staged.cpu(), atol=1e-3, rtol=1e-2, equal_nan=True)
        # The leading one-token row and trailing token padding are not owned by this kernel.
        if fused:
            torch.testing.assert_close(
                output[:, 1:terminal].cpu(), expected_output[:, 1:terminal].cpu(), atol=1e-3, rtol=1e-2
            )
            assert output[:, terminal:].isnan().all() and output[:, :1].isnan().all()
        else:
            torch.testing.assert_close(
                actual_h[:, :live_chunks].cpu(), expected_h[:, :live_chunks].cpu(), atol=1e-3, rtol=1e-2
            )
            torch.testing.assert_close(
                actual_v[:, 1:terminal].cpu(), expected_v[:, 1:terminal].cpu(), atol=1e-3, rtol=1e-2
            )
