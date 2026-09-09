# SPDX-License-Identifier: Apache-2.0
"""Real Ascend compile/capture/replay test against a CPU attention reference."""

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.ops.triton.graph_attention import graph_paged_attention
from vllm_ascend.ops.triton.graph_metadata import allocate_attention_work, refresh_attention_metadata


@pytest.mark.parametrize("dim", [128, 256])
@pytest.mark.parametrize("window", [None, 32])
@pytest.mark.parametrize("block_size", [16, 96, 128])
def test_device_attention_replays_changing_mixed_lengths(dim, window, block_size):
    torch.manual_seed(122)
    capacity, rows, heads, key_heads, columns = 256, 64, 4, 2, (768 + block_size - 1) // block_size
    q = (torch.randn(capacity, heads, dim) * 0.2).bfloat16()
    k = torch.randn(rows * columns, block_size, key_heads, dim).bfloat16()
    v = torch.randn_like(k)
    blocks = torch.randperm(rows * columns).reshape(rows, columns).int()
    qn, kn, vn = (tensor.to("npu") for tensor in (q, k, v))
    metadata = SimpleNamespace(
        query_start_loc=torch.zeros(rows + 2, dtype=torch.int64, device="npu"),
        seq_lens_device=torch.zeros(rows + 1, dtype=torch.int64, device="npu"),
        block_tables=torch.zeros(rows + 1, columns, dtype=torch.int32, device="npu"),
        slot_mapping=torch.empty(capacity, dtype=torch.int64, device="npu"),
        attention_work=allocate_attention_work(capacity, rows, "npu"),
    )
    device_blocks = blocks.to("npu")

    def update(cu, lens, count, actual):
        source = SimpleNamespace(
            query_start_loc=cu.to("npu"),
            seq_lens_device=lens.to("npu"),
            block_tables=device_blocks,
            slot_mapping=torch.arange(actual, dtype=torch.int64, device="npu"),
        )
        refresh_attention_metadata(metadata, source, count, actual)

    update(torch.tensor([0, 61]), torch.tensor([61]), 1, 61)
    work_address = metadata.attention_work.data_ptr()
    output = torch.empty_like(qn)

    def run():
        return graph_paged_attention(
            qn, kn, vn, metadata, output, num_heads=heads, scale=dim**-0.5, sliding_window=window
        )

    run()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        run()
    # One graph must cover a fresh prompt, mixed continuation, c64 and arrivals.
    for lengths in ([61], [1, 63, 65, 2], [1] * 62 + [31, 64], [2, 7]):
        prefixes = [(row % 3) * 127 for row in range(len(lengths))]
        starts = [0, *torch.tensor(lengths).cumsum(0).tolist()]
        cu = torch.tensor(starts + [sum(lengths)] * (rows - len(lengths)) + [capacity], dtype=torch.int64)
        lens = torch.tensor(
            [length + prefix for length, prefix in zip(lengths, prefixes)] + [0] * (rows + 1 - len(lengths))
        )
        update(cu, lens, len(lengths), sum(lengths))
        assert metadata.attention_work.data_ptr() == work_address
        output.fill_(float("nan"))
        graph.replay()
        actual = output.cpu()
        for row, (length, prefix) in enumerate(zip(lengths, prefixes)):
            keys = (
                k[blocks[row]]
                .reshape(-1, key_heads, dim)[: prefix + length]
                .float()
                .repeat_interleave(heads // key_heads, 1)
            )
            values = (
                v[blocks[row]]
                .reshape(-1, key_heads, dim)[: prefix + length]
                .float()
                .repeat_interleave(heads // key_heads, 1)
            )
            query = q[starts[row] : starts[row + 1]].float()
            scores = torch.einsum("qhd,khd->hqk", query, keys) * dim**-0.5
            positions = torch.arange(prefix + length)
            allowed = positions[None, :] <= prefix + torch.arange(length)[:, None]
            if window:
                allowed &= positions[None, :] > prefix + torch.arange(length)[:, None] - window
            probability = scores.masked_fill(~allowed, -float("inf")).softmax(-1)
            expected = torch.einsum("hqk,khd->qhd", probability, values)
            torch.testing.assert_close(actual[starts[row] : starts[row + 1]].float(), expected, atol=0.012, rtol=0.02)
        assert actual[sum(lengths) :].isnan().all()
