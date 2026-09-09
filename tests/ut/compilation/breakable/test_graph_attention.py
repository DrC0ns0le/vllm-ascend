# SPDX-License-Identifier: Apache-2.0
"""Execute the device-length attention kernel with CPU pointer semantics."""

from math import gcd
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from test_capacity_metadata import Launch, TensorLanguage, load_source
from test_full_prefill_attention import load_functions


@pytest.mark.parametrize("lengths", [[61], [1, 63, 65, 2], [1] * 62 + [31, 64]])
@pytest.mark.parametrize("dim", [64, 128, 256])
@pytest.mark.parametrize("window", [0, 32])
@pytest.mark.parametrize("block_size", [16, 96, 128])
def test_device_attention_matches_paged_causal_reference(lengths, dim, window, block_size):
    torch.manual_seed(121)
    heads, key_heads = 4, 2
    capacity = sum(lengths) + 13
    rows = 64
    prefixes = [(i % 3) * 127 for i in range(len(lengths))]
    kv_lengths = [length + prefix for length, prefix in zip(lengths, prefixes)]
    columns = max((length + block_size - 1) // block_size for length in kv_lengths)
    blocks = torch.randperm(rows * columns).reshape(rows, columns)
    q = (torch.randn(capacity, heads, dim) * 0.2).bfloat16()
    k = torch.randn(rows * columns, block_size, key_heads, dim).bfloat16()
    v = torch.randn_like(k)
    # Never touch empty-request block rows or padded Q rows.
    blocks[len(lengths) :] = -1
    q[sum(lengths) :] = float("nan")
    for row, length in enumerate(kv_lengths):
        if length % block_size:
            page = blocks[row, length // block_size]
            k[page, length % block_size :] = float("nan")
            v[page, length % block_size :] = float("nan")
    output = torch.full_like(q, float("nan"))
    expected = torch.empty(sum(lengths), heads, dim)
    cu = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()] + [sum(lengths)] * (rows - len(lengths)))
    seq = torch.tensor(kv_lengths + [0] * (rows - len(lengths)))
    for row, (length, prefix) in enumerate(zip(lengths, prefixes)):
        keys = (
            k[blocks[row]]
            .reshape(-1, key_heads, dim)[: length + prefix]
            .float()
            .repeat_interleave(heads // key_heads, 1)
        )
        values = (
            v[blocks[row]]
            .reshape(-1, key_heads, dim)[: length + prefix]
            .float()
            .repeat_interleave(heads // key_heads, 1)
        )
        query = q[cu[row] : cu[row + 1]].float()
        scores = torch.einsum("qhd,khd->hqk", query, keys) * dim**-0.5
        positions = torch.arange(prefix + length)
        allowed = positions[None, :] <= prefix + torch.arange(length)[:, None]
        if window:
            allowed &= positions[None, :] > prefix + torch.arange(length)[:, None] - window
        probability = scores.masked_fill(~allowed, -float("inf")).softmax(-1)
        expected[cu[row] : cu[row + 1]] = torch.einsum("hqk,khd->qhd", probability, values)
    tl = TensorLanguage()
    namespace = load_source("vllm_ascend/ops/triton/graph_attention.py", {"_graph_attention_kernel"}, dict(tl=tl))
    metadata_code = dict(
        tl=tl, torch=torch, triton=SimpleNamespace(cdiv=tl.cdiv, next_power_of_2=lambda n: 1 << (n - 1).bit_length())
    )
    load_source(
        "vllm_ascend/ops/triton/graph_metadata.py",
        {
            "_refresh_attention_metadata_kernel",
            "refresh_attention_metadata",
            "allocate_attention_work",
            "METADATA_BLOCK_SIZE",
            "ATTENTION_QUERY_TILE",
            "ATTENTION_WORK_BLOCK",
        },
        metadata_code,
    )
    metadata_code["_refresh_attention_metadata_kernel"] = Launch(
        metadata_code["_refresh_attention_metadata_kernel"], tl
    )
    target = SimpleNamespace(
        query_start_loc=torch.zeros(rows + 2, dtype=torch.int64),
        seq_lens_device=torch.zeros(rows + 1, dtype=torch.int64),
        block_tables=torch.zeros(rows + 1, columns, dtype=blocks.dtype),
        slot_mapping=torch.empty(capacity, dtype=torch.int64),
        attention_work=metadata_code["allocate_attention_work"](capacity, rows, "cpu"),
    )
    source = SimpleNamespace(
        query_start_loc=cu, seq_lens_device=seq, block_tables=blocks, slot_mapping=torch.arange(sum(lengths))
    )
    metadata_code["refresh_attention_metadata"](target, source, len(lengths), sum(lengths))
    work = [(row, offset) for row, length in enumerate(lengths) for offset in range(0, length, 16)]
    assert target.attention_work[0, 0] == len(work)
    assert target.attention_work[1 : len(work) + 1].tolist() == [list(item) for item in work]
    assert (target.attention_work[len(work) + 1 :, 0] == -1).all()
    load = tl.load

    def scalar_page_load(pointer, *args, **kwargs):
        if pointer.tensor.data_ptr() == blocks.data_ptr():
            assert pointer.offset.numel() == 1, "Attention must look up one page per affine K/V tile"
        return load(pointer, *args, **kwargs)

    tl.load = scalar_page_load
    Launch(namespace["_graph_attention_kernel"], tl)[(32,)](
        q,
        k,
        v,
        cu,
        seq,
        blocks,
        target.attention_work,
        output,
        H=heads,
        HK=key_heads,
        D=dim,
        QT=q.stride(0),
        QH=q.stride(1),
        QD=q.stride(2),
        KB=k.stride(0),
        KT=k.stride(1),
        KH=k.stride(2),
        KD=k.stride(3),
        VB=v.stride(0),
        VT=v.stride(1),
        VH=v.stride(2),
        VD=v.stride(3),
        OT=output.stride(0),
        OH=output.stride(1),
        OD=output.stride(2),
        BLOCK_SIZE=block_size,
        COLUMNS=columns,
        BQ=16,
        BK=gcd(64, block_size),
        SCALE=dim**-0.5,
        WINDOW=window,
        PROGRAMS=32,
    )
    torch.testing.assert_close(output[: sum(lengths)].float(), expected, atol=0.012, rtol=0.02)
    assert output[sum(lengths) :].isnan().all()


@pytest.mark.parametrize("block_size, tile", [(16, 16), (96, 32), (128, 64), (17, None)])
def test_attention_launcher_keeps_tiles_inside_cache_pages(block_size, tile):
    launch = Mock()
    kernel = Mock()
    kernel.__getitem__ = Mock(return_value=launch)
    namespace = load_source(
        "vllm_ascend/ops/triton/graph_attention.py",
        {"graph_paged_attention", "ATTENTION_PROGRAMS", "KEY_TILE", "MIN_KEY_TILE"},
        dict(torch=torch, gcd=gcd, ATTENTION_QUERY_TILE=16, _graph_attention_kernel=kernel),
    )
    q = torch.empty(64, 4, 128, dtype=torch.bfloat16)
    k = torch.empty(4, block_size, 2, 128, dtype=q.dtype)
    output = torch.empty_like(q)
    work = torch.empty(70, 2, dtype=torch.int32)
    metadata = SimpleNamespace(
        query_start_loc=object(), seq_lens_device=object(), block_tables=torch.empty(65, 6), attention_work=work
    )
    if tile is None:
        with pytest.raises(ValueError, match="page size divisible by 16"):
            namespace["graph_paged_attention"](q, k, k, metadata, output, num_heads=4, scale=0.1)
        launch.assert_not_called()
        return
    assert namespace["graph_paged_attention"](q, k, k, metadata, output, num_heads=4, scale=0.1) is output
    assert launch.call_args.kwargs["BK"] == tile
    assert launch.call_args.args[6] is work


def test_capacity_attention_bypasses_host_fia_even_during_capture():
    device_attention = Mock(return_value="result")
    namespace = load_functions(
        "vllm_ascend/attention/attention_v1.py",
        {"forward_fused_infer_attention"},
        dict(_EXTRA_CTX=SimpleNamespace(capturing=True), graph_paged_attention=device_attention),
        owner="AscendAttentionBackendImpl",
    )
    impl = SimpleNamespace(key_cache="key", value_cache="value", num_heads=8, scale=0.1, sliding_window=None)
    metadata = SimpleNamespace(full_graph_token_capacity=128)
    assert namespace["forward_fused_infer_attention"](impl, "query", None, None, metadata, "out") == "result"
    device_attention.assert_called_once_with(
        "query", "key", "value", metadata, "out", num_heads=8, scale=0.1, sliding_window=None
    )
