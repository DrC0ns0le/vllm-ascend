# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for packed native-GDN metadata; no NPU arithmetic emulation."""

import ast
from dataclasses import dataclass
from itertools import accumulate
from pathlib import Path
from random import Random
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[4]


def metadata_namespace():
    source = ast.parse((ROOT / "vllm_ascend/ops/gdn_attn_builder.py").read_text())
    names = {
        "GDNChunkedPrefillMetadata",
        "_chunk_indices_from_lengths",
        "_upload_chunk_metadata",
        "_build_non_spec_chunked_prefill_metadata",
    }
    nodes = [
        node
        for node in source.body
        if getattr(node, "name", None) in names
        or isinstance(node, ast.Assign)
        and any(getattr(target, "id", "").startswith("_GDN_") for target in node.targets)
    ]
    namespace = dict(torch=torch, dataclass=dataclass)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "gdn_attn_builder.py", "exec"), namespace)
    return namespace


def runtime_tables():
    source = ast.parse((ROOT / "vllm_ascend/ops/triton/fla/utils.py").read_text())
    nodes = [node for node in source.body if isinstance(node, ast.FunctionDef) and node.name.startswith("prepare_")]
    namespace = dict(torch=torch, triton=SimpleNamespace(cdiv=lambda x, y: (x + y - 1) // y))
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "fla/utils.py", "exec"), namespace)
    return namespace


def builder(heads=16):
    return SimpleNamespace(
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(hf_text_config=SimpleNamespace(linear_num_value_heads=heads)),
            parallel_config=SimpleNamespace(tensor_parallel_size=1),
        )
    )


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("heads", [16, 32])
def test_packed_tables_match_runtime_for_mixed_lengths_and_empty_rows(dtype, heads):
    namespace, runtime = metadata_namespace(), runtime_tables()
    build = namespace["_build_non_spec_chunked_prefill_metadata"]
    random = Random(17)
    cases = [[61], [1, 1, 59], [0], [0, 65, 0, 129, 0], [1] * 63 + [767], [4096]]
    cases += [[random.randrange(769) if random.randrange(3) else 0 for _ in range(64)] for _ in range(10)]
    for lengths in cases:
        cu = torch.tensor([0, *accumulate(lengths)], dtype=dtype)
        meta = build(builder(heads), cu, torch.device("cpu"))
        cumsum_chunks = max(1, (2**18) // (heads * 64))
        cumsum_size = 1 << (cumsum_chunks - 1).bit_length()
        expected = {
            "chunk_indices_chunk64": runtime["prepare_chunk_indices"](cu, 64),
            "chunk_offsets_chunk64": runtime["prepare_chunk_offsets"](cu, 64),
            "update_chunk_offsets_chunk64": runtime["prepare_update_chunk_offsets"](cu, 64),
            "final_chunk_indices_chunk64": runtime["prepare_final_chunk_indices"](cu, 64),
            "chunk_indices_large_block": runtime["prepare_chunk_indices"](cu, 1216),
            "block_indices_cumsum": runtime["prepare_chunk_indices"](cu, cumsum_size),
        }
        for name, reference in expected.items():
            torch.testing.assert_close(getattr(meta, name), reference)
        assert meta.cu_seqlens_host == tuple(cu.tolist())
        assert meta.chunk_indices_chunk64_host == tuple(expected["chunk_indices_chunk64"].flatten().tolist())
        assert meta.num_decodes == lengths.count(1)
        live = torch.tensor(lengths) > 0
        if all(lengths):
            assert meta.keep_meta is None and meta.cu_seqlens_kern is None
        else:
            torch.testing.assert_close(meta.keep_meta, live)
            assert meta.cu_seqlens_kern == (0, *cu[1:][live].tolist())
        # All device views retain the same slab, with original dtypes and shapes.
        storages = {getattr(meta, name).untyped_storage().data_ptr() for name in expected}
        assert len(storages) == 1


def test_one_upload_per_build_and_previous_step_tables_remain_owned(monkeypatch):
    build = metadata_namespace()["_build_non_spec_chunked_prefill_metadata"]
    original_to = torch.Tensor.to
    uploads = []

    def track_to(tensor, *args, **kwargs):
        uploads.append((tensor.dtype, kwargs))
        return original_to(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", track_to)
    first = build(builder(), torch.tensor([0, 61], dtype=torch.int32), torch.device("cpu"))
    before = first.chunk_indices_chunk64.clone()
    second = build(builder(), torch.tensor([0, 1, 65, 193], dtype=torch.int32), torch.device("cpu"))
    assert len(uploads) == 2
    assert all(dtype == torch.uint8 and kwargs["non_blocking"] for dtype, kwargs in uploads)
    assert (
        first.chunk_indices_chunk64.untyped_storage().data_ptr()
        != second.chunk_indices_chunk64.untyped_storage().data_ptr()
    )
    torch.testing.assert_close(first.chunk_indices_chunk64, before)


@pytest.mark.parametrize("offsets,live", [([0, 61], 61), ([0, 1, 2, 61], 61), ([0, 61, 64], 61)])
def test_attention_builder_reuses_uploaded_query_offsets_including_padding(monkeypatch, offsets, live):
    source = ast.parse((ROOT / "vllm_ascend/attention/attention_v1.py").read_text())
    cls = next(
        node for node in source.body if isinstance(node, ast.ClassDef) and node.name == "AscendAttentionMetadataBuilder"
    )
    fn = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "build")
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), fn],
        type_ignores=[],
    )
    namespace = dict(torch=torch, CrossAttentionSpec=type("CrossAttentionSpec", (), {}))
    exec(compile(ast.fix_missing_locations(module), "attention_v1.py", "exec"), namespace)
    count = len(offsets) - 1
    query = torch.tensor([*offsets, 999], dtype=torch.int32)
    seq_lens = torch.tensor(offsets[1:], dtype=torch.int32)
    common = SimpleNamespace(
        num_reqs=count,
        num_actual_tokens=live,
        query_start_loc=query,
        query_start_loc_cpu=query.clone(),
        _seq_lens_cpu=seq_lens,
        block_table_tensor=torch.zeros(count, 1, dtype=torch.int32),
        slot_mapping=torch.arange(offsets[-1]),
        attn_state="prefill",
        causal=True,
        max_query_len=max(b - a for a, b in zip(offsets, offsets[1:])),
    )
    meta_builder = SimpleNamespace(
        _split_decodes_and_prefills=lambda _: (0, count, 0, live),
        kv_cache_spec=None,
        speculative_config=None,
        attn_mask_builder=SimpleNamespace(get_attention_mask=lambda *args: None),
        model_config=SimpleNamespace(runner_type="generate"),
        _build_backend_metadata=lambda *args, **kwargs: {},
        metadata_cls=SimpleNamespace,
    )
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda *args, **kwargs: pytest.fail("duplicate upload"))
    result = namespace["build"](meta_builder, 0, common)
    assert result.query_start_loc.data_ptr() == query.data_ptr()
    torch.testing.assert_close(result.query_start_loc, torch.tensor(offsets, dtype=torch.int32))
    assert result.actual_seq_lengths_q == offsets[1:]
    assert result.slot_mapping.numel() == live
