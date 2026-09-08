# SPDX-License-Identifier: Apache-2.0
"""Execute production metadata/state kernels with CPU tensor pointer semantics.

This checks device indexing and graph contracts, not Triton compilation or
NPU arithmetic. Kernel compilation and replay still need hardware validation.
"""

import ast
import copy
import math
from contextlib import nullcontext
from dataclasses import dataclass, is_dataclass
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

ROOT = Path(__file__).resolve().parents[4]


def load_source(path, names, namespace):
    tree = ast.parse((ROOT / path).read_text())
    selected = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
            if isinstance(node, ast.FunctionDef):
                node.decorator_list = []
            selected.append(node)
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *selected],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(ROOT / path), "exec"), namespace)
    return namespace


class Pointer:
    def __init__(self, tensor, offset=0):
        self.tensor, self.offset = tensor.reshape(-1), torch.as_tensor(offset)
        self.dtype = SimpleNamespace(element_ty=tensor.dtype)

    def __add__(self, offset):
        return Pointer(self.tensor, self.offset + offset)


class BlockPointer(Pointer):
    def __init__(self, base, shape, strides, offsets, block_shape, order):
        coords = torch.meshgrid(
            *(torch.arange(size) + offset for size, offset in zip(block_shape, offsets, strict=True)), indexing="ij"
        )
        super().__init__(
            base.tensor, base.offset + sum(coord * stride for coord, stride in zip(coords, strides, strict=True))
        )
        self.bounds = [(coord >= 0) & (coord < size) for coord, size in zip(coords, shape, strict=True)]

    def mask(self, boundary_check):
        result = torch.ones_like(self.offset, dtype=torch.bool)
        for axis in boundary_check:
            result &= self.bounds[axis]
        return result


class TensorLanguage:
    int32 = torch.int32
    int64 = torch.int64
    float32 = torch.float32
    arange = staticmethod(torch.arange)
    cumsum = staticmethod(
        lambda x, axis, reverse=False: x.flip((axis,)).cumsum(axis).flip((axis,)) if reverse else x.cumsum(axis)
    )
    reshape = staticmethod(torch.reshape)
    trans = staticmethod(lambda x, dims=None: x.permute(dims if dims is not None else tuple(reversed(range(x.ndim)))))
    sum = staticmethod(torch.sum)
    where = staticmethod(torch.where)
    cdiv = staticmethod(lambda x, y: (x + y - 1) // y)
    exp = staticmethod(torch.exp)
    zeros = staticmethod(torch.zeros)
    make_block_ptr = BlockPointer
    dot = staticmethod(lambda a, b, **kwargs: a.float() @ b.float())

    def __init__(self):
        self.program = (0, 0)

    def program_id(self, axis):
        return self.program[axis]

    @staticmethod
    def load(pointer, mask=True, other=0, boundary_check=(), padding_option=None):
        if isinstance(pointer, BlockPointer):
            mask = pointer.mask(boundary_check)
            # Poison undefined padding so a missing zero-fill is observable.
            other = 0 if padding_option == "zero" else float("nan")
        offsets, mask = torch.broadcast_tensors(pointer.offset, torch.as_tensor(mask))
        return torch.where(
            mask,
            pointer.tensor[torch.where(mask, offsets, 0).long()],
            torch.as_tensor(other, dtype=pointer.tensor.dtype),
        )

    @staticmethod
    def store(pointer, values, mask=True, boundary_check=()):
        if isinstance(pointer, BlockPointer):
            assert values.dtype == pointer.tensor.dtype, "block-pointer store dtype mismatch"
            mask = pointer.mask(boundary_check)
        offsets, values, mask = torch.broadcast_tensors(pointer.offset, torch.as_tensor(values), torch.as_tensor(mask))
        pointer.tensor[offsets[mask].long()] = values[mask].to(pointer.tensor.dtype)


class Launch:
    def __init__(self, function, tl):
        self.function, self.tl = function, tl

    def __getitem__(self, grid):
        def run(*args, **kwargs):
            kwargs.pop("num_warps", None)
            kwargs.pop("num_stages", None)
            for first in range(grid[0]):
                for second in range(grid[1] if len(grid) > 1 else 1):
                    self.tl.program = first, second
                    self.function(
                        *(Pointer(arg) if isinstance(arg, torch.Tensor) else arg for arg in args),
                        **{
                            key: Pointer(value) if isinstance(value, torch.Tensor) else value
                            for key, value in kwargs.items()
                        },
                    )

        return run


@pytest.fixture
def graph_code():
    namespace = dict(torch=torch, dataclass=dataclass, __name__=__name__)
    load_source("vllm_ascend/ops/gdn_graph_metadata.py", {"GDNFullGraphMetadata"}, namespace)
    tl = TensorLanguage()
    namespace.update(
        tl=tl,
        triton=SimpleNamespace(cdiv=tl.cdiv, next_power_of_2=lambda n: 1 << (n - 1).bit_length()),
        CHUNK_SIZE=64,
        SOLVE_BLOCK_SIZE=1216,
        CUMSUM_WORKING_SET=2**18,
        STATE_BLOCK_SIZE=1024,
        METADATA_BLOCK_SIZE=32,
    )
    load_source(
        "vllm_ascend/ops/triton/fla/graph.py",
        {
            "_chunk_metadata_kernel",
            "cumsum_block_size",
            "allocate_graph_metadata",
            "update_graph_metadata",
            "_state_transfer_kernel",
            "transfer_state",
        },
        namespace,
    )
    namespace["_chunk_metadata_kernel"] = Launch(namespace["_chunk_metadata_kernel"], tl)
    namespace["_state_transfer_kernel"] = Launch(namespace["_state_transfer_kernel"], tl)
    return namespace


@pytest.mark.parametrize("lengths", [[1, 65, 127], [64, 128], [1] * 64, [1217, 1, 1], [1536], [1, 0, 63, 0, 64]])
def test_chunk_tables_cover_each_real_chunk_once_and_pad_with_empty_sequence(graph_code, lengths):
    code = graph_code
    target = code["allocate_graph_metadata"](2048, 64, 16, "cpu")
    cu = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()])
    indices = torch.arange(len(lengths), dtype=torch.int32)
    code["update_graph_metadata"](target, cu, indices, torch.ones(len(lengths), dtype=torch.bool), 16)
    for table, chunk in ((target.chunk_indices, 64), (target.solve_indices, 1216), (target.cumsum_indices, 256)):
        expected = [(seq, block) for seq, length in enumerate(lengths) for block in range(math.ceil(length / chunk))]
        assert table[: len(expected)].tolist() == [list(row) for row in expected]
        assert table[len(expected) :].tolist() == [[64, 0]] * (len(table) - len(expected))
    assert target.query_start_loc[64] == target.query_start_loc[65] == sum(lengths)
    counts = torch.tensor([math.ceil(length / 64) for length in lengths] + [0] * (65 - len(lengths)))
    torch.testing.assert_close(target.chunk_offsets, torch.cat((torch.tensor([0]), counts.cumsum(0))).int())


def test_metadata_update_clears_removed_requests_and_preserves_addresses(graph_code):
    target = graph_code["allocate_graph_metadata"](256, 4, 16, "cpu")
    addresses = {name: tensor.data_ptr() for name, tensor in vars(target).items()}
    update = graph_code["update_graph_metadata"]
    update(target, torch.tensor([0, 1, 2, 129, 193]), torch.tensor([7, 2, 9, 3]), torch.ones(4, dtype=torch.bool), 16)
    update(target, torch.tensor([0, 3]), torch.tensor([5]), torch.tensor([False]), 16)
    assert target.state_read_indices.tolist() == [5, -1, -1, -1, -1]
    assert target.state_write_indices.tolist() == [5, -1, -1, -1, -1]
    assert target.query_start_loc.tolist() == [0, 3, 3, 3, 3, 3]
    assert not target.has_initial_state.any()
    assert addresses == {name: tensor.data_ptr() for name, tensor in vars(target).items()}


def test_state_transfer_uses_distinct_anchors_and_never_writes_padding(graph_code):
    code = graph_code
    metadata = code["allocate_graph_metadata"](256, 4, 1, "cpu")
    code["update_graph_metadata"](
        metadata, torch.tensor([0, 1, 4]), torch.tensor([3, 1]), torch.tensor([True, False]), 1
    )
    cache = torch.arange(6 * 2 * 3).reshape(6, 1, 3, 2).float()
    packed = torch.full((5, 1, 2, 3), -100.0)
    code["transfer_state"](cache, packed, metadata, write=False)
    torch.testing.assert_close(packed[0], cache[3].transpose(-1, -2))
    torch.testing.assert_close(packed[1], torch.zeros_like(packed[1]))
    torch.testing.assert_close(packed[2:], torch.full_like(packed[2:], -100))
    before = cache.clone()
    metadata.state_write_indices[:2] = torch.tensor([4, 2])
    packed[:2].add_(10)
    code["transfer_state"](cache, packed, metadata, write=True)
    torch.testing.assert_close(cache[4], packed[0].transpose(-1, -2))
    torch.testing.assert_close(cache[2], packed[1].transpose(-1, -2))
    torch.testing.assert_close(cache[[0, 1, 3, 5]], before[[0, 1, 3, 5]])


class AttentionState(Enum):
    ChunkedPrefill = 0
    PrefillNoCache = 1
    DecodeOnly = 2


class GDNAttentionMetadata(SimpleNamespace):
    pass


class AscendMetadata(SimpleNamespace):
    pass


def live_context(lengths, *, fresh=False):
    count, total = len(lengths), sum(lengths)
    cu = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()])
    indices = torch.arange(count, dtype=torch.int32) + 1
    flags = torch.full((count,), not fresh)
    num_decodes = 0 if fresh else sum(length == 1 for length in lengths)
    conv = SimpleNamespace(query_start_loc=cu, cache_indices=indices, initial_state_mode=flags)
    gdn = GDNAttentionMetadata(
        num_decodes=num_decodes,
        num_prefills=count - num_decodes,
        num_actual_tokens=total,
        non_spec_prefill_metadata=SimpleNamespace(causal_conv1d=conv),
        non_spec_state_indices_tensor=indices,
    )
    attention = AscendMetadata(
        num_decodes=num_decodes,
        num_prefills=count - num_decodes,
        num_actual_tokens=total,
        num_decode_tokens=num_decodes,
        actual_seq_lengths_q=cu[1:].tolist(),
        seq_lens_list=(cu[1:] - cu[:-1] + 5).tolist(),
        seq_lens=cu[1:] - cu[:-1] + 5,
        query_start_loc=cu,
        max_query_len=max(lengths),
        block_tables=torch.ones(count, 2, dtype=torch.int32),
        slot_mapping=torch.arange(total),
        attn_mask=torch.eye(4, dtype=torch.bool),
        causal=True,
        model_runner_type="generate",
        attn_state=AttentionState.PrefillNoCache if fresh else AttentionState.ChunkedPrefill,
    )
    context = SimpleNamespace(
        batch_descriptor=SimpleNamespace(num_tokens=256),
        attn_metadata={"gdn": gdn, "attention": attention},
        no_compile_layers={"gdn": SimpleNamespace(kv_cache=(torch.empty(8, 3, 8), torch.empty(8, 16, 128, 128)))},
    )
    return context


@pytest.fixture
def adapter_code(graph_code):
    namespace = dict(
        copy=copy,
        math=math,
        torch=torch,
        MAX_GRAPH_REQUESTS=64,
        GDN_GRAPH_HEAD_DIM=128,
        MAX_GDN_GRAPH_HEADS=64,
        FIA_CACHE_BLOCK_SIZE=128,
        allocate_graph_metadata=graph_code["allocate_graph_metadata"],
        update_graph_metadata=graph_code["update_graph_metadata"],
    )
    return load_source(
        "vllm_ascend/compilation/full_graph_metadata.py",
        {"capacity_metadata_signature", "FullGraphMetadataAdapter"},
        namespace,
    )


def test_adapter_reuses_capacity_across_request_arrivals_lengths_and_classification(adapter_code):
    code = adapter_code
    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_seqs=4), cache_config=SimpleNamespace(block_size=128)
    )
    context = live_context([65, 127], fresh=True)
    signature = code["capacity_metadata_signature"](context, config)
    adapter = code["FullGraphMetadataAdapter"](context, signature)
    adapter.update(context.attn_metadata)
    target = adapter.metadata["attention"]
    addresses = (
        target.block_tables.data_ptr(),
        target.slot_mapping.data_ptr(),
        adapter.metadata["gdn"].query_start_loc.data_ptr(),
    )
    for lengths in ([1, 193], [1, 1, 63, 127], [128], [256]):
        context = live_context(lengths)
        assert code["capacity_metadata_signature"](context, config) == signature
        adapter.update(context.attn_metadata)
        count, actual = len(lengths), sum(lengths)
        assert target.actual_seq_lengths_q == torch.tensor(lengths).cumsum(0).tolist() + [actual] * (4 - count) + [256]
        assert target.seq_lens_list[-1] == 256 - actual
        assert target.attn_state == AttentionState.ChunkedPrefill
        assert target.full_graph_token_capacity == 256
        assert target.num_actual_tokens == actual
        assert target.slot_mapping[actual:].tolist() == [-1] * (256 - actual)
        assert not target.block_tables[count:].any()
        assert addresses == (
            target.block_tables.data_ptr(),
            target.slot_mapping.data_ptr(),
            adapter.metadata["gdn"].query_start_loc.data_ptr(),
        )


@dataclass(frozen=True)
class Descriptor:
    num_tokens: int = 256
    has_lora: bool = False


class Mode(Enum):
    PIECEWISE = 0
    FULL = 1


def test_production_cache_captures_once_across_fresh_stateful_and_mixed_layouts(adapter_code, monkeypatch):
    sync = Mock()
    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(
            get_device_name=lambda: "Ascend910B4",
            current_stream=lambda: SimpleNamespace(synchronize=sync, wait_stream=Mock()),
            current_device=lambda: 0,
            Stream=Mock(),
            stream=lambda target: nullcontext(),
        ),
        raising=False,
    )
    namespace = dict(adapter_code)
    namespace.update(
        torch=torch,
        copy=copy,
        dataclass=dataclass,
        is_dataclass=is_dataclass,
        Enum=Enum,
        __name__=__name__,
        CUDAGraphMode=Mode,
        DEFAULT_FULL_PREFILL_GRAPH_MAX_ENTRIES=8,
        logger=Mock(),
        AscendAttentionBackendImpl=SimpleNamespace(update_graph_params=Mock()),
        _new_graph_params=lambda sizes: SimpleNamespace(attn_params={size: [] for size in sizes}),
    )
    names = {
        "_flatten",
        "_clone",
        "FullPrefillEntry",
        "_stateful_gdn_layers",
        "_snapshot_recurrent_states",
        "_restore_recurrent_states",
        "FullPrefillGraphCache",
    }
    load_source("vllm_ascend/compilation/full_prefill.py", names, namespace)
    config = SimpleNamespace(
        additional_config={},
        scheduler_config=SimpleNamespace(max_num_seqs=4),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            data_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
        ),
        cache_config=SimpleNamespace(mamba_cache_mode="none", block_size=128),
        speculative_config=None,
        kv_transfer_config=None,
        lora_config=None,
        model_config=SimpleNamespace(enforce_eager=False),
    )
    cache = namespace["FullPrefillGraphCache"](config)
    captures, replays = [], []
    model_layers = live_context([64]).no_compile_layers
    for lengths, fresh in (
        ([64, 128], True),
        ([1, 193], False),
        ([1, 1, 63, 127], False),
        ([128], False),
        ([1, 1], False),
    ):
        context = live_context(lengths, fresh=fresh)
        context.batch_descriptor = Descriptor()
        context.cudagraph_runtime_mode, context.capturing = Mode.PIECEWISE, False
        context.no_compile_layers = model_layers
        gdn = context.attn_metadata["gdn"]
        if lengths == [1, 1]:
            # FIA classifies all one-token queries as decode. GDN must still
            # initialize the newly arrived request rather than replay decode.
            gdn.num_decodes, gdn.num_prefills = 1, 1
            gdn.non_spec_prefill_metadata.causal_conv1d.initial_state_mode[1] = False
            context.attn_metadata["attention"].attn_state = AttentionState.DecodeOnly
        gdn.num_spec_decodes = 0
        prefill_lengths = lengths[gdn.num_decodes :]
        gdn.non_spec_prefill_metadata.chunk = SimpleNamespace(
            cu_seqlens_host=tuple([0, *torch.tensor(prefill_lengths).cumsum(0).tolist()]), fresh_prefill=fresh
        )
        live_metadata = context.attn_metadata

        def capture(entry, args, kwargs, context=context):
            captures.append(entry)
            context.capturing = True
            entry.capture = SimpleNamespace(num_graphs=1, num_eager_breaks=0)

        def replay(entry, args, kwargs, context=context, lengths=lengths):
            replays.append(entry)
            assert context.cudagraph_runtime_mode == Mode.FULL
            packed = context.attn_metadata["gdn"]
            assert packed.query_start_loc[: len(lengths) + 1].tolist() == [0, *torch.tensor(lengths).cumsum(0).tolist()]
            return "result"

        assert cache.run(context, (torch.zeros(256),), {}, runnable=Mock(), capture=capture, replay=replay) == (
            True,
            "result",
        )
        assert context.attn_metadata is live_metadata and context.cudagraph_runtime_mode == Mode.PIECEWISE
    assert len(captures) == len(cache.entries) == 1
    assert len(replays) == 5
    assert all(entry is captures[0] for entry in replays)


@pytest.mark.parametrize("output_dtype", [torch.float32, torch.bfloat16])
def test_cumsum_stores_destination_dtype_for_bf16_input(graph_code, output_dtype):
    lengths, capacity, heads = [1, 65, 63], 160, 32
    total = sum(lengths)
    metadata = graph_code["allocate_graph_metadata"](capacity, 4, heads, "cpu")
    graph_code["update_graph_metadata"](
        metadata, torch.tensor([0, 1, 66, total]), torch.arange(3), torch.zeros(3, dtype=torch.bool), heads
    )
    values = torch.full((1, capacity, heads), 1 / 256, dtype=torch.bfloat16)
    values[:, ::4] = 1
    values[:, total:] = float("nan")
    output = torch.full(values.shape, float("nan"), dtype=output_dtype)
    expected = torch.empty((1, total, heads), dtype=output_dtype)
    start = 0
    for length in lengths:
        for offset in range(0, length, 64):
            lo, hi = start + offset, start + min(offset + 64, length)
            expected[:, lo:hi] = values[:, lo:hi].float().cumsum(1).to(output_dtype)
        start += length
    tl = TensorLanguage()
    namespace = dict(tl=tl)
    load_source("vllm_ascend/ops/triton/fla/cumsum.py", {"chunk_local_cumsum_scalar_kernel"}, namespace)
    Launch(namespace["chunk_local_cumsum_scalar_kernel"], tl)[(len(metadata.cumsum_indices), 1)](
        s=values,
        o=output,
        scale=None,
        cu_seqlens=metadata.query_start_loc,
        chunk_indices=metadata.cumsum_indices,
        T=capacity,
        H=heads,
        BLOCK_T=graph_code["cumsum_block_size"](heads),
        REVERSE=False,
        HAS_SCALE=False,
        IS_VARLEN=True,
        HEAD_FIRST=False,
        CHUNK_SIZE=64,
    )
    torch.testing.assert_close(output[:, :total], expected, atol=0, rtol=0)
    assert output[:, total:].isnan().all()


def test_triton_state_and_output_stages_match_recurrence_with_nan_padding(graph_code):
    torch.manual_seed(7)
    lengths, capacity, heads, key_heads, dim = [1, 63, 65], 160, 2, 1, 128
    total = sum(lengths)
    meta = graph_code["allocate_graph_metadata"](capacity, 4, heads, "cpu")
    graph_code["update_graph_metadata"](
        meta, torch.tensor([0, 1, 64, total]), torch.tensor([0, 1, 2]), torch.ones(3, dtype=torch.bool), heads
    )
    q = torch.nn.functional.normalize(torch.randn(1, capacity, key_heads, dim), dim=-1).bfloat16()
    k = torch.nn.functional.normalize(torch.randn_like(q.float()), dim=-1).bfloat16()
    v = torch.randn(1, capacity, heads, dim).bfloat16()
    g = -torch.rand(1, capacity, heads) * 0.05
    beta = torch.rand(1, capacity, heads)
    initial = torch.randn(5, heads, dim, dim) * 0.05
    initial[3:] = float("nan")
    w = torch.full((1, capacity, heads, dim), float("nan"), dtype=k.dtype)
    u = torch.full_like(v, float("nan"))
    cumulative_g = torch.full_like(g, float("nan"))
    expected_output = torch.empty(1, total, heads, dim)
    expected_state = initial.clone()
    start = 0
    for seq, length in enumerate(lengths):
        state = initial[seq].clone()
        for token in range(start, start + length):
            kt = k[0, token, 0].float().expand(heads, -1)
            state *= g[0, token].exp()[:, None, None]
            delta = (v[0, token].float() - torch.einsum("hk,hkv->hv", kt, state)) * beta[0, token, :, None]
            state += kt[:, :, None] * delta[:, None, :]
            expected_output[0, token] = torch.einsum("k,hkv->hv", q[0, token, 0].float(), state) / math.sqrt(dim)
        expected_state[seq] = state
        for block in range(math.ceil(length / 64)):
            lo, hi = start + block * 64, min(start + length, start + (block + 1) * 64)
            for head in range(heads):
                kk = k[0, lo:hi, 0].float()
                gg = g[0, lo:hi, head].cumsum(0)
                bb = beta[0, lo:hi, head]
                lower = torch.tril(bb[:, None] * (kk @ kk.T) * (gg[:, None] - gg[None, :]).exp(), diagonal=-1)
                inverse = torch.linalg.inv(torch.eye(hi - lo) + lower)
                w[0, lo:hi, head] = (inverse @ (bb[:, None] * kk * gg.exp()[:, None])).bfloat16()
                u[0, lo:hi, head] = (inverse @ (bb[:, None] * v[0, lo:hi, head].float())).bfloat16()
                cumulative_g[0, lo:hi, head] = gg
        start += length
    q[:, total:] = k[:, total:] = float("nan")
    v[:, total:] = float("nan")
    tl = TensorLanguage()
    namespace = dict(tl=tl, safe_exp=lambda x: torch.where(x <= 0, x, -float("inf")).exp())
    load_source(
        "vllm_ascend/ops/triton/fla/chunk_delta_h.py", {"chunk_gated_delta_rule_fwd_kernel_h_blockdim64"}, namespace
    )
    load_source("vllm_ascend/ops/triton/fla/chunk_o.py", {"chunk_fwd_kernel_o"}, namespace)
    h = torch.full((1, len(meta.chunk_indices), heads, dim, dim), float("nan"), dtype=k.dtype)
    final = torch.full_like(initial, float("nan"))
    v_new = torch.full_like(v, float("nan"))
    common = dict(
        cu_seqlens=meta.query_start_loc,
        chunk_offsets=meta.chunk_offsets,
        T=capacity,
        H=heads,
        Hg=key_heads,
        K=dim,
        V=dim,
        BT=64,
        USE_G=True,
        IS_VARLEN=True,
    )
    Launch(namespace["chunk_gated_delta_rule_fwd_kernel_h_blockdim64"], tl)[(1, 5 * heads)](
        k=k,
        v=u,
        w=w,
        v_new=v_new,
        g=cumulative_g.transpose(1, 2).contiguous(),
        h=h,
        h0=initial,
        ht=final,
        h_update=None,
        USE_INITIAL_STATE=True,
        STORE_FINAL_STATE=True,
        SAVE_NEW_VALUE=True,
        **common,
    )
    output = torch.full_like(v, float("nan"))
    Launch(namespace["chunk_fwd_kernel_o"], tl)[(1, 5 * heads)](
        q=q,
        k=k,
        v=v_new,
        h=h,
        g=cumulative_g.transpose(1, 2).contiguous(),
        o=output,
        scale=dim**-0.5,
        BK=128,
        BV=128,
        **common,
    )
    torch.testing.assert_close(final[:3], expected_state[:3], atol=0.025, rtol=0.025)
    torch.testing.assert_close(output[:, :total].float(), expected_output, atol=0.01, rtol=0.025)
    assert output[:, total:].isnan().all() and final[3:].isnan().all()
