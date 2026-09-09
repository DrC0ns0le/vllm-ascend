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
        size = tensor.untyped_storage().nbytes() // tensor.element_size()
        self.tensor = tensor.as_strided((size,), (1,), storage_offset=0)
        self.offset = torch.as_tensor(offset) + tensor.storage_offset()
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
    static_range = staticmethod(range)
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
        assert mask.dtype == torch.bool, "Triton load mask must be an i1 predicate"
        result = torch.broadcast_to(torch.as_tensor(other, dtype=pointer.tensor.dtype), offsets.shape).clone()
        result[mask] = pointer.tensor[offsets[mask].long()]
        return result

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
    namespace["VALUE_TILE"] = 32
    load_source(
        "vllm_ascend/ops/triton/fla/single_token.py", {"_single_token_gdn_kernel", "single_token_gdn"}, namespace
    )
    namespace["_single_token_gdn_kernel"] = Launch(namespace["_single_token_gdn_kernel"], tl)
    return namespace


@pytest.mark.parametrize("heads", [1, 16, 32, 64])
@pytest.mark.parametrize("skip_single_token", [False, True])
@pytest.mark.parametrize("lengths", [[], [1, 65, 127], [64, 128], [1] * 64, [1217, 1, 1], [1536], [1, 0, 63, 0, 64]])
def test_chunk_tables_cover_each_real_chunk_once_and_pad_with_empty_sequence(
    graph_code, lengths, heads, skip_single_token
):
    code = graph_code
    target = code["allocate_graph_metadata"](2048, 64, heads, "cpu")
    cu = torch.tensor([0, *torch.tensor(lengths, dtype=torch.int64).cumsum(0).tolist()])
    indices = torch.arange(len(lengths), dtype=torch.int32)
    code["update_graph_metadata"](
        target, cu, indices, torch.ones(len(lengths), dtype=torch.bool), heads, skip_single_token=skip_single_token
    )
    for table, chunk in (
        (target.chunk_indices, 64),
        (target.solve_indices, 1216),
        (target.cumsum_indices, code["cumsum_block_size"](heads)),
    ):
        expected = [
            (seq, block)
            for seq, length in enumerate(lengths)
            if not skip_single_token or length > 1
            for block in range(math.ceil(length / chunk))
        ]
        assert table[: len(expected)].tolist() == [list(row) for row in expected]
        assert table[len(expected) :].tolist() == [[64, 0]] * (len(table) - len(expected))
    assert target.query_start_loc[64] == target.query_start_loc[65] == sum(lengths)
    counts = torch.tensor(
        [math.ceil(length / 64) if not skip_single_token or length > 1 else 0 for length in lengths]
        + [0] * (65 - len(lengths))
    )
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


def test_gdn_refresh_is_one_launch_for_changing_request_counts(graph_code):
    target = graph_code["allocate_graph_metadata"](768, 64, 16, "cpu")
    launcher = Mock(wraps=graph_code["_chunk_metadata_kernel"].__getitem__)

    class CountedKernel:
        def __getitem__(self, grid):
            return launcher(grid)

    graph_code["_chunk_metadata_kernel"] = CountedKernel()
    for count in (1, 64, 3, 0):
        graph_code["update_graph_metadata"](
            target, torch.arange(count + 1), torch.arange(count), torch.zeros(count, dtype=torch.bool), 16
        )
    assert launcher.call_count == 4
    assert all(call.args == launcher.call_args_list[0].args for call in launcher.call_args_list)


def test_gdn_refresh_accepts_strided_single_column_builder_indices(graph_code):
    target = graph_code["allocate_graph_metadata"](768, 4, 16, "cpu")
    # Poison gaps in runner views so accidentally treating them as contiguous
    # changes boundaries/slots/flags rather than passing by coincidence.
    cu = torch.tensor([0, -99, 63, -99, 128, -99])[::2]
    indices = torch.tensor([[3, -99], [8, -99]])[:, :1]
    flags = torch.tensor([True, False, False, True])[::2]
    graph_code["update_graph_metadata"](target, cu, indices, flags, 16)
    assert target.query_start_loc.tolist() == [0, 63, 128, 128, 128, 128]
    assert target.state_read_indices.tolist() == [3, 8, -1, -1, -1]
    assert target.state_write_indices.tolist() == [3, 8, -1, -1, -1]
    assert target.has_initial_state.tolist() == [True, False, False, False, False]
    assert target.chunk_indices[:3].tolist() == [[0, 0], [1, 0], [1, 1]]
    with pytest.raises(ValueError, match="one cache slot"):
        graph_code["update_graph_metadata"](target, cu, torch.zeros(2, 2), flags, 16)


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
    # The production non-spec builder gives conv1d the (N, 1) block table.
    conv = SimpleNamespace(query_start_loc=cu, cache_indices=indices[:, None], initial_state_mode=flags)
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
    tl = TensorLanguage()
    attention_code = dict(tl=tl, triton=SimpleNamespace(cdiv=tl.cdiv), METADATA_BLOCK_SIZE=256)
    load_source(
        "vllm_ascend/ops/triton/graph_metadata.py",
        {"_refresh_attention_metadata_kernel", "refresh_attention_metadata"},
        attention_code,
    )
    attention_code["_refresh_attention_metadata_kernel"] = Launch(
        attention_code["_refresh_attention_metadata_kernel"], tl
    )
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
        refresh_attention_metadata=attention_code["refresh_attention_metadata"],
    )
    return load_source(
        "vllm_ascend/compilation/full_graph_metadata.py",
        {"capacity_metadata_signature", "FullGraphMetadataAdapter"},
        namespace,
    )


@pytest.mark.parametrize("seq_dtype", [torch.int32, torch.int64])
def test_adapter_reuses_capacity_across_request_arrivals_lengths_and_classification(adapter_code, seq_dtype):
    code = adapter_code
    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_seqs=4), cache_config=SimpleNamespace(block_size=128)
    )
    context = live_context([65, 127], fresh=True)
    context.attn_metadata["attention"].seq_lens = context.attn_metadata["attention"].seq_lens.to(seq_dtype)
    signature = code["capacity_metadata_signature"](context, config)
    adapter = code["FullGraphMetadataAdapter"](context, signature)
    adapter.update(context.attn_metadata)
    target = adapter.metadata["attention"]
    addresses = (
        target.block_tables.data_ptr(),
        target.slot_mapping.data_ptr(),
        target.seq_lens.data_ptr(),
        adapter.metadata["gdn"].query_start_loc.data_ptr(),
    )
    for lengths in ([1, 193], [1, 1, 63, 127], [128], [256]):
        context = live_context(lengths)
        source = context.attn_metadata["attention"]
        storage = torch.full((len(lengths), 2), -99, dtype=seq_dtype)
        storage[:, 0] = source.seq_lens
        source.seq_lens = storage[:, 0]
        assert code["capacity_metadata_signature"](context, config) == signature
        adapter.update(context.attn_metadata)
        count, actual = len(lengths), sum(lengths)
        assert target.actual_seq_lengths_q == torch.tensor(lengths).cumsum(0).tolist() + [actual] * (4 - count) + [256]
        assert target.seq_lens_list[-1] == 256 - actual
        expected_lengths = source.seq_lens_list + [0] * (4 - count) + [256 - actual]
        assert target.seq_lens.tolist() == target.seq_lens_cpu.tolist() == expected_lengths
        assert target.seq_lens_cpu is target.seq_lens
        assert target.seq_lens.dtype == seq_dtype
        assert target.attn_state == AttentionState.ChunkedPrefill
        assert target.full_graph_token_capacity == 256
        assert target.num_actual_tokens == actual
        assert target.slot_mapping[actual:].tolist() == [-1] * (256 - actual)
        assert not target.block_tables[count:].any()
        assert addresses == (
            target.block_tables.data_ptr(),
            target.slot_mapping.data_ptr(),
            target.seq_lens.data_ptr(),
            adapter.metadata["gdn"].query_start_loc.data_ptr(),
        )


@pytest.mark.parametrize("decode_tokens", [4, 10, 64])
def test_chunked_prefill_metadata_reuses_graph_buffers_through_decode(adapter_code, decode_tokens):
    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_seqs=64), cache_config=SimpleNamespace(block_size=128)
    )
    # Logical requests use nonconsecutive cache slots. Requests B and C
    # arrive while A is still prefilling; the final step reorders live rows.
    steps = [
        [(3, 63)],
        [(3, 64), (8, 1)],
        [(3, 2), (8, 127), (11, 63)],
        [(8, 65), (11, 129)],
        [(11, 3)],
    ]
    steps += [[(11, 1), (3, 1), (8, 1)] for _ in range(decode_tokens)]
    completed = {}
    adapter, signature = None, None
    for step in steps:
        slots, lengths = zip(*step, strict=True)
        context = live_context(list(lengths))
        context.batch_descriptor.num_tokens = 768
        gdn = context.attn_metadata["gdn"].non_spec_prefill_metadata.causal_conv1d
        flags = torch.tensor([slot in completed for slot in slots])
        gdn.cache_indices = torch.tensor(slots, dtype=torch.int32)[:, None]
        gdn.initial_state_mode = flags
        source = context.attn_metadata["attention"]
        source.seq_lens_list = [completed.get(slot, 0) + length for slot, length in step]
        source.seq_lens = torch.tensor(source.seq_lens_list)
        # Exercise noncontiguous runner tables and slot views too.
        source.block_tables = torch.arange(len(slots) * 8, dtype=torch.int32).reshape(len(slots), 8)[:, ::2]
        source.slot_mapping = torch.arange(sum(lengths) * 2)[::2]
        current = adapter_code["capacity_metadata_signature"](context, config)
        if adapter is None:
            signature = current
            adapter = adapter_code["FullGraphMetadataAdapter"](context, signature)
            addresses = {name: tensor.data_ptr() for name, tensor in vars(adapter.metadata["gdn"]).items()}
        assert current == signature
        adapter.update(context.attn_metadata)
        target = adapter.metadata["gdn"]
        n = len(slots)
        assert target.has_initial_state[:n].tolist() == flags.tolist()
        assert target.state_read_indices[:n].tolist() == list(slots)
        assert target.state_write_indices[:n].tolist() == list(slots)
        assert target.state_write_indices[n:].eq(-1).all()
        expected_chunks = [
            [row, chunk] for row, length in enumerate(lengths) if length > 1 for chunk in range(math.ceil(length / 64))
        ]
        assert target.chunk_indices[: len(expected_chunks)].tolist() == expected_chunks
        assert target.chunk_indices[len(expected_chunks) :].tolist() == [[64, 0]] * (
            len(target.chunk_indices) - len(expected_chunks)
        )
        assert addresses == {name: tensor.data_ptr() for name, tensor in vars(target).items()}
        attention = adapter.metadata["attention"]
        torch.testing.assert_close(attention.block_tables[:n, :4], source.block_tables)
        assert not attention.block_tables[n:].any()
        assert not attention.block_tables[:n, 4:].any()
        torch.testing.assert_close(attention.slot_mapping[: sum(lengths)], source.slot_mapping)
        assert attention.slot_mapping[sum(lengths) :].eq(-1).all()
        assert attention.query_start_loc.tolist() == [0, *torch.tensor(lengths).cumsum(0).tolist()] + [sum(lengths)] * (
            64 - n
        ) + [768]
        assert attention.seq_lens_list[:n] == source.seq_lens_list
        assert attention.seq_lens.tolist() == source.seq_lens_list + [0] * (64 - n) + [768 - sum(lengths)]
        for slot, length in step:
            completed[slot] = completed.get(slot, 0) + length


@dataclass(frozen=True)
class Descriptor:
    num_tokens: int = 256
    has_lora: bool = False


class Mode(Enum):
    PIECEWISE = 0
    FULL = 1
    NONE = 2


@pytest.mark.parametrize("full_only", [False, True])
def test_production_cache_captures_once_across_fresh_stateful_and_mixed_layouts(adapter_code, monkeypatch, full_only):
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
    config.compilation_config = SimpleNamespace(cudagraph_mode=Mode.FULL if full_only else Mode.PIECEWISE)
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
        runtime_mode = Mode.FULL if full_only else Mode.PIECEWISE
        context.cudagraph_runtime_mode, context.capturing = runtime_mode, False
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
        assert context.attn_metadata is live_metadata and context.cudagraph_runtime_mode == runtime_mode
        if full_only:
            cache.seal()
    assert len(captures) == len(cache.entries) == 1
    assert len(replays) == 5
    assert all(entry is captures[0] for entry in replays)
    if full_only:
        with pytest.raises(RuntimeError, match="not captured at startup"):
            cache.run(context, (torch.zeros(257),), {}, runnable=Mock(), capture=Mock(), replay=Mock())


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


@pytest.mark.parametrize("steps", [4, 10, 64])
@pytest.mark.parametrize("cache_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("flag_dtype", [torch.bool, torch.int8])
def test_single_token_recurrence_survives_reordering_and_slot_reuse(graph_code, steps, cache_dtype, flag_dtype):
    torch.manual_seed(19)
    capacity, heads, key_heads, dim = 160, 4, 2, 128
    metadata = graph_code["allocate_graph_metadata"](capacity, 4, heads, "cpu")
    metadata.has_initial_state = metadata.has_initial_state.to(flag_dtype)
    addresses = {name: value.data_ptr() for name, value in vars(metadata).items()}
    # Exercise noncontiguous cache layout and grouped query/key heads.
    cache = (torch.randn(8, heads, dim, dim) * 0.05).to(cache_dtype).transpose(-1, -2)
    expected_cache = cache.clone()
    slots = [1, 4, 2, 5]
    for step in range(steps):
        fresh = set()
        if step == 2:
            slots[0] = 6  # A completed decode is replaced by a fresh request.
            fresh.add(0)
        elif step == 3:
            slots[1] = 1  # Reuse the first request's former cache slot.
            fresh.add(1)
        for request in fresh:
            cache[slots[request]] = expected_cache[slots[request]] = float("nan")
        before = cache.clone()
        # A prefill enters/leaves the batch, and packed row order changes.
        order = [2, 1, 3, 0] if step % 2 else [0, 1, 2]
        lengths = [1 if request < 2 else (63 if request == 2 else 65) for request in order]
        boundaries = [0, *torch.tensor(lengths).cumsum(0).tolist()]
        graph_code["update_graph_metadata"](
            metadata,
            torch.tensor(boundaries),
            torch.tensor([slots[request] for request in order])[:, None],
            torch.tensor([request not in fresh for request in order]),
            heads,
            skip_single_token=True,
        )
        q = torch.nn.functional.normalize(torch.randn(1, capacity, key_heads, dim), dim=-1).bfloat16()
        k = torch.nn.functional.normalize(torch.randn_like(q.float()), dim=-1).bfloat16()
        v = torch.randn(1, capacity, heads, dim).bfloat16()
        # Gates and beta may be strided views of fused projection results.
        g = (-torch.rand(1, capacity, heads * 2) * 0.05)[..., ::2]
        beta = torch.rand(1, capacity, heads * 2)[..., 1::2]
        output = torch.full_like(v, float("nan"))
        expected_output = torch.full_like(v, float("nan"))
        step_reference = before.clone()
        for row, request in enumerate(order):
            if lengths[row] != 1:
                continue
            token, slot = boundaries[row], slots[request]
            qt = q[0, token].float().repeat_interleave(heads // key_heads, dim=0)
            kt = k[0, token].float().repeat_interleave(heads // key_heads, dim=0)
            for reference in (step_reference, expected_cache):
                state = reference[slot].float().clone()
                if request in fresh:
                    state.zero_()
                # Independent recurrence uses cache orientation [head, value, key].
                state *= g[0, token].exp()[:, None, None]
                prediction = torch.einsum("hvk,hk->hv", state, kt)
                delta = beta[0, token, :, None] * (v[0, token].float() - prediction)
                state += delta[:, :, None] * kt[:, None, :]
                expected_output[0, token] = torch.einsum("hvk,hk->hv", state, qt) / math.sqrt(dim)
                reference[slot] = state.to(cache_dtype)
        graph_code["single_token_gdn"](q, k, v, g, beta, cache, metadata, output)
        torch.testing.assert_close(output, expected_output, atol=0.001, rtol=0.01, equal_nan=True)
        # Check each update against the same incoming state. BF16 persistence
        # allows one rounding unit for different FP32 reduction orders.
        state_rtol = torch.finfo(cache_dtype).eps if cache_dtype == torch.bfloat16 else 1e-5
        torch.testing.assert_close(cache, step_reference, atol=1e-6, rtol=state_rtol, equal_nan=True)
        # Also bound trajectory drift from an independently evolving cache.
        # Near-zero elements make an elementwise relative bound unsuitable
        # after repeated BF16 rounding; use relative RMS for the state vector.
        active = cache[[slots[0], slots[1]]].float()
        expected_active = expected_cache[[slots[0], slots[1]]].float()
        relative_rms = (active - expected_active).square().mean().sqrt() / expected_active.square().mean().sqrt()
        assert relative_rms < (torch.finfo(torch.bfloat16).eps if cache_dtype == torch.bfloat16 else 1e-5)
        untouched = torch.ones(cache.shape[0], dtype=torch.bool)
        untouched[[slots[0], slots[1]]] = False
        torch.testing.assert_close(cache[untouched], before[untouched], atol=0, rtol=0, equal_nan=True)
        assert addresses == {name: value.data_ptr() for name, value in vars(metadata).items()}


def test_packed_graph_uses_direct_state_without_staging_or_gate_layout_copies():
    tensor = torch.empty(1, 64, 2, 128)
    state = torch.empty(8, 2, 128, 128)
    meta = SimpleNamespace(
        query_start_loc=object(),
        cumsum_indices=object(),
        chunk_indices=object(),
        solve_indices=object(),
        chunk_offsets=object(),
    )
    h_stage = Mock(return_value=(tensor, tensor, state))
    o_stage = Mock(return_value=tensor)
    namespace = dict(
        GDN_GRAPH_HEAD_DIM=128,
        CHUNK_SIZE=64,
        l2norm_fwd=lambda value: value,
        single_token_gdn=Mock(),
        chunk_local_cumsum=Mock(return_value=tensor),
        chunk_scaled_dot_kkt_fwd=Mock(return_value=tensor),
        solve_tril=Mock(return_value=tensor),
        recompute_w_u_fwd=Mock(return_value=(tensor, tensor)),
        chunk_gated_delta_rule_fwd_h=h_stage,
        chunk_fwd_o=o_stage,
        transfer_state=Mock(side_effect=AssertionError("FULL must not stage recurrent state")),
        torch=SimpleNamespace(empty=Mock(side_effect=AssertionError("FULL must not allocate packed states"))),
    )
    load_source("vllm_ascend/ops/triton/fla/graph.py", {"chunk_gated_delta_rule_graph"}, namespace)
    assert (
        namespace["chunk_gated_delta_rule_graph"](tensor, tensor, tensor, tensor, tensor, state, meta, output=tensor)
        is tensor
    )
    assert h_stage.call_args.kwargs["state_cache"] is state
    assert h_stage.call_args.kwargs["state_metadata"] is meta
    assert h_stage.call_args.kwargs["token_major_g"] and o_stage.call_args.kwargs["token_major_g"]
    namespace["transfer_state"].assert_not_called()


@pytest.mark.parametrize("requests", [2, 3, 4, 8, 16, 32, 64])
@pytest.mark.parametrize("cache_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("split_single_token", [False, True])
@pytest.mark.parametrize("direct_state", [False, True])
def test_triton_state_and_output_stages_match_recurrence_with_nan_padding(
    graph_code, requests, cache_dtype, split_single_token, direct_state
):
    torch.manual_seed(7)
    # Many decodes coexist with a continuing partial prefill and a fresh
    # prefill. At concurrency two, use one decode and one fresh prefill.
    lengths = [1, 65] if requests == 2 else [1] * (requests - 2) + [63, 65]
    heads, key_heads, dim = 2, 1, 128
    total = sum(lengths)
    capacity = math.ceil((total + 1) / 64) * 64
    meta = graph_code["allocate_graph_metadata"](capacity, requests, heads, "cpu")
    slots = torch.randperm(requests + 1)[:requests]
    flags = torch.ones(requests, dtype=torch.bool)
    flags[-1] = False
    if requests >= 4:
        flags[0] = False  # A newly arrived one-token request also uses the fast path.
    graph_code["update_graph_metadata"](
        meta,
        torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()]),
        slots[:, None],
        flags,
        heads,
        skip_single_token=split_single_token,
    )
    q = torch.nn.functional.normalize(torch.randn(1, capacity, key_heads, dim), dim=-1).bfloat16()
    k = torch.nn.functional.normalize(torch.randn_like(q.float()), dim=-1).bfloat16()
    v = torch.randn(1, capacity, heads, dim).bfloat16()
    g = -torch.rand(1, capacity, heads) * 0.05
    beta = torch.rand(1, capacity, heads)
    cache = (torch.randn(2 * requests + 3, heads, dim, dim) * 0.05).to(cache_dtype)
    if direct_state:
        cache = cache.transpose(-1, -2)
    write_slots = slots + requests + 1 if direct_state and requests == 3 else slots
    meta.state_write_indices[:requests].copy_(write_slots)
    cache[slots[~flags]] = float("nan")
    before = cache.clone()
    initial = torch.full((requests + 1, heads, dim, dim), float("nan"), dtype=cache_dtype)
    if not direct_state:
        graph_code["transfer_state"](cache, initial, meta, write=False, skip_single_token=split_single_token)
    expected_initial = torch.where(flags[:, None, None, None], before[slots].transpose(-1, -2), 0)
    chunk_rows = torch.tensor([not split_single_token or length > 1 for length in lengths])
    if not direct_state:
        torch.testing.assert_close(initial[:requests][chunk_rows], expected_initial[chunk_rows])
    if split_single_token:
        assert initial[:requests][~chunk_rows].isnan().all()
    w = torch.full((1, capacity, heads, dim), float("nan"), dtype=k.dtype)
    u = torch.full_like(v, float("nan"))
    cumulative_g = torch.full_like(g, float("nan"))
    expected_output = torch.empty(1, total, heads, dim)
    expected_state = expected_initial.float().clone()
    start = 0
    for seq, length in enumerate(lengths):
        state = expected_initial[seq].float().clone()
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
    output = torch.full_like(v, float("nan"))
    if split_single_token:
        graph_code["single_token_gdn"](q, k, v, g, beta, cache, meta, output)
    tl = TensorLanguage()
    namespace = dict(tl=tl, safe_exp=lambda x: torch.where(x <= 0, x, -float("inf")).exp())
    load_source(
        "vllm_ascend/ops/triton/fla/chunk_delta_h.py", {"chunk_gated_delta_rule_fwd_kernel_h_blockdim64"}, namespace
    )
    load_source("vllm_ascend/ops/triton/fla/chunk_o.py", {"chunk_fwd_kernel_o"}, namespace)
    h = torch.full((1, len(meta.chunk_indices), heads, dim, dim), float("nan"), dtype=k.dtype)
    final = torch.full_like(initial, float("nan"), dtype=torch.float32)
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
        SKIP_SINGLE_TOKEN=split_single_token,
        G_TOKEN_STRIDE=cumulative_g.stride(1) if direct_state else 1,
        G_HEAD_STRIDE=cumulative_g.stride(2) if direct_state else 0,
    )
    Launch(namespace["chunk_gated_delta_rule_fwd_kernel_h_blockdim64"], tl)[(1, (requests + 1) * heads)](
        k=k,
        v=u,
        w=w,
        v_new=v_new,
        g=cumulative_g if direct_state else cumulative_g.transpose(1, 2).contiguous(),
        h=h,
        h0=cache if direct_state else initial,
        ht=cache if direct_state else final,
        h_update=None,
        USE_INITIAL_STATE=True,
        STORE_FINAL_STATE=True,
        SAVE_NEW_VALUE=True,
        DIRECT_STATE=direct_state,
        state_read_indices=meta.state_read_indices,
        state_write_indices=meta.state_write_indices,
        state_initial_flags=meta.has_initial_state.to(torch.int8),
        STATE_N=cache.stride(0),
        STATE_H=cache.stride(1),
        STATE_V=cache.stride(2),
        STATE_K=cache.stride(3),
        **common,
    )
    Launch(namespace["chunk_fwd_kernel_o"], tl)[(1, (requests + 1) * heads)](
        q=q,
        k=k,
        v=v_new,
        h=h,
        g=cumulative_g if direct_state else cumulative_g.transpose(1, 2).contiguous(),
        o=output,
        scale=dim**-0.5,
        BK=128,
        BV=128,
        **common,
    )
    if not direct_state:
        torch.testing.assert_close(final[:requests][chunk_rows], expected_state[chunk_rows], atol=0.025, rtol=0.025)
    torch.testing.assert_close(output[:, :total].float(), expected_output, atol=0.01, rtol=0.025)
    assert output[:, total:].isnan().all() and final[requests:].isnan().all()
    if split_single_token:
        assert final[:requests][~chunk_rows].isnan().all()
    if not direct_state:
        graph_code["transfer_state"](cache, final, meta, write=True, skip_single_token=split_single_token)
    torch.testing.assert_close(cache[write_slots].float(), expected_state.transpose(-1, -2), atol=0.025, rtol=0.025)
    inactive = torch.ones(cache.shape[0], dtype=torch.bool)
    inactive[write_slots] = False
    torch.testing.assert_close(cache[inactive], before[inactive], atol=0, rtol=0, equal_nan=True)
