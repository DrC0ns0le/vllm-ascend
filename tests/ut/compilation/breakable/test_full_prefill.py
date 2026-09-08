# SPDX-License-Identifier: Apache-2.0
"""Exercise production prefill cache ownership/routing with a simulated graph."""

import importlib.util
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest


@dataclass(frozen=True)
class Device:
    type: str = "npu"


class Tensor:
    device = Device()
    dtype = "int64"

    def __init__(self, values):
        self.values = list(values)
        self.shape = (len(values),)

    def stride(self):
        return (1,)

    def clone(self):
        result = Tensor(self.values)
        result.device, result.dtype = self.device, self.dtype
        return result

    def reshape(self, *args):
        return self

    def tolist(self):
        return self.values[:]

    def copy_(self, other, non_blocking=False):
        self.values[:] = other.values

    def to(self, *, dtype):
        result = self.clone()
        result.dtype = dtype
        return result

    def index_select(self, dim, indices):
        assert dim == 0
        return Tensor([self.values[index] for index in indices.values])

    def index_copy_(self, dim, indices, saved):
        assert dim == 0 and indices.dtype == "int64"
        for index, value in zip(indices.values, saved.values, strict=True):
            self.values[index] = value


@dataclass
class GDNChunkedPrefillMetadata:
    cu_seqlens_host: tuple = (0, 127, 256)
    fresh_prefill: bool = True


@dataclass
class Prefill:
    chunk: GDNChunkedPrefillMetadata


@dataclass
class GDNAttentionMetadata:
    non_spec_prefill_metadata: Prefill
    prefill_state_indices: Tensor
    num_prefills: int = 2
    num_decodes: int = 0
    num_spec_decodes: int = 0
    non_spec_state_indices_tensor: Tensor | None = None


class AttentionState(Enum):
    ChunkedPrefill = 1
    PrefillNoCache = 2


@dataclass
class AscendMetadata:
    seq_lens: Tensor
    seq_lens_cpu: Tensor
    seq_lens_list: list[int]
    actual_seq_lengths_q: list[int]
    block_tables: Tensor
    attn_state: AttentionState = AttentionState.ChunkedPrefill
    num_prefills: int = 2
    num_decodes: int = 0


@dataclass(frozen=True)
class Descriptor:
    num_tokens: int = 256
    has_lora: bool = False


class Mode(Enum):
    NONE = 0
    PIECEWISE = 1
    FULL = 2


@pytest.fixture
def setup(monkeypatch):
    sync = Mock()
    active_stream = SimpleNamespace(synchronize=sync, wait_stream=Mock())

    @contextmanager
    def stream(target):
        nonlocal active_stream
        previous, active_stream = active_stream, target
        try:
            yield
        finally:
            active_stream = previous

    modules = {
        "torch": dict(
            Tensor=Tensor,
            int64="int64",
            npu=SimpleNamespace(
                current_stream=lambda: active_stream,
                current_device=lambda: 0,
                get_device_name=lambda: "Ascend910B4",
                Stream=Mock(side_effect=lambda **kwargs: SimpleNamespace(synchronize=sync, wait_stream=Mock())),
                stream=stream,
            ),
        ),
        "vllm.config": dict(CUDAGraphMode=Mode),
        "vllm.logger": dict(logger=Mock()),
        "vllm_ascend.attention.attention_v1": dict(
            AscendAttentionBackendImpl=SimpleNamespace(update_graph_params=Mock())
        ),
        "vllm_ascend.compilation.acl_graph": dict(
            _new_graph_params=lambda sizes: SimpleNamespace(attn_params={size: [] for size in sizes}),
        ),
        "vllm_ascend.compilation.full_graph_metadata": dict(
            FullGraphMetadataAdapter=Mock(),
            capacity_metadata_signature=lambda *args: None,
        ),
    }
    for name, attrs in modules.items():
        module = ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
    root = Path(__file__).resolve().parents[4]
    spec = importlib.util.spec_from_file_location(
        "tested_full_prefill", root / "vllm_ascend/compilation/full_prefill.py"
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "tested_full_prefill", module)
    spec.loader.exec_module(module)
    config = SimpleNamespace(
        additional_config={},
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            data_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
        ),
        cache_config=SimpleNamespace(mamba_cache_mode="none"),
        speculative_config=None,
        kv_transfer_config=None,
        lora_config=None,
        model_config=SimpleNamespace(enforce_eager=False),
    )
    metadata = GDNAttentionMetadata(Prefill(GDNChunkedPrefillMetadata()), Tensor([4, 9]))
    context = SimpleNamespace(
        cudagraph_runtime_mode=Mode.PIECEWISE,
        batch_descriptor=Descriptor(),
        capturing=False,
        attn_metadata={"gdn.0": metadata, "gdn.1": metadata},
    )
    return module, config, context, sync


def add_live_states(context, *, mixed=False):
    metadata = context.attn_metadata["gdn.0"]
    metadata.num_decodes = int(mixed)
    metadata.non_spec_prefill_metadata.chunk.fresh_prefill = mixed
    metadata.non_spec_state_indices_tensor = Tensor([4, 9, 2] if mixed else [4, 9])
    metadata.non_spec_state_indices_tensor.dtype = "int32"
    states = (Tensor(range(12)), Tensor(range(100, 112)))
    context.no_compile_layers = {name: SimpleNamespace(kv_cache=states) for name in context.attn_metadata}
    return states


def callbacks(context):
    def capture(entry, args, kwargs):
        assert context.cudagraph_runtime_mode == Mode.FULL
        assert context.full_prefill_graph and not context.capturing
        entry.capture = SimpleNamespace(num_graphs=1, num_eager_breaks=0)

    def replay(entry, args, kwargs):
        assert context.attn_metadata is entry.inputs[2]
        return args[0].values[:], context.attn_metadata["gdn.0"].prefill_state_indices.values[:]

    return dict(runnable=Mock(), capture=Mock(side_effect=capture), replay=Mock(side_effect=replay))


@pytest.mark.parametrize("mixed", [False, True])
def test_stateful_capture_advances_active_rows_once_and_replay_uses_new_slots(setup, mixed):
    module, config, context, _ = setup
    states = add_live_states(context, mixed=mixed)
    before = [state.values[:] for state in states]
    cache = module.FullPrefillGraphCache(config)
    ops = callbacks(context)

    def advance(*args):
        indices = context.attn_metadata["gdn.0"].non_spec_state_indices_tensor.values
        for state in states:
            for index in indices:
                state.values[index] += 10

    def capture(entry, args, kwargs):
        # Exercise the case where capture executes device work as well.
        advance()
        entry.capture = SimpleNamespace(num_graphs=1, num_eager_breaks=0)

    ops["runnable"].side_effect = advance
    ops["capture"].side_effect = capture
    ops["replay"].side_effect = advance
    assert cache.run(context, (Tensor([1]),), {}, **ops)[0]
    indices = context.attn_metadata["gdn.0"].non_spec_state_indices_tensor.values
    for state, original in zip(states, before, strict=True):
        assert state.values == [value + (10 if index in indices else 0) for index, value in enumerate(original)]

    # Reordering and slot reuse must update only the newly scheduled rows.
    new_indices = Tensor([8, 1, 5] if mixed else [8, 1])
    new_indices.dtype = "int32"
    context.attn_metadata["gdn.0"].non_spec_state_indices_tensor = new_indices
    before = [state.values[:] for state in states]
    assert cache.run(context, (Tensor([2]),), {}, **ops)[0]
    for state, original in zip(states, before, strict=True):
        assert state.values == [
            value + (10 if index in new_indices.values else 0) for index, value in enumerate(original)
        ]
    ops["capture"].assert_called_once()


def test_fresh_to_continuation_does_not_specialize_on_megagdn_eligibility(setup):
    module, config, context, _ = setup
    add_live_states(context)
    chunk = context.attn_metadata["gdn.0"].non_spec_prefill_metadata.chunk
    chunk.fresh_prefill = True
    cache = module.FullPrefillGraphCache(config)
    ops = callbacks(context)
    cache.run(context, (Tensor([1]),), {}, **ops)
    chunk.fresh_prefill = False
    cache.run(context, (Tensor([2]),), {}, **ops)
    ops["capture"].assert_called_once()
    ops["runnable"].assert_called_once()
    owned = next(iter(cache.entries.values())).inputs[2]["gdn.0"]
    assert not owned.non_spec_prefill_metadata.chunk.fresh_prefill


@pytest.mark.parametrize("failure", ["warmup", "capture", "split"])
def test_stateful_capture_failure_rewinds_live_state(setup, failure):
    module, config, context, _ = setup
    states = add_live_states(context, mixed=True)
    before = [state.values[:] for state in states]
    ops = callbacks(context)

    def mutate_and_fail(*args):
        for state in states:
            for index in context.attn_metadata["gdn.0"].non_spec_state_indices_tensor.values:
                state.values[index] = -100
        if failure == "split":
            args[0].capture = SimpleNamespace(num_graphs=2, num_eager_breaks=1)
        else:
            raise RuntimeError("failed")

    ops["runnable" if failure == "warmup" else "capture"].side_effect = mutate_and_fail
    cache = module.FullPrefillGraphCache(config)
    with pytest.raises(RuntimeError):
        cache.run(context, (Tensor([1]),), {}, **ops)
    assert [state.values for state in states] == before
    assert not cache.entries
    assert context.cudagraph_runtime_mode == Mode.PIECEWISE
    assert context.full_prefill_graph_params is None
    ops["replay"].assert_not_called()


@pytest.mark.parametrize("fail", [False, True])
def test_lazy_capture_stream_orders_state_restore_and_is_reused(setup, monkeypatch, fail):
    module, config, context, _ = setup
    add_live_states(context, mixed=True)
    npu = module.torch.npu
    request_stream = npu.current_stream()
    capture_stream = SimpleNamespace(wait_stream=Mock())
    npu.Stream.return_value = capture_stream
    npu.Stream.side_effect = None
    events = []
    capture_stream.wait_stream.side_effect = lambda other: events.append(("capture waits", other))
    request_stream.wait_stream.side_effect = lambda other: events.append(("request waits", other))
    restore = module._restore_recurrent_states

    def restore_on_request(snapshots):
        assert npu.current_stream() is request_stream
        events.append("restore")
        restore(snapshots)

    monkeypatch.setattr(module, "_restore_recurrent_states", restore_on_request)
    cache = module.FullPrefillGraphCache(config)
    ops = callbacks(context)

    def capture(entry, args, kwargs):
        assert npu.current_stream() is capture_stream
        events.append("capture")
        if fail:
            raise RuntimeError("capture failed")
        entry.capture = SimpleNamespace(num_graphs=1, num_eager_breaks=0)

    ops["capture"].side_effect = capture
    if fail:
        with pytest.raises(RuntimeError, match="capture failed"):
            cache.run(context, (Tensor([1]),), {}, **ops)
        assert not cache.entries
        ops["replay"].assert_not_called()
    else:
        cache.run(context, (Tensor([1]),), {}, **ops)
        # Another input shape admits a second graph on the same stream.
        cache.run(context, (Tensor([1, 2]),), {}, **ops)
    expected = ["restore", ("capture waits", request_stream), "capture", ("request waits", capture_stream), "restore"]
    assert events == expected * (1 if fail else 2)
    assert npu.current_stream() is request_stream
    npu.Stream.assert_called_once_with(device=0)


def test_growing_kv_lengths_reuse_owned_attention_entry_and_update_after_replay(setup):
    module, config, context, sync = setup
    attention = AscendMetadata(Tensor([127, 256]), Tensor([127, 256]), [127, 256], [127, 256], Tensor([4, 9]))
    # CPU length values must not enter the signature when FIA updates them.
    attention.seq_lens_cpu.device = Device("cpu")
    context.attn_metadata["attention"] = attention
    ops = callbacks(context)
    events = []

    def capture(entry, args, kwargs):
        assert context.full_prefill_graph_params is entry.graph_params
        entry.graph_params.attn_params[256].append("handle")
        entry.capture = SimpleNamespace(num_graphs=1, num_eager_breaks=0)
        context.capturing = True

    ops["capture"].side_effect = capture
    ops["replay"].side_effect = lambda *args: events.append("replay")

    def update(stream, active_context, num_tokens, active_config):
        assert stream is cache.update_stream and active_config is config
        assert not active_context.capturing
        assert active_context.full_prefill_graph_params is entry.graph_params
        events.append(active_context.attn_metadata["attention"].seq_lens_list[:])

    cache = module.FullPrefillGraphCache(config)
    # On the first update the entry has already been admitted.
    module.AscendAttentionBackendImpl.update_graph_params.side_effect = lambda *args: events.append("first update")
    assert cache.run(context, (Tensor([1]),), {}, **ops)[0]
    entry = next(iter(cache.entries.values()))
    owned = entry.inputs[2]["attention"]
    pointers = owned.seq_lens, owned.block_tables, owned.seq_lens_list
    module.AscendAttentionBackendImpl.update_graph_params.side_effect = update
    attention.seq_lens = Tensor([200, 300])
    attention.seq_lens_cpu.values[:] = [200, 300]
    attention.seq_lens_list[:] = [200, 300]
    attention.block_tables = Tensor([8, 1])
    assert cache.run(context, (Tensor([2]),), {}, **ops)[0]
    assert events == ["replay", "first update", "replay", [200, 300]]
    assert all(a is b for a, b in zip(pointers, (owned.seq_lens, owned.block_tables, owned.seq_lens_list), strict=True))
    assert owned.block_tables.values == [8, 1]
    assert len(cache.entries) == 1
    ops["capture"].assert_called_once()
    assert context.full_prefill_graph_params is None


def test_same_token_count_different_layout_owns_separate_attention_handles(setup):
    module, config, context, _ = setup
    attention = AscendMetadata(Tensor([127, 129]), Tensor([127, 129]), [127, 129], [127, 256], Tensor([4, 9]))
    context.attn_metadata["attention"] = attention
    ops = callbacks(context)
    cache = module.FullPrefillGraphCache(config)
    cache.run(context, (Tensor([1]),), {}, **ops)
    first = next(iter(cache.entries.values()))
    attention.actual_seq_lengths_q[:] = [128, 256]
    context.attn_metadata["gdn.0"].non_spec_prefill_metadata.chunk.cu_seqlens_host = (0, 128, 256)
    cache.run(context, (Tensor([1]),), {}, **ops)
    second = list(cache.entries.values())[1]
    assert first.graph_params is not second.graph_params
    assert first.graph_params.attn_params[256] is not second.graph_params.attn_params[256]
    assert first.inputs[2]["attention"].block_tables is not second.inputs[2]["attention"].block_tables


def test_changing_per_layer_length_aliases_cannot_overwrite_another_layers_lengths(setup):
    module, config, context, _ = setup
    lengths = [127, 129]
    first = AscendMetadata(Tensor(lengths), Tensor(lengths), lengths, [127, 256], Tensor([4, 9]))
    second = AscendMetadata(Tensor(lengths), Tensor(lengths), lengths, [127, 256], Tensor([4, 9]))
    context.attn_metadata.update(attention1=first, attention2=second)
    ops = callbacks(context)
    cache = module.FullPrefillGraphCache(config)
    cache.run(context, (Tensor([1]),), {}, **ops)
    # The two layers now require different KV lengths, so their previously
    # shared captured list cannot be updated in place for both of them.
    second.seq_lens_list = [200, 300]
    cache.run(context, (Tensor([2]),), {}, **ops)
    assert len(cache.entries) == 2
    current = list(cache.entries.values())[-1].inputs[2]
    assert current["attention1"].seq_lens_list == lengths
    assert current["attention2"].seq_lens_list == [200, 300]


def test_capture_then_replay_refreshes_current_tokens_and_cache_slots(setup):
    module, config, context, sync = setup
    cache = module.FullPrefillGraphCache(config)
    ops = callbacks(context)
    live = context.attn_metadata
    assert cache.run(context, (Tensor([1, 2]),), {}, **ops) == (True, ([1, 2], [4, 9]))
    entry = next(iter(cache.entries.values()))
    assert entry.inputs[2] is not live
    assert entry.inputs[2]["gdn.0"] is entry.inputs[2]["gdn.1"]
    static_indices = entry.inputs[2]["gdn.0"].prefill_state_indices
    context.attn_metadata["gdn.0"].prefill_state_indices = Tensor([7, 3])
    assert cache.run(context, (Tensor([8, 6]),), {}, **ops) == (True, ([8, 6], [7, 3]))
    assert entry.inputs[2]["gdn.0"].prefill_state_indices is static_indices
    ops["capture"].assert_called_once()
    ops["runnable"].assert_called_once()
    assert ops["replay"].call_count == 2
    sync.assert_called_once()  # No synchronization on the steady replay path.
    assert context.attn_metadata is live
    assert context.cudagraph_runtime_mode == Mode.PIECEWISE
    assert not context.capturing and not context.full_prefill_graph


def test_different_packed_layout_never_reuses_captured_launch_dimensions(setup):
    module, config, context, _ = setup
    cache = module.FullPrefillGraphCache(config)
    ops = callbacks(context)
    cache.run(context, (Tensor([1, 2]),), {}, **ops)
    # Same bucket and sequence count, different per-sequence chunk boundaries.
    context.attn_metadata["gdn.0"].non_spec_prefill_metadata.chunk.cu_seqlens_host = (0, 128, 256)
    cache.run(context, (Tensor([1, 2]),), {}, **ops)
    assert ops["capture"].call_count == 2
    assert len(cache.entries) == 2


@pytest.mark.parametrize("count", [1, 2, 4, 8, 16, 32, 64])
def test_full_prefill_accepts_packed_sequence_counts(setup, count):
    module, config, context, _ = setup
    meta = context.attn_metadata["gdn.0"]
    meta.num_prefills = count
    boundaries = [0]
    for index in range(count):
        boundaries.append(boundaries[-1] + 1 + index % 4)
    meta.non_spec_prefill_metadata.chunk.cu_seqlens_host = tuple(boundaries)
    meta.prefill_state_indices = Tensor(range(count))
    ops = callbacks(context)
    handled, _ = module.FullPrefillGraphCache(config).run(context, (Tensor(range(boundaries[-1])),), {}, **ops)
    assert handled
    ops["capture"].assert_called_once()
    ops["replay"].assert_called_once()


def test_clear_releases_entries_only_after_replays_complete(setup):
    module, config, context, sync = setup
    cache = module.FullPrefillGraphCache(config)
    cache.run(context, (Tensor([1, 2]),), {}, **callbacks(context))
    sync.reset_mock()
    sync.side_effect = lambda: len(cache.entries) == 1 or pytest.fail("entries released before synchronization")
    cache.clear()
    sync.assert_called_once()
    assert cache.entries == {}


@pytest.mark.parametrize("limit", [-1, True, "8"])
def test_invalid_cache_limits_are_rejected(setup, limit):
    module, config, _, _ = setup
    config.additional_config["full_prefill_graph_max_entries"] = limit
    with pytest.raises(ValueError, match="non-negative integer"):
        module.FullPrefillGraphCache(config)


def test_zero_cache_limit_disables_full_prefill(setup):
    module, config, context, _ = setup
    config.additional_config["full_prefill_graph_max_entries"] = 0
    ops = callbacks(context)
    assert module.FullPrefillGraphCache(config).run(context, (Tensor([1]),), {}, **ops) == (False, None)
    for op in ops.values():
        op.assert_not_called()


def test_cache_capacity_uses_piecewise_for_new_layouts(setup):
    module, config, context, _ = setup
    config.additional_config["full_prefill_graph_max_entries"] = 1
    cache = module.FullPrefillGraphCache(config)
    ops = callbacks(context)
    cache.run(context, (Tensor([1, 2]),), {}, **ops)
    context.attn_metadata["gdn.0"].non_spec_prefill_metadata.chunk.cu_seqlens_host = (0, 128, 256)
    assert cache.run(context, (Tensor([1, 2]),), {}, **ops) == (False, None)
    assert context.cudagraph_runtime_mode == Mode.PIECEWISE
    ops["capture"].assert_called_once()


@pytest.mark.parametrize(
    "condition",
    [
        "stateful",
        "mixed",
        "speculative",
        "none",
        "full",
        "tp",
        "pp",
        "unknown",
        "all_mode",
        "all_mode_metadata",
        "draft",
        "c8",
    ],
)
def test_unsupported_batches_stay_on_existing_path(setup, condition):
    module, config, context, _ = setup
    meta = context.attn_metadata["gdn.0"]
    if condition == "stateful":
        meta.non_spec_prefill_metadata.chunk.fresh_prefill = False
    elif condition == "mixed":
        meta.num_decodes = 1
    elif condition == "speculative":
        meta.num_spec_decodes = 1
    elif condition in ("none", "full"):
        context.cudagraph_runtime_mode = Mode[condition.upper()]
    elif condition in ("tp", "pp"):
        setattr(config.parallel_config, "tensor_parallel_size" if condition == "tp" else "pipeline_parallel_size", 2)
    elif condition == "all_mode":
        config.cache_config.mamba_cache_mode = "all"
    elif condition == "all_mode_metadata":
        meta.all_state_indices_tensor = Tensor([4, 9])
    elif condition == "draft":
        context.is_draft_model = True
    elif condition == "c8":
        config.quant_config = SimpleNamespace(enable_c8_quant=True)
    else:
        context.attn_metadata["unknown"] = object()
    ops = callbacks(context)
    assert module.FullPrefillGraphCache(config).run(context, (Tensor([1]),), {}, **ops) == (False, None)
    for op in ops.values():
        op.assert_not_called()


@pytest.mark.parametrize("fail_warmup", [False, True])
def test_failed_capture_restores_context_and_is_not_cached(setup, fail_warmup):
    module, config, context, _ = setup
    cache = module.FullPrefillGraphCache(config)
    live = context.attn_metadata
    ops = callbacks(context)
    if fail_warmup:
        ops["runnable"].side_effect = RuntimeError("warmup failed")
        with pytest.raises(RuntimeError, match="warmup failed"):
            cache.run(context, (Tensor([1]),), {}, **ops)
        ops["capture"].assert_not_called()
    else:
        ops["capture"].side_effect = RuntimeError("capture failed")
        with pytest.raises(RuntimeError, match="capture failed"):
            cache.run(context, (Tensor([1]),), {}, **ops)
    assert cache.entries == {}
    assert context.attn_metadata is live
    assert context.cudagraph_runtime_mode == Mode.PIECEWISE
    assert not context.capturing and not context.full_prefill_graph
