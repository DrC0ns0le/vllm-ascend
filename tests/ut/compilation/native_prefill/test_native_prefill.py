"""CPU coverage for graph planning, padding, ownership and decode isolation."""

import ast
import importlib.util
import random
import sys
from contextlib import nullcontext
from enum import Enum
from functools import partial
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

ROOT = Path(__file__).resolve().parents[4]


class Mode(Enum):
    NONE = 0
    FULL = 1
    FULL_AND_PIECEWISE = 2
    FULL_DECODE_ONLY = 3
    PIECEWISE = 4

    def has_full_cudagraphs(self):
        return self not in (Mode.NONE, Mode.PIECEWISE)


class GDNAttentionMetadata(SimpleNamespace):
    pass


class AscendMetadata(SimpleNamespace):
    pass


def load_module(monkeypatch, name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def modules(monkeypatch):
    layout = load_module(
        monkeypatch,
        "vllm_ascend.compilation.native_prefill_layout",
        "vllm_ascend/compilation/native_prefill_layout.py",
    )
    dependencies = {
        "vllm.config": {"CUDAGraphMode": Mode},
        "vllm.forward_context": {"BatchDescriptor": SimpleNamespace},
        "vllm.logger": {"logger": Mock()},
        "vllm_ascend.ascend_forward_context": {"set_ascend_forward_context": lambda *args, **kwargs: nullcontext()},
        "vllm_ascend.attention.attention_v1": {"AscendAttentionState": SimpleNamespace(PrefillNoCache="fresh")},
        "vllm_ascend.ops.gdn_attn_builder": {
            "_build_non_spec_chunked_prefill_metadata": lambda builder, starts, device: SimpleNamespace(
                cu_seqlens_host=tuple(starts.tolist())
            )
        },
        "vllm_ascend.ops.rotary_embedding": {"update_cos_sin": Mock()},
    }
    for name, attributes in dependencies.items():
        module = ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
    runtime = load_module(monkeypatch, "native_prefill_test_runtime", "vllm_ascend/compilation/native_prefill_graph.py")
    return SimpleNamespace(layout=layout, runtime=runtime)


def config():
    return SimpleNamespace(
        additional_config={"native_prefill_graph": True},
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(model_type="qwen3_5_text"), enforce_eager=False, max_model_len=128
        ),
        compilation_config=SimpleNamespace(cudagraph_mode=Mode.FULL_DECODE_ONLY, cudagraph_capture_sizes=[1, 4, 8]),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=32, max_num_seqs=8),
        parallel_config=SimpleNamespace(tensor_parallel_size=1),
        cache_config=SimpleNamespace(mamba_cache_mode="none"),
        speculative_config=None,
        lora_config=None,
        kv_transfer_config=None,
        quant_config=None,
    )


def inputs(tokens=8, embedded=False):
    return {
        "input_ids": None if embedded else torch.arange(tokens),
        "inputs_embeds": torch.arange(tokens * 3).reshape(tokens, 3).float() if embedded else None,
        "positions": torch.arange(tokens),
        "intermediate_tensors": None,
    }


def context(lengths=(3, 2), totals=None, padded_boundary=False):
    boundaries = torch.tensor(lengths).cumsum(0).tolist()
    if padded_boundary:
        boundaries.append(boundaries[-1])
    gdn = GDNAttentionMetadata(non_spec_state_indices_tensor=torch.tensor([[5], [2]], dtype=torch.int32))
    attention = AscendMetadata(
        attn_state="fresh",
        num_actual_tokens=sum(lengths),
        actual_seq_lengths_q=boundaries,
        seq_lens_list=list(lengths if totals is None else totals),
        slot_mapping=torch.arange(sum(lengths), dtype=torch.int32) + 20,
        block_tables=torch.ones((len(lengths), 4), dtype=torch.int32),
    )
    return SimpleNamespace(attn_metadata={"gdn": gdn, "gdn_alias": gdn, "attention": attention}, capturing=False)


@pytest.mark.parametrize("seed", range(8))
def test_ragged_fresh_requests_round_up_to_bounded_registry(modules, seed):
    plan = modules.layout.NativePrefillPlan([1, 4, 16, 32], 256, 16, 64)
    rng = random.Random(seed)
    lengths = [rng.randint(1, 15) for _ in range(16)]
    boundaries = [0, *torch.tensor(lengths).cumsum(0).tolist()]
    batches = plan.batches(boundaries, lengths)
    requests = []
    for shape, items in batches:
        assert shape in plan.shapes
        assert shape.tokens <= plan.max_tokens
        assert len(items) <= shape.requests <= plan.max_requests
        for item in items:
            assert item.length == lengths[item.request] <= shape.width
            assert item.start == boundaries[item.request]
            requests.append(item.request)
    assert sorted(requests) == list(range(16))


@pytest.mark.parametrize("lengths,totals", [([1, 1], [20, 12]), ([3, 1], [3, 12]), ([3], [7])])
def test_decode_mixed_and_continuation_batches_are_not_admitted(modules, lengths, totals):
    plan = modules.layout.NativePrefillPlan([1, 4], 32, 8, 16)
    assert plan.batches([0, *torch.tensor(lengths).cumsum(0).tolist()], totals) is None


@pytest.mark.parametrize("counts", [[], [0], [-1], [True], [1.5], [9]])
def test_invalid_request_counts_fail(modules, counts):
    with pytest.raises(ValueError, match="request_counts"):
        modules.layout.NativePrefillPlan([4], 32, 8, 16, counts)


def test_budget_capacity_and_non_power_of_two_limits(modules):
    plan = modules.layout.NativePrefillPlan([8], 32, 5, 16)
    assert max(shape.requests for shape in plan.shapes) == 5
    assert plan.batches([0, 17], [17]) is None
    assert plan.batches([0, 16, 32, 48], [16, 16, 16]) is None
    assert plan.batches([0, 1, 2, 3, 4, 5, 6], [1] * 6) is None


def test_empty_standard_sizes_bucket_long_contexts_without_maximum_padding(modules):
    plan = modules.layout.NativePrefillPlan([], 16384, 4, 8192, [1, 4])
    for length, width in [(300, 512), (1025, 2048), (3000, 4096), (8192, 8192)]:
        batches = plan.batches([0, length], [length])
        assert len(batches) == 1
        assert batches[0][0].width == width
        assert batches[0][0].requests == 1
    assert plan.batches([0, 8193], [8193]) is None
    assert all(shape.tokens <= 16384 for shape in plan.shapes)


def test_explicit_context_sizes_are_independent_of_standard_capture(modules):
    cfg = config()
    cfg.model_config.max_model_len = 8192
    cfg.scheduler_config.max_num_batched_tokens = 8192
    cfg.additional_config.update(
        native_prefill_graph_max_context=8192,
        native_prefill_graph_context_sizes=[256, 512, 1024, 2048, 4096],
    )
    cache = modules.runtime.NativePrefillGraphCache(cfg)
    assert cache.plan.widths == (1, 256, 512, 1024, 2048, 4096, 8192)
    shape, _ = cache.plan.batches([0, 3000], [3000])[0]
    assert shape.width == 4096


@pytest.mark.parametrize("sizes", [[], [0], [-1], [True], [1.5], [8193], "1024"])
def test_invalid_native_context_sizes_fail(modules, sizes):
    with pytest.raises(ValueError, match="native_prefill_graph_context_sizes"):
        modules.layout.NativePrefillPlan([], 8192, 4, 8192, [1], sizes)


def test_padding_preserves_recurrent_state_and_real_outputs(modules):
    generator = torch.Generator().manual_seed(12)
    real_tokens, padded_tokens = 3, 8
    query = torch.randn((1, padded_tokens, 2, 4), generator=generator)
    key = torch.nn.functional.normalize(torch.randn(query.shape, generator=generator), dim=-1)
    value = torch.randn(query.shape, generator=generator)
    gate = -torch.rand((1, padded_tokens, 2), generator=generator)
    beta = torch.rand(gate.shape, generator=generator)
    valid = torch.arange(padded_tokens) < real_tokens
    padded = modules.layout.neutral_gdn_padding(query, key, value, gate, beta, valid)

    def recurrence(tensors, tokens):
        query, key, value, gate, beta = tensors
        state = torch.zeros(2, 4, 4)
        outputs = []
        for token in range(tokens):
            state = state * gate[0, token].exp()[:, None, None]
            residual = value[0, token] - torch.einsum("hk,hkv->hv", key[0, token], state)
            state = state + torch.einsum("hk,hv->hkv", key[0, token], residual * beta[0, token, :, None])
            outputs.append(torch.einsum("hk,hkv->hv", query[0, token], state))
        return torch.stack(outputs), state

    expected, expected_state = recurrence((query, key, value, gate, beta), real_tokens)
    actual, actual_state = recurrence(padded, padded_tokens)
    torch.testing.assert_close(actual[:real_tokens], expected)
    torch.testing.assert_close(actual_state, expected_state)
    assert torch.count_nonzero(actual[real_tokens:]) == 0


def test_conv_history_uses_last_real_tokens_and_zeroes_short_rows(modules):
    values = torch.arange(16).reshape(2, 4, 2)
    history = modules.layout.fresh_conv_history(values, torch.tensor([1, 3]), 3)
    assert history.tolist() == [[[0, 0], [0, 0], [0, 1]], [[8, 9], [10, 11], [12, 13]]]


@pytest.mark.parametrize("embedded", [False, True])
def test_entry_owns_inputs_and_masks_padding_without_mutating_source(modules, embedded):
    owner = modules.runtime.NativePrefillGraphCache(config())
    source = context()
    model_inputs = inputs(embedded=embedded)
    shape = modules.layout.PrefillGraphShape(4, 4)
    entry = modules.runtime.NativePrefillEntry(owner, shape, source, model_inputs)
    items = (modules.layout.PrefillRequest(0, 0, 3), modules.layout.PrefillRequest(1, 3, 2))
    output_sources, output_targets = entry.update(source, model_inputs, items)
    assert entry.metadata["gdn"] is entry.metadata["gdn_alias"]
    assert entry.metadata["gdn"].non_spec_state_indices_tensor.tolist() == [5, 2, -1, -1]
    assert entry.metadata["attention"].slot_mapping.tolist() == [20, 21, 22, -1, 23, 24, -1, -1] + [-1] * 8
    assert source.attn_metadata["gdn"].non_spec_state_indices_tensor.tolist() == [[5], [2]]
    assert source.attn_metadata["attention"].slot_mapping.tolist() == [20, 21, 22, 23, 24]
    assert output_sources.tolist() == [0, 1, 2, 4, 5]
    assert output_targets.tolist() == [0, 1, 2, 3, 4]
    model_inputs["positions"].fill_(99)
    assert entry.inputs["positions"].tolist()[:8] == [0, 1, 2, 2, 3, 4, 4, 4]
    assert entry.chunk.cu_seqlens_host == (0, 4, 8, 12, 16)
    assert entry.conv_indices.tolist() == [1, 2, 3, 4]
    assert entry.metadata["attention"].seq_lens_gpu.data_ptr() != entry.lengths.data_ptr()


def make_replay_cache(modules):
    owner = modules.runtime.NativePrefillGraphCache(config())
    source = context(padded_boundary=True)
    model_inputs = inputs()
    for shape in owner.plan.shapes:
        entry = modules.runtime.NativePrefillEntry(owner, shape, source, model_inputs)
        entry.output = torch.zeros((shape.tokens, 1))

        def replay(entry=entry):
            entry.output.copy_(entry.inputs["input_ids"][:, None])

        entry.graph = SimpleNamespace(replay=Mock(side_effect=replay))
        owner.entries[shape] = entry
    owner.ready = True
    return owner, source, model_inputs


def test_replay_scatter_restores_token_order_and_reuses_sealed_registry(modules):
    owner, source, model_inputs = make_replay_cache(modules)
    registry = tuple(owner.entries)
    for _ in range(3):
        output = owner.replay(source, model_inputs)
        assert output[:, 0].tolist() == [0, 1, 2, 3, 4, 0, 0, 0]
    assert tuple(owner.entries) == registry
    assert sum(entry.replays for entry in owner.entries.values()) == 3


def test_first_replay_reports_actual_and_padded_tokens_once(modules):
    owner, source, model_inputs = make_replay_cache(modules)
    for _ in range(3):
        owner.replay(source, model_inputs)
    modules.runtime.logger.info.assert_called_once_with(
        "Native prefill first replay: requests=%d captured_requests=%d width=%d "
        "actual_tokens=%d captured_tokens=%d padding_ratio=%.2f",
        2,
        2,
        4,
        5,
        8,
        8 / 5,
    )


def test_long_prefill_replay_preserves_every_token(modules):
    cfg = config()
    cfg.model_config.max_model_len = 8192
    cfg.scheduler_config.max_num_batched_tokens = 8192
    cfg.compilation_config.cudagraph_capture_sizes = []
    cfg.additional_config["native_prefill_graph_max_context"] = 8192
    owner = modules.runtime.NativePrefillGraphCache(cfg)
    source, model_inputs = context((3000,)), inputs(3000)
    shape, _ = owner.plan.batches([0, 3000], [3000])[0]
    entry = modules.runtime.NativePrefillEntry(owner, shape, source, model_inputs)
    entry.output = torch.zeros((shape.tokens, 1), dtype=torch.long)
    entry.graph = SimpleNamespace(
        replay=Mock(side_effect=lambda: entry.output.copy_(entry.inputs["input_ids"][:, None]))
    )
    owner.entries[shape] = entry
    owner.ready = True
    for offset in (0, 1):
        model_inputs["input_ids"] = torch.arange(3000) + offset
        output = owner.replay(source, model_inputs)
        torch.testing.assert_close(output[:, 0], model_inputs["input_ids"])
    assert entry.replays == 2
    assert shape.width == 4096


@pytest.mark.parametrize(
    "case,reason",
    [
        ("cached", "cached_context"),
        ("long", "context_capacity"),
        ("batch", "batch_capacity"),
        ("decode", "not_fresh_prefill"),
        ("inputs", "input_layout"),
        ("boundaries", "query_boundaries"),
        ("empty", "empty_batch"),
        ("missing", "missing_attention_metadata"),
        ("gdn_only", "missing_ascend_metadata"),
    ],
)
def test_fallback_reports_reason_once_without_mutation(modules, case, reason):
    owner, source, model_inputs = make_replay_cache(modules)
    attention = source.attn_metadata["attention"]
    if case == "cached":
        attention.seq_lens_list = [10, 9]
    elif case == "long":
        source = context((33,))
    elif case == "batch":
        source = context((20, 20))
    elif case == "decode":
        attention.attn_state = "decode"
    elif case == "inputs":
        model_inputs["extra"] = torch.zeros(1)
    elif case == "boundaries":
        attention.actual_seq_lengths_q = [3, 6]
    elif case == "empty":
        attention.num_actual_tokens = 0
    elif case == "missing":
        source.attn_metadata = None
    elif case == "gdn_only":
        del source.attn_metadata["attention"]
    for entry in owner.entries.values():
        entry.update = Mock(side_effect=AssertionError("Fallback must not mutate inputs"))
    for _ in range(3):
        assert owner.replay(source, model_inputs) is None
    assert all(entry.replays == 0 for entry in owner.entries.values())
    modules.runtime.logger.info.assert_called_once()
    assert modules.runtime.logger.info.call_args.args[1] == reason


@pytest.mark.parametrize("totals", [(10, 9), (3, 9), (8, 2)])
def test_fallback_does_not_update_or_replay_any_entry(modules, totals):
    owner, source, model_inputs = make_replay_cache(modules)
    source.attn_metadata["attention"].seq_lens_list = list(totals)
    for entry in owner.entries.values():
        entry.update = Mock(side_effect=AssertionError("Fallback must not mutate graph inputs"))
    assert owner.replay(source, model_inputs) is None
    assert all(entry.replays == 0 for entry in owner.entries.values())


def test_unsupported_inputs_fall_back_before_state_changes(modules):
    owner, source, model_inputs = make_replay_cache(modules)
    model_inputs["inputs_embeds"] = torch.zeros(8, 4)
    assert owner.replay(source, model_inputs) is None
    assert all(entry.replays == 0 for entry in owner.entries.values())


def test_decode_metadata_returns_without_even_planning_prefill(modules):
    owner, source, model_inputs = make_replay_cache(modules)
    source.attn_metadata["attention"].attn_state = "decode"
    owner.plan.batches = Mock(side_effect=AssertionError("Decode must retain the existing path"))
    assert owner.replay(source, model_inputs) is None


def test_incomplete_registry_fails_before_replay(modules):
    owner, source, model_inputs = make_replay_cache(modules)
    owner.entries.clear()
    with pytest.raises(RuntimeError, match="incomplete"):
        owner.replay(source, model_inputs)


def test_configuration_is_opt_in_and_rejects_speculation(modules):
    cfg = config()
    cfg.additional_config = {}
    assert not modules.runtime.NativePrefillGraphCache(cfg).enabled
    cfg.additional_config = {"native_prefill_graph": "true"}
    with pytest.raises(ValueError, match="boolean"):
        modules.runtime.NativePrefillGraphCache(cfg)
    cfg.additional_config = {"native_prefill_graph": True}
    cfg.speculative_config = SimpleNamespace(num_speculative_tokens=3)
    with pytest.raises(ValueError, match="no speculation"):
        modules.runtime.NativePrefillGraphCache(cfg)


def test_warmup_seals_prefill_only_registry_without_touching_live_slots(modules, monkeypatch):
    owner = modules.runtime.NativePrefillGraphCache(config())
    source = context()
    stream = Mock()
    graph = Mock()
    npu = SimpleNamespace(
        get_device_name=lambda: "Ascend910B",
        Stream=lambda: stream,
        current_stream=lambda: stream,
        stream=lambda stream: nullcontext(),
        NPUGraph=lambda: graph,
        graph=lambda *args, **kwargs: nullcontext(),
    )
    monkeypatch.setattr(torch, "npu", npu, raising=False)
    output = owner.warmup(source, inputs(), lambda **kwargs: kwargs["input_ids"][:, None].float())
    assert owner.ready
    assert len(owner.entries) == len(owner.plan.shapes)
    assert output.shape[1] == 1
    for entry in owner.entries.values():
        assert torch.all(entry.metadata["gdn"].non_spec_state_indices_tensor == -1)
        assert torch.all(entry.metadata["attention"].slot_mapping == -1)
        assert entry.metadata["gdn"].num_decodes == 0
    with pytest.raises(RuntimeError, match="once"):
        owner.warmup(source, inputs(), Mock())


def load_runner_method(name, namespace):
    path = ROOT / "vllm_ascend/worker/model_runner_v1.py"
    tree = ast.parse(path.read_text())
    runner = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "NPUModelRunner")
    method = next(node for node in runner.body if isinstance(node, ast.FunctionDef) and node.name == name)
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias("annotations")], level=0), method], type_ignores=[]
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[name]


@pytest.mark.parametrize("standard_bytes", [0, 10])
def test_runner_keeps_standard_capture_and_adds_prefill_afterward(standard_bytes):
    events = []
    native = SimpleNamespace(enabled=True, warming=False, ready=False)
    runner = SimpleNamespace(native_prefill_graph=native, encoder_cudagraph_manager=None)

    def dummy(*args, **kwargs):
        assert native.warming
        assert kwargs == {"force_attention": True, "cudagraph_runtime_mode": Mode.NONE}
        events.append("prefill")
        native.ready = True

    runner._dummy_run = dummy
    namespace = {
        "_get_gpu_model_runner_module_name": lambda runner: "test",
        "_torch_cuda_wrapper": nullcontext,
        "_replace_gpu_model_runner_function_wrapper": lambda name: nullcontext(),
        "GPUModelRunner": SimpleNamespace(capture_model=lambda runner: events.append("standard") or standard_bytes),
        "torch": SimpleNamespace(npu=SimpleNamespace(mem_get_info=Mock(side_effect=[(100, 200), (80, 200)]))),
        "CUDAGraphMode": Mode,
    }
    assert load_runner_method("capture_model", namespace)(runner) == standard_bytes + 20
    assert events == ["standard", "prefill"]
    assert not native.warming
    assert not runner._engram_capture_active


@pytest.mark.parametrize("native_result", [None, "prefill-output"])
@pytest.mark.parametrize("enable_enpu", [False, True])
def test_runner_keeps_decode_updates_and_retires_metadata_submission(native_result, enable_enpu):
    events = []
    executor = SimpleNamespace(submission_in_flight=True, release=lambda: events.append("release"))
    forward_context = SimpleNamespace(cudagraph_runtime_mode=Mode.FULL, device_metadata_executor=executor)
    native = SimpleNamespace(enabled=True, warming=False, replay=lambda *args: native_result)

    class Model:
        def __call__(self, **kwargs):
            events.append("standard-model")
            return "standard-output"

    runner = SimpleNamespace(
        model=Model(),
        native_prefill_graph=native,
        enable_enpu=enable_enpu,
        _update_full_graph_params_if_needed=lambda *args: events.append("update"),
    )
    namespace = {
        "get_forward_context": lambda: forward_context,
        "CUDAGraphMode": Mode,
        "partial": partial,
    }
    output = load_runner_method("_model_forward", namespace)(runner, 8, **inputs())
    if native_result is None:
        assert output == "standard-output"
        expected = ["update", "standard-model"] if enable_enpu else ["standard-model", "update"]
        assert events == [*expected, "release"]
    else:
        assert output == native_result
        assert events == ["release"]


@pytest.mark.parametrize("use_fla", [False, True])
def test_gdn_native_prefill_uses_zero_initial_state_and_masked_cache_writes(modules, use_fla):
    shape = modules.layout.PrefillGraphShape(4, 4)
    lengths = torch.tensor([3, 2, 0, 0])
    valid = (torch.arange(4)[None, :] < lengths[:, None]).reshape(-1)
    layout = SimpleNamespace(
        shape=shape,
        lengths=lengths,
        valid=valid,
        query_start=torch.arange(5, dtype=torch.int32) * 4,
        conv_indices=torch.arange(1, 5, dtype=torch.int32),
        has_initial_state=torch.zeros(4, dtype=torch.bool),
        chunk=object(),
    )
    slots = torch.tensor([5, 2, -1, -1], dtype=torch.int32)
    conv_backing = torch.full((7, 25), float("nan"))
    conv_cache = conv_backing[:, :21].view(7, 3, 7)
    state_backing = torch.full((7, 10), float("nan"))
    state_cache = state_backing[:, :6].view(7, 1, 3, 2)
    conv_layer = SimpleNamespace(weight=torch.ones(7, 1, 4), bias=None)
    packed_weights = torch.ones(4, 7)
    mixed_qkv = torch.arange(16 * 7).reshape(16, 7).float()
    final_state = torch.arange(24).reshape(4, 1, 3, 2).float()

    def conv(inputs, weights, bias, **kwargs):
        assert weights is packed_weights
        assert kwargs["conv_states"].shape[0] == 5
        assert kwargs["conv_states"].data_ptr() != conv_cache.data_ptr()
        assert kwargs["cache_indices"].tolist() == [1, 2, 3, 4]
        assert not kwargs["has_initial_state"].any()
        return inputs

    def split(inputs):
        query, key, value = inputs.split((2, 2, 3), dim=-1)
        return query.view(1, 16, 1, 2), key.view(1, 16, 1, 2), value.view(1, 16, 1, 3)

    def prefill(**kwargs):
        assert torch.count_nonzero(kwargs["initial_state"]) == 0
        for name in ("q", "k", "v", "g", "beta"):
            assert torch.count_nonzero(kwargs[name][:, ~valid]) == 0
        assert kwargs["prebuilt_meta"] is layout.chunk
        return kwargs["v"], final_state if use_fla else final_state.transpose(-1, -2)

    def writeback(source, cache, indices):
        for row, slot in enumerate(indices.tolist()):
            if slot >= 0:
                cache[slot].copy_(source[row])

    layer = SimpleNamespace(
        kv_cache=(conv_cache, state_cache),
        activation=True,
        rearrange_mixed_qkv=split,
        A_log=torch.zeros(1),
        dt_bias=torch.zeros(1),
    )
    namespace = {
        "torch": torch,
        "_get_base_conv1d": lambda layer: conv_layer,
        "_get_packed_conv_weights": lambda layer: packed_weights,
        "fresh_conv_history": modules.layout.fresh_conv_history,
        "neutral_gdn_padding": modules.layout.neutral_gdn_padding,
        "causal_conv1d_fn": conv,
        "write_fresh_states": writeback,
        "PAD_SLOT_ID": -1,
        "DeviceOperator": SimpleNamespace(
            fused_gdn_gating=lambda *args: (torch.full((1, 16, 1), -0.2), torch.full((1, 16, 1), 0.5)),
            fla_gdn_prefill=prefill,
        ),
        "get_current_hardware_profile": lambda: SimpleNamespace(supports=lambda capability: use_fla),
        "HardwareCapability": SimpleNamespace(FLA_GDN_PREFILL="fla"),
        "_get_fla_gdn_prefill_op": lambda: object(),
        "chunk_gated_delta_rule": prefill,
    }
    path = ROOT / "vllm_ascend/ops/gdn.py"
    tree = ast.parse(path.read_text())
    attention = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "AscendGatedDeltaNetAttention"
    )
    method = next(
        node for node in attention.body if isinstance(node, ast.FunctionDef) and node.name == "_forward_native_prefill"
    )
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    output = torch.empty(16, 1, 3)
    namespace["_forward_native_prefill"](
        layer,
        mixed_qkv,
        None,
        None,
        output,
        SimpleNamespace(native_prefill=layout, non_spec_state_indices_tensor=slots),
    )
    torch.testing.assert_close(state_cache[5], final_state[0])
    torch.testing.assert_close(state_cache[2], final_state[1])
    assert state_cache[[0, 1, 3, 4, 6]].isnan().all()
    assert state_backing[:, 6:].isnan().all()
    assert conv_backing[:, 21:].isnan().all()
    assert conv_cache[[0, 1, 3, 4, 6]].isnan().all()
    expected_history = modules.layout.fresh_conv_history(mixed_qkv.view(4, 4, 7), lengths, 3)
    torch.testing.assert_close(conv_cache[5], expected_history[0])
    torch.testing.assert_close(conv_cache[2], expected_history[1])
    assert torch.count_nonzero(output[~valid]) == 0


@pytest.mark.parametrize("mode", [Mode.NONE, Mode.PIECEWISE, Mode.FULL_DECODE_ONLY])
def test_clef_pooling_has_independent_backbone_registry(modules, mode):
    cfg = config()
    cfg.model_config.architecture = "ClefFlashForJointSchema"
    cfg.model_config.runner_type = "pooling"
    cfg.compilation_config.cudagraph_mode = mode
    # NONE can clear the standard dispatcher's capture sizes during config init.
    cfg.compilation_config.cudagraph_capture_sizes = []
    owner = modules.runtime.NativePrefillGraphCache(cfg)
    assert owner.enabled and owner.plan.shapes
    assert owner.plan.widths == (1, 2, 4, 8, 16, 32)
    assert cfg.compilation_config.cudagraph_mode == mode


@pytest.mark.parametrize(
    "architecture,runner,eager,model_type",
    [
        ("Qwen3_5ForConditionalGeneration", "generate", False, "qwen3_5_text"),
        ("OtherPoolingModel", "pooling", False, "qwen3_5_text"),
        ("ClefFlashForJointSchema", "generate", False, "qwen3_5_text"),
        ("ClefFlashForJointSchema", "pooling", True, "qwen3_5_text"),
        ("ClefFlashForJointSchema", "pooling", False, "qwen3_5_moe_text"),
    ],
)
def test_clef_exception_does_not_relax_other_graph_guards(modules, architecture, runner, eager, model_type):
    cfg = config()
    cfg.compilation_config.cudagraph_mode = Mode.NONE
    cfg.model_config.architecture = architecture
    cfg.model_config.runner_type = runner
    cfg.model_config.enforce_eager = eager
    cfg.model_config.hf_text_config.model_type = model_type
    with pytest.raises(ValueError, match="Native prefill graphs require"):
        modules.runtime.NativePrefillGraphCache(cfg)


def test_clef_forward_capture_excludes_pooler_and_preserves_all_rows(modules, monkeypatch):
    cfg = config()
    cfg.model_config.architecture = "ClefFlashForJointSchema"
    cfg.model_config.runner_type = "pooling"
    cfg.compilation_config.cudagraph_mode = Mode.NONE
    cfg.compilation_config.cudagraph_capture_sizes = []
    owner = modules.runtime.NativePrefillGraphCache(cfg)
    stream = Mock()
    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(
            get_device_name=lambda: "Ascend910B",
            Stream=lambda: stream,
            current_stream=lambda: stream,
            stream=lambda stream: nullcontext(),
            NPUGraph=Mock,
            graph=lambda *a, **k: nullcontext(),
        ),
        raising=False,
    )

    class Backbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.pooler = Mock(side_effect=AssertionError("Pooler must not run inside backbone capture"))

        def forward(self, input_ids, **kwargs):
            return torch.stack([input_ids, input_ids + 100], dim=-1).float()

    model = Backbone()
    runner = SimpleNamespace(model=model, native_prefill_graph=owner, enable_enpu=False)
    source = context()
    source.cudagraph_runtime_mode = Mode.NONE
    namespace = {
        "get_forward_context": lambda: source,
        "CUDAGraphMode": Mode,
        "partial": partial,
        "ACLGraphWrapper": type("Wrapper", (), {}),
        "BreakableACLGraphWrapper": type("Breakable", (), {}),
    }
    forward = load_runner_method("_model_forward", namespace)
    owner.warming = True
    forward(runner, 8, **inputs())
    owner.warming = False
    assert owner.ready
    # Simulate device graph replay on CPU, with the same owned input/output buffers.
    for entry in owner.entries.values():
        entry.graph.replay.side_effect = lambda entry=entry: entry.output.copy_(entry.execute(model))
    actual = forward(runner, 8, **inputs())
    torch.testing.assert_close(actual[:5], model(**inputs())[:5])
    assert actual.shape == (8, 2)  # All valid tokens reach pooling, not just last tokens.
    assert sum(entry.replays for entry in owner.entries.values()) > 0
    model.pooler.assert_not_called()
