# SPDX-License-Identifier: Apache-2.0
"""Native FULL planning, numerical padding, metadata ownership and replay tests.

CPU numerical references do not validate CANN compilation or ACLGraph replay;
that boundary is covered by the real-model NPU integration test.
"""

import ast
import copy
import importlib.util
import random
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from math import prod
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

ROOT = Path(__file__).resolve().parents[4]


def load_layout():
    spec = importlib.util.spec_from_file_location(
        "native_layout_tested", ROOT / "vllm_ascend/compilation/native_full_layout.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


layout = load_layout()


class Mode(Enum):
    NONE = 0
    FULL = 1
    FULL_AND_PIECEWISE = 2
    PIECEWISE = 3


class GDNAttentionMetadata(SimpleNamespace):
    pass


class AscendMetadata(SimpleNamespace):
    pass


def load_runtime():
    workspace_path = ROOT / "vllm_ascend/ops/pto_chunk_gdn/workspace.py"
    workspace_tree = ast.parse(workspace_path.read_text())
    workspace_tree.body = [n for n in workspace_tree.body if not isinstance(n, (ast.Import, ast.ImportFrom))]
    workspace_namespace = dict(__name__=__name__, torch=torch, dataclass=dataclass, prod=prod, CHUNK_SIZE=128)
    exec(compile(workspace_tree, str(workspace_path), "exec"), workspace_namespace)
    path = ROOT / "vllm_ascend/compilation/native_full_graph.py"
    tree = ast.parse(path.read_text())
    tree.body = [node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))]
    namespace = dict(
        torch=torch,
        copy=copy,
        contextmanager=contextmanager,
        SimpleNamespace=SimpleNamespace,
        CUDAGraphMode=Mode,
        BatchDescriptor=lambda **kwargs: SimpleNamespace(**kwargs),
        AscendAttentionState=SimpleNamespace(PrefillNoCache="fresh"),
        NativeGraphPlan=layout.NativeGraphPlan,
        DEFAULT_NATIVE_FULL_MAX_CONTEXT=layout.DEFAULT_NATIVE_FULL_MAX_CONTEXT,
        cached_attention_mask=layout.cached_attention_mask,
        padded_token_indices=layout.padded_token_indices,
        GDNCausalConv1dMetadata=lambda *args: SimpleNamespace(
            query_start_loc=args[0], cache_indices=args[1], initial_state_mode=args[2]
        ),
        GDNDecodeMetadata=lambda conv, lengths: SimpleNamespace(causal_conv1d=conv, actual_seq_lengths=lengths),
        GDNPrefillMetadata=lambda conv, chunk: SimpleNamespace(causal_conv1d=conv, chunk=chunk),
        _build_non_spec_chunked_prefill_metadata=lambda builder, cu, device: SimpleNamespace(
            cu_seqlens_host=tuple(cu.tolist()),
            chunk_indices_chunk64=torch.zeros((len(cu) - 1, 2), dtype=torch.int32),
            chunk_indices_large_block=torch.zeros((len(cu) - 1, 2), dtype=torch.int32),
            block_indices_cumsum=torch.zeros((len(cu) - 1, 2), dtype=torch.int32),
        ),
        logger=Mock(),
        CHUNK_SIZE=128,
        MegaGDNGraphWorkspace=workspace_namespace["MegaGDNGraphWorkspace"],
    )
    exec(compile(tree, str(path), "exec"), namespace)
    return SimpleNamespace(**namespace)


def config():
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(model_type="qwen3_5_text"), enforce_eager=False, max_model_len=768
        ),
        compilation_config=SimpleNamespace(
            cudagraph_mode=Mode.FULL_AND_PIECEWISE, cudagraph_capture_sizes=[1, 64, 128, 196, 256]
        ),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=256, max_num_seqs=64),
        additional_config={},
        parallel_config=SimpleNamespace(tensor_parallel_size=1, pipeline_parallel_size=1, data_parallel_size=1),
        cache_config=SimpleNamespace(mamba_cache_mode="none"),
        speculative_config=None,
        kv_transfer_config=None,
        lora_config=None,
        quant_config=None,
    )


@pytest.mark.parametrize("seed", range(8))
def test_every_mixed_shape_routes_to_startup_registry_and_rounds_up(seed):
    rng = random.Random(seed)
    plan = layout.NativeGraphPlan([1, 64, 128, 196, 256, 384, 512, 768], 4096, 64, 768)
    sizes = [rng.randint(1, 704) for _ in range(64)]
    sizes[::3] = [1] * len(sizes[::3])
    contexts = [rng.randint(1, 64) if size == 1 else 0 for size in sizes]
    boundaries = [0]
    for size in sizes:
        boundaries.append(boundaries[-1] + size)
    batches = plan.batches(boundaries, [size + ctx for size, ctx in zip(sizes, contexts)])
    seen = [0] * 64
    for shape, items in batches:
        assert shape in plan.shapes
        assert shape.tokens <= 4096
        assert shape.requests >= len(items)
        assert len({item.request for item in items}) == len(items)
        for item in items:
            assert item.start == boundaries[item.request] + seen[item.request]
            assert item.context == contexts[item.request] + seen[item.request]
            assert shape.fresh == (item.context == 0)
            assert 0 < item.length <= shape.width
            seen[item.request] += item.length
    assert seen == sizes
    assert len(plan.shapes) < 100


def test_counts_are_live_not_capacity_sized():
    plan = layout.NativeGraphPlan([1, 64, 128], 4096, 64, 768)
    batches = plan.batches([0, 61], [61])
    assert batches[0][0] == layout.NativeGraphShape(1, 64, True)
    batches = plan.batches([0, 61, 122, 183, 244, 305], [61] * 5)
    assert [shape.requests for shape, _ in batches] == [8]


@pytest.mark.parametrize("backend", ["ascendc", "megagdn"])
def test_large_fresh_batch_uses_one_captured_rectangle(backend):
    cfg = config()
    cfg.scheduler_config.max_num_batched_tokens = 49152
    cfg.additional_config["native_full_graph_backend"] = backend
    owner = load_runtime().NativeFullGraphCache(cfg)
    assert owner.plan.max_tokens == cfg.scheduler_config.max_num_batched_tokens
    batches = owner.plan.batches([i * 768 for i in range(33)], [768] * 32)
    assert len(batches) == 1
    shape, items = batches[0]
    assert shape == layout.NativeGraphShape(32, 768, True)
    assert shape in owner.plan.shapes
    assert [item.request for item in items] == list(range(32))
    assert [item.start for item in items] == [i * 768 for i in range(32)]
    assert all(item.length == 768 and item.context == 0 for item in items)


@pytest.mark.parametrize("boundaries,lengths", [([0, 0], [1]), ([0, 3], [2]), ([1, 3], [3])])
def test_invalid_shapes_fail_before_execution(boundaries, lengths):
    with pytest.raises(ValueError):
        layout.NativeGraphPlan([64], 256, 64, 768).batches(boundaries, lengths)


def recurrence(q, k, v, g, beta, state):
    outputs = []
    for index in range(q.shape[1]):
        state = state * g[:, index].exp()[..., None, None]
        prediction = torch.einsum("bhk,bhkv->bhv", k[:, index], state)
        state = state + torch.einsum("bhk,bhv->bhkv", k[:, index], (v[:, index] - prediction) * beta[:, index, :, None])
        outputs.append(torch.einsum("bhk,bhkv->bhv", q[:, index], state))
    return torch.stack(outputs, 1), state


@pytest.mark.parametrize("length", [1, 2, 3, 61, 64, 65, 127])
def test_padding_preserves_real_gdn_outputs_and_final_state_with_stale_nans(length):
    torch.manual_seed(length)
    width = 128
    q, k = [torch.randn(1, width, 2, 4, dtype=torch.float64) * 0.1 for _ in range(2)]
    v = torch.randn(1, width, 2, 3, dtype=torch.float64)
    g = -torch.rand(1, width, 2, dtype=torch.float64)
    beta = torch.rand_like(g)
    state = torch.randn(1, 2, 4, 3, dtype=torch.float64)
    expected, expected_state = recurrence(
        q[:, :length], k[:, :length], v[:, :length], g[:, :length], beta[:, :length], state
    )
    for tensor in (q, k, v, g, beta):
        tensor[:, length:] = float("nan")
    neutral = layout.neutral_gdn_padding(q, k, v, g, beta, torch.arange(width) < length)
    output, final = recurrence(*neutral, state)
    torch.testing.assert_close(output[:, :length], expected, rtol=0, atol=0)
    torch.testing.assert_close(final, expected_state, rtol=0, atol=0)
    assert output.isfinite().all()


def test_cached_attention_mask_matches_unpadded_causal_attention():
    torch.manual_seed(5)
    width, capacity = 8, 32
    lengths, contexts = torch.tensor([1, 3, 7]), torch.tensor([17, 4, 0])
    mask = layout.cached_attention_mask(lengths, contexts, width, capacity)
    q, k, v = torch.randn(3, 2, width, 4), torch.randn(3, 2, capacity, 4), torch.randn(3, 2, capacity, 4)
    actual = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=~mask)
    for row, (length, ctx) in enumerate(zip(lengths.tolist(), contexts.tolist())):
        total = length + ctx
        allowed = torch.arange(total)[None, :] <= ctx + torch.arange(length)[:, None]
        expected = torch.nn.functional.scaled_dot_product_attention(
            q[row, :, :length], k[row, :, :total], v[row, :, :total], attn_mask=allowed
        )
        torch.testing.assert_close(actual[row, :, :length], expected)
    assert actual.isfinite().all()


def sources():
    gdn = GDNAttentionMetadata(non_spec_state_indices_tensor=torch.tensor([7, 3, 5], dtype=torch.int32))
    attention = AscendMetadata(
        block_tables=torch.tensor([[4, 5, 0], [6, 7, 0], [8, 9, 0]], dtype=torch.int32),
        slot_mapping=torch.arange(10, dtype=torch.int32) + 100,
        actual_seq_lengths_q=[3, 4, 10],
        seq_lens_list=[3, 9, 6],
        num_actual_tokens=10,
    )
    return SimpleNamespace(
        attn_metadata={"gdn0": gdn, "gdn1": gdn, "attn": attention},
        cudagraph_runtime_mode=Mode.PIECEWISE,
        capturing=False,
        batch_descriptor=SimpleNamespace(num_tokens=10),
        num_tokens=10,
    )


def inputs(tokens=10):
    return dict(
        input_ids=torch.arange(tokens),
        positions=torch.arange(tokens).expand(3, -1),
        inputs_embeds=None,
        intermediate_tensors=None,
    )


def test_entry_metadata_is_owned_and_refreshes_slots_flags_and_input_positions():
    runtime = load_runtime()
    owner = runtime.NativeFullGraphCache(config())
    context, model_inputs = sources(), inputs()
    shape = layout.NativeGraphShape(2, 8, True)
    entry = runtime.NativeFullEntry(owner, shape, context, model_inputs)
    items = (layout.RequestSlice(2, 4, 6, 0), layout.RequestSlice(0, 0, 3, 0))
    src, dst = entry.update(context, model_inputs, items)
    entry.execute(lambda **kwargs: kwargs["input_ids"])
    assert entry.metadata["gdn0"] is entry.metadata["gdn1"]
    assert entry.metadata["gdn0"] is not context.attn_metadata["gdn0"]
    assert entry.metadata["gdn0"].non_spec_state_indices_tensor.tolist() == [5, 7]
    assert entry.metadata["attn"].slot_mapping.tolist() == [
        104,
        105,
        106,
        107,
        108,
        109,
        -1,
        -1,
        100,
        101,
        102,
        -1,
        -1,
        -1,
        -1,
        -1,
    ]
    assert src.tolist() == [0, 1, 2, 3, 4, 5, 8, 9, 10]
    assert dst.tolist() == [4, 5, 6, 7, 8, 9, 0, 1, 2]
    addresses = [entry.input_ids.data_ptr(), entry.positions.data_ptr(), entry.lengths.data_ptr()]
    context.attn_metadata["gdn0"].non_spec_state_indices_tensor.add_(1)
    entry.update(context, model_inputs, items)
    assert addresses == [entry.input_ids.data_ptr(), entry.positions.data_ptr(), entry.lengths.data_ptr()]
    assert entry.metadata["gdn0"].non_spec_state_indices_tensor.tolist() == [6, 8]


def test_native_decode_lengths_keep_leading_zero_and_per_sequence_lengths():
    runtime = load_runtime()
    entry = runtime.NativeFullEntry(
        runtime.NativeFullGraphCache(config()), layout.NativeGraphShape(4, 1, False), sources(), inputs()
    )
    metadata = entry.metadata["gdn0"]
    assert metadata.num_prefills == 0
    assert metadata.num_decodes == 4
    assert metadata.non_spec_decode_metadata.actual_seq_lengths.tolist() == [0, 1, 1, 1, 1]


def test_context_restores_capture_and_geometry_on_failure():
    runtime = load_runtime()
    context = sources()
    old = vars(context).copy()
    with pytest.raises(RuntimeError), runtime.native_context(context, {}, 64):
        assert context.num_tokens == 64
        assert context.cudagraph_runtime_mode == Mode.FULL
        raise RuntimeError("capture failure")
    assert vars(context) == old


def test_serving_replays_sealed_entries_without_recapture_and_scatter_is_ordered():
    runtime = load_runtime()
    owner = runtime.NativeFullGraphCache(config())
    context, model_inputs = sources(), inputs()
    expected_batches = owner.plan.batches([0, 3, 4, 10], [3, 9, 6])
    for shape, _ in expected_batches:
        entry = runtime.NativeFullEntry(owner, shape, context, model_inputs)
        entry.output = torch.empty(shape.tokens, 1)
        # This tests dispatch/ownership, not hardware graph behavior.
        entry.graph = SimpleNamespace(replay=lambda entry=entry: entry.output.copy_(entry.input_ids[:, None] * 7))
        owner.entries[shape] = entry
    owner.ready = True
    keys = tuple(owner.entries)
    for offset in (0, 10):
        model_inputs["input_ids"] = torch.arange(10) + offset
        result = owner.run(context, model_inputs)
        torch.testing.assert_close(result[:, 0], model_inputs["input_ids"].float() * 7)
        assert tuple(owner.entries) == keys
    assert all(entry.replays == 2 for entry in owner.entries.values())
    owner.entries.pop(next(iter(owner.entries)))
    before = [entry.replays for entry in owner.entries.values()]
    with pytest.raises(RuntimeError, match="registry is incomplete"):
        owner.run(context, model_inputs)
    assert before == [entry.replays for entry in owner.entries.values()]


@pytest.mark.parametrize(
    "field,value", [("speculative_config", object()), ("lora_config", object()), ("kv_transfer_config", object())]
)
def test_unsupported_configuration_fails_before_warmup(field, value):
    runtime = load_runtime()
    cfg = config()
    setattr(cfg, field, value)
    with pytest.raises(ValueError, match="single-rank"):
        runtime.NativeFullGraphCache(cfg)


def test_native_cache_gather_never_reads_invalid_pages_or_propagates_padding_nans():
    path = ROOT / "vllm_ascend/ops/native_full_attention.py"
    tree = ast.parse(path.read_text())
    node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "gather_native_cache")
    namespace = dict(torch=torch)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    cache = torch.full((4, 4, 2, 3), float("nan"))
    cache[1] = torch.arange(24).reshape(4, 2, 3)
    cache[2] = torch.arange(24, 48).reshape(4, 2, 3)
    blocks = torch.tensor([[1, 2, 99999], [2, -100, 99999]])
    result = namespace["gather_native_cache"](cache, blocks, torch.tensor([5, 2]), 12)
    assert result.isfinite().all()
    torch.testing.assert_close(result[0, :, :5], torch.cat((cache[1], cache[2, :1])).transpose(0, 1))
    torch.testing.assert_close(result[1, :, :2], cache[2, :2].transpose(0, 1))
    assert result[0, :, 5:].eq(0).all() and result[1, :, 2:].eq(0).all()


@pytest.mark.parametrize("backend", ["ascendc", "megagdn"])
def test_startup_captures_all_geometries_once_without_touching_live_state(monkeypatch, backend):
    runtime = load_runtime()
    cfg = config()
    cfg.scheduler_config.max_num_seqs = 2
    cfg.scheduler_config.max_num_batched_tokens = 8
    cfg.compilation_config.cudagraph_capture_sizes = [1, 4, 8]
    cfg.additional_config["native_full_graph_backend"] = backend
    owner = runtime.NativeFullGraphCache(cfg)
    context = sources()
    state = torch.ones(2, 2, 2)
    context.no_compile_layers = {"gdn0": SimpleNamespace(kv_cache=(state,)), "gdn1": SimpleNamespace(kv_cache=(state,))}
    events = []
    stream = SimpleNamespace(wait_stream=lambda other: events.append("wait"), synchronize=lambda: events.append("sync"))

    @contextmanager
    def stream_context(*args, **kwargs):
        yield

    @contextmanager
    def graph_context(*args, **kwargs):
        events.append("capture")
        yield

    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(
            get_device_name=lambda: "Ascend910B4",
            Stream=lambda: stream,
            current_stream=lambda: stream,
            stream=stream_context,
            graph=graph_context,
            NPUGraph=lambda: SimpleNamespace(pool=lambda: "shared"),
        ),
        raising=False,
    )

    def model(**kwargs):
        assert context.cudagraph_runtime_mode == Mode.FULL
        assert context.num_tokens == kwargs["input_ids"].numel()
        assert all(
            (m.non_spec_state_indices_tensor == -1).all()
            for m in context.attn_metadata.values()
            if type(m).__name__ == "GDNAttentionMetadata"
        )
        native = context.attn_metadata["gdn0"].native_full
        assert native.megagdn_workspace is owner.megagdn_workspace
        if backend == "megagdn" and native.shape.fresh:
            native.megagdn_workspace.views(
                device="cpu",
                tokens=native.shape.tokens,
                heads=1,
                hidden_size=128,
                chunks=native.shape.requests,
                block_dim=1,
            )
        return kwargs["input_ids"][:, None].float()

    old = vars(context).copy()
    owner.warmup(context, inputs(1), model)
    assert owner.ready and set(owner.entries) == set(owner.plan.shapes)
    assert events.count("capture") == len(owner.plan.shapes)
    assert state.eq(1).all()
    assert vars(context) == old
    if backend == "megagdn":
        assert owner.megagdn_workspace.sealed and owner.megagdn_workspace.reserved_bytes > 0
        assert len(owner.megagdn_workspace.buffers) == 1
    with pytest.raises(RuntimeError, match="once"):
        owner.warmup(context, inputs(1), model)


def test_megagdn_capacity_covers_chunks_from_short_high_concurrency_rectangles():
    runtime = load_runtime()
    cfg = config()
    cfg.additional_config["native_full_graph_backend"] = "megagdn"
    owner = runtime.NativeFullGraphCache(cfg)
    workspace = owner.megagdn_workspace
    assert workspace.max_tokens == 256
    assert workspace.max_chunks == 64  # 64 one-token rows, versus two chunks for T=256/R=1.
    assert workspace.reserved_bytes == 0  # Allocation waits for the warmup stream.
    for shape in owner.plan.shapes:
        entry = runtime.NativeFullEntry(owner, shape, sources(), inputs())
        assert entry.megagdn_workspace is workspace
    cfg.additional_config["native_full_graph_megagdn_shared_workspace"] = False
    assert runtime.NativeFullGraphCache(cfg).megagdn_workspace is None
    cfg.additional_config["native_full_graph_megagdn_shared_workspace"] = "false"
    with pytest.raises(ValueError, match="must be a boolean"):
        runtime.NativeFullGraphCache(cfg)


def test_native_runner_bypasses_standard_dispatch_and_mutable_attention_updates():
    path = ROOT / "vllm_ascend/worker/model_runner_v1.py"
    tree = ast.parse(path.read_text())
    namespace = dict(
        CUDAGraphMode=Mode,
        BatchDescriptor=lambda tokens: SimpleNamespace(num_tokens=tokens),
        partial=__import__("functools").partial,
        record_function_or_nullcontext=lambda name: __import__("contextlib").nullcontext(),
        get_forward_context=lambda: SimpleNamespace(),
    )
    for name in ("_model_forward", "_determine_batch_execution_and_padding"):
        node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
        module = ast.Module(
            body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node],
            type_ignores=[],
        )
        exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    model = Mock(return_value="native replay")
    model.native_full = SimpleNamespace(enabled=True, ready=True, warming=False)
    runner = SimpleNamespace(
        model=model, _update_full_graph_params_if_needed=Mock(side_effect=AssertionError("unexpected task update"))
    )
    mode, descriptor, *_ = namespace["_determine_batch_execution_and_padding"](runner, 61, 1, None, 61, False)
    assert mode == Mode.PIECEWISE and descriptor.num_tokens == 61
    assert namespace["_model_forward"](runner, 61, input_ids="ids", positions="pos") == "native replay"
    runner._update_full_graph_params_if_needed.assert_not_called()


def test_native_core_routes_without_eager_break_and_binds_qwen_helpers():
    path = ROOT / "vllm_ascend/ops/gdn.py"
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    methods = {node.name: node for node in cls.body if isinstance(node, ast.FunctionDef)}
    for name in ("_forward_core", "_forward_native_graph", "_forward_native_decode_graph"):
        assert not methods[name].decorator_list
    assert any(
        isinstance(node, ast.Name) and node.id == "eager_break_during_capture"
        for node in methods["_forward_compatibility"].decorator_list
    )
    for name in ("_forward_native_graph", "_forward_native_decode_graph"):
        assert not any(
            isinstance(node, ast.Attribute) and node.attr in ("item", "tolist", "num_actual_tokens", "num_prefills")
            for node in ast.walk(methods[name])
        )
    method = methods["_forward_core"]
    namespace = {"get_forward_context": lambda: SimpleNamespace(attn_metadata={"layer": metadata})}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    layer = SimpleNamespace(
        prefix="layer", _forward_native_graph=Mock(), _forward_native_decode_graph=Mock(), _forward_compatibility=Mock()
    )
    for width, fresh, target in (
        (64, True, "_forward_native_graph"),
        (1, True, "_forward_native_graph"),
        (1, False, "_forward_native_decode_graph"),
    ):
        metadata = SimpleNamespace(native_full=SimpleNamespace(shape=SimpleNamespace(width=width, fresh=fresh)))
        namespace["_forward_core"](layer, 1, 2, 3, 4)
        getattr(layer, target).assert_called_with(1, 2, 3, 4, metadata)
    layer._forward_compatibility.assert_not_called()
    patch = (ROOT / "vllm_ascend/patch/worker/patch_qwen3_5.py").read_text()
    for name in ("_forward_core", "_forward_native_graph", "_forward_native_decode_graph", "_forward_compatibility"):
        assert f"_GDN_PATCH_TARGET.{name} = AscendGatedDeltaNetAttention.{name}" in patch


def test_native_text_preprocessing_preserves_token_ids_for_multimodal_qwen():
    path = ROOT / "vllm_ascend/worker/model_runner_v1.py"
    tree = ast.parse(path.read_text())
    method = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "_preprocess")
    namespace = {}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    runner = SimpleNamespace(
        model=SimpleNamespace(native_full=SimpleNamespace(enabled=True, ready=True)),
        input_batch=SimpleNamespace(req_prompt_embeds={}),
        requests={"r": SimpleNamespace(mm_features=[])},
        uses_mrope=True,
        mrope_positions=SimpleNamespace(gpu=torch.arange(12).view(3, 4)),
        input_ids=SimpleNamespace(gpu=torch.arange(4)),
    )
    scheduler = SimpleNamespace(num_scheduled_tokens={"r": 3})
    ids, embeds, positions, intermediate, kwargs, ec = namespace["_preprocess"](runner, scheduler, 3)
    assert ids.tolist() == [0, 1, 2]
    assert positions.tolist() == [[0, 1, 2], [4, 5, 6], [8, 9, 10]]
    assert embeds is intermediate is ec is None and kwargs == {}
    runner.requests["r"].mm_features = ["image"]
    with pytest.raises(ValueError, match="text token IDs"):
        namespace["_preprocess"](runner, scheduler, 3)
    runner.requests["r"].mm_features = []
    runner.input_batch.req_prompt_embeds = {0: torch.ones(1)}
    with pytest.raises(ValueError, match="text token IDs"):
        namespace["_preprocess"](runner, scheduler, 3)


def test_configured_request_counts_round_up_and_never_capture_stateful_prefill():
    plan = layout.NativeGraphPlan([1, 64, 128], 4096, 64, 768, [1, 4, 16])
    assert plan.batches([0, 61, 122], [61, 61])[0][0].requests == 4
    assert plan.batches([0, 61, 122, 183, 244, 305], [61] * 5)[0][0].requests == 16
    assert all(s.fresh or s.width == 1 for s in plan.shapes)
    assert plan.batches([0, 61], [65]) is None
    assert plan.batches([0, 769], [769]) is None
    padded_only = layout.NativeGraphPlan([64], 4096, 64, 768, [4, 16])
    assert padded_only.batches([0, 61], [61])[0][0].requests == 4
    with pytest.raises(ValueError, match="positive integers"):
        layout.NativeGraphPlan([64], 4096, 64, 768, [0, 16])


def test_dummy_requests_cannot_address_live_state_or_kv_slots():
    runtime = load_runtime()
    entry = runtime.NativeFullEntry(
        runtime.NativeFullGraphCache(config()), layout.NativeGraphShape(4, 8, True), sources(), inputs()
    )
    item = layout.RequestSlice(0, 0, 3, 0)
    source, target = entry.update(sources(), inputs(), (item,))
    entry.execute(lambda **kwargs: None)
    assert entry.lengths.tolist() == [3, 0, 0, 0]
    assert entry.metadata["gdn0"].non_spec_state_indices_tensor.tolist() == [7, -1, -1, -1]
    assert entry.metadata["attn"].slot_mapping.tolist() == [100, 101, 102] + [-1] * 29
    assert source.tolist() == target.tolist() == [0, 1, 2]
    assert not entry.has_initial_state.any()
    decode = runtime.NativeFullEntry(
        runtime.NativeFullGraphCache(config()), layout.NativeGraphShape(4, 1, False), sources(), inputs()
    )
    decode.update(sources(), inputs(), (layout.RequestSlice(1, 3, 1, 8),))
    decode.execute(lambda **kwargs: None)
    assert decode.metadata["gdn0"].non_spec_decode_metadata.actual_seq_lengths.tolist() == [0, 1, 0, 0, 0]
    assert (~decode.mask).any(dim=-1).all(), "Dummy attention rows must not be fully masked"


def test_fresh_convolution_padding_keeps_last_real_inputs():
    x = torch.arange(4 * 8 * 2).reshape(4, 8, 2).float()
    lengths = torch.tensor([0, 1, 2, 6])
    actual = layout.fresh_conv_history(x, lengths, 3)
    for row, length in enumerate(lengths.tolist()):
        reference = torch.cat((torch.zeros(3, 2), x[row, :length]))[-3:]
        torch.testing.assert_close(actual[row], reference)


def test_nonfresh_prefill_uses_compatibility_before_replaying_any_graph():
    runtime = load_runtime()
    owner = runtime.NativeFullGraphCache(config())
    owner.ready = True
    context = sources()
    context.attn_metadata["attn"].seq_lens_list = [7, 9, 6]
    before = vars(context).copy()
    fallback = Mock(side_effect=lambda **kwargs: context.cudagraph_runtime_mode)
    assert owner.run(context, inputs(), fallback) == Mode.NONE
    fallback.assert_called_once()
    assert vars(context) == before and owner.compatibility_steps == 1
