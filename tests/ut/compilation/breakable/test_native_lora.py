# SPDX-License-Identifier: Apache-2.0
"""Adapter selection, eviction and graph ownership without an NPU runtime."""

import ast
from collections.abc import Callable
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from test_native_full import ROOT, config, inputs, layout, load_runtime, sources
from test_native_mtp import load_draft_runtime, mtp_config, mtp_sources


def wrapper():
    return SimpleNamespace(
        _token_lora_indices=torch.tensor([0, 0, 0, 0, -1, -1, -1, 1]),
        _embeddings_indices=torch.tensor([0, 0, 0, 0, 0, 0, 0, 100]),
        _sampler_indices=torch.tensor([0, 0, 0, 0, -1, 1]),
        _sampler_indices_padded=torch.arange(6),
        indices_len=[8, 6, 6, 8],
        is_prefill=True,
        no_lora=True,
        lora_config=SimpleNamespace(max_loras=2, max_lora_rank=8, fully_sharded_loras=False),
        enable_native_full_graph=Mock(),
    )


@pytest.mark.parametrize("rank", [8, 16, 32, 64])
def test_lora_slot_capacity_is_independent_of_graph_shapes(rank):
    cfg = config()
    cfg.lora_config = wrapper().lora_config
    cfg.lora_config.max_lora_rank = rank
    assert (
        load_runtime().NativeFullGraphCache(cfg).plan.shapes
        == load_runtime().NativeFullGraphCache(config()).plan.shapes
    )
    cfg.lora_config.fully_sharded_loras = True
    with pytest.raises(ValueError, match="unsharded"):
        load_runtime().NativeFullGraphCache(cfg)
    cfg.lora_config.fully_sharded_loras = False
    cfg.lora_config.max_lora_rank = 128
    with pytest.raises(ValueError, match="max_lora_rank"):
        load_runtime().NativeFullGraphCache(cfg)


def test_warmup_discovers_shared_punica_once_and_requires_target_adapters():
    runtime = load_runtime()
    punica = wrapper()
    model = SimpleNamespace(modules=lambda: [SimpleNamespace(punica_wrapper=punica)] * 4)
    assert runtime.native_lora_wrappers(model, required=True) == (punica,)
    punica.enable_native_full_graph.assert_called_once()
    model = SimpleNamespace(modules=lambda: [])
    with pytest.raises(ValueError, match="before warmup"):
        runtime.native_lora_wrappers(model, required=True)
    assert runtime.native_lora_wrappers(model, required=False) == (), "The MTP head need not carry target adapters"


def test_native_lora_rejects_mismatched_kernel_weight_dtype():
    cfg = config()
    cfg.model_config.dtype = torch.bfloat16
    cfg.lora_config = wrapper().lora_config
    cfg.lora_config.lora_dtype = torch.float16
    with pytest.raises(ValueError, match="model dtype"):
        load_runtime().NativeFullGraphCache(cfg)
    cfg.lora_config.lora_dtype = "auto"
    assert load_runtime().NativeFullGraphCache(cfg).enabled


@pytest.mark.parametrize("embedding_planes", [1, 2])
def test_binding_keeps_packed_runner_and_sampler_mapping_untouched(embedding_planes):
    runtime = load_runtime()
    punica = wrapper()
    if embedding_planes == 2:
        punica._embeddings_indices = punica._embeddings_indices.repeat(2, 1)
    original = vars(punica).copy()
    mapping = runtime.NativeGraphLoRA(punica, layout.NativeGraphShape(4, 4, True))
    tokens = torch.tensor([7, 7, 7, 7, 0, 1, 2, 3, 4, 5, 6, 6, 0, 0, 0, 0])
    valid = torch.tensor([1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]).bool()
    mapping.update(tokens, valid)
    assert mapping.tokens.tolist() == [1, -1, -1, -1, 0, 0, 0, 0] + [-1] * 8
    assert mapping.sampler.tolist() == [1, 0, -1, -1]
    expected_embeddings = torch.tensor([100] + [0] * 15)
    if embedding_planes == 2:
        expected_embeddings = expected_embeddings.repeat(2, 1)
    torch.testing.assert_close(mapping.embeddings, expected_embeddings)
    with pytest.raises(RuntimeError, match="capture failed"), mapping.bind():
        assert punica._token_lora_indices is mapping.tokens
        assert punica._sampler_indices is mapping.sampler
        assert not punica.no_lora and not punica.is_prefill
        raise RuntimeError("capture failed")
    assert all(getattr(punica, key) is value for key, value in original.items())


@pytest.mark.parametrize("method,count", [("mtp", 1), ("mtp", 2), ("mtp", 3), ("ngram", 3)])
def test_mixed_padded_replay_and_adapter_eviction_preserve_original_order(method, count):
    runtime, context, model_inputs = load_runtime(), mtp_sources(), inputs(8)
    cfg = mtp_config(count, method)
    punica = wrapper()
    cfg.lora_config = punica.lora_config
    owner = runtime.NativeFullGraphCache(cfg)
    owner.lora_wrappers = (punica,)
    width = count + 1
    context.attn_metadata["attn"].actual_seq_lengths_q = [width, 7, 8]
    context.attn_metadata["attn"].seq_lens_list = [32 + width, 7 - width, 20]
    flags = context.attn_metadata["gdn0"].native_is_prefilling
    batches = owner.plan.batches([0, width, 7, 8], [32 + width, 7 - width, 20], flags.tolist())
    # A captured closure holds the same physical weight storage across swaps.
    weights = torch.tensor([10.0, 30.0])
    for shape, _ in batches:
        entry = runtime.NativeFullEntry(owner, shape, context, model_inputs)
        entry.output = torch.empty(shape.tokens, 1)
        mapping = entry.lora[0]

        def replay(e=entry, indices=mapping.tokens):
            delta = torch.where(indices >= 0, weights[indices.clamp(min=0)], 0)
            e.output[:, 0].copy_(e.input_ids + delta)

        entry.graph = SimpleNamespace(replay=replay)
        owner.entries[shape] = entry
    owner.ready = True
    keys = tuple(owner.entries)
    addresses = [e.lora[0].tokens.data_ptr() for e in owner.entries.values()]
    source_sampler = punica._sampler_indices.clone()
    for slots, values in (
        ([0] * width + [-1] * (7 - width) + [1], [10, 30]),
        ([1] * width + [0] * (7 - width) + [-1], [50, 70]),
        ([-1] * 8, [50, 70]),
        ([0] * 8, [-10, 70]),
    ):
        punica._token_lora_indices.copy_(torch.tensor(slots))
        weights.copy_(torch.tensor(values))  # Evict/reload an existing physical slot.
        result = owner.run(context, model_inputs)[:, 0]
        expected = torch.arange(8) + torch.tensor([0 if slot < 0 else values[slot] for slot in slots])
        torch.testing.assert_close(result, expected.float())
        torch.testing.assert_close(punica._sampler_indices, source_sampler)
        assert tuple(owner.entries) == keys and owner.compatibility_steps == 0
        assert addresses == [e.lora[0].tokens.data_ptr() for e in owner.entries.values()]


@pytest.mark.parametrize("count", [1, 2, 3])
def test_draft_lora_mapping_tracks_requests_through_acceptance_and_skinny_passes(count):
    context = sources()
    context.attn_metadata = {"draft": context.attn_metadata["attn"]}
    runtime = load_draft_runtime(context)
    punica = wrapper()
    seen = []

    def model(**kwargs):
        seen.append((punica._token_lora_indices.clone(), punica._sampler_indices.clone()))
        return kwargs["hidden_states"] + 1

    proposer = SimpleNamespace(
        model=model,
        compute_draft_token_ids=lambda h: h.sum(-1).long() + punica._sampler_indices.clamp(min=0),
        model_returns_tuple=lambda: False,
        pass_hidden_states_to_model=True,
    )
    owner = runtime.NativeMTPGraphCache(mtp_config(count), proposer)
    owner.lora_wrappers, owner.block_size = (punica,), 4
    model_inputs = inputs(8)
    model_inputs["hidden_states"] = torch.zeros(8, 2)
    entry = runtime.NativeMTPEntry(owner, layout.NativeGraphShape(4, 4, False), context, model_inputs)
    context.attn_metadata["draft"].native_seq_lens = torch.tensor([36, 3, 20])
    items = [layout.RequestSlice(2, 7, 1, 19), layout.RequestSlice(0, 0, 4, 32)]
    entry.update_draft(context, model_inputs, torch.tensor([1, 6, 7]), items)
    result = entry.execute(proposer)
    assert seen[0][0].tolist() == [1, -1, -1, -1, 0, 0] + [-1] * 10
    assert all(sampler.tolist() == [1, 0, -1, -1] for _, sampler in seen)
    assert all(tokens.tolist() == [1, 0, -1, -1] for tokens, _ in seen[1:])
    assert result[0].tolist() == [3 + 2 * i for i in range(count)]
    assert punica._sampler_indices.tolist() == [0, 0, 0, 0, -1, 1]


def test_native_punica_metadata_uses_base_map_without_device_prefill_grouping():
    path = ROOT / "vllm_ascend/lora/punica_npu.py"
    cls = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef))

    class Base:
        def _update_base_metadata(self, *args):
            self.base_updates += 1

        def update_metadata(self, *args, **kwargs):
            self.compatibility_updates += 1

    namespace = dict(torch=torch, Callable=Callable, PunicaWrapperBase=Base)
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), "exec"), namespace)
    punica = object.__new__(namespace["PunicaWrapperNPU"])
    punica._native_full_graph = True
    punica._single_lora_mask = None
    punica.base_updates = punica.compatibility_updates = 0
    for mapping in (SimpleNamespace(is_prefill=True), SimpleNamespace(is_prefill=False)):
        punica.update_metadata(mapping, [1, None], 2, 100)
        assert not punica.is_prefill and not punica.no_lora
    assert (punica.base_updates, punica.compatibility_updates) == (2, 0)
    punica._native_full_graph = False
    punica.update_metadata(None, [], 1, 100)
    assert punica.compatibility_updates == 1


def test_runner_allows_native_speculation_but_retains_legacy_breakable_guard():
    path = ROOT / "vllm_ascend/worker/model_runner_v1.py"
    tree = ast.parse(path.read_text())
    guard = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and any(
            isinstance(child, ast.Raise) and "does not support speculative decoding" in ast.unparse(child)
            for child in node.body
        )
    )
    condition = compile(ast.Expression(guard.test), str(path), "eval")
    for native, expected in ((True, False), (False, True)):
        assert (
            eval(
                condition,
                {
                    "self": SimpleNamespace(speculative_config=object(), vllm_config=None),
                    "native_full_requested": lambda cfg, native=native: native,
                },
            )
            is expected
        )
