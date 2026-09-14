# SPDX-License-Identifier: Apache-2.0
"""CPU regression tests for padded MTP and n-gram graph integration."""

import ast
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from test_native_full import ROOT, AscendMetadata, config, inputs, layout, load_runtime, sources


def mtp_config(count=3, method="mtp"):
    cfg = config()
    cfg.speculative_config = SimpleNamespace(method=method, num_speculative_tokens=count, parallel_drafting=False)
    return cfg


@pytest.mark.parametrize(
    "method,count,parallel",
    [
        ("mtp", 0, False),
        ("mtp", 4, False),
        ("eagle", 3, False),
        ("mtp", 3, True),
        ("ngram", 0, False),
        ("ngram", 16, False),
        ("ngram_gpu", 16, False),
        ("mtp", True, False),
    ],
)
def test_native_mtp_configuration_is_explicit(method, count, parallel):
    cfg = mtp_config()
    cfg.speculative_config = SimpleNamespace(method=method, num_speculative_tokens=count, parallel_drafting=parallel)
    with pytest.raises(ValueError, match="autoregressive MTP with num_speculative_tokens=1/2/3"):
        load_runtime().NativeFullGraphCache(cfg)


@pytest.mark.parametrize("method", ["mtp", "qwen3_5_mtp", "ngram", "ngram_gpu"])
@pytest.mark.parametrize("count", [1, 2, 3])
def test_speculative_counts_capture_width_and_no_draft_steps(method, count):
    runtime = load_runtime()
    owner = runtime.NativeFullGraphCache(mtp_config(count, method))
    width = count + 1
    assert {s.width for s in owner.plan.shapes if not s.fresh} == {width}
    context = mtp_sources()
    context.attn_metadata["gdn0"].native_num_accepted_tokens.fill_(width)
    # N-gram may find no match after the previous round accepted all drafts.
    # Its next single-token query must still read the last checkpoint.
    entry = runtime.NativeFullEntry(owner, layout.NativeGraphShape(2, width, False), context, inputs(8))
    entry.update(context, inputs(8), [layout.RequestSlice(2, 7, 1, 19)])
    entry.prepare()
    metadata = entry.metadata["gdn0"]
    assert metadata.num_accepted_tokens.tolist() == [width, 1]
    assert metadata.spec_state_indices_tensor.tolist() == [list(range(31, 31 + width)), [-1] * width]
    assert metadata.spec_decode_metadata.actual_seq_lengths.tolist() == [0, width, 0]
    assert owner.plan.batches([0, 1], [20], [False])[0][0].width == width


@pytest.mark.parametrize("method", ["ngram", "ngram_gpu"])
def test_ngram_supports_native_checkpoint_limit(method):
    plan = load_runtime().NativeFullGraphCache(mtp_config(15, method)).plan
    assert plan.decode_width == 16
    assert plan.batches([0, 16, 17], [48, 50], [False, False])[0][0].width == 16


def test_mtp_three_has_four_token_verification_and_separate_fresh_buckets():
    plan = load_runtime().NativeFullGraphCache(mtp_config()).plan
    assert {shape.width for shape in plan.shapes if not shape.fresh} == {4}
    boundaries, totals = [0, 4, 65, 66, 68], [36, 61, 72, 101]
    batches = plan.batches(boundaries, totals, [False, True, False, False])
    assert [(s.width, s.fresh) for s, _ in batches] == [(4, False), (64, True)]
    assert [item.length for item in batches[0][1]] == [4, 1, 2]
    assert all(shape in plan.shapes for shape, _ in batches)
    assert plan.batches([0, 4], [100], [True]) is None, "Continuation prefill is not verification"
    assert plan.batches([0, 4], [100]) is None, "Verification needs scheduler classification"
    assert plan.batches([0, 5], [100], [False]) is None


def mtp_sources():
    context = sources()
    gdn = context.attn_metadata["gdn0"]
    # Intentionally make the compatibility partitions unusable by original row.
    gdn.non_spec_state_indices_tensor = torch.tensor([19], dtype=torch.int32)
    gdn.spec_state_indices_tensor = torch.tensor([[7, 8, 9, 10], [31, 32, 33, 34]], dtype=torch.int32)
    gdn.native_state_indices = torch.tensor([[7, 8, 9, 10], [19, 20, 21, 22], [31, 32, 33, 34]], dtype=torch.int32)
    gdn.native_num_accepted_tokens = torch.tensor([4, 1, 2], dtype=torch.int32)
    gdn.native_is_prefilling = torch.tensor([False, True, False])
    attn = context.attn_metadata["attn"]
    attn.actual_seq_lengths_q = [4, 7, 8]
    attn.num_actual_tokens = 8
    attn.seq_lens_list = [36, 3, 20]
    # Deliberately optimistic CPU totals: attention uses the corrected device view.
    attn.seq_lens = torch.tensor(attn.seq_lens_list)
    attn.native_seq_lens = torch.tensor([33, 3, 18])
    return context


def test_mtp_slots_survive_partitioning_reorder_rejection_and_short_verification():
    runtime = load_runtime()
    owner, context = runtime.NativeFullGraphCache(mtp_config()), mtp_sources()
    entry = runtime.NativeFullEntry(owner, layout.NativeGraphShape(4, 4, False), context, inputs(8))
    metadata = entry.metadata["gdn0"]
    addresses = [metadata.spec_state_indices_tensor.data_ptr(), metadata.num_accepted_tokens.data_ptr()]
    items = (layout.RequestSlice(2, 7, 1, 19), layout.RequestSlice(0, 0, 4, 32))
    for accepted in ([4, 1, 2], [1, 1, 4], [3, 1, 1]):
        context.attn_metadata["gdn0"].native_num_accepted_tokens.copy_(torch.tensor(accepted))
        src, dst = entry.update(context, inputs(8), items)
        entry.prepare()
        assert metadata.spec_state_indices_tensor.tolist() == [[31, 32, 33, 34], [7, 8, 9, 10], [-1] * 4, [-1] * 4]
        assert metadata.num_accepted_tokens.tolist() == [accepted[2], accepted[0], 1, 1]
        assert metadata.spec_decode_metadata.actual_seq_lengths.tolist() == [0, 4, 4, 0, 0]
        assert entry.contexts.tolist() == [17, 29, 0, 0]
        assert src.tolist() == [0, 4, 5, 6, 7] and dst.tolist() == [7, 0, 1, 2, 3]
        assert entry.metadata["attn"].slot_mapping.tolist() == [107, -1, -1, -1, 100, 101, 102, 103] + [-1] * 8
        assert addresses == [metadata.spec_state_indices_tensor.data_ptr(), metadata.num_accepted_tokens.data_ptr()]
    fresh = runtime.NativeFullEntry(owner, layout.NativeGraphShape(1, 4, True), context, inputs(8))
    fresh.update(context, inputs(8), (layout.RequestSlice(1, 4, 3, 0),))
    assert fresh.metadata["gdn0"].non_spec_state_indices_tensor.tolist() == [19]


def test_mtp_mixed_step_replays_without_losing_original_token_order():
    runtime = load_runtime()
    owner, context, model_inputs = runtime.NativeFullGraphCache(mtp_config()), mtp_sources(), inputs(8)
    for shape, _ in owner.plan.batches([0, 4, 7, 8], [36, 3, 20], [False, True, False]):
        entry = runtime.NativeFullEntry(owner, shape, context, model_inputs)
        entry.output = torch.empty(shape.tokens, 1)
        entry.graph = SimpleNamespace(replay=lambda e=entry: e.output.copy_(e.input_ids[:, None]))
        owner.entries[shape] = entry
    owner.ready = True
    torch.testing.assert_close(owner.run(context, model_inputs)[:, 0], torch.arange(8).float())
    assert owner.compatibility_steps == 0
    assert all(entry.replays == 1 for entry in owner.entries.values())


def test_cached_four_token_core_cannot_enter_fresh_prefill():
    path = ROOT / "vllm_ascend/ops/gdn.py"
    tree = ast.parse(path.read_text())
    method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_forward_core")
    metadata = SimpleNamespace(native_full=SimpleNamespace(shape=layout.NativeGraphShape(4, 4, False)))
    context = SimpleNamespace(attn_metadata={"gdn": metadata})
    namespace = {"get_forward_context": lambda: context}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    layer = SimpleNamespace(
        prefix="gdn", _forward_native_decode_graph=Mock(), _forward_native_graph=Mock(), _forward_compatibility=Mock()
    )
    namespace["_forward_core"](layer, "qkv", "b", "a", "out")
    layer._forward_native_decode_graph.assert_called_once_with("qkv", "b", "a", "out", metadata)
    layer._forward_native_graph.assert_not_called()
    layer._forward_compatibility.assert_not_called()


def test_attention_builder_keeps_authoritative_device_lengths_separate_from_cpu_fia_abi():
    path = ROOT / "vllm_ascend/attention/attention_v1.py"
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "AscendAttentionMetadataBuilder")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "build")
    method.returns = None
    for arg in method.args.args:
        arg.annotation = None
    namespace = dict(torch=torch, CrossAttentionSpec=type("CrossAttentionSpec", (), {}))
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    common = SimpleNamespace(
        num_reqs=2,
        num_actual_tokens=5,
        query_start_loc_cpu=torch.tensor([0, 4, 5]),
        query_start_loc=torch.tensor([0, 4, 5]),
        block_table_tensor=torch.zeros(2, 3, dtype=torch.int32),
        _seq_lens_cpu=torch.tensor([68, 13]),
        seq_lens_cpu=None,
        seq_lens=torch.tensor([66, 10]),
        slot_mapping=torch.arange(5),
        attn_state="spec",
        causal=True,
        max_query_len=4,
    )
    builder = SimpleNamespace(
        _split_decodes_and_prefills=lambda common: (2, 0, 5, 0),
        kv_cache_spec=object(),
        speculative_config=SimpleNamespace(parallel_drafting=False),
        attn_mask_builder=SimpleNamespace(get_attention_mask=lambda *args: None),
        model_config=SimpleNamespace(runner_type="generate"),
        _build_backend_metadata=lambda *args, **kwargs: {},
        metadata_cls=AscendMetadata,
    )
    metadata = namespace["build"](builder, 0, common)
    assert metadata.seq_lens_list == [68, 13]
    assert metadata.seq_lens.data_ptr() == common._seq_lens_cpu.data_ptr()
    assert metadata.native_seq_lens is common.seq_lens
    assert metadata.native_seq_lens.tolist() == [66, 10]


def load_draft_runtime(context):
    runtime = load_runtime()
    path = ROOT / "vllm_ascend/compilation/native_mtp_graph.py"
    tree = ast.parse(path.read_text())
    tree.body = [n for n in tree.body if not isinstance(n, (ast.Import, ast.ImportFrom))]
    namespace = vars(runtime).copy()
    namespace.update(
        NativeGraphShape=layout.NativeGraphShape,
        get_forward_context=lambda: context,
    )
    exec(compile(tree, str(path), "exec"), namespace)
    return SimpleNamespace(**namespace)


@pytest.mark.parametrize("count", [1, 2, 3])
def test_merged_draft_uses_requested_passes_and_device_accepted_lengths(count):
    context = sources()
    context.attn_metadata = {"draft": context.attn_metadata["attn"]}
    context.attn_metadata["draft"].block_tables = torch.zeros(3, 5, dtype=torch.int32)
    runtime = load_draft_runtime(context)
    calls = []

    def model(input_ids, positions, hidden_states, inputs_embeds):
        entry = context.attn_metadata["draft"].native_full
        calls.append(
            (
                input_ids.numel(),
                entry.lengths.clone(),
                entry.contexts.clone(),
                entry.metadata["draft"].slot_mapping.clone(),
            )
        )
        return hidden_states + input_ids[:, None] + positions[0, :, None]

    proposer = SimpleNamespace(
        model=model,
        compute_draft_token_ids=lambda h: h.sum(-1).to(torch.int64) % 97,
        model_returns_tuple=lambda: False,
        pass_hidden_states_to_model=True,
    )
    owner = runtime.NativeMTPGraphCache(mtp_config(count), proposer)
    owner.block_size = 4
    model_inputs = inputs(8)
    model_inputs["hidden_states"] = torch.arange(16).view(8, 2).float()
    entry = runtime.NativeMTPEntry(owner, layout.NativeGraphShape(4, 4, False), context, model_inputs)
    source = AscendMetadata(
        block_tables=torch.tensor([[2, 3, 4, 5, 6], [8, 9, 10, 11, 12], [13, 14, 15, 16, 17]]),
        native_seq_lens=torch.tensor([12, 15, 10]),
        slot_mapping=torch.arange(8) + 100,
    )
    source_context = SimpleNamespace(attn_metadata={"draft": source})
    items = (layout.RequestSlice(0, 0, 4, 8), layout.RequestSlice(1, 4, 3, 12), layout.RequestSlice(2, 7, 1, 9))
    # Accept 2/3 first-pass tokens; row 2 has no valid sample (discard/EOS).
    entry.update_draft(source_context, model_inputs, torch.tensor([1, 6, 6]), items)
    result = entry.execute(proposer)
    assert [call[0] for call in calls] == [16] + [4] * (count - 1)
    assert calls[0][1].tolist() == [2, 3, 0, 0]
    for index, call in enumerate(calls[1:]):
        assert call[1].tolist() == [1, 1, 0, 0]
        assert call[2].tolist() == [10 + index, 15 + index, 0, 0]
        assert call[3].tolist() == [18 + index, 47 + index, -1, -1]
    assert calls[0][3].tolist() == [100, 101, -1, -1, 104, 105, 106, -1] + [-1] * 8
    for row, selected in enumerate([1, 6]):
        hidden = model_inputs["hidden_states"][selected].clone()
        token, position = selected, selected
        expected = []
        for _ in range(count):
            hidden += token + position
            token = int(hidden.sum()) % 97
            expected.append(token)
            position += 1
        assert result[row].tolist() == expected
    torch.testing.assert_close(model_inputs["hidden_states"], torch.arange(16).view(8, 2).float())


def test_mtp_capture_warmup_never_writes_draft_cache():
    context = sources()
    context.attn_metadata = {"draft": context.attn_metadata["attn"]}
    runtime = load_draft_runtime(context)
    slots = []

    def model(**kwargs):
        slots.append(context.attn_metadata["draft"].slot_mapping.clone())
        return kwargs["hidden_states"] + 1

    proposer = SimpleNamespace(
        model=model,
        compute_draft_token_ids=lambda h: h.sum(-1).long(),
        model_returns_tuple=lambda: False,
        pass_hidden_states_to_model=True,
    )
    owner = runtime.NativeMTPGraphCache(mtp_config(), proposer)
    owner.block_size = 4
    model_inputs = dict(inputs(), hidden_states=torch.ones(10, 2))
    entry = runtime.NativeMTPEntry(owner, layout.NativeGraphShape(4, 64, True), context, model_inputs)
    entry.execute(proposer)
    assert len(slots) == 3 and all((mapping == -1).all() for mapping in slots)


def test_mixed_draft_groups_preserve_inputs_and_restore_request_order():
    context = sources()
    context.attn_metadata = {"draft": context.attn_metadata["attn"]}
    runtime = load_draft_runtime(context)
    ids = torch.arange(13)
    hidden = torch.arange(26).view(13, 2).float()
    positions = ids.expand(3, -1)
    proposer = SimpleNamespace(
        input_ids=ids,
        hidden_states=hidden,
        _get_positions=lambda n: positions[:, :n],
        model=lambda input_ids, positions, hidden_states, inputs_embeds: hidden_states + input_ids[:, None],
        compute_draft_token_ids=lambda h: h.sum(-1).long() % 97,
        model_returns_tuple=lambda: False,
        pass_hidden_states_to_model=True,
    )
    owner = runtime.NativeMTPGraphCache(mtp_config(), proposer)
    owner.block_size = 16
    common = SimpleNamespace(
        num_reqs=3,
        query_start_loc_cpu=torch.tensor([0, 3, 7, 13]),
        seq_lens_cpu_upper_bound=torch.tensor([3, 16, 6]),
        seq_lens=torch.tensor([3, 14, 6]),
        is_prefilling=torch.tensor([True, False, True]),
        block_table_tensor=torch.tensor([[2, 3, 4], [5, 6, 7], [8, 9, 10]], dtype=torch.int32),
        slot_mapping=torch.arange(13, dtype=torch.int32),
    )
    model_inputs = dict(input_ids=ids, positions=positions, hidden_states=hidden)
    batches = owner.plan.batches([0, 3, 7, 13], [3, 16, 6], [True, False, True])
    # Simulate a shared output pool to catch an omitted scatter between groups.
    shared_output = torch.empty(1, 3, dtype=torch.int64)
    for shape, _ in batches:
        entry = runtime.NativeMTPEntry(owner, shape, context, model_inputs)
        entry.output = shared_output
        entry.graph = SimpleNamespace(replay=lambda e=entry: shared_output.copy_(e.execute(proposer)))
        owner.entries[shape] = entry
    owner.ready = True
    keys = tuple(owner.entries)
    samples = torch.tensor([2, 4, 12])
    for _ in range(2):
        actual = owner.propose(common, 13, samples)
        for row, sample in enumerate(samples.tolist()):
            state, token = hidden[sample].clone(), int(ids[sample])
            expected = []
            for _ in range(3):
                state += token
                token = int(state.sum()) % 97
                expected.append(token)
            assert actual[row].tolist() == expected
        torch.testing.assert_close(ids, torch.arange(13))
        torch.testing.assert_close(hidden, torch.arange(26).view(13, 2).float())
        assert tuple(owner.entries) == keys
        assert owner.compatibility_steps == 0
    common.seq_lens_cpu_upper_bound[1] = owner.plan.max_context - 1
    before = [e.replays for e in owner.entries.values()]
    assert owner.propose(common, 13, samples) is None
    assert [e.replays for e in owner.entries.values()] == before
