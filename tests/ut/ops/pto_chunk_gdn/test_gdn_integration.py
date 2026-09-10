# SPDX-License-Identifier: Apache-2.0
"""Exercise the production GDN core's slicing and state writeback on CPU.

Only device operators are faked; the mixed/decode/prefill control flow is the
actual method from gdn.py. NPU arithmetic is covered by numerical.py on hardware.
"""

import ast
import logging
from functools import wraps
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch


def load_core(context, monkeypatch, saved, eager_break=lambda fn: fn):
    path = Path(__file__).resolve().parents[4] / "vllm_ascend/ops/gdn.py"
    tree = ast.parse(path.read_text())
    core = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "_forward_core")
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), core],
        type_ignores=[],
    )
    decode = Mock(side_effect=lambda **kwargs: kwargs["value"] + 20)
    monkeypatch.setattr(
        torch.ops._C_ascend,
        "npu_causal_conv1d_custom",
        lambda out, data, weights, **kwargs: out.copy_(data),
        raising=False,
    )
    monkeypatch.setattr(torch.ops._C_ascend, "npu_recurrent_gated_delta_rule", decode, raising=False)
    baseline = Mock(side_effect=lambda **kwargs: (kwargs["v"] + 10, kwargs["initial_state"] + 5))
    namespace = dict(
        torch=torch,
        eager_break_during_capture=eager_break,
        get_forward_context=lambda: context,
        GDNAttentionMetadata=SimpleNamespace,
        logging=logging,
        logger=logging.getLogger("gdn_contract"),
        get_pcp_group=lambda: SimpleNamespace(world_size=1),
        PAD_SLOT_ID=-1,
        HEAD_DIM=128,
        DeviceOperator=SimpleNamespace(fused_gdn_gating=lambda A, a, b, bias: (a.unsqueeze(0), b.unsqueeze(0))),
        l2norm_fwd=lambda tensor: tensor,
        clear_ssm_states=lambda state, flags: state.masked_fill_(~flags[:, None, None, None], 0),
        chunk_gated_delta_rule=baseline,
        maybe_save_kv_layer_to_connector=lambda *args: saved.append("saved"),
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace["_forward_core"], baseline, decode


def layer():
    def reshape(data):
        if data is None:
            return None, None, None
        shaped = data.reshape(1, data.shape[0], 1, 2)
        return shaped, shaped, shaped

    return SimpleNamespace(
        prefix="gdn",
        kv_cache=(torch.zeros(3, 1, 2), torch.arange(12).reshape(3, 1, 2, 2).float()),
        conv1d=SimpleNamespace(weight=torch.ones(2, 1, 2), bias=None),
        activation=None,
        A_log=torch.zeros(1),
        dt_bias=torch.zeros(1),
        rearrange_mixed_qkv=reshape,
    )


@pytest.mark.parametrize("topology_supported", [False, True])
def test_profile_prepares_from_local_tensor_geometry(monkeypatch, topology_supported):
    model = layer()
    model.num_v_heads, model.num_k_heads = 32, 16
    query = torch.empty(1, 3, 8, 128)
    value = torch.empty(1, 3, 16, 128)
    model.rearrange_mixed_qkv = Mock(return_value=(query, query, value))
    model.pto_gdn_backend = SimpleNamespace(topology_supported=topology_supported, prepare=Mock())
    core, baseline, decode = load_core(SimpleNamespace(attn_metadata=None), monkeypatch, [])
    core(model, torch.empty(3, 4096), torch.empty(3, 16), torch.empty(3, 16), torch.empty_like(value))
    if topology_supported:
        model.pto_gdn_backend.prepare.assert_called_once_with(query.device, 16, 8, 128)
    else:
        model.rearrange_mixed_qkv.assert_not_called()
        model.pto_gdn_backend.prepare.assert_not_called()
    baseline.assert_not_called()
    decode.assert_not_called()


@pytest.mark.parametrize("tp,pp,expected", [(1, 1, True), (2, 1, False), (4, 1, False), (1, 2, False)])
def test_production_constructor_keeps_megagdn_tp1_pp1_only(tp, pp, expected):
    path = Path(__file__).resolve().parents[4] / "vllm_ascend/ops/gdn.py"
    tree = ast.parse(path.read_text())
    cls = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "AscendGatedDeltaNetAttention"
    )
    cls.body = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "__init__"]

    class Base:
        def __init__(self):
            self.tp_size = tp
            self.prefix = "gdn"

    config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=pp, prefill_context_parallel_size=1, decode_context_parallel_size=1
        ),
        cache_config=SimpleNamespace(mamba_cache_mode="none"),
        speculative_config=None,
    )
    factory = Mock()
    namespace = dict(
        GatedDeltaNetAttention=Base,
        envs=SimpleNamespace(VLLM_ASCEND_PTO_CHUNK_GDN=True),
        get_current_vllm_config=lambda: config,
        MegaGDNBackend=factory,
    )
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), "exec"), namespace)
    namespace["AscendGatedDeltaNetAttention"]()
    factory.assert_called_once_with(topology_supported=expected, prefix="gdn")


@pytest.mark.parametrize("enable_mega", [False, True])
@pytest.mark.parametrize("stateful", [False, True])
def test_mixed_batch_only_sends_prefill_slice_and_rebased_state_to_backend(monkeypatch, enable_mega, stateful):
    model = layer()
    before = model.kv_cache[1].clone()
    saved = []
    conv = SimpleNamespace(
        query_start_loc=torch.tensor([0, 1, 4]),
        cache_indices=torch.tensor([2, 0]),
        initial_state_mode=torch.tensor([True, stateful]),
    )
    metadata = SimpleNamespace(
        spec_sequence_masks=None,
        spec_token_indx=None,
        non_spec_token_indx=None,
        spec_state_indices_tensor=None,
        non_spec_state_indices_tensor=torch.tensor([2, 0]),
        num_actual_tokens=4,
        num_decodes=1,
        num_prefills=1,
        num_decode_tokens=1,
        non_spec_prefill_metadata=SimpleNamespace(
            causal_conv1d=conv, chunk=SimpleNamespace(fresh_prefill=not stateful)
        ),
        non_spec_decode_metadata=SimpleNamespace(actual_seq_lengths=torch.tensor([0, 1])),
        prefill_query_start_loc=torch.tensor([0, 3]),
        prefill_state_indices=torch.tensor([0]),
        prefill_has_initial_state=torch.tensor([stateful]),
    )
    core, baseline, decode = load_core(SimpleNamespace(attn_metadata={"gdn": metadata}), monkeypatch, saved)
    if enable_mega:
        model.pto_gdn_backend = Mock(side_effect=lambda **kwargs: (kwargs["v"] + 10, kwargs["initial_state"] + 5))
    inputs = torch.arange(12).reshape(6, 2).float()
    output = torch.zeros(6, 1, 2)
    core(model, inputs, torch.zeros(6, 1), torch.zeros(6, 1), output)
    selected = model.pto_gdn_backend if enable_mega else baseline
    selected.assert_called_once()
    passed = selected.call_args.kwargs
    torch.testing.assert_close(passed["v"].squeeze(0).squeeze(1), inputs[1:4])
    initial_state = before[:1].transpose(-1, -2) if stateful else torch.zeros(1, 1, 2, 2)
    torch.testing.assert_close(passed["initial_state"], initial_state)
    torch.testing.assert_close(passed["cu_seqlens"], torch.tensor([0, 3]))
    if enable_mega:
        assert passed["fresh_prefill"] is not stateful
        baseline.assert_not_called()
    torch.testing.assert_close(decode.call_args.kwargs["ssm_state_indices"], torch.tensor([2]))
    torch.testing.assert_close(output[0, 0], inputs[0] + 20)
    torch.testing.assert_close(output[1:4, 0], inputs[1:4] + 10)
    torch.testing.assert_close(output[4:], torch.zeros_like(output[4:]))
    torch.testing.assert_close(model.kv_cache[1][:1], (initial_state + 5).transpose(-1, -2))
    torch.testing.assert_close(model.kv_cache[1][1:], before[1:])
    assert saved == ["saved"]


def test_pure_decode_does_not_call_megagdn(monkeypatch):
    model = layer()
    model.pto_gdn_backend = Mock(side_effect=AssertionError("decode must not use MegaGDN"))
    conv = SimpleNamespace(query_start_loc=torch.tensor([0, 1, 2]), cache_indices=torch.tensor([0, 2]))
    metadata = SimpleNamespace(
        spec_sequence_masks=None,
        spec_token_indx=None,
        non_spec_token_indx=None,
        spec_state_indices_tensor=None,
        non_spec_state_indices_tensor=torch.tensor([0, 2]),
        num_actual_tokens=2,
        num_decodes=2,
        num_prefills=0,
        num_decode_tokens=2,
        non_spec_decode_metadata=SimpleNamespace(causal_conv1d=conv, actual_seq_lengths=torch.tensor([0, 1, 1])),
    )
    core, baseline, decode = load_core(SimpleNamespace(attn_metadata={"gdn": metadata}), monkeypatch, [])
    inputs = torch.arange(4).reshape(2, 2).float()
    output = torch.zeros(2, 1, 2)
    core(model, inputs, torch.zeros(2, 1), torch.zeros(2, 1), output)
    torch.testing.assert_close(output[:, 0], inputs + 20)
    decode.assert_called_once()
    baseline.assert_not_called()
    model.pto_gdn_backend.assert_not_called()


def test_native_mixed_core_reuses_padded_buffers_across_arrivals_and_state_slots(monkeypatch):
    model = layer()
    model.kv_cache = (torch.zeros(70, 1, 2), torch.arange(280).reshape(70, 1, 2, 2).float())
    model.pto_gdn_backend = None
    context = SimpleNamespace(attn_metadata={})
    core, baseline, decode = load_core(context, monkeypatch, [])
    cases = [
        ([61], 0),
        ([17, 44], 0),
        ([1, 1, 59], 2),
        ([1] * 63 + [137], 63),
        ([1] * 63 + [65], 63),
        ([1, 767], 1),
        ([7], 0),
    ]
    for iteration, (lengths, num_decodes) in enumerate(cases):
        baseline.reset_mock()
        decode.reset_mock()
        actual = sum(lengths)
        capacity = next(size for size in (64, 128, 256, 768) if actual <= size)
        inputs = torch.arange(capacity * 2).reshape(capacity, 2).float()
        inputs[actual:] = float("nan")
        output = torch.zeros(capacity, 1, 2)
        slots = torch.arange(len(lengths) - 1, -1, -1) + iteration % 3
        flags = (torch.arange(len(lengths)) + iteration) % 2 == 0
        flags[:num_decodes] = True
        cu = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()])
        prefill_cu = cu[num_decodes:] - num_decodes
        metadata = SimpleNamespace(
            spec_sequence_masks=None,
            spec_token_indx=None,
            non_spec_token_indx=None,
            spec_state_indices_tensor=None,
            non_spec_state_indices_tensor=slots,
            num_actual_tokens=actual,
            num_decodes=num_decodes,
            num_prefills=len(lengths) - num_decodes,
            num_decode_tokens=num_decodes,
            non_spec_prefill_metadata=SimpleNamespace(
                causal_conv1d=SimpleNamespace(query_start_loc=cu, cache_indices=slots, initial_state_mode=flags),
                chunk=SimpleNamespace(fresh_prefill=not flags[num_decodes:].any()),
            ),
            non_spec_decode_metadata=SimpleNamespace(actual_seq_lengths=cu[: num_decodes + 1]),
            prefill_query_start_loc=prefill_cu,
            prefill_state_indices=slots[num_decodes:],
            prefill_has_initial_state=flags[num_decodes:],
        )
        context.attn_metadata["gdn"] = metadata
        before = model.kv_cache[1].clone()
        core(model, inputs, torch.zeros(capacity, 1), torch.zeros(capacity, 1), output)
        baseline.assert_called_once()
        call = baseline.call_args.kwargs
        assert call["prebuilt_meta"] is metadata.non_spec_prefill_metadata.chunk
        torch.testing.assert_close(call["cu_seqlens"], prefill_cu)
        torch.testing.assert_close(call["v"][0, :, 0], inputs[num_decodes:actual])
        initial = before[slots[num_decodes:]].transpose(-1, -2).contiguous()
        initial.masked_fill_(~flags[num_decodes:, None, None, None], 0)
        torch.testing.assert_close(call["initial_state"], initial)
        expected_state = before.clone()
        expected_state[slots[num_decodes:]] = (initial + 5).transpose(-1, -2)
        torch.testing.assert_close(model.kv_cache[1], expected_state)
        torch.testing.assert_close(output[num_decodes:actual, 0], inputs[num_decodes:actual] + 10)
        if num_decodes:
            torch.testing.assert_close(decode.call_args.kwargs["ssm_state_indices"], slots[:num_decodes])
            torch.testing.assert_close(output[:num_decodes, 0], inputs[:num_decodes] + 20)
        else:
            decode.assert_not_called()
        assert output[actual:].eq(0).all()


@pytest.mark.parametrize("upstream_break", [False, True])
def test_warmup_without_metadata_records_gdn_for_live_piecewise_replay(monkeypatch, upstream_break):
    # Model the upstream capture boundary contract: nested decorators do not
    # add another eager segment, and the callable runs again on each replay.
    # Device graph operations and NPU arithmetic are not emulated here.
    context = SimpleNamespace(attn_metadata=None)
    capturing = True
    segments = []

    def eager_break(fn):
        @wraps(fn)
        def call(*args, **kwargs):
            nonlocal capturing
            if not capturing:
                return fn(*args, **kwargs)
            capturing = False
            try:
                result = fn(*args, **kwargs)
                segments.append(lambda: fn(*args, **kwargs))
                return result
            finally:
                capturing = True

        return call

    model = layer()
    core, baseline, decode = load_core(context, monkeypatch, [], eager_break=eager_break)
    # Old upstream calls _forward_core directly; new upstream adds its own
    # decorator around the custom op. Both must record exactly one callback.
    dispatch = eager_break(core) if upstream_break else core
    inputs = torch.arange(128).reshape(64, 2).float()
    gates = torch.zeros(64, 1)
    output = torch.zeros(64, 1, 2)
    dispatch(model, inputs, gates, gates, output)
    assert len(segments) == 1, "GDN vanished from the warmup graph when metadata was None"
    baseline.assert_not_called()
    capturing = False
    for length, slot, stateful in [(61, 1, False), (17, 0, True), (44, 2, False)]:
        baseline.reset_mock()
        before = model.kv_cache[1].clone()
        flags = torch.tensor([stateful])
        cu = torch.tensor([0, length])
        metadata = SimpleNamespace(
            spec_sequence_masks=None,
            spec_token_indx=None,
            non_spec_token_indx=None,
            spec_state_indices_tensor=None,
            non_spec_state_indices_tensor=torch.tensor([slot]),
            num_actual_tokens=length,
            num_decodes=0,
            num_prefills=1,
            num_decode_tokens=0,
            non_spec_prefill_metadata=SimpleNamespace(
                causal_conv1d=SimpleNamespace(
                    query_start_loc=cu, cache_indices=torch.tensor([slot]), initial_state_mode=flags
                ),
                chunk=SimpleNamespace(fresh_prefill=not stateful),
            ),
            prefill_query_start_loc=cu,
            prefill_state_indices=torch.tensor([slot]),
            prefill_has_initial_state=flags,
        )
        context.attn_metadata = {"gdn": metadata}
        inputs.add_(1)
        output.zero_()  # The preceding captured segment resets this buffer.
        for replay in segments:
            replay()
        baseline.assert_called_once()
        assert baseline.call_args.kwargs["prebuilt_meta"] is metadata.non_spec_prefill_metadata.chunk
        torch.testing.assert_close(output[:length, 0], inputs[:length] + 10)
        assert output[length:].eq(0).all()
        initial = before[slot] if stateful else torch.zeros_like(before[slot])
        torch.testing.assert_close(model.kv_cache[1][slot], initial + 5)
    assert len(segments) == 1
    decode.assert_not_called()
