# SPDX-License-Identifier: Apache-2.0
"""Exercise the production GDN core's slicing and state writeback on CPU.

Only device operators are faked; the mixed/decode/prefill control flow is the
actual method from gdn.py. NPU arithmetic is covered by numerical.py on hardware.
"""

import ast
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch


def load_core(context, monkeypatch, saved):
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
        get_forward_context=lambda: context,
        GDNAttentionMetadata=SimpleNamespace,
        logging=logging,
        logger=logging.getLogger("gdn_contract"),
        get_pcp_group=lambda: SimpleNamespace(world_size=1),
        PAD_SLOT_ID=-1,
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


@pytest.mark.parametrize("enable_mega", [False, True])
def test_mixed_batch_only_sends_prefill_slice_and_rebased_state_to_backend(monkeypatch, enable_mega):
    model = layer()
    before = model.kv_cache[1].clone()
    saved = []
    conv = SimpleNamespace(
        query_start_loc=torch.tensor([0, 1, 4]),
        cache_indices=torch.tensor([2, 0]),
        initial_state_mode=torch.tensor([True, False]),
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
        non_spec_prefill_metadata=SimpleNamespace(causal_conv1d=conv, chunk=SimpleNamespace(fresh_prefill=True)),
        non_spec_decode_metadata=SimpleNamespace(actual_seq_lengths=torch.tensor([0, 1])),
        prefill_query_start_loc=torch.tensor([0, 3]),
        prefill_state_indices=torch.tensor([0]),
        prefill_has_initial_state=torch.tensor([False]),
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
    torch.testing.assert_close(passed["initial_state"], torch.zeros(1, 1, 2, 2))
    torch.testing.assert_close(passed["cu_seqlens"], torch.tensor([0, 3]))
    if enable_mega:
        assert passed["fresh_prefill"] is True
        baseline.assert_not_called()
    torch.testing.assert_close(decode.call_args.kwargs["ssm_state_indices"], torch.tensor([2]))
    torch.testing.assert_close(output[0, 0], inputs[0] + 20)
    torch.testing.assert_close(output[1:4, 0], inputs[1:4] + 10)
    torch.testing.assert_close(output[4:], torch.zeros_like(output[4:]))
    torch.testing.assert_close(model.kv_cache[1][0], torch.full((1, 2, 2), 5.0))
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
