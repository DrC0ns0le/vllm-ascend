# SPDX-License-Identifier: Apache-2.0
"""Check inference avoids materializing discarded native GDN intermediates."""

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch


@pytest.mark.parametrize("suppress_level", [0, 2, 3])
def test_native_chunk_preserves_output_without_copying_unused_intermediates(monkeypatch, suppress_level):
    path = Path(__file__).resolve().parents[4] / "vllm_ascend/ops/triton/fla/chunk.py"
    tree = ast.parse(path.read_text())
    fn = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "chunk_gated_delta_rule_fwd"
    )
    q = torch.arange(24, dtype=torch.bfloat16).reshape(1, 3, 2, 4)
    native_output = q.transpose(1, 2).contiguous()
    h = Mock(wraps=native_output)
    v_new = Mock(wraps=native_output)
    state = torch.arange(16).reshape(1, 2, 2, 4).float()
    h_op = Mock(return_value=(h, v_new, state))
    o_op = Mock(return_value=native_output)
    monkeypatch.setattr(torch.ops._C_ascend, "chunk_gated_delta_rule_fwd_h", h_op, raising=False)
    monkeypatch.setattr(torch.ops._C_ascend, "chunk_fwd_o", o_op, raising=False)
    namespace = dict(
        torch=torch,
        SUPPRESS_LEVEL=suppress_level,
        get_forward_context=lambda: SimpleNamespace(attn_metadata=None),
        get_pcp_group=lambda: SimpleNamespace(world_size=1),
        chunk_local_cumsum=lambda g, **kwargs: g,
        chunk_scaled_dot_kkt_fwd=lambda **kwargs: q,
        solve_tril=lambda **kwargs: q,
        recompute_w_u_fwd=lambda **kwargs: (q, q),
    )
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), "exec"), namespace)
    result = namespace[fn.name](q, q, q, q[..., 0], q[..., 0], 0.5, state, True)
    torch.testing.assert_close(result[1], q)
    assert result[3] is state
    assert o_op.call_args.args[2] is v_new
    assert o_op.call_args.args[3] is h
    if suppress_level < 3:
        h.to.assert_not_called()
        v_new.to.assert_not_called()
        assert result[4:] == (None, None, None)
    else:
        h.to.assert_called_once_with(torch.bfloat16)
        v_new.to.assert_called_once_with(torch.bfloat16)
        torch.testing.assert_close(result[5], q)
        torch.testing.assert_close(result[6], q)
