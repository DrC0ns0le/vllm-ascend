# SPDX-License-Identifier: Apache-2.0
"""Exercise the actual MRV1 routing method with an isolated dispatcher."""

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch


@pytest.mark.parametrize(
    "computed,scheduled,breakable,expected",
    [
        ([0], [1], True, False),
        ([100], [1], True, True),
        ([0, 0], [90, 90], True, False),
        ([10, 20, 30, 40], [1, 1, 1, 1], True, True),
        ([10, 20, 30, 0], [1, 1, 1, 120], True, False),
        ([10, 0], [1, 1], True, False),
        ([0], [1], False, True),
    ],
)
def test_only_live_decodes_use_uniform_full_descriptor(computed, scheduled, breakable, expected):
    root = Path(__file__).resolve().parents[4]
    tree = ast.parse((root / "vllm_ascend/worker/model_runner_v1.py").read_text())
    method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_determine_batch_execution_and_padding"
    )
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), method],
        type_ignores=[],
    )
    captured = []

    def dispatch(**kwargs):
        captured.append(kwargs)
        return "FULL" if kwargs["uniform_decode"] else "PIECEWISE", SimpleNamespace(num_tokens=kwargs["num_tokens"])

    namespace = dict(
        np=SimpleNamespace(all=lambda tensor: bool(torch.all(tensor))),
        breakable_cudagraph=SimpleNamespace(is_breakable_cudagraph_enabled=lambda: breakable),
        enable_sp=lambda config: False,
        logger=Mock(),
    )
    exec(compile(ast.fix_missing_locations(module), "model_runner_v1.py", "exec"), namespace)
    runner = SimpleNamespace(
        _pad_for_sequence_parallelism=lambda n: n,
        input_batch=SimpleNamespace(num_computed_tokens_cpu=torch.tensor(computed), lora_id_to_lora_request={}),
        speculative_config=None,
        uniform_decode_query_len=1,
        model_config=SimpleNamespace(is_encoder_decoder=False),
        cudagraph_dispatcher=SimpleNamespace(dispatch=dispatch),
        vllm_config=SimpleNamespace(
            parallel_config=SimpleNamespace(data_parallel_size=1),
            observability_config=SimpleNamespace(cudagraph_metrics=False),
        ),
    )
    namespace[method.name](
        runner,
        num_tokens=sum(scheduled),
        num_reqs=len(scheduled),
        num_scheduled_tokens_np=scheduled,
        max_num_scheduled_tokens=max(scheduled),
        use_cascade_attn=False,
    )
    assert bool(captured[0]["uniform_decode"]) is expected
