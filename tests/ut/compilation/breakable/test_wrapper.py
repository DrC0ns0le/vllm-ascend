# SPDX-License-Identifier: Apache-2.0
"""CPU contract tests; these do not substitute for ACLGraph hardware tests."""

import importlib.util
import sys
from contextlib import nullcontext
from enum import Enum
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.fixture
def wrapper_module(monkeypatch):
    events = []

    class Mode(Enum):
        FULL = 1
        PIECEWISE = 2

    context = SimpleNamespace(cudagraph_runtime_mode=Mode.FULL, capturing=False)

    class Base:
        def __init__(self, **kwargs):
            pass

        def _capture(self, entry, args, kwargs):
            events.append(("capture", context.capturing))
            if kwargs.get("fail"):
                raise RuntimeError("operator failed")
            entry.capture = SimpleNamespace(num_graphs=1, num_eager_breaks=0, segments=kwargs.get("segments", []))
            return "captured"

        def _replay(self, entry, args, kwargs):
            events.append("replay")

    modules = {
        "vllm": dict(__path__=[]),
        "torch": dict(
            npu=SimpleNamespace(current_stream=lambda: SimpleNamespace(synchronize=lambda: events.append("sync")))
        ),
        "vllm.compilation.breakable_cudagraph": dict(BreakableCUDAGraphWrapper=Base),
        "vllm.envs": dict(VLLM_CUSTOM_SCOPES_FOR_PROFILING=False),
        "vllm.v1.utils": dict(record_function_or_nullcontext=lambda name: nullcontext()),
        "vllm.config": dict(CUDAGraphMode=Mode, VllmConfig=object),
        "vllm.forward_context": dict(get_forward_context=lambda: context),
        "vllm.logger": dict(logger=Mock()),
        "vllm_ascend.ascend_forward_context": dict(_EXTRA_CTX=SimpleNamespace(is_draft_model=False)),
        "vllm_ascend.compilation.acl_graph": dict(
            get_graph_params=lambda: "main",
            get_draft_graph_params=lambda: "draft",
            get_draft_graph_prefill_params=lambda: "prefill",
            weak_ref_workspaces=lambda p: events.append(p),
        ),
    }
    for name, attributes in modules.items():
        module = ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
    sys.modules["vllm"].envs = sys.modules["vllm.envs"]
    path = Path(__file__).resolve().parents[4] / "vllm_ascend/compilation/breakable_aclgraph.py"
    spec = importlib.util.spec_from_file_location("tested_breakable", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, context, Mode, events


def test_full_capture_preserves_workspace_and_caller_update_contract(wrapper_module):
    module, context, mode, events = wrapper_module
    wrapper = module.BreakableACLGraphWrapper(None, None)
    assert wrapper._capture(SimpleNamespace(batch_descriptor="4"), (), {}) == "captured"
    assert events == [("capture", True), "main", "draft", "prefill"]
    assert context.capturing is True


def test_piecewise_does_not_set_full_capture_flag(wrapper_module):
    module, context, mode, events = wrapper_module
    context.cudagraph_runtime_mode = mode.PIECEWISE
    module.BreakableACLGraphWrapper(None, None)._capture(SimpleNamespace(batch_descriptor="192"), (), {})
    assert events == [("capture", False)]


def test_capture_failure_restores_flag(wrapper_module):
    module, context, mode, events = wrapper_module
    with pytest.raises(RuntimeError, match="operator failed"):
        module.BreakableACLGraphWrapper(None, None)._capture(SimpleNamespace(batch_descriptor="4"), (), {"fail": True})
    assert context.capturing is False


@pytest.mark.parametrize(
    "full,enpu,expected", [(True, False, ["sync", "replay"]), (True, True, ["replay"]), (False, False, ["replay"])]
)
def test_replay_order_and_output_alias(wrapper_module, full, enpu, expected):
    module, context, mode, events = wrapper_module
    context.cudagraph_runtime_mode = mode.FULL if full else mode.PIECEWISE
    sentinel = object()
    result = module.BreakableACLGraphWrapper(None, None, enable_enpu=enpu)._replay(
        SimpleNamespace(
            output=sentinel, batch_descriptor="test", capture=SimpleNamespace(num_graphs=1, num_eager_breaks=0)
        ),
        (),
        {},
    )
    assert result is sentinel
    assert events == expected


@pytest.mark.parametrize("profiling", [False, True])
def test_segment_scopes_preserve_order_and_are_absent_from_normal_serving(wrapper_module, monkeypatch, profiling):
    from contextlib import contextmanager

    module, context, mode, events = wrapper_module
    context.cudagraph_runtime_mode = mode.PIECEWISE
    module.vllm_envs.VLLM_CUSTOM_SCOPES_FOR_PROFILING = profiling
    labels = []

    @contextmanager
    def record(name):
        labels.append(name)
        yield

    monkeypatch.setattr(module, "record_function_or_nullcontext", record)

    def replay():
        events.append("graph")

    def eager():
        events.append("eager")

    original = [replay, eager, replay]
    entry = SimpleNamespace(batch_descriptor=SimpleNamespace(num_tokens=128))
    wrapper = module.BreakableACLGraphWrapper(None, None)
    wrapper._capture(entry, (), {"segments": original})
    events.clear()
    for segment in entry.capture.segments:
        segment()
    assert events == ["graph", "eager", "graph"]
    if profiling:
        assert labels == [
            "ascend::PIECEWISE::tokens=128::segment=0::graph",
            "ascend::PIECEWISE::tokens=128::segment=1::eager",
            "ascend::PIECEWISE::tokens=128::segment=2::graph",
        ]
    else:
        assert entry.capture.segments is original
        assert labels == []


@pytest.mark.parametrize("enpu", [False, True])
def test_model_scopes_preserve_attention_update_order(enpu):
    import ast
    from functools import partial

    path = Path(__file__).resolve().parents[4] / "vllm_ascend/worker/model_runner_v1.py"
    tree = ast.parse(path.read_text())
    method = next(
        node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "_model_forward"
    )
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), method],
        type_ignores=[],
    )
    context = SimpleNamespace(flash_comm_v1_enabled=False)
    namespace = dict(
        partial=partial,
        get_forward_context=lambda: context,
        record_function_or_nullcontext=lambda name: nullcontext(),
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    events = []
    output = object()

    def run(**kwargs):
        events.append("model")
        return output

    runner = SimpleNamespace(
        model=run,
        enable_enpu=enpu,
        _update_full_graph_params_if_needed=lambda *args: events.append("update"),
    )
    assert namespace[method.name](runner, 64) is output
    assert events == (["update", "model"] if enpu else ["model", "update"])
