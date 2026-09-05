# SPDX-License-Identifier: Apache-2.0
"""CPU contract tests; these do not substitute for ACLGraph hardware tests."""

import importlib.util
import sys
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
            entry.capture = SimpleNamespace(num_graphs=1, num_eager_breaks=0)
            return "captured"

        def _replay(self, entry, args, kwargs):
            events.append("replay")

    modules = {
        "torch": dict(
            npu=SimpleNamespace(current_stream=lambda: SimpleNamespace(synchronize=lambda: events.append("sync")))
        ),
        "vllm.compilation.breakable_cudagraph": dict(BreakableCUDAGraphWrapper=Base),
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
