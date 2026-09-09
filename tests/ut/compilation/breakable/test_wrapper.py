# SPDX-License-Identifier: Apache-2.0
"""CPU contract tests; these do not substitute for ACLGraph hardware tests."""

import ast
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
        NONE = 0
        FULL = 1
        PIECEWISE = 2

    context = SimpleNamespace(cudagraph_runtime_mode=Mode.FULL, capturing=False)
    monitor = ModuleType("vllm.compilation.monitor")
    monitor.cudagraph_capturing_enabled = True
    monitor.set_cudagraph_capturing_enabled = Mock(
        side_effect=lambda enabled: setattr(monitor, "cudagraph_capturing_enabled", enabled)
    )

    class Base:
        def __init__(self, **kwargs):
            self.runnable = kwargs["runnable"]

        def __call__(self, *args, **kwargs):
            events.append("existing path")
            return "existing result"

        def _capture(self, entry, args, kwargs):
            if not monitor.cudagraph_capturing_enabled:
                raise RuntimeError("capture disabled")
            events.append(("capture", context.capturing))
            if kwargs.get("fail"):
                raise RuntimeError("operator failed")
            entry.capture = SimpleNamespace(num_graphs=1, num_eager_breaks=0)
            return "captured"

        def _replay(self, entry, args, kwargs):
            events.append("replay")

    modules = {
        "vllm.compilation": dict(monitor=monitor),
        "vllm.compilation.monitor": vars(monitor),
        "torch": dict(
            npu=SimpleNamespace(current_stream=lambda: SimpleNamespace(synchronize=lambda: events.append("sync")))
        ),
        "vllm.compilation.breakable_cudagraph": dict(BreakableCUDAGraphWrapper=Base),
        "vllm.config": dict(CUDAGraphMode=Mode, VllmConfig=object),
        "vllm.forward_context": dict(get_forward_context=lambda: context, is_forward_context_available=lambda: True),
        "vllm.logger": dict(logger=Mock()),
        "vllm_ascend.ascend_forward_context": dict(_EXTRA_CTX=SimpleNamespace(is_draft_model=False)),
        "vllm_ascend.compilation.acl_graph": dict(
            get_graph_params=lambda: "main",
            get_draft_graph_params=lambda: "draft",
            get_draft_graph_prefill_params=lambda: "prefill",
            weak_ref_workspaces=lambda p: events.append(p),
        ),
        "vllm_ascend.compilation.full_prefill": dict(
            FullPrefillGraphCache=lambda config: Mock(full_only=False, sealed=False)
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


def test_full_prefill_uses_one_capture_without_decode_task_updates(wrapper_module):
    module, context, mode, events = wrapper_module
    context.full_prefill_graph = True
    entry = SimpleNamespace(batch_descriptor="256", output="result")
    wrapper = module.BreakableACLGraphWrapper(None, None)
    wrapper._capture(entry, (), {})
    assert entry.capture.num_graphs == 1
    assert entry.capture.num_eager_breaks == 0
    assert wrapper._replay(entry, (), {}) == "result"
    assert events == [("capture", False), "replay"]
    assert context.capturing is False


def test_cached_prefill_captures_attention_tasks_without_touching_draft_workspaces(wrapper_module):
    module, context, mode, events = wrapper_module
    context.full_prefill_graph = True
    context.full_prefill_graph_params = object()
    entry = SimpleNamespace(batch_descriptor="256", output="result")
    wrapper = module.BreakableACLGraphWrapper(None, None)
    wrapper._capture(entry, (), {})
    assert context.capturing
    assert wrapper._replay(entry, (), {}) == "result"
    # FullPrefillGraphCache owns synchronization and updates for this path.
    assert events == [("capture", True), "main", "replay"]


@pytest.mark.parametrize("handled", [False, True])
def test_prefill_cache_dispatches_before_existing_wrapper(wrapper_module, handled):
    module, context, mode, events = wrapper_module
    context.cudagraph_runtime_mode = mode.PIECEWISE
    wrapper = module.BreakableACLGraphWrapper(Mock(), None)
    wrapper.full_prefill.run.return_value = handled, "prefill result"
    assert wrapper() == ("prefill result" if handled else "existing result")
    wrapper.full_prefill.run.assert_called_once()
    assert events == ([] if handled else ["existing path"])


@pytest.mark.parametrize("mode_name,uniform", [("PIECEWISE", False), ("FULL", False)])
def test_full_only_never_enters_existing_fallback(wrapper_module, mode_name, uniform):
    module, context, mode, events = wrapper_module
    context.cudagraph_runtime_mode = mode[mode_name]
    context.attn_metadata = {"gdn": type("GDNAttentionMetadata", (), {})()}
    context.batch_descriptor = SimpleNamespace(uniform=uniform)
    wrapper = module.BreakableACLGraphWrapper(Mock(), None)
    wrapper.full_prefill.full_only = True
    wrapper.full_prefill.sealed = True
    wrapper.full_prefill.run.return_value = False, None
    with pytest.raises(RuntimeError, match="cannot fall back"):
        wrapper()
    assert not events


def test_full_only_retains_standard_uniform_decode(wrapper_module):
    module, context, _, events = wrapper_module
    context.attn_metadata = {"gdn": object()}
    context.batch_descriptor = SimpleNamespace(uniform=True)
    wrapper = module.BreakableACLGraphWrapper(Mock(), None)
    wrapper.full_prefill.full_only = True
    wrapper.full_prefill.run.return_value = False, None
    assert wrapper() == "existing result"
    assert events == ["existing path"]


@pytest.mark.parametrize("reuse", [False, True])
def test_outer_acl_wrapper_reuses_native_registry_before_ordinary_dispatch(wrapper_module, monkeypatch, reuse):
    module, context, mode, events = wrapper_module
    inner = module.BreakableACLGraphWrapper(Mock(), None)
    inner.full_prefill.run.return_value = True, "native replay"
    registry = inner.full_prefill
    path = Path(__file__).resolve().parents[4] / "vllm_ascend/compilation/acl_graph.py"
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ACLGraphWrapper")
    namespace = dict(
        get_forward_context=Mock(side_effect=AssertionError("ordinary ACL path was entered")),
        CUDAGraphMode=mode,
        envs=SimpleNamespace(VLLM_LOGGING_LEVEL="INFO"),
        current_platform=SimpleNamespace(get_global_graph_pool=lambda: None),
        CUDAGraphOptions=SimpleNamespace,
        check_gdn_layer=lambda config: True,
        _acl_graph_wrappers=set(),
    )
    monkeypatch.setitem(sys.modules, "vllm_ascend.compilation.breakable_aclgraph", module)
    definitions = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), cls]
    exec(
        compile(ast.fix_missing_locations(ast.Module(body=definitions, type_ignores=[])), str(path), "exec"), namespace
    )
    config = SimpleNamespace(compilation_config=SimpleNamespace(cudagraph_mode=mode.FULL))
    outer = namespace["ACLGraphWrapper"](inner if reuse else inner.runnable, config, runtime_mode=mode.FULL)
    if reuse:
        assert outer._full_prefill_wrapper is inner and outer.full_prefill is registry
    else:
        inner = outer._full_prefill_wrapper
        registry = inner.full_prefill
        registry.run.return_value = True, "native replay"
    assert outer("input") == "native replay"
    assert inner.full_prefill is registry
    registry.run.assert_called_once_with(
        context, ("input",), {}, runnable=inner.runnable, capture=inner._capture, replay=inner._replay
    )
    inner.runnable.assert_not_called()
    assert events == []


def test_capture_failure_restores_flag(wrapper_module):
    module, context, mode, events = wrapper_module
    with pytest.raises(RuntimeError, match="operator failed"):
        module.BreakableACLGraphWrapper(None, None)._capture(SimpleNamespace(batch_descriptor="4"), (), {"fail": True})
    assert context.capturing is False


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_lazy_admission_restores_capture_monitor(wrapper_module, enabled, fail):
    module, context, _, _ = wrapper_module
    monitor = module.cudagraph_monitor
    monitor.cudagraph_capturing_enabled = enabled
    context.full_prefill_graph = True
    context.full_prefill_graph_params = object()
    wrapper = module.BreakableACLGraphWrapper(None, None)
    entry = SimpleNamespace(batch_descriptor="256")
    if fail:
        with pytest.raises(RuntimeError, match="operator failed"):
            wrapper._capture(entry, (), {"fail": True})
        assert not context.capturing
    else:
        wrapper._capture(entry, (), {})
    assert monitor.cudagraph_capturing_enabled is enabled
    assert [call.args for call in monitor.set_cudagraph_capturing_enabled.call_args_list] == [(True,), (enabled,)]


@pytest.mark.parametrize("mode_name", ["FULL", "PIECEWISE"])
def test_ordinary_capture_cannot_bypass_disabled_monitor(wrapper_module, mode_name):
    module, context, mode, _ = wrapper_module
    monitor = module.cudagraph_monitor
    monitor.cudagraph_capturing_enabled = False
    context.cudagraph_runtime_mode = getattr(mode, mode_name)
    with pytest.raises(RuntimeError, match="capture disabled"):
        module.BreakableACLGraphWrapper(None, None)._capture(SimpleNamespace(batch_descriptor="256"), (), {})
    monitor.set_cudagraph_capturing_enabled.assert_not_called()
    assert not monitor.cudagraph_capturing_enabled


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
