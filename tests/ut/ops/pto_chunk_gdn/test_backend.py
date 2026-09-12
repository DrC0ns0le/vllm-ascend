# SPDX-License-Identifier: Apache-2.0
import sys
from enum import Enum
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_ascend.ops.pto_chunk_gdn import backend
from vllm_ascend.ops.pto_chunk_gdn.eligibility import fallback_reason, total_chunks


class RuntimeMode(Enum):
    NONE = 0
    PIECEWISE = 1
    FULL = 2


@pytest.fixture
def runtime_context(monkeypatch):
    context = SimpleNamespace(
        cudagraph_runtime_mode=RuntimeMode.PIECEWISE,
        batch_descriptor=SimpleNamespace(num_tokens=64),
    )
    config = ModuleType("vllm.config")
    config.CUDAGraphMode = RuntimeMode
    forward = ModuleType("vllm.forward_context")
    forward.is_forward_context_available = Mock(return_value=True)
    forward.get_forward_context = Mock(return_value=context)
    monkeypatch.setitem(sys.modules, "vllm.config", config)
    monkeypatch.setitem(sys.modules, "vllm.forward_context", forward)
    return context, forward


@pytest.fixture
def npu_eligibility(monkeypatch):
    # Substitute only the CPU device type. Keep all production shape, topology,
    # freshness and runtime checks active while using a fake kernel below.
    def check(**kwargs):
        return fallback_reason(**(kwargs | {"device_type": "npu"}))

    monkeypatch.setattr(backend, "fallback_reason", check)


def eligible(**overrides):
    values = dict(
        device_type="npu",
        dtype="torch.bfloat16",
        q_shape=(1, 180, 8, 128),
        k_shape=(1, 180, 8, 128),
        v_shape=(1, 180, 16, 128),
        g_shape=(1, 180, 16),
        beta_shape=(1, 180, 16),
        cu_shape=(3,),
        cu_host=(0, 90, 180),
        fresh_prefill=True,
        topology_supported=True,
    )
    values.update(overrides)
    return fallback_reason(**values)


@pytest.mark.parametrize(
    "changes,reason",
    [
        ({}, None),
        ({"fresh_prefill": False}, "stateful_or_unknown_prefill"),
        ({"fresh_prefill": None}, "stateful_or_unknown_prefill"),
        ({"topology_supported": False}, "topology_or_cache_mode"),
        ({"device_type": "cpu"}, "device"),
        ({"dtype": "torch.float32"}, "dtype"),
        ({"head_first": True}, "layout"),
        ({"cu_host": None}, "missing_sequence_boundaries"),
        ({"cu_host": (0, 0, 180)}, "empty_or_invalid_sequence"),
        ({"cu_host": (0, 90, 181)}, "sequence_extent"),
        ({"k_shape": (1, 179, 8, 128)}, "shape"),
        ({"v_shape": (1, 180, 16, 64)}, "head_dimension"),
        ({"v_shape": (1, 180, 17, 128)}, "head_count"),
        ({"beta_shape": (1, 180, 8)}, "gate_shape"),
    ],
)
def test_eligibility(changes, reason):
    assert eligible(**changes) == reason


@pytest.mark.parametrize("count", [1, 2, 4, 8, 16, 32, 64])
def test_fresh_packed_sequence_counts_remain_eligible(count):
    total = 1024
    assert (
        eligible(
            q_shape=(1, total, 8, 128),
            k_shape=(1, total, 8, 128),
            v_shape=(1, total, 16, 128),
            g_shape=(1, total, 16),
            beta_shape=(1, total, 16),
            cu_shape=(count + 1,),
            cu_host=tuple(range(0, total + 1, total // count)),
        )
        is None
    )


@pytest.mark.parametrize(
    "boundaries,expected",
    [
        ((0, 127), 1),
        ((0, 128), 1),
        ((0, 129), 2),
        ((0, 90, 180), 2),
        ((0, 129, 258), 4),
        ((0, 50, 125, 250, 430), 5),
    ],
)
def test_chunk_counts_are_per_sequence(boundaries, expected):
    assert total_chunks(boundaries) == expected


def arguments():
    q = torch.ones(1, 3, 1, 128, dtype=torch.bfloat16)
    return dict(
        q=q,
        k=q.clone(),
        v=q.clone(),
        g=torch.zeros(1, 3, 1),
        beta=torch.ones(1, 3, 1),
        initial_state=torch.zeros(1, 1, 128, 128),
        output_final_state=True,
        cu_seqlens=torch.tensor([0, 3]),
        prebuilt_meta=SimpleNamespace(cu_seqlens_host=(0, 3)),
        head_first=False,
        use_qk_l2norm_in_kernel=False,
        fresh_prefill=True,
    )


def test_fallback_preserves_arguments_and_does_not_prepare_kernel(monkeypatch):
    instance = backend.MegaGDNBackend(topology_supported=True, prefix="test")
    instance.prepare = Mock(side_effect=AssertionError("must not compile on CPU"))
    args = arguments()
    sentinel = object()
    fallback = Mock(return_value=sentinel)
    assert instance(**args, fallback=fallback) is sentinel
    passed = fallback.call_args.kwargs
    for key, value in args.items():
        if key != "fresh_prefill":
            assert passed[key] is value
    assert instance.counts == {"fallback:device": 1}


def test_fast_path_preserves_state_dtype_and_uses_rebased_host_boundaries(monkeypatch, runtime_context):
    monkeypatch.setattr(backend, "fallback_reason", lambda **kwargs: None)
    instance = backend.MegaGDNBackend(topology_supported=True, prefix="test")
    args = arguments()
    instance.kernel = SimpleNamespace(run=Mock(return_value=(args["q"].half(), args["initial_state"].half())))
    result, state = instance(**args, fallback=Mock(side_effect=AssertionError))
    assert result.dtype == torch.bfloat16
    assert state.dtype == torch.float32
    assert instance.kernel.run.call_args.kwargs["cu_seqlens_host"] == (0, 3)
    assert instance.kernel.run.call_args.kwargs["scale"] == 128**-0.5
    assert instance.counts["megagdn"] == 1


def test_compile_failure_is_not_a_silent_fallback(monkeypatch, runtime_context):
    monkeypatch.setattr(backend, "fallback_reason", lambda **kwargs: None)
    instance = backend.MegaGDNBackend(topology_supported=True, prefix="test")
    instance.prepare = Mock(side_effect=RuntimeError("compile failed"))
    fallback = Mock()
    with pytest.raises(RuntimeError, match="compile failed"):
        instance(**arguments(), fallback=fallback)
    fallback.assert_not_called()


def test_packed_debug_uses_host_boundaries_and_current_graph_context(monkeypatch, runtime_context):
    monkeypatch.setattr(backend, "fallback_reason", lambda **kwargs: None)
    logger = Mock()
    logger.isEnabledFor.return_value = True
    monkeypatch.setattr(backend, "logger", logger)
    instance = backend.MegaGDNBackend(topology_supported=True, prefix="test")
    args = arguments()
    args["cu_seqlens"] = torch.tensor([0, 1, 3])
    args["initial_state"] = torch.zeros(2, 1, 128, 128)
    args["prebuilt_meta"].cu_seqlens_host = (0, 1, 3)
    instance.kernel = SimpleNamespace(run=Mock(return_value=(args["q"].half(), args["initial_state"].half())))
    instance(**args, fallback=Mock(side_effect=AssertionError))
    record = next(call.args for call in logger.debug.call_args_list if call.args[0].startswith("MegaGDN decision:"))
    assert record[1:] == ("test", "megagdn", "PIECEWISE", 64, 2, 3, 2, 1, 1, 128)
    logger.info.assert_not_called()


@pytest.mark.parametrize("count", [1, 2, 4, 8, 16, 32, 64])
def test_piecewise_fresh_packed_prefill_must_execute_megagdn(npu_eligibility, runtime_context, count):
    args = arguments()
    # Ragged sequences include partial chunks and lengths crossing a chunk.
    lengths = (1, 127, 128, 129)
    boundaries = [0]
    for index in range(count):
        boundaries.append(boundaries[-1] + lengths[index % len(lengths)])
    tokens = boundaries[-1]
    args.update(
        q=torch.ones(1, tokens, 1, 128, dtype=torch.bfloat16),
        k=torch.ones(1, tokens, 1, 128, dtype=torch.bfloat16),
        v=torch.ones(1, tokens, 2, 128, dtype=torch.bfloat16),
        g=torch.zeros(1, tokens, 2),
        beta=torch.ones(1, tokens, 2),
        initial_state=torch.zeros(count, 2, 128, 128),
        cu_seqlens=torch.tensor(boundaries),
        prebuilt_meta=SimpleNamespace(cu_seqlens_host=tuple(boundaries)),
    )
    instance = backend.MegaGDNBackend(topology_supported=True, prefix="test")
    instance.kernel = SimpleNamespace(run=Mock(return_value=(args["v"].half(), args["initial_state"].half())))
    instance.prepare = Mock()
    fallback = Mock(side_effect=AssertionError("eligible prefill must execute MegaGDN"))
    instance(**args, fallback=fallback)
    instance.prepare.assert_called_once_with(args["q"].device, 2, 1, 128)
    instance.kernel.run.assert_called_once()
    assert instance.kernel.run.call_args.kwargs["cu_seqlens_host"] == tuple(boundaries)
    assert instance.counts == {"megagdn": 1}
    fallback.assert_not_called()


@pytest.mark.parametrize(
    "mode,available,topology,fresh,reason",
    [
        (RuntimeMode.NONE, True, True, True, "runtime_not_piecewise"),
        (RuntimeMode.FULL, True, True, True, "runtime_not_piecewise"),
        (RuntimeMode.PIECEWISE, False, True, True, "runtime_not_piecewise"),
        (RuntimeMode.PIECEWISE, True, False, True, "topology_or_cache_mode"),
        (RuntimeMode.PIECEWISE, True, True, False, "stateful_or_unknown_prefill"),
    ],
)
def test_unsupported_runtime_never_prepares_or_runs_kernel(
    monkeypatch, npu_eligibility, runtime_context, mode, available, topology, fresh, reason
):
    logger = Mock()
    logger.isEnabledFor.return_value = True
    monkeypatch.setattr(backend, "logger", logger)
    context, forward = runtime_context
    context.cudagraph_runtime_mode = mode
    forward.is_forward_context_available.return_value = available
    if not available:
        forward.get_forward_context.side_effect = AssertionError("context unavailable")
    instance = backend.MegaGDNBackend(topology_supported=topology, prefix="test")
    instance.prepare = Mock(side_effect=AssertionError("must not compile"))
    instance.kernel = SimpleNamespace(run=Mock(side_effect=AssertionError("must not execute")))
    args = arguments()
    args["fresh_prefill"] = fresh
    fallback = Mock(return_value=object())
    result = instance(**args, fallback=fallback)
    assert result is fallback.return_value
    fallback.assert_called_once()
    instance.prepare.assert_not_called()
    instance.kernel.run.assert_not_called()
    assert instance.counts == {f"fallback:{reason}": 1}
    record = next(call.args for call in logger.debug.call_args_list if call.args[0].startswith("MegaGDN decision:"))
    assert record[1:] == (
        "test",
        reason,
        mode.name if available else None,
        64 if available else None,
        1,
        3,
        1,
        1,
        1,
        128,
    )
    logger.info.assert_not_called()


@pytest.mark.parametrize(
    "field,dtype,reason", [("k", torch.float32, "dtype_mismatch"), ("cu_seqlens", torch.float32, "sequence_dtype")]
)
def test_incompatible_tensor_dtype_falls_back_before_compile(monkeypatch, field, dtype, reason):
    monkeypatch.setattr(backend, "fallback_reason", lambda **kwargs: None)
    instance = backend.MegaGDNBackend(topology_supported=True, prefix="test")
    instance.prepare = Mock(side_effect=AssertionError("must not compile"))
    args = arguments()
    args[field] = args[field].to(dtype)
    fallback = Mock()
    instance(**args, fallback=fallback)
    assert instance.counts == {f"fallback:{reason}": 1}
    assert fallback.call_args.kwargs[field] is args[field]


def test_explicit_fresh_full_graph_runs_megagdn_without_fallback(npu_eligibility, runtime_context):
    context, _ = runtime_context
    context.cudagraph_runtime_mode = RuntimeMode.FULL
    args = arguments()
    instance = backend.MegaGDNBackend(topology_supported=True, prefix="graph")
    instance.kernel = SimpleNamespace(run=Mock(return_value=(args["q"].half(), args["initial_state"].half())))
    fallback = Mock(side_effect=AssertionError("explicit graph backend must not fall back"))
    workspace = object()
    output, state = instance(**args, native_graph=True, fallback=fallback, workspace=workspace)
    assert output.dtype == torch.bfloat16 and state.dtype == torch.float32
    instance.kernel.run.assert_called_once()
    assert instance.kernel.run.call_args.kwargs["workspace"] is workspace
    fallback.assert_not_called()
    args["fresh_prefill"] = False
    with pytest.raises(ValueError, match="stateful_or_unknown"):
        instance(**args, native_graph=True, fallback=fallback)
