# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_ascend.ops.pto_chunk_gdn import backend
from vllm_ascend.ops.pto_chunk_gdn.eligibility import fallback_reason, total_chunks


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


def test_fast_path_preserves_state_dtype_and_uses_rebased_host_boundaries(monkeypatch):
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


def test_compile_failure_is_not_a_silent_fallback(monkeypatch):
    monkeypatch.setattr(backend, "fallback_reason", lambda **kwargs: None)
    instance = backend.MegaGDNBackend(topology_supported=True, prefix="test")
    instance.prepare = Mock(side_effect=RuntimeError("compile failed"))
    fallback = Mock()
    with pytest.raises(RuntimeError, match="compile failed"):
        instance(**arguments(), fallback=fallback)
    fallback.assert_not_called()


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
