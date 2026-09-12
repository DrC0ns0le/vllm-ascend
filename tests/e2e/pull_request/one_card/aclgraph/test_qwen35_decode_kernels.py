# SPDX-License-Identifier: Apache-2.0
"""910B compile, numerical, and changed-input replay gates (no timing claims)."""

from unittest.mock import patch

import pytest
import torch
import torch_npu  # noqa: F401

import vllm_ascend.ops.qwen35_decode  # noqa: F401

# Stored [N,K] projection shapes, including the otherwise uncaptured lm_head.
SHAPES = (
    (12288, 2048),
    (2048, 6144),
    (8192, 2048),
    (32, 2048),
    (2048, 2048),
    (5120, 2048),
    (248320, 2048),
    (18432, 2560),
    (2560, 9216),
    (12288, 2560),
    (64, 2560),
    (2560, 4096),
    (10240, 2560),
    (248320, 2560),
)


@pytest.fixture(autouse=True)
def require_910b():
    if not torch.npu.is_available() or "910B" not in torch.npu.get_device_name():
        pytest.skip("Requires Ascend 910B and Triton-Ascend")


@pytest.fixture(params=[torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
def dtype(request):
    return request.param


def check_linear_replay(m, n, k, dtype, split_k=1, force_cube=False, has_bias=False):
    torch.manual_seed(123)
    x = torch.randn(m, k, device="npu", dtype=dtype) * 0.1
    weight = torch.randn(n, k, device="npu", dtype=dtype) * 0.02
    bias = torch.randn(n, device="npu", dtype=dtype) if has_bias else None
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        # JIT preparation must finish before graph capture.
        torch.ops.vllm.qwen35_decode_linear(x, weight, bias, split_k, force_cube)
        stream.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, stream=stream):
            output = torch.ops.vllm.qwen35_decode_linear(x, weight, bias, split_k, force_cube)
    torch.npu.current_stream().wait_stream(stream)
    for rows in (m, 1, m):
        x.normal_(std=0.1)
        x[rows:].zero_()  # Padded rows must not affect active rows.
        expected = torch.nn.functional.linear(x, weight, bias)
        output.fill_(float("nan"))
        # Replay must execute captured device work without re-entering Python.
        with patch("vllm_ascend.ops.triton.qwen35_decode_linear.decode_linear", side_effect=AssertionError("eager")):
            graph.replay()
        assert output.dtype == dtype
        torch.testing.assert_close(output, expected, rtol=1e-2 if dtype == torch.bfloat16 else 2e-3, atol=2e-3)


@pytest.mark.parametrize("n,k", SHAPES)
@pytest.mark.parametrize("m", [1, 2, 4, 8])
def test_model_linear_shapes_capture_replay(m, n, k, dtype):
    check_linear_replay(m, n, k, dtype)


@pytest.mark.parametrize("m", [1, 2, 4, 8])
@pytest.mark.parametrize("split_k", [1, 2, 4])
def test_cube_split_k_and_bias_capture_replay(m, split_k, dtype):
    check_linear_replay(m, 2560, 9216, dtype, split_k=split_k, force_cube=True, has_bias=True)


@pytest.mark.parametrize("split_k", [2, 4])
def test_vector_split_k_and_bias_capture_replay(split_k, dtype):
    check_linear_replay(1, 2048, 6144, dtype, split_k=split_k, has_bias=True)


@pytest.mark.parametrize("m", [1, 2, 4, 8])
@pytest.mark.parametrize("k,heads,qkv_size", [(2048, 16, 6144), (2560, 32, 8192)])
@pytest.mark.parametrize("has_bias", [False, True])
def test_gdn_ba_prepare_capture_replay(m, k, heads, qkv_size, has_bias, dtype):
    x = torch.randn(m, k, device="npu", dtype=dtype) * 0.1
    qkvz = torch.randn(m, qkv_size + heads * 128, device="npu", dtype=dtype)
    weight = torch.randn(2 * heads, k, device="npu", dtype=dtype) * 0.02
    bias = torch.randn(2 * heads, device="npu", dtype=dtype) if has_bias else None
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        torch.ops.vllm.qwen35_gdn_ba_prepare(x, qkvz, weight, bias, qkv_size, 128)
        stream.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, stream=stream):
            outputs = torch.ops.vllm.qwen35_gdn_ba_prepare(x, qkvz, weight, bias, qkv_size, 128)
    torch.npu.current_stream().wait_stream(stream)
    for rows in (m, 1, m):
        x.normal_(std=0.1)
        x[rows:].zero_()
        qkvz.normal_()
        expected_b, expected_a = torch.nn.functional.linear(x, weight, bias).chunk(2, -1)
        for tensor in outputs:
            tensor.fill_(float("nan"))
        with patch("vllm_ascend.ops.triton.qwen35_gdn_prepare.gdn_ba_prepare", side_effect=AssertionError("eager")):
            graph.replay()
        qkv, z, b, a = outputs
        assert all(t.is_contiguous() and t.dtype == dtype for t in outputs)
        torch.testing.assert_close(qkv, qkvz[:, :qkv_size], rtol=0, atol=0)
        torch.testing.assert_close(z.flatten(1), qkvz[:, qkv_size:], rtol=0, atol=0)
        rtol = 1e-2 if dtype == torch.bfloat16 else 2e-3
        torch.testing.assert_close(b, expected_b, rtol=rtol, atol=2e-3)
        torch.testing.assert_close(a, expected_a, rtol=rtol, atol=2e-3)
