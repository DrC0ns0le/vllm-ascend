# SPDX-License-Identifier: Apache-2.0
"""NPU integration smoke through the actual registered Qwen GDN custom op.

Uses a synthetic recurrent core to isolate segmentation and metadata re-entry.
Real-weight serving and MegaGDN numerical validation are separate gates.
Run in a fresh process with VLLM_USE_BREAKABLE_CUDAGRAPH=1 before import.
"""

import os
from types import SimpleNamespace
from unittest.mock import patch

import torch


def main():
    if os.environ.get("VLLM_USE_BREAKABLE_CUDAGRAPH") != "1":
        raise RuntimeError("Export VLLM_USE_BREAKABLE_CUDAGRAPH=1 before starting this process")
    # Lazy imports keep import-time activation explicit for this NPU process.
    import torch_npu  # noqa: F401
    from vllm.compilation import breakable_cudagraph
    from vllm.config import CUDAGraphMode
    from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn

    from vllm_ascend.worker.model_runner_v1 import _torch_cuda_wrapper

    torch.npu.set_device(0)
    context = SimpleNamespace(cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE, lengths=(192,))

    class Core:
        def __init__(self):
            self.calls = 0

        def _forward_core(self, mixed_qkv, b, a, core_attn_out):
            self.calls += 1
            start = 0
            for index, length in enumerate(context.lengths):
                core_attn_out[start : start + length].copy_(mixed_qkv[start : start + length] + index + 1)
                start += length
            core_attn_out[start:].zero_()

    core = Core()
    context.no_compile_layers = {"test_gdn": core}
    x = torch.ones(192, 1, 128, device="npu", dtype=torch.bfloat16)
    output = torch.zeros_like(x)
    gates = torch.zeros(192, 1, device="npu", dtype=torch.bfloat16)
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with (
        _torch_cuda_wrapper(),
        torch.npu.stream(stream),
        patch.object(breakable_cudagraph, "get_forward_context", return_value=context),
        patch.object(breakable_cudagraph, "is_forward_context_available", return_value=True),
        patch.object(qwen_gdn_linear_attn, "get_forward_context", return_value=context),
    ):
        capture = breakable_cudagraph.BreakableCUDAGraphCapture()
        with capture:
            projected = x * 2
            torch.ops.vllm.qwen_gdn_attention_core(projected, gates, gates, output, "test_gdn", False)
            result = output + 3
        assert capture.num_eager_breaks == 1, "Qwen registration did not activate its eager break"
        assert capture.num_graphs == 2
        for lengths in ((192,), (96, 96), (64, 64, 64), (48, 48, 48, 48), (90, 90), (133,), (1,)):
            context.lengths = lengths
            previous = core.calls
            capture.replay()
            stream.synchronize()
            assert core.calls == previous + 1, "Replay did not re-enter the current recurrent metadata"
            expected = torch.full_like(result, 3)
            start = 0
            for index, length in enumerate(lengths):
                expected[start : start + length] = 6 + index
                start += length
            torch.testing.assert_close(result, expected, rtol=0, atol=0)

        context.cudagraph_runtime_mode = CUDAGraphMode.FULL
        context.lengths = (192,)
        full = breakable_cudagraph.BreakableCUDAGraphCapture()
        with full:
            torch.ops.vllm.qwen_gdn_attention_core(x, gates, gates, output, "test_gdn", False)
        assert full.num_eager_breaks == 0, "FULL decode must remain monolithic"
        previous = core.calls
        full.replay()
        stream.synchronize()
        assert core.calls == previous
    print("PASS: registered Qwen op re-enters metadata across 1/2/3/4-request PIECEWISE reuse; FULL has no eager break")


if __name__ == "__main__":
    main()
