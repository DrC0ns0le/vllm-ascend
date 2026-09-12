# SPDX-License-Identifier: Apache-2.0
"""Real-weight FULL_DECODE_ONLY wiring and replay coverage."""

import pytest
import torch
from vllm import SamplingParams

from tests.e2e.conftest import VllmRunner, wait_until_npu_memory_free


def inspect_and_track_decode(worker):
    runner = worker.model_runner
    wrapper = runner.model
    selected = [
        module.prefix
        for module in runner.get_model().modules()
        if getattr(module, "_ascend_decode_config", None) is not None
    ]
    captures = {
        desc.num_tokens: entry.capture.num_eager_breaks
        for desc, entry in wrapper.entries.items()
        if entry.capture is not None
    }
    wrapper._decode_kernel_test_replays = []
    original = wrapper._replay

    def replay(entry, args, kwargs):
        wrapper._decode_kernel_test_replays.append(entry.batch_descriptor.num_tokens)
        return original(entry, args, kwargs)

    wrapper._replay = replay
    return {"selected": selected, "captures": captures}


def replayed_buckets(worker):
    return worker.model_runner.model._decode_kernel_test_replays


@wait_until_npu_memory_free()
@pytest.mark.parametrize("model,layers,gdn_layers", [("Qwen/Qwen3.5-2B", 24, 18), ("Qwen/Qwen3.5-4B", 32, 24)])
@pytest.mark.parametrize("dtype", ["bfloat16", "float16"])
def test_qwen35_decode_model_capture_and_replay(monkeypatch, model, layers, gdn_layers, dtype):
    if not torch.npu.is_available() or "910B" not in torch.npu.get_device_name():
        pytest.skip("Requires Ascend 910B and Qwen3.5 weights")
    monkeypatch.setenv("VLLM_USE_BREAKABLE_CUDAGRAPH", "1")
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "0")
    options = dict(
        dtype=dtype,
        max_model_len=128,
        max_num_seqs=8,
        max_num_batched_tokens=512,
        enable_chunked_prefill=False,
        enable_prefix_caching=False,
        mamba_cache_mode="none",
        compilation_config={"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [1, 2, 4, 8, 64]},
    )
    sampling = SamplingParams(temperature=0, max_tokens=8, ignore_eos=True)
    prompts = [
        "Answer briefly: what is the capital of France?",
        "Calculate 12 plus 13.",
        "Write the next number: 2, 4, 6,",
        "Name one primary color.",
    ] * 2

    def generate(runner):
        return [
            [out.outputs[0].token_ids for out in runner.model.generate(prompts[:m], sampling, use_tqdm=False)]
            for m in (1, 2, 4, 8)
        ]

    with VllmRunner(model, **options) as runner:
        expected = generate(runner)
    with VllmRunner(
        model,
        **options,
        additional_config={
            "qwen35_decode": {
                "linear_backend": "triton",
                "fuse_gdn_ba_prepare": True,
                "linear_projections": ["gate_up_proj", "down_proj", "lm_head"],
            }
        },
    ) as runner:
        status = runner.model.collective_rpc(inspect_and_track_decode)[0]
        assert sum(p.endswith("gate_up_proj") for p in status["selected"]) == layers
        assert sum(p.endswith("down_proj") for p in status["selected"]) == layers
        assert sum(p.endswith("in_proj_ba") for p in status["selected"]) == gdn_layers
        assert sum(p.endswith("lm_head") for p in status["selected"]) == 1
        assert all(status["captures"].get(m) == 0 for m in (1, 2, 4, 8)), "Missing FULL capture or eager break"
        assert generate(runner) == expected
        replays = runner.model.collective_rpc(replayed_buckets)[0]
        assert all(m in replays for m in (1, 2, 4, 8)), "Serving did not replay the small decode buckets"
