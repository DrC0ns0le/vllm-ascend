# SPDX-License-Identifier: Apache-2.0
"""Real-weight Qwen3.5 startup/capture regression; requires Ascend 910B."""

import re

import pytest
import torch
from vllm import SamplingParams

from tests.e2e.conftest import VllmRunner, wait_until_npu_memory_free


@wait_until_npu_memory_free()
def test_qwen3_5_2b_full_startup_and_replay(monkeypatch, capfd):
    if not torch.npu.is_available() or "910B" not in torch.npu.get_device_name():
        pytest.skip("Native Qwen3.5 FULL capture requires Ascend 910B")
    monkeypatch.setenv("VLLM_USE_BREAKABLE_CUDAGRAPH", "1")
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "0")
    monkeypatch.setenv("VLLM_LOGGING_LEVEL", "DEBUG")
    with VllmRunner(
        "Qwen/Qwen3.5-2B",
        dtype="bfloat16",
        block_size=128,
        max_model_len=768,
        max_num_seqs=4,
        max_num_batched_tokens=256,
        enable_chunked_prefill=True,
        enable_prefix_caching=False,
        mamba_cache_mode="none",
        compilation_config={
            "cudagraph_mode": "FULL",
            "cudagraph_capture_sizes": [1, 4, 64, 128, 256],
        },
    ) as runner:
        # Construction must reach native prefill capture and finish startup,
        # not merely pass compatibility checks or silently select another mode.
        startup = capfd.readouterr()
        startup_log = startup.out + startup.err
        assert "FULL prefill graph captured:" in startup_log
        assert "FULL-only mixed graph capacities ready:" in startup_log
        captured_counts = re.findall(r"FULL prefill graph captured:.*entries=(\d+)", startup_log)
        assert captured_counts
        registry_size = int(captured_counts[-1])

        # These totals are not exact buckets; the longest prompt also needs
        # multiple prefill chunks. Repeat to exercise warmed replay/state reuse.
        prompts = [{"prompt_token_ids": [1] * length} for length in (17, 137, 319)]
        sampling = SamplingParams(temperature=0, max_tokens=8, ignore_eos=True)
        for _ in range(2):
            outputs = runner.model.generate(prompts, sampling, use_tqdm=False)
            assert len(outputs) == len(prompts)
            assert all(len(output.outputs[0].token_ids) == 8 for output in outputs)
        serving = capfd.readouterr()
        serving_log = serving.out + serving.err
        assert "FULL prefill graph captured:" not in serving_log
        replay_counts = re.findall(r"FULL prefill replay hit:.*sealed=True entries=(\d+)", serving_log)
        assert replay_counts, "Serving never replayed a warmed FULL prefill graph"
        assert all(int(count) == registry_size for count in replay_counts)
