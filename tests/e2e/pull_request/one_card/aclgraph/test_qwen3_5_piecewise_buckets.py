# SPDX-License-Identifier: Apache-2.0
"""Real-weight native GDN regression: padded buckets, chunking and arrivals."""

import pytest
import torch
from vllm import SamplingParams

from tests.e2e.conftest import VllmRunner, wait_until_npu_memory_free
from vllm_ascend.ops.gdn_attn_builder import _upload_chunk_metadata


def test_packed_gdn_metadata_views_on_npu():
    if not torch.npu.is_available():
        pytest.skip("Requires NPU")
    device = torch.device("npu")
    expected = [
        torch.tensor([[0, 0], [1, 0], [1, 1]], dtype=torch.int32),
        torch.tensor([0, 1, 3], dtype=torch.int64),
        torch.tensor([True, False, True], dtype=torch.bool),
    ]
    first = _upload_chunk_metadata(expected, device)
    # Submit another step before reading the first back. Its tables must not
    # overwrite or release storage still owned by the first metadata object.
    second = _upload_chunk_metadata([table.flip(0) for table in expected], device)
    for actual, reference in zip(first, expected):
        torch.testing.assert_close(actual.cpu(), reference)
    for actual, reference in zip(second, expected):
        torch.testing.assert_close(actual.cpu(), reference.flip(0))


def run_arrivals(llm):
    engine = llm.llm_engine
    prompts = [{"prompt_token_ids": [1] * length} for length in ([319, 719] + list(range(17, 79)))]
    sampling = SamplingParams(temperature=0, max_tokens=8, ignore_eos=True)
    continuing = SamplingParams(temperature=0, max_tokens=64, ignore_eos=True)
    finished = {}

    def collect():
        outputs = engine.step()
        for output in outputs:
            if output.finished:
                finished[output.request_id] = output.outputs[0].token_ids
        return outputs

    for i in range(32):
        # The longest prompt leaves room for eight output tokens; the other
        # initial requests remain decoding while the second wave arrives.
        engine.add_request(str(i), prompts[i], sampling if i == 1 else continuing)
    for _ in range(256):
        outputs = collect()
        if any(not output.finished for output in outputs):
            break
    else:
        pytest.fail("No ongoing decode observed before injecting arrivals")
    assert engine.has_unfinished_requests()
    for i in range(32, len(prompts)):
        engine.add_request(str(i), prompts[i], sampling)
    for _ in range(2048):
        if not engine.has_unfinished_requests():
            break
        collect()
    assert len(finished) == len(prompts)
    return finished


@wait_until_npu_memory_free()
def test_qwen3_5_piecewise_warmup_padding_and_mixed_arrivals(monkeypatch, capfd):
    if not torch.npu.is_available() or "910B" not in torch.npu.get_device_name():
        pytest.skip("Qwen3.5 native graph validation requires Ascend 910B")
    monkeypatch.setenv("VLLM_USE_BREAKABLE_CUDAGRAPH", "1")
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "0")
    monkeypatch.setenv("VLLM_ASCEND_PTO_CHUNK_GDN", "0")
    monkeypatch.setenv("VLLM_LOGGING_LEVEL", "DEBUG")
    options = dict(
        dtype="bfloat16",
        max_model_len=768,
        max_num_seqs=64,
        max_num_batched_tokens=256,
        enable_chunked_prefill=True,
        enable_prefix_caching=False,
        mamba_cache_mode="none",
    )
    with VllmRunner("Qwen/Qwen3.5-2B", enforce_eager=True, **options) as runner:
        expected = run_arrivals(runner.model)
    capfd.readouterr()
    with VllmRunner(
        "Qwen/Qwen3.5-2B",
        **options,
        compilation_config={"cudagraph_mode": "FULL_AND_PIECEWISE", "cudagraph_capture_sizes": [1, 64, 128, 196]},
    ) as runner:
        startup = capfd.readouterr()
        assert "Piecewise ACLGraph token buckets ready: [1, 64, 128, 196, 256]" in startup.out + startup.err
        for _ in range(2):
            assert run_arrivals(runner.model) == expected
        serving = capfd.readouterr()
        log = serving.out + serving.err
        assert "Breakable ACLGraph captured:" not in log, "Serving captured a new graph instead of replaying warmup"
        assert "Breakable ACLGraph replay: mode=CUDAGraphMode.PIECEWISE" in log
        assert "GDN step:" in log, "Serving bypassed the native GDN path"
