"""Real-weight prefill replay and unchanged decode acceptance on Ascend 910B."""

import pytest
import torch
from vllm import SamplingParams

from tests.e2e.conftest import VllmRunner, wait_until_npu_memory_free
from vllm_ascend.ops.triton.fla.prefill_state_writeback import write_fresh_states


def registry_status(worker):
    runner = worker.model_runner
    cache = runner.native_prefill_graph
    wrapper = runner.model
    entries = getattr(wrapper, "concrete_aclgraph_entries", None)
    if entries is None:
        entries = getattr(wrapper, "entries", {})
    return {
        "ready": cache.ready,
        "shapes": sorted((shape.requests, shape.width) for shape in cache.entries),
        "replays": sum(entry.replays for entry in cache.entries.values()),
        "standard_entries": len(entries),
    }


@wait_until_npu_memory_free()
def test_padded_prefill_preserves_standard_decode(monkeypatch):
    if not torch.npu.is_available() or "910B" not in torch.npu.get_device_name():
        pytest.skip("Requires Ascend 910B and real Qwen3.5-2B weights")
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "0")
    options = dict(
        dtype="bfloat16",
        max_model_len=256,
        max_num_seqs=4,
        max_num_batched_tokens=256,
        enable_prefix_caching=False,
        mamba_cache_mode="none",
        compilation_config={"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [1, 2, 4]},
    )
    prompts = [{"prompt_token_ids": [1] * length} for length in (1, 3, 17, 61, 129)]
    sampling = SamplingParams(temperature=0, max_tokens=8, ignore_eos=True)
    with VllmRunner("Qwen/Qwen3.5-2B", **options) as runner:
        expected = [
            runner.model.generate([prompt], sampling, use_tqdm=False)[0].outputs[0].token_ids for prompt in prompts
        ]
        expected_batch = runner.model.generate(prompts[:3], sampling, use_tqdm=False)
        expected_batch = [output.outputs[0].token_ids for output in expected_batch]
    with VllmRunner(
        "Qwen/Qwen3.5-2B",
        **options,
        additional_config={
            "native_prefill_graph": True,
            "native_prefill_graph_max_context": 128,
            "native_prefill_graph_request_counts": [1, 4],
        },
    ) as runner:
        before = runner.model.collective_rpc(registry_status)[0]
        assert before["ready"] and before["shapes"] and before["standard_entries"] > 0
        for _ in range(2):
            actual = [
                runner.model.generate([prompt], sampling, use_tqdm=False)[0].outputs[0].token_ids for prompt in prompts
            ]
            assert actual == expected
            actual_batch = runner.model.generate(prompts[:3], sampling, use_tqdm=False)
            assert [output.outputs[0].token_ids for output in actual_batch] == expected_batch
        after = runner.model.collective_rpc(registry_status)[0]
        assert after["shapes"] == before["shapes"]
        assert after["standard_entries"] == before["standard_entries"]
        assert after["replays"] > before["replays"]


def test_prefill_state_writeback_masks_padding_and_preserves_physical_page_gaps():
    if not torch.npu.is_available():
        pytest.skip("Requires NPU")
    source = torch.randn(4, 2, 16, 16, device="npu")
    backing = torch.randn(7, 2 * 16 * 16 + 64, device="npu")
    cache = backing[:, : 2 * 16 * 16].view(7, 2, 16, 16)
    slots = torch.full((4,), -1, device="npu", dtype=torch.int32)
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    before_capture = backing.clone()
    with torch.npu.stream(stream):
        write_fresh_states(source, cache, slots)
        stream.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, stream=stream):
            write_fresh_states(source, cache, slots)
    torch.npu.current_stream().wait_stream(stream)
    torch.testing.assert_close(backing, before_capture, rtol=0, atol=0)
    for indices in ([5, 2, -1, -1], [1, -1, -1, -1], [-1] * 4):
        expected = backing.clone()
        slots.copy_(torch.tensor(indices, dtype=torch.int32, device="npu"))
        source.add_(1)
        for row, slot in enumerate(indices):
            if slot >= 0:
                expected[slot, : 2 * 16 * 16] = source[row].reshape(-1)
        graph.replay()
        torch.testing.assert_close(backing, expected, rtol=0, atol=0)
