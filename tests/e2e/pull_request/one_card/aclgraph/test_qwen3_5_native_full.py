# SPDX-License-Identifier: Apache-2.0
"""Real-weight startup/replay gate for native FULL, including padded request counts and c64 arrivals."""

import pytest
import torch
from vllm import SamplingParams

from tests.e2e.conftest import VllmRunner, wait_until_npu_memory_free
from tests.e2e.pull_request.one_card.aclgraph.test_qwen3_5_piecewise_buckets import run_arrivals


def registry_status(worker):
    wrapper = worker.model_runner.model
    cache = wrapper.native_full
    return dict(
        ready=cache.ready,
        entries=[(s.requests, s.width, s.fresh) for s in cache.entries],
        replays=sum(e.replays for e in cache.entries.values()),
        fresh_replays=sum(e.replays for s, e in cache.entries.items() if s.fresh),
        compatibility_steps=cache.compatibility_steps,
        decode_replays=sum(e.replays for s, e in cache.entries.items() if not s.fresh and s.width == 1),
        ordinary_entries=len(wrapper.entries),
        megagdn_scratch_bytes=cache.megagdn_workspace.reserved_bytes if cache.megagdn_workspace else 0,
        megagdn_scratch_geometries=len(cache.megagdn_workspace.buffers) if cache.megagdn_workspace else 0,
    )


@wait_until_npu_memory_free()
@pytest.mark.parametrize("backend", ["ascendc", "megagdn"])
def test_qwen3_5_fresh_full_capture_replay_and_mixed_arrivals(monkeypatch, capfd, backend):
    if not torch.npu.is_available() or "910B" not in torch.npu.get_device_name():
        pytest.skip("Requires Ascend 910B and Qwen3.5-2B weights")
    monkeypatch.setenv("VLLM_USE_BREAKABLE_CUDAGRAPH", "1")
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "0")
    monkeypatch.setenv("VLLM_ASCEND_PTO_CHUNK_GDN", "0")
    monkeypatch.setenv("VLLM_LOGGING_LEVEL", "DEBUG")
    options = dict(
        dtype="bfloat16",
        max_model_len=768,
        max_num_seqs=64,
        max_num_batched_tokens=8192,
        enable_chunked_prefill=False,
        enable_prefix_caching=False,
        mamba_cache_mode="none",
    )
    # Irregular single-request shapes plus concurrent arrivals. Running a second
    # wave with the same IDs exercises recycled cache rows, not just zero state.
    prompts = [{"prompt_token_ids": [1] * length} for length in (1, 2, 3, 61, 65, 129, 197, 319, 719)]
    sampling = SamplingParams(temperature=0, max_tokens=8, ignore_eos=True)
    with VllmRunner("Qwen/Qwen3.5-2B", enforce_eager=True, **options) as runner:
        expected = run_arrivals(runner.model)
        singles = [
            runner.model.generate([prompt], sampling, use_tqdm=False)[0].outputs[0].token_ids for prompt in prompts
        ]
    capfd.readouterr()
    with VllmRunner(
        "Qwen/Qwen3.5-2B",
        **options,
        additional_config={"native_full_graph_backend": backend, "native_full_graph_request_counts": [1, 4, 8]},
        compilation_config={"cudagraph_mode": "FULL", "cudagraph_capture_sizes": [1, 64, 128, 196, 384, 768]},
    ) as runner:
        startup = capfd.readouterr()
        assert "Native FULL registry sealed:" in startup.out + startup.err
        before = runner.model.collective_rpc(registry_status)[0]
        assert before["ready"] and before["entries"] and before["ordinary_entries"] == 0
        if backend == "megagdn":
            assert before["megagdn_scratch_bytes"] > 0 and before["megagdn_scratch_geometries"] == 1
        for _ in range(2):
            assert run_arrivals(runner.model) == expected
            actual = [
                runner.model.generate([prompt], sampling, use_tqdm=False)[0].outputs[0].token_ids for prompt in prompts
            ]
            assert actual == singles
        after = runner.model.collective_rpc(registry_status)[0]
        assert after["entries"] == before["entries"]
        assert after["ordinary_entries"] == 0
        assert after["megagdn_scratch_bytes"] == before["megagdn_scratch_bytes"]
        assert after["megagdn_scratch_geometries"] == before["megagdn_scratch_geometries"]
        assert all(after[name] > before[name] for name in ("replays", "fresh_replays", "decode_replays"))
        assert after["compatibility_steps"] == 0
        serving = capfd.readouterr()
        log = serving.out + serving.err
        assert "Native FULL replay hit:" in log
        assert "Breakable ACLGraph captured:" not in log
        assert "Breakable ACLGraph replay:" not in log
        assert "Qwen GDN step:" not in log, "GDN Python core executed during serving"


def test_fresh_state_writeback_capture_skips_dummy_and_recycled_slots():
    from vllm_ascend.ops.triton.fla.graph_state_writeback import write_fresh_states

    if not torch.npu.is_available():
        pytest.skip("Requires NPU")
    source = torch.randn(8, 2, 128, 128, device="npu")
    cache = torch.randn(11, 2, 128, 128, device="npu")
    slots = torch.full((8,), -1, device="npu", dtype=torch.int32)
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        write_fresh_states(source, cache, slots)
        stream.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, stream=stream):
            write_fresh_states(source, cache, slots)
    torch.npu.current_stream().wait_stream(stream)
    for indices in ([7, 2, 9, -1, -1, -1, -1, -1], [1, -1, -1, -1, -1, -1, -1, -1], [-1] * 8):
        before = cache.clone()
        slots.copy_(torch.tensor(indices, dtype=torch.int32))
        source.add_(1)
        graph.replay()
        expected = before
        for row, slot in enumerate(indices):
            if slot >= 0:
                expected[slot] = source[row]
        torch.testing.assert_close(cache, expected, rtol=0, atol=0)
