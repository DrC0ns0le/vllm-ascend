# SPDX-License-Identifier: Apache-2.0
"""External 910B gate for Qwen3.5 MTP=3 capture, rejection and mixed arrivals."""

import pytest
import torch
from vllm import SamplingParams

from tests.e2e.conftest import VllmRunner, wait_until_npu_memory_free


def mtp_registry_status(worker):
    runner = worker.model_runner
    target, draft = runner.model.native_full, runner.drafter.native_mtp
    return {
        name: dict(
            ready=cache.ready,
            entries=sorted((s.requests, s.width, s.fresh) for s in cache.entries),
            fresh_replays=sum(e.replays for s, e in cache.entries.items() if s.fresh),
            verify_replays=sum(e.replays for s, e in cache.entries.items() if not s.fresh and s.width == 4),
            compatibility_steps=cache.compatibility_steps,
        )
        for name, cache in (("target", target), ("draft", draft))
    }


def mtp_arrivals(llm):
    engine = llm.llm_engine
    finished = {}

    def step():
        for output in engine.step():
            if output.finished:
                finished[output.request_id] = output.outputs[0].token_ids

    for index, length in enumerate([1, 3, 61, 129]):
        engine.add_request(
            str(index),
            {"prompt_token_ids": [1] * length},
            SamplingParams(temperature=0, max_tokens=24, ignore_eos=True),
        )
    step()
    step()
    assert engine.has_unfinished_requests()
    for index, (length, output) in enumerate(zip([2, 65, 193, 319], [1, 2, 5, 8]), start=4):
        engine.add_request(
            str(index),
            {"prompt_token_ids": [1] * length},
            SamplingParams(temperature=0, max_tokens=output, ignore_eos=True),
        )
    for _ in range(256):
        if not engine.has_unfinished_requests():
            break
        step()
    assert len(finished) == 8
    return finished


@wait_until_npu_memory_free()
@pytest.mark.parametrize("model", ["Qwen/Qwen3.5-2B", "Qwen/Qwen3.5-4B"])
@pytest.mark.parametrize("backend", ["ascendc", "megagdn"])
def test_qwen3_5_native_mtp_three_startup_replay_and_accuracy(monkeypatch, capfd, model, backend):
    if not torch.npu.is_available() or "910B" not in torch.npu.get_device_name():
        pytest.skip("Requires Ascend 910B and Qwen3.5 weights including the MTP head")
    monkeypatch.setenv("VLLM_USE_BREAKABLE_CUDAGRAPH", "1")
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "0")
    monkeypatch.setenv("VLLM_ASCEND_PTO_CHUNK_GDN", "0")
    monkeypatch.setenv("VLLM_LOGGING_LEVEL", "DEBUG")
    options = dict(
        dtype="bfloat16",
        max_model_len=768,
        max_num_seqs=8,
        max_num_batched_tokens=8192,
        enable_chunked_prefill=False,
        enable_prefix_caching=False,
        mamba_cache_mode="none",
    )
    # Compare final generation with the target alone and the existing eager
    # speculative path; separate kernel tests force every acceptance length.
    with VllmRunner(model, enforce_eager=True, **options) as runner:
        expected = mtp_arrivals(runner.model)
        stop_prompt = {"prompt_token_ids": [1]}
        stop_sampling = SamplingParams(temperature=0, max_tokens=16, stop_token_ids=[expected["0"][0]])
        expected_stop = runner.model.generate([stop_prompt], stop_sampling, use_tqdm=False)[0].outputs[0].token_ids
    speculation = dict(method="mtp", num_speculative_tokens=3)
    with VllmRunner(model, enforce_eager=True, speculative_config=speculation, **options) as runner:
        assert mtp_arrivals(runner.model) == expected
    capfd.readouterr()
    with VllmRunner(
        model,
        **options,
        speculative_config=speculation,
        additional_config={"native_full_graph_backend": backend, "native_full_graph_request_counts": [1, 2, 4, 8]},
        compilation_config={"cudagraph_mode": "FULL", "cudagraph_capture_sizes": [1, 64, 128, 196, 384, 768]},
    ) as runner:
        startup = capfd.readouterr()
        assert "Native MTP registry sealed:" in startup.out + startup.err
        before = runner.model.collective_rpc(mtp_registry_status)[0]
        assert all(cache["ready"] for cache in before.values())
        assert all(any(width == 4 and not fresh for _, width, fresh in c["entries"]) for c in before.values())
        for _ in range(2):
            assert mtp_arrivals(runner.model) == expected
            assert (
                runner.model.generate([stop_prompt], stop_sampling, use_tqdm=False)[0].outputs[0].token_ids
                == expected_stop
            )
        after = runner.model.collective_rpc(mtp_registry_status)[0]
        for name in before:
            assert after[name]["entries"] == before[name]["entries"]
            assert after[name]["compatibility_steps"] == 0
            assert after[name]["fresh_replays"] > before[name]["fresh_replays"]
            assert after[name]["verify_replays"] > before[name]["verify_replays"]
        serving = capfd.readouterr()
        assert "Qwen GDN step:" not in serving.out + serving.err


@pytest.mark.parametrize("value_heads", [16, 32])
def test_native_mtp_three_recurrence_graph_checkpoint_selection(value_heads):
    if not torch.npu.is_available():
        pytest.skip("Requires NPU and native recurrent_gated_delta_rule")
    torch.manual_seed(27)
    count, width, heads, dim = 4, 4, 16, 128
    tokens = count * width
    q = torch.nn.functional.normalize(torch.randn(tokens, heads, dim), dim=-1).bfloat16()
    k = torch.nn.functional.normalize(torch.randn_like(q.float()), dim=-1).bfloat16()
    v = torch.randn(tokens, value_heads, dim).bfloat16()
    g = -torch.rand(tokens, value_heads)
    beta = torch.rand(tokens, value_heads).bfloat16()
    expected_state = torch.randn(24, value_heads, dim, dim) * 0.1
    state = expected_state.npu()
    qn, kn, vn, gn, bn = (tensor.npu() for tensor in (q, k, v, g, beta))
    q = q.repeat_interleave(value_heads // heads, dim=1).float()
    k = k.repeat_interleave(value_heads // heads, dim=1).float()
    slots = torch.full((tokens,), -1, dtype=torch.int32, device="npu")
    accepted = torch.ones(count, dtype=torch.int32, device="npu")
    lengths = torch.zeros(count + 1, dtype=torch.int32, device="npu")

    def run():
        return torch.ops._C_ascend.npu_recurrent_gated_delta_rule(
            query=qn,
            key=kn,
            value=vn,
            g=gn,
            beta=bn,
            state=state,
            scale=dim**-0.5,
            actual_seq_lengths=lengths,
            ssm_state_indices=slots,
            num_accepted_tokens=accepted,
        )

    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        run()
        stream.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, stream=stream):
            output = run()
    torch.npu.current_stream().wait_stream(stream)
    checkpoints = torch.randperm(expected_state.shape[0])[:tokens].reshape(count, width)
    # All acceptance lengths, reordered requests, padding and recycled slots.
    for iteration, requests in enumerate(([0, 1, 2, 3], [3, 1, 0], [2], [], [1, 3, 0, 2])):
        slot_cpu = torch.full((count, width), -1, dtype=torch.int32)
        accepted_cpu = torch.ones(count, dtype=torch.int32)
        expected = torch.zeros_like(v.float())
        for row, request in enumerate(requests):
            slot_cpu[row] = checkpoints[request]
            accepted_cpu[row] = 1 + (iteration + row) % width
            recurrent = expected_state[slot_cpu[row, accepted_cpu[row] - 1]].clone()
            for offset in range(width):
                token = row * width + offset
                recurrent *= g[token].exp()[:, None, None]
                delta = (v[token].float() - (recurrent * k[token].float()[:, None, :]).sum(-1)) * beta[token, :, None]
                recurrent += delta[:, :, None] * k[token].float()[:, None, :]
                expected_state[slot_cpu[row, offset]] = recurrent
                expected[token] = (recurrent * q[token].float()[:, None, :] * dim**-0.5).sum(-1)
        slots.copy_(slot_cpu.flatten())
        accepted.copy_(accepted_cpu)
        lengths.copy_(torch.tensor([0] + [width] * len(requests) + [0] * (count - len(requests))))
        graph.replay()
        live_tokens = len(requests) * width
        torch.testing.assert_close(output[:live_tokens].float().cpu(), expected[:live_tokens], rtol=1e-2, atol=1e-3)
        torch.testing.assert_close(state.cpu(), expected_state, rtol=1e-2, atol=1e-3)


def test_native_mtp_three_convolution_graph_acceptance_and_padding():
    if not torch.npu.is_available():
        pytest.skip("Requires NPU and native causal_conv1d")
    torch.manual_seed(31)
    count, width, channels, history = 4, 4, 128, 3
    x = torch.randn(count, width, channels).bfloat16()
    weight = (torch.randn(history + 1, channels) * 0.1).bfloat16()
    expected_cache = torch.randn(7, history + width - 1, channels).bfloat16()
    xn, wn, cache = x.flatten(0, 1).npu(), weight.npu(), expected_cache.npu()
    slots = torch.full((count,), -1, dtype=torch.int32, device="npu")
    accepted = torch.ones(count, dtype=torch.int32, device="npu")
    query_start = torch.arange(count + 1, dtype=torch.int32, device="npu") * width
    output = torch.zeros_like(xn)

    def run():
        torch.ops._C_ascend.npu_causal_conv1d_custom(
            output,
            xn,
            wn,
            conv_state=cache,
            bias_opt=None,
            query_start_loc_opt=query_start,
            cache_indices_opt=slots,
            initial_state_mode_opt=None,
            num_accepted_tokens_opt=accepted,
            activation_mode=1,
            pad_slot_id=-1,
            run_mode=1,
        )

    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        run()
        stream.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, stream=stream):
            run()
    torch.npu.current_stream().wait_stream(stream)
    for iteration, indices in enumerate(([5, 2, 6, 1], [2, -1, 5, -1], [-1] * 4, [1, 6, 2, 5])):
        accepted_cpu = 1 + (torch.arange(count, dtype=torch.int32) + iteration) % width
        expected_outputs = {}
        for row, slot in enumerate(indices):
            if slot < 0:
                continue
            start = int(accepted_cpu[row]) - 1
            previous = expected_cache[slot, start : start + history].clone()
            sequence = torch.cat((previous, x[row])).float()
            expected_output = torch.stack(
                [
                    torch.nn.functional.silu((sequence[token : token + history + 1] * weight.float()).sum(0))
                    for token in range(width)
                ]
            )
            expected_cache[slot] = torch.cat((previous[1:], x[row]))
            expected_outputs[row] = expected_output
        slots.copy_(torch.tensor(indices, dtype=torch.int32))
        accepted.copy_(accepted_cpu)
        graph.replay()
        for row, slot in enumerate(indices):
            if slot >= 0:
                torch.testing.assert_close(
                    output.view(count, width, channels)[row].float().cpu(), expected_outputs[row], rtol=1e-2, atol=3e-3
                )
        torch.testing.assert_close(cache.cpu(), expected_cache, rtol=0, atol=0)
