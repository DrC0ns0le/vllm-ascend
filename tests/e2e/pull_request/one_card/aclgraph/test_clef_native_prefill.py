# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Real 9B weights: prove backbone graph replay and eager joint-head parity."""

from types import SimpleNamespace

import pytest
import torch
from vllm import PoolingParams

from tests.e2e.conftest import VllmRunner, wait_until_npu_memory_free
from tests.e2e.pull_request.one_card.aclgraph.test_qwen3_5_native_prefill import registry_status
from vllm_ascend.entrypoints.generate.structured_decisions.protocol import StructuredDecisionRequest
from vllm_ascend.entrypoints.generate.structured_decisions.serving import parse_questions
from vllm_ascend.entrypoints.openai.decisions.strategies import ClefFlashStrategy, ReadContext


def observe_pooler(worker):
    model = worker.model_runner.get_model()
    observations = {"calls": 0}

    def check_outside_capture(module, args):
        assert not torch.npu.is_current_stream_capturing(), "Clef joint head was captured"
        observations["calls"] += 1

    model.pooler.register_forward_pre_hook(check_outside_capture)
    worker.clef_pooler_observations = observations


def pooler_calls(worker):
    return worker.clef_pooler_observations["calls"]


def encode_requests(llm, capacity):
    tokenizer = llm.get_tokenizer()
    strategy = ClefFlashStrategy(
        ReadContext(
            SimpleNamespace(model_config=SimpleNamespace(max_model_len=2 * capacity)),
            SimpleNamespace(renderer=SimpleNamespace(get_tokenizer=lambda: tokenizer)),
            None,
            "auto",
            {},
        )
    )
    prompts, params = [], []
    # Different schemas and lengths within capture capacity; one over it.
    for state, levels in [
        ("A duplicate charge needs a refund.", 2),
        ("Please investigate my billing ticket. " * 20, 3),
        ("A customer submitted a billing ticket. " * (capacity // 16), 4),
        ("A customer submitted a billing ticket. " * capacity, 4),
    ]:
        request = StructuredDecisionRequest(
            state=state,
            questions={
                "refund": {"type": "noul", "instructions": "Does the customer request a refund?"},
                "team": {"type": "choice", "criteria": {"shipping": "deliveries", "billing": "payments"}},
                "severity": {"type": "score", "criteria": [str(i) for i in range(levels)]},
            },
        )
        questions = parse_questions(request, strategy.limits())
        ids, schema, _ = strategy.encode(questions, state)
        prompts.append({"prompt_token_ids": ids})
        params.append(PoolingParams(task="token_classify", extra_kwargs={"joint_schema": schema}))
    assert all(len(p["prompt_token_ids"]) <= capacity for p in prompts[:-1])
    assert len(prompts[-1]["prompt_token_ids"]) > capacity
    if capacity > 1024:
        assert len(prompts[-2]["prompt_token_ids"]) > 1024
    return prompts, params


def probabilities(llm, prompts, params):
    results = llm.encode(prompts, pooling_params=params, pooling_task="token_classify", use_tqdm=False)
    return [result.outputs.data.float().cpu().clone() for result in results]


@wait_until_npu_memory_free()
@pytest.mark.parametrize("capacity", [1024, 4096])
def test_clef_native_capture_replay_and_probability_parity(monkeypatch, capacity):
    if not torch.npu.is_available() or "910B" not in torch.npu.get_device_name():
        pytest.skip("Requires Ascend 910B and real Cloudflare/clef-flash weights")
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "0")
    options = dict(
        runner="pooling",
        dtype="bfloat16",
        max_model_len=2 * capacity,
        max_num_seqs=4,
        max_num_batched_tokens=2 * capacity,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        mamba_cache_mode="none",
        limit_mm_per_prompt={"image": 0, "video": 0},
        compilation_config={"mode": 0, "cudagraph_mode": "NONE"},
    )
    with VllmRunner("Cloudflare/clef-flash", **options) as runner:
        prompts, params = encode_requests(runner.model, capacity)
        expected = probabilities(runner.model, prompts, params)
    with VllmRunner(
        "Cloudflare/clef-flash",
        **options,
        additional_config={
            "native_prefill_graph": True,
            "native_prefill_graph_max_context": capacity,
            "native_prefill_graph_request_counts": [1, 4],
        },
    ) as runner:
        before = runner.model.collective_rpc(registry_status)[0]
        assert before["ready"] and before["shapes"]
        assert before["standard_entries"] == 0
        runner.model.collective_rpc(observe_pooler)
        for _ in range(2):
            # Check both individual and ragged batched requests across graph reuse.
            actual = [probabilities(runner.model, [p], [s])[0] for p, s in zip(prompts, params)]
            actual_batch = probabilities(runner.model, prompts[:-1], params[:-1])
            for output, reference in zip(actual + actual_batch, expected + expected[:-1]):
                torch.testing.assert_close(output, reference, atol=5e-3, rtol=2e-2)
            for output, param in zip(actual, params):
                counts = param.extra_kwargs["joint_schema"]["option_counts"]
                for question in output.split(counts):
                    torch.testing.assert_close(question.sum(), torch.tensor(1.0), atol=1e-5, rtol=0)
        after = runner.model.collective_rpc(registry_status)[0]
        assert after["shapes"] == before["shapes"]
        assert after["replays"] > before["replays"]  # Success cannot be eager fallback alone.
        # Prove the long prompt itself replays, rather than only the short cases.
        probabilities(runner.model, prompts[-2:-1], params[-2:-1])
        long_replay = runner.model.collective_rpc(registry_status)[0]
        assert long_replay["replays"] > after["replays"]
        assert runner.model.collective_rpc(pooler_calls)[0] >= 8
        # A prompt above capacity must fall back without changing the sealed registry.
        probabilities(runner.model, prompts[-1:], params[-1:])
        fallback = runner.model.collective_rpc(registry_status)[0]
        assert fallback["replays"] == long_replay["replays"]
        assert fallback["shapes"] == after["shapes"]
