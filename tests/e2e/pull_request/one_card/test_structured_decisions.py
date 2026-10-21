# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real-weight HTTP smoke and label-score parity on Ascend (vLLM 0.30)."""

import math

import pytest
import requests
from transformers import AutoTokenizer

from tests.e2e.conftest import RemoteOpenAIServer
from vllm_ascend.entrypoints.openai.decisions.strategies import reply_label_ids

PLUGIN_ALLOWLIST = (
    "ascend,ascend_kv_connector,ascend_model_loader,ascend_service_profiling,ascend_model,ascend_structured_decisions"
)
REQUEST_TIMEOUT = 180


@pytest.fixture(scope="module", params=["Qwen/Qwen3.5-0.8B", "Qwen/Qwen3-0.6B"])
def decision_server(request):
    model = request.param
    with RemoteOpenAIServer(
        model,
        [
            "--served-model-name",
            "decision-model",
            "--max-model-len",
            "4096",
            "--max-num-seqs",
            "16",
            "--enable-prefix-caching",
            "--logprobs-mode",
            "raw_logprobs",
        ],
        env_dict={"VLLM_PLUGINS": PLUGIN_ALLOWLIST},
    ) as server:
        yield server, AutoTokenizer.from_pretrained(model)


def post(server, path, payload):
    response = requests.post(server.url_for(path), json=payload, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.json()


@pytest.mark.parametrize("long_state", [False, True])
def test_decision_endpoints_and_label_scores(decision_server, long_state):
    server, tokenizer = decision_server
    state = "The customer's card was charged twice for the same order."
    if long_state:
        # Cross the hybrid prefix-warming threshold with real Qwen tokenization.
        state = "This support ticket concerns a single order. " * 32 + state
    question = "Which team should handle this ticket?"
    payload = {
        "model": "decision-model",
        "state": state,
        "questions": {
            "team": {
                "type": "choice",
                "instructions": question,
                "criteria": {"billing": "payments and refunds", "shipping": "deliveries"},
            },
            "urgent": {"type": "noul", "instructions": "Is a refund needed?"},
            "severity": {
                "type": "score",
                "instructions": "How severe is the problem?",
                "criteria": ["minor", "moderate", "severe"],
            },
        },
    }
    # Repeat to exercise prefix-cache reuse as well as the cold path.
    for _ in range(2):
        result = post(server, "v1/systemone", payload)
        assert result["usage"]["output_tokens"] == 3
        assert set(result["answers"]) == {"team", "urgent", "severity"}
        for key, answer in result["answers"].items():
            probs = answer["probabilities"]
            assert sum(probs.values()) == pytest.approx(1.0)
            assert all(math.isfinite(p) and 0 <= p <= 1 for p in probs.values())
            mass = result["diagnostics"][key]["label_mass"]
            assert 0 <= mass <= 1.00001
            assert answer["confidence"] == pytest.approx(max(probs.values()) * mass)

    decision = post(
        server,
        "v1/decisions",
        {
            "model": "decision-model",
            "input": state,
            "questions": [
                {"type": "predicate", "name": "refund", "instructions": "Is a refund needed?"},
                {
                    "type": "choice",
                    "name": "team",
                    "instructions": question,
                    "choices": [{"value": "billing"}, {"value": "shipping"}],
                },
                {
                    "type": "score",
                    "instructions": "How severe is the problem?",
                    "levels": [{"label": "minor"}, {"label": "severe"}],
                },
            ],
        },
    )
    assert [a["name"] for a in decision["answers"]] == ["refund", "team", None]
    assert decision["usage"]["output_tokens"] == 3
    assert 0 <= decision["answers"][0]["probability"] <= 1

    # Independent oracle: query the exact same prompt through /v1/completions
    # with explicit token IDs, including labels that are outside top-k.
    content = (
        f"{state}\n\nQuestion: {question}\nA: billing - payments and refunds\n"
        "B: shipping - deliveries\nAnswer with the letter of one option only."
    )
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        add_generation_prompt=True,
        enable_thinking=False,
        tokenize=True,
        return_dict=False,
    )
    _, ids = reply_label_ids(tokenizer, prompt, ["A", "B"])
    reference = post(
        server,
        "v1/completions",
        {
            "model": "decision-model",
            "prompt": prompt,
            "max_tokens": 1,
            "temperature": 0,
            "logprob_token_ids": ids,
            "logprobs": len(ids),
            "return_tokens_as_token_ids": True,
        },
    )
    lp = reference["choices"][0]["logprobs"]["top_logprobs"][0]
    weights = [math.exp(lp[f"token_id:{i}"]) for i in ids]
    probs = result["answers"]["team"]["probabilities"]
    # NPU batching/prefix reuse can change floating-point results slightly.
    assert probs["billing"] == pytest.approx(weights[0] / sum(weights), abs=0.02)
    assert result["diagnostics"]["team"]["label_mass"] == pytest.approx(sum(weights), abs=0.02)
