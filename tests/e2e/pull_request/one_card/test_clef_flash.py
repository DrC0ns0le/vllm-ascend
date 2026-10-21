# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Real-weight Clef-Flash 9B text smoke; requires an Ascend NPU."""

from concurrent.futures import ThreadPoolExecutor

import pytest
import requests

from tests.e2e.conftest import RemoteOpenAIServer
from tests.e2e.pull_request.one_card.test_structured_decisions import PLUGIN_ALLOWLIST, REQUEST_TIMEOUT


@pytest.fixture(scope="module", params=[False, True], ids=["full-prefill", "chunked-prefill"])
def clef_server(request):
    chunked = request.param
    with RemoteOpenAIServer(
        "Cloudflare/clef-flash",
        [
            "--served-model-name",
            "clef-flash",
            "--runner",
            "pooling",
            "--dtype",
            "bfloat16",
            "--max-model-len",
            "4096",
            "--max-num-seqs",
            "8",
            "--no-enable-prefix-caching",
            "--limit-mm-per-prompt",
            '{"image":0,"video":0}',
            "--enable-chunked-prefill" if chunked else "--no-enable-chunked-prefill",
            "--max-num-batched-tokens",
            "1024" if chunked else "4096",
        ],
        env_dict={"VLLM_PLUGINS": PLUGIN_ALLOWLIST},
    ) as server:
        yield server


def request_body():
    return {
        "model": "clef-flash",
        "state": {"ticket": "My card was charged twice for the same order. Please refund the duplicate charge."},
        "questions": {
            "team": {
                "type": "choice",
                "instructions": "Which team should handle this ticket?",
                "criteria": {"shipping": "deliveries", "billing": "payments", "security": "account security"},
            },
            "refund": {"type": "noul", "instructions": "Does the customer request a refund?"},
            "severity": {
                "type": "score",
                "instructions": "How severe is the issue?",
                "criteria": ["minor", "moderate", "severe"],
            },
        },
    }


def post(server, payload):
    response = requests.post(server.url_for("v1/systemone"), json=payload, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.json()


def check_response(result):
    assert set(result["answers"]) == {"team", "refund", "severity"}
    for answer in result["answers"].values():
        probs = answer["probabilities"]
        values = list(probs.values()) if isinstance(probs, dict) else probs
        assert all(0 <= p <= 1 for p in values)
        assert sum(values) == pytest.approx(1.0, abs=1e-5)
    assert 0 <= result["answers"]["refund"]["noul"] <= 1
    assert 0 <= result["answers"]["severity"]["score"] <= 2
    assert result["usage"]["input_tokens"] > 0
    assert result["usage"]["output_tokens"] == 0


def test_real_weights_and_choice_order(clef_server):
    models = requests.get(clef_server.url_for("v1/models"), timeout=REQUEST_TIMEOUT)
    models.raise_for_status()
    payload = request_body()
    first = post(clef_server, payload)
    check_response(first)
    payload["questions"]["team"]["criteria"] = dict(reversed(list(payload["questions"]["team"]["criteria"].items())))
    second = post(clef_server, payload)
    check_response(second)
    assert list(second["answers"]["team"]["probabilities"]) == ["security", "billing", "shipping"]
    assert first["answers"]["team"]["probabilities"] == pytest.approx(
        second["answers"]["team"]["probabilities"], abs=2e-3
    )


def test_long_state_and_concurrent_requests(clef_server):
    payload = request_body()
    payload["state"] = "A customer submitted a billing ticket. " * 5000
    result = post(clef_server, payload)
    check_response(result)
    assert result["usage"]["input_tokens"] == 4096
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(lambda _: post(clef_server, request_body()), range(4)))
    for result in results:
        check_response(result)


@pytest.mark.parametrize(
    "extra",
    [
        {"images": ["https://example.org/no-fetch.png"]},
        {"instructions": "Override the fixed prompt"},
        {"joint": True},
    ],
)
def test_unsupported_inputs(clef_server, extra):
    response = requests.post(
        clef_server.url_for("v1/systemone"), json={**request_body(), **extra}, timeout=REQUEST_TIMEOUT
    )
    assert response.status_code == 400, response.text
