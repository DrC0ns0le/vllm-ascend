# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Clef text requests through the real v0.30 serving and pooling contracts."""

import asyncio
import json
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from vllm.outputs import PoolingOutput, PoolingRequestOutput

from vllm_ascend.entrypoints.generate.structured_decisions.protocol import StructuredDecisionRequest
from vllm_ascend.entrypoints.generate.structured_decisions.serving import parse_questions
from vllm_ascend.entrypoints.openai.decisions.strategies import (
    ClefFlashStrategy,
    ReadContext,
    StructuredDecisionError,
    select_read_strategy,
)
from vllm_ascend.entrypoints.plugin import StructuredDecisionsPlugin

from .test_serving import Engine, Renderer, Tokenizer, body
from .test_serving import server as server

JOINT_SCHEMA_PROMPT = """<|im_start|>system
Read the complete state and schema. Decide every field jointly. Each answer \
must be exactly one of that field's allowed options.<|im_end|>
<|im_start|>user
STATE:
{"customer":{"tier":"gold"},"ticket":"Charged twice"}

SCHEMA FIELDS:

FIELD 1
ID: team
TYPE: choice
INSTRUCTION: Which team?
ALLOWED OPTIONS:
OPTION 1: {"description":"payments","option_id":"billing"}
OPTION 2: {"option_id":"security"}
OPTION 3: {"description":"deliveries","option_id":"shipping"}
END FIELD

FIELD 2
ID: urgent
TYPE: noul
INSTRUCTION: urgent
ALLOWED OPTIONS:
OPTION 1: {"description":"money is moving","option_id":"true"}
OPTION 2: {"description":"The proposition is false or the answer is no.",\
"option_id":"false"}
END FIELD

FIELD 3
ID: tone
TYPE: score
INSTRUCTION: How upset?
ALLOWED OPTIONS:
OPTION 1: {"description":"calm","option_id":"0"}
OPTION 2: {"description":"annoyed","option_id":"1"}
OPTION 3: {"description":"furious","option_id":"2"}
END FIELD

<|im_end|>
<|im_start|>assistant
<think>

</think>

JOINT SCHEMA DECISIONS:"""

JOINT_SCHEMA_REQUEST: dict[str, Any] = dict(
    state={"ticket": "Charged twice", "customer": {"tier": "gold"}},
    questions={
        "team": {
            "type": "choice",
            "instructions": "Which team?",
            "criteria": {
                "shipping": "deliveries",
                "billing": "payments",
                "security": None,
            },
        },
        "urgent": {"type": "noul", "criteria": {"true": "money is moving"}},
        "tone": {
            "type": "score",
            "instructions": "How upset?",
            "criteria": ["calm", "annoyed", "furious"],
        },
    },
)


class PoolingEngine(Engine):
    def __init__(self):
        super().__init__()
        self.model_config.architecture = "ClefFlashForJointSchema"
        self.vllm_config.cache_config.enable_prefix_caching = False
        self.encoded = []

    async def get_supported_tasks(self):
        return ("token_classify",)

    async def encode(self, prompt, params, request_id, **kwargs):
        self.encoded.append((prompt, params, request_id, kwargs))
        await asyncio.sleep(0)
        if self.failure == "no_result":
            return
        scores = {
            "billing": 0.7,
            "security": 0.1,
            "shipping": 0.2,
            "true": 0.9,
            "false": 0.1,
            "0": 0.2,
            "1": 0.5,
            "2": 0.3,
        }
        ids = prompt["prompt_token_ids"]
        spans = params.extra_kwargs["joint_schema"]["option_spans"]
        options = [json.loads(Tokenizer().decode(ids[s:e])) for s, e in spans]
        data = torch.tensor([scores[o["option_id"]] for o in options])
        if self.failure == "bad_count":
            data = data[:-1]
        if self.failure == "bad_shape":
            data = data.unsqueeze(0)
        result = PoolingRequestOutput(request_id, PoolingOutput(data), ids, 0, self.failure != "unfinished")
        if self.failure == "error":
            result.error = SimpleNamespace(message="head failed")
        yield result


@pytest.fixture
def pooling_engine(monkeypatch):
    # Reuse the real endpoint-plugin/FastAPI fixture with only its engine replaced.
    monkeypatch.setattr("tests.ut.entrypoints.structured_decisions.test_serving.Engine", PoolingEngine)


@pytest.fixture
def clef_server(pooling_engine, request):
    return request.getfixturevalue("server")


def test_pooling_plugin_and_selection():
    assert "token_classify" in StructuredDecisionsPlugin.required_tasks
    assert select_read_strategy(PoolingEngine().model_config, ("token_classify",)) is ClefFlashStrategy
    with pytest.raises(ValueError, match="need the token_classify"):
        select_read_strategy(PoolingEngine().model_config, ("generate",))


def test_joint_request(clef_server):
    response = clef_server.client.post(
        "/v1/systemone", json=dict(model="test-model", **JOINT_SCHEMA_REQUEST, cache_salt="salt", priority=2)
    )
    assert response.status_code == 200, response.text
    result = response.json()
    engine = clef_server.engine
    assert engine.admitted == [1]
    assert len(engine.encoded) == 1
    assert engine.generated == []
    prompt, params, request_id, kwargs = engine.encoded[0]
    ids = prompt["prompt_token_ids"]
    assert Tokenizer().decode(ids) == JOINT_SCHEMA_PROMPT
    assert prompt["cache_salt"] == "salt"
    assert request_id.endswith("-0")
    assert kwargs["priority"] == 2
    schema = params.extra_kwargs["joint_schema"]
    assert schema["question_types"] == [1, 0, 2]
    assert schema["option_counts"] == [3, 2, 3]
    assert [Tokenizer().decode(ids[s:e]) for s, e in schema["question_spans"]] == [
        "Which team?",
        "urgent",
        "How upset?",
    ]
    team = result["answers"]["team"]
    assert team["choice"] == "billing"
    assert list(team["probabilities"]) == ["shipping", "billing", "security"]
    assert team["probabilities"] == pytest.approx({"shipping": 0.2, "billing": 0.7, "security": 0.1})
    assert result["answers"]["urgent"]["noul"] == pytest.approx(0.9)
    assert result["answers"]["tone"]["score"] == pytest.approx(1.1)
    assert result["usage"]["input_tokens"] == len(ids)
    assert result["usage"]["output_tokens"] == 0


@pytest.mark.parametrize(
    "extra",
    [
        {"instructions": "Be strict"},
        {"images": ["https://example.org/a.png"]},
        {"audio": ["https://example.org/a.wav"]},
        {"videos": ["https://example.org/a.mp4"]},
        {"joint": True},
        {"questions": {}},
    ],
)
def test_invalid_requests(clef_server, extra):
    response = clef_server.client.post("/v1/systemone", json={"model": "test-model", **JOINT_SCHEMA_REQUEST, **extra})
    assert response.status_code == 400, response.text
    assert clef_server.engine.encoded == []


@pytest.mark.parametrize("failure", ["no_result", "unfinished", "bad_count", "bad_shape", "error"])
def test_engine_failures(clef_server, failure):
    clef_server.engine.failure = failure
    response = clef_server.client.post("/v1/systemone", json=dict(model="test-model", **JOINT_SCHEMA_REQUEST))
    assert response.status_code == 500, response.text


def test_openai_decisions_not_supported(clef_server):
    response = clef_server.client.post("/v1/decisions", json=body("decisions"))
    assert response.status_code == 501


def test_state_truncation_and_schema_limits():
    strategy = ClefFlashStrategy(ReadContext(PoolingEngine(), Renderer(), None, "auto", {}))
    request = StructuredDecisionRequest(**JOINT_SCHEMA_REQUEST)
    questions = parse_questions(request, strategy.limits())
    prompt, schema, _ = strategy.encode(questions, "word " * 100)
    strategy.max_length = len(prompt) - 50
    truncated, truncated_schema, _ = strategy.encode(questions, "word " * 100)
    assert len(truncated) == strategy.max_length
    assert truncated[-200:] == prompt[-200:]
    assert truncated_schema == schema
    strategy.max_length = 100
    with pytest.raises(StructuredDecisionError, match="questions take"):
        strategy.encode(questions, "word")
