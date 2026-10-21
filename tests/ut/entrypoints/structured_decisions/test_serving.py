# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise the actual vLLM 0.30 HTTP/serving contract with a deterministic engine."""

import asyncio
import logging
import math
from argparse import Namespace
from importlib import metadata
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.datastructures import State
from vllm import envs
from vllm.entrypoints.openai.models.protocol import BaseModelPath
from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.entrypoints.serve.exception_handling.register import init_exception_handler
from vllm.entrypoints.serve.utils.request_logger import RequestLogger
from vllm.inputs import tokens_input
from vllm.logger import logger
from vllm.logprobs import Logprob
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.plugins import load_endpoint_plugins
from vllm.plugins.endpoint_plugins.interface import EndpointPlugin
from vllm.sampling_params import SamplingParams

from vllm_ascend.entrypoints.generate.label_reads import next_token_label_reads
from vllm_ascend.entrypoints.openai.decisions.strategies import (
    MIN_WARM_PREFIX_TOKENS,
    NextTokenStrategy,
    ReadContext,
    reply_label_ids,
    select_read_strategy,
)
from vllm_ascend.entrypoints.plugin import StructuredDecisionsPlugin


class Tokenizer:
    """A controllable reply boundary; real Qwen tokenizers are covered by E2E."""

    def encode(self, text, **kwargs):
        for label, token_id in (("yes", 1001), ("no", 1002)):
            if text.endswith(label):
                return [ord(c) for c in text[: -len(label)]] + [token_id]
        return [ord(c) for c in text]

    def decode(self, ids):
        return "".join({1001: "yes", 1002: "no"}.get(i, chr(i)) for i in ids)

    def get_added_vocab(self):
        return {"<reply>": 0}

    def apply_chat_template(self, messages, **kwargs):
        return self.encode("\n".join(m["content"] for m in messages)) + [0, 10, 10]


class Engine:
    errored = False
    renderer = input_processor = None

    def __init__(self):
        self.model_config = SimpleNamespace(
            architecture="Qwen3_5ForConditionalGeneration",
            logprobs_mode="raw_logprobs",
            is_encoder_decoder=False,
            is_hybrid=True,
            max_model_len=4096,
        )
        self.vllm_config = SimpleNamespace(cache_config=SimpleNamespace(enable_prefix_caching=True))
        self.generated = []
        self.admitted = []
        self.failure = None
        self.closed = []

    async def get_supported_tasks(self):
        return ("generate",)

    def check_admission(self, count):
        self.admitted.append(count)

    async def generate(self, prompt, params, request_id, **kwargs):
        self.generated.append((prompt, params, request_id, kwargs))
        try:
            await asyncio.sleep(0)
            if self.failure == "no_result":
                return
            if self.failure == "bad_request":
                raise ValueError("Invalid generation input")
            ids = params.logprob_token_ids or [ord("A")]
            # Total label mass 0.6; confidence must not be the normalized score.
            logprobs = {i: Logprob(math.log(0.6 / len(ids))) for i in ids}
            if self.failure == "missing_label":
                logprobs.pop(ids[-1])
            output = CompletionOutput(
                index=0,
                text="A",
                token_ids=[ids[0]],
                cumulative_logprob=None,
                logprobs=None if self.failure == "no_logprobs" else [logprobs],
                finish_reason="error" if self.failure == "error" else "length",
            )
            yield RequestOutput(
                request_id=request_id,
                prompt=None,
                prompt_token_ids=prompt["prompt_token_ids"],
                prompt_logprobs=None,
                outputs=[] if self.failure == "no_output" else [output],
                finished=True,
                num_cached_tokens=3,
                num_cache_creation_tokens=2,
            )
        finally:
            self.closed.append(request_id)


class Renderer:
    def __init__(self):
        self.renderer = SimpleNamespace(get_tokenizer=Tokenizer)
        self.requests = []
        self.bad_tail = False

    async def preprocess_chat(self, request, messages, **kwargs):
        self.requests.append((request, messages, kwargs))
        ids = Tokenizer().apply_chat_template(messages)
        if self.bad_tail:
            ids.append(99)
        return [], [tokens_input(ids, cache_salt=request.cache_salt)]


def body(endpoint, count=1):
    if endpoint == "decisions":
        return {
            "model": "test-model",
            "input": "a ticket",
            "questions": [
                {
                    "type": "choice",
                    "instructions": f"Question {i}",
                    "choices": [{"value": "billing"}, {"value": "shipping"}],
                }
                for i in range(count)
            ],
        }
    return {
        "model": "test-model",
        "state": "a ticket",
        "questions": {
            str(i): {"type": "choice", "instructions": f"Question {i}", "criteria": {"billing": None, "shipping": None}}
            for i in range(count)
        },
    }


@pytest.fixture
def server():
    engine, renderer = Engine(), Renderer()
    app = FastAPI()
    app.state.args = Namespace(log_error_stack=False)
    app.state.online_renderer = renderer
    app.state.openai_serving_models = OpenAIServingModels(
        engine, [BaseModelPath(name="test-model", model_path="test-model")]
    )
    plugin = StructuredDecisionsPlugin()
    plugin.attach_router(app)
    args = Namespace(
        enable_log_requests=False,
        max_log_len=None,
        chat_template=None,
        chat_template_content_format="auto",
        default_chat_template_kwargs=None,
    )
    asyncio.run(plugin.init_state(engine, app.state, args))
    init_exception_handler(app)
    with TestClient(app, headers={"X-Request-Id": "unit-test"}) as client:
        yield SimpleNamespace(client=client, engine=engine, renderer=renderer, app=app)


@pytest.mark.parametrize(
    "architecture",
    [
        "Qwen3ForCausalLM",
        "Qwen3_5ForConditionalGeneration",
        "Qwen3_5MoeForConditionalGeneration",
    ],
)
@pytest.mark.parametrize("mode", ["raw_logprobs", "processed_logprobs"])
def test_supported_strategies(architecture, mode):
    assert (
        select_read_strategy(SimpleNamespace(architecture=architecture, logprobs_mode=mode), ("generate",))
        is NextTokenStrategy
    )


@pytest.mark.parametrize(
    "architecture,mode,tasks",
    [
        ("LlamaForCausalLM", "raw_logprobs", ("generate",)),
        ("Qwen3ForCausalLM", "raw_logits", ("generate",)),
        ("Qwen3ForCausalLM", "processed_logits", ("generate",)),
        ("Qwen3ForCausalLM", "raw_logprobs", ("embed",)),
    ],
)
def test_unsupported_strategies(architecture, mode, tasks):
    with pytest.raises(ValueError):
        select_read_strategy(SimpleNamespace(architecture=architecture, logprobs_mode=mode), tasks)


@pytest.mark.parametrize("endpoint", ["systemone", "decisions"])
def test_http_success_and_metadata(server, endpoint):
    payload = body(endpoint, 3)
    payload["cache_salt"] = "tenant-a"
    response = server.client.post(
        f"/v1/{endpoint}",
        json=payload,
        headers={
            "traceparent": "00-0123456789abcdef0123456789abcdef-0123456789abcdef-01",
            "authorization": "Bearer private",
        },
    )
    assert response.status_code == 200, response.text
    data = response.json()
    assert len(data["answers"]) == 3
    assert data["usage"]["output_tokens"] == 3
    answers = data["answers"].values() if endpoint == "systemone" else data["answers"]
    for answer in answers:
        assert answer["confidence"] == pytest.approx(0.3)
    if endpoint == "systemone":
        assert data["diagnostics"]["0"]["label_mass"] == pytest.approx(0.6)
    else:
        assert data["usage"]["input_tokens_details"] == {"cached_tokens": 9, "cache_write_tokens": 6}
    assert server.engine.admitted == [3]
    for prompt, params, _, kwargs in server.engine.generated:
        assert params.max_tokens == 1 and params.temperature == 0.0
        assert params.logprob_token_ids == [ord("A"), ord("B")]
        assert prompt["cache_salt"] == "tenant-a"
        assert set(kwargs["trace_headers"]) == {"traceparent"}
    assert len(server.engine.closed) == 3


@pytest.mark.parametrize("endpoint", ["systemone", "decisions"])
@pytest.mark.parametrize(
    "failure,code",
    [
        ("bad_request", 400),
        ("no_result", 500),
        ("no_output", 500),
        ("error", 500),
        ("no_logprobs", 500),
        ("missing_label", 500),
    ],
)
def test_generation_errors(server, endpoint, failure, code):
    server.engine.failure = failure
    response = server.client.post(f"/v1/{endpoint}", json=body(endpoint))
    assert response.status_code == code, response.text


@pytest.mark.parametrize("endpoint", ["systemone", "decisions"])
def test_validation_before_engine_admission(server, endpoint):
    for payload in (body(endpoint, 65), {**body(endpoint), "joint": True}, {**body(endpoint), "unknown": True}):
        response = server.client.post(f"/v1/{endpoint}", json=payload)
        assert response.status_code == 400, response.text
    assert server.engine.admitted == []
    assert server.engine.generated == []
    payload = {**body(endpoint), "model": "missing"}
    assert server.client.post(f"/v1/{endpoint}", json=payload).status_code == 404


@pytest.mark.parametrize(
    "field,value",
    [
        ("images", ["https://example.org/image.png"]),
        ("audio", ["data:audio/wav;base64,AA=="]),
        ("videos", ["https://example.org/video.mp4"]),
        ("chat_template_kwargs", {"enable_thinking": True}),
    ],
)
def test_unsupported_media_and_thinking(server, field, value):
    response = server.client.post("/v1/systemone", json={**body("systemone"), field: value})
    assert response.status_code == 400
    assert not server.engine.generated


def test_hybrid_prefix_warmup_and_prompt_order(server):
    payload = body("systemone", 3)
    payload["state"] = "x" * (MIN_WARM_PREFIX_TOKENS + 20)
    payload["cache_salt"] = "tenant"
    payload["priority"] = 5
    response = server.client.post("/v1/systemone", json=payload)
    assert response.status_code == 200, response.text
    calls = server.engine.generated
    assert len(calls) == 4 and calls[0][2].endswith("-prefix")
    assert calls[0][1].logprob_token_ids is None
    for i, (prompt, _, _, kwargs) in enumerate(calls[1:]):
        assert f"Question {i}" in Tokenizer().decode(prompt["prompt_token_ids"])
        assert prompt["cache_salt"] == "tenant" and kwargs["priority"] == 5


def test_reply_boundary_validation(server):
    server.renderer.bad_tail = True
    assert server.client.post("/v1/systemone", json=body("systemone")).status_code == 400
    assert not server.engine.generated
    tokenizer = Tokenizer()
    tokenizer.encode = lambda *args, **kwargs: [0, 7]
    with pytest.raises(ValueError, match="one distinct token"):
        reply_label_ids(tokenizer, [0], ["A", "B"])


@pytest.mark.parametrize("native", [(), ("/v1/systemone",), ("/v1/systemone", "/v1/decisions")])
def test_plugin_preserves_native_routes(native):
    app = FastAPI()
    for path in native:
        app.add_api_route(path, lambda: {"native": True}, methods=["POST"])
    plugin = StructuredDecisionsPlugin()
    plugin.attach_router(app)
    plugin.attach_router(app)
    paths = [getattr(route, "path", None) for route in app.routes]
    assert paths.count("/v1/systemone") == paths.count("/v1/decisions") == 1
    assert plugin.needs_state == (len(native) != 2)


@pytest.mark.asyncio
async def test_unsupported_model_initializes_without_blocking_server():
    plugin = StructuredDecisionsPlugin()
    app = FastAPI()
    plugin.attach_router(app)
    engine = Engine()
    engine.model_config.architecture = "LlamaForCausalLM"
    args = Namespace(
        enable_log_requests=False,
        max_log_len=None,
        chat_template=None,
        chat_template_content_format="auto",
        default_chat_template_kwargs=None,
    )
    state = State()
    await plugin.init_state(engine, state, args)
    assert state.ascend_serving_decisions is None
    assert state.ascend_serving_structured_decisions is None
    app.state.args = Namespace(log_error_stack=False)
    init_exception_handler(app)
    with TestClient(app) as client:
        for endpoint in ("systemone", "decisions"):
            assert client.post(f"/v1/{endpoint}", json=body(endpoint)).status_code == 501


def test_chat_options_disable_thinking_and_reserve_output_token():
    strategy = NextTokenStrategy(ReadContext(Engine(), Renderer(), None, "auto", {}))
    request = strategy._read_request(None)
    chat = request.build_chat_params(None, "auto")
    assert chat.chat_template_kwargs["enable_thinking"] is False
    assert chat.chat_template_kwargs["add_generation_prompt"] is True
    tok = request.build_tok_params(strategy.context.engine_client.model_config)
    assert tok.max_output_tokens == 1


@pytest.mark.parametrize(
    "hybrid,enabled,count,long_state,expected_warmups",
    [
        (True, True, 2, True, 1),
        (False, True, 2, True, 0),
        (True, False, 2, True, 0),
        (True, True, 1, True, 0),
        (True, True, 2, False, 0),
    ],
)
def test_prefix_warming_conditions(server, hybrid, enabled, count, long_state, expected_warmups):
    server.engine.model_config.is_hybrid = hybrid
    server.engine.vllm_config.cache_config.enable_prefix_caching = enabled
    payload = body("systemone", count)
    payload["state"] = "x" * (MIN_WARM_PREFIX_TOKENS + 20) if long_state else "x"
    assert server.client.post("/v1/systemone", json=payload).status_code == 200
    assert len(server.engine.generated) == count + expected_warmups


@pytest.mark.parametrize(
    "plugins,tasks,loaded",
    [
        (None, ("generate",), False),
        ([], ("generate",), False),
        (["ascend_structured_decisions"], ("generate",), True),
        (["ascend_structured_decisions"], ("embed",), False),
    ],
)
def test_endpoint_plugin_discovery(monkeypatch, plugins, tasks, loaded):
    entry = SimpleNamespace(
        name="ascend_structured_decisions",
        value="vllm_ascend.entrypoints.plugin:StructuredDecisionsPlugin",
        load=lambda: StructuredDecisionsPlugin,
    )
    monkeypatch.setattr(metadata, "entry_points", lambda **kwargs: [entry])
    monkeypatch.setattr(envs, "VLLM_PLUGINS", plugins)
    discovered = load_endpoint_plugins(tasks)
    assert bool(discovered) == loaded
    if loaded:
        assert isinstance(discovered[0], EndpointPlugin)


@pytest.mark.asyncio
async def test_cancelling_label_reads_closes_all_engine_streams():
    started, closed = set(), set()
    ready = asyncio.Event()

    async def generate(prompt, params, request_id, **kwargs):
        try:
            started.add(request_id)
            if len(started) == 2:
                ready.set()
            await asyncio.Event().wait()
            yield None
        finally:
            closed.add(request_id)

    engine = SimpleNamespace(generate=generate)
    task = asyncio.create_task(
        next_token_label_reads(
            engine,
            [tokens_input([1]), tokens_input([2])],
            [SamplingParams(max_tokens=1, logprob_token_ids=[3])] * 2,
            "cancel",
        )
    )
    await asyncio.wait_for(ready.wait(), timeout=5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert closed == started == {"cancel-0", "cancel-1"}


def test_request_logging_keeps_metadata_and_respects_limit(server, caplog, monkeypatch):
    handler = server.app.state.ascend_serving_decisions
    handler.request_logger = RequestLogger(max_log_len=None)
    monkeypatch.setattr(logger, "propagate", True)
    payload = {**body("decisions"), "safety_identifier": "test-user"}
    with caplog.at_level(logging.DEBUG, logger=logger.name):
        assert server.client.post("/v1/decisions", json=payload).status_code == 200
    assert '"safety_identifier":"test-user"' in caplog.text
    caplog.clear()
    handler.request_logger.max_log_len = 10
    with caplog.at_level(logging.DEBUG, logger=logger.name):
        assert server.client.post("/v1/decisions", json=payload).status_code == 200
    assert "safety_identifier" not in caplog.text
