# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vLLM 0.30 endpoint plugin for the upstream structured decision APIs."""

from argparse import Namespace

from fastapi import FastAPI
from starlette.datastructures import State
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.launchers.cli_args import resolve_default_chat_template_kwargs
from vllm.entrypoints.serve.utils.request_logger import RequestLogger
from vllm.tasks import SupportedTask

from .generate.structured_decisions.api_router import (
    init_structured_decisions_state,
    register_structured_decisions_api_router,
)
from .openai.decisions.api_router import register_decisions_api_router


class StructuredDecisionsPlugin:
    name = "ascend_structured_decisions"
    required_tasks: tuple[SupportedTask, ...] = ("generate", "token_classify")

    def __init__(self):
        self.needs_state = False

    def attach_router(self, app: FastAPI) -> None:
        # A newer vLLM may already provide these endpoints. Never shadow it.
        paths = {getattr(route, "path", None) for route in app.routes}
        for path, register in (
            ("/v1/systemone", register_structured_decisions_api_router),
            ("/v1/decisions", register_decisions_api_router),
        ):
            if path not in paths:
                register(app)
                self.needs_state = True

    async def init_state(self, engine_client: EngineClient | None, state: State, args: Namespace) -> None:
        if not self.needs_state or engine_client is None:
            return
        request_logger = RequestLogger(max_log_len=args.max_log_len) if args.enable_log_requests else None
        init_structured_decisions_state(
            engine_client,
            state,
            args,
            request_logger,
            await engine_client.get_supported_tasks(),
            resolve_default_chat_template_kwargs(args),
        )
