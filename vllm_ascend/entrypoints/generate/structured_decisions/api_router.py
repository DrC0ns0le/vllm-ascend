# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from argparse import Namespace
from http import HTTPStatus
from typing import Any

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.datastructures import State
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.chat_utils import load_chat_template
from vllm.entrypoints.serve.engine.protocol import ErrorResponse
from vllm.entrypoints.serve.utils.api_utils import (
    load_aware_call,
    validate_json_request,
    with_cancellation,
)
from vllm.entrypoints.serve.utils.request_logger import RequestLogger
from vllm.tasks import SupportedTask

from vllm_ascend.entrypoints.openai.decisions.state import init_decisions_state

from .protocol import StructuredDecisionRequest
from .serving import ServingStructuredDecisions


def structured_decisions(request: Request) -> ServingStructuredDecisions | None:
    return getattr(request.app.state, "ascend_serving_structured_decisions", None)


@with_cancellation
@load_aware_call
async def create_structured_decision(request: StructuredDecisionRequest, raw_request: Request):
    handler = structured_decisions(raw_request)
    if handler is None:
        raise NotImplementedError("The model does not support the structured decision API")
    result = await handler.create_decision(request, raw_request)
    if isinstance(result, ErrorResponse):
        return JSONResponse(content=result.model_dump(), status_code=result.error.code)
    return JSONResponse(content=result.model_dump())


def register_structured_decisions_api_router(app: FastAPI):
    app.add_api_route(
        "/v1/systemone",
        create_structured_decision,
        methods=["POST"],
        dependencies=[Depends(validate_json_request)],
        responses={
            HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
            HTTPStatus.NOT_FOUND.value: {"model": ErrorResponse},
            HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
            HTTPStatus.NOT_IMPLEMENTED.value: {"model": ErrorResponse},
        },
    )


def init_structured_decisions_state(
    engine_client: EngineClient,
    state: State,
    args: Namespace,
    request_logger: RequestLogger | None,
    supported_tasks: tuple[SupportedTask, ...],
    default_chat_template_kwargs: dict[str, Any],
):
    strategy = init_decisions_state(
        engine_client,
        state,
        args,
        request_logger,
        load_chat_template(args.chat_template),
        default_chat_template_kwargs,
        supported_tasks,
    )
    state.ascend_serving_structured_decisions = (
        ServingStructuredDecisions(state.openai_serving_models, strategy, request_logger=request_logger)
        if strategy is not None
        else None
    )
