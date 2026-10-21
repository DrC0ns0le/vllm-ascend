# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Read strategies: how a model is asked for label probabilities.

Only models in READ_STRATEGIES are currently supported.
"""

import asyncio
import json
import math
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property
from types import MappingProxyType
from typing import Any, ClassVar

import vllm.envs as envs
from PIL import Image
from vllm.config import ModelConfig
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.chat_utils import ChatTemplateContentFormatOption
from vllm.entrypoints.serve.engine.protocol import OpenAIBaseModel
from vllm.inputs import EngineInput, tokens_input
from vllm.lora.request import LoRARequest
from vllm.multimodal.media import (
    MEDIA_CONNECTOR_REGISTRY,
    MediaConnector,
)
from vllm.outputs import RequestOutput
from vllm.renderers import ChatParams, TokenizeParams, merge_kwargs
from vllm.renderers.inputs.preprocess import extract_prompt_components
from vllm.renderers.online_renderer import OnlineRenderer
from vllm.sampling_params import SamplingParams
from vllm.tasks import SupportedTask
from vllm.tokenizers import TokenizerLike

from vllm_ascend.entrypoints.generate.label_reads import next_token_label_reads

from .question_types import (
    LABELS,
    QUESTION_TYPES,
    Question,
    StructuredDecisionError,
    label_softmax,
)

# Fewer shared tokens are not worth an extra round trip before the reads.
MIN_WARM_PREFIX_TOKENS = 128


def common_prefix_len(prompts: Sequence[Sequence[int]]) -> int:
    """The number of leading token ids that every prompt starts with."""
    shortest = min(prompts, key=len)
    for i, token_id in enumerate(shortest):
        if any(p[i] != token_id for p in prompts):
            return i
    return len(shortest)


@dataclass
class ReadContext:
    engine_client: EngineClient
    online_renderer: OnlineRenderer
    chat_template: str | None
    chat_template_content_format: ChatTemplateContentFormatOption
    default_chat_template_kwargs: dict[str, Any]


@dataclass(frozen=True)
class DecisionLimits:
    max_questions: int
    max_options: int
    #: Whether a read scores each option by its label's token, so a question
    #: has at most as many options as its type has labels.
    labeled: bool = True


@dataclass(frozen=True)
class JointSlot:
    canvas: int
    width: int
    position: int


@dataclass
class QuestionRead:
    probs: list[float]
    label_mass: float
    argmax_is_label: bool
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    cache_write_tokens: int = 0
    #: Where a joint read found this answer. Usage sits on each canvas's first read.
    slot: JointSlot | None = None


class ReadPromptRequest(OpenAIBaseModel):
    """Chat options for one read, ending at the generation prompt so the label
    is the reply's first token. Thinking is off unless the request enables it."""

    chat_template_kwargs: dict[str, Any] | None = None
    cache_salt: str | None = None

    def build_chat_params(
        self,
        default_template: str | None,
        default_template_content_format: ChatTemplateContentFormatOption,
    ) -> ChatParams:
        return ChatParams(
            chat_template=default_template,
            chat_template_content_format=default_template_content_format,
            chat_template_kwargs=merge_kwargs(
                merge_kwargs({"enable_thinking": False}, self.chat_template_kwargs),
                dict(add_generation_prompt=True, continue_final_message=False),
            ),
        )

    def build_tok_params(self, model_config: ModelConfig) -> TokenizeParams:
        return TokenizeParams(
            max_total_tokens=model_config.max_model_len,
            max_output_tokens=1,
            add_special_tokens=False,
        )


def state_text(state: Any) -> str:
    return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)


class ReadStrategy(ABC):
    #: The engine task reads run as.
    task: ClassVar[SupportedTask] = "generate"
    #: The modalities a request may attach media of.
    modalities: ClassVar[frozenset[str]] = frozenset()
    #: Whether read_joint answers several questions from one canvas.
    joint_reads: bool = False

    def __init__(self, context: ReadContext):
        self.context = context

    @cached_property
    def _connector(self) -> MediaConnector:
        model_config = self.context.engine_client.model_config
        return MEDIA_CONNECTOR_REGISTRY.load(
            envs.VLLM_MEDIA_CONNECTOR,
            media_io_kwargs=model_config.get_multimodal_config().media_io_kwargs,
            allowed_local_media_path=model_config.allowed_local_media_path,
            allowed_media_domains=model_config.allowed_media_domains,
        )

    @abstractmethod
    def limits(self) -> DecisionLimits: ...

    def render_state(self, state: Any) -> str:
        """The request's state as the model reads it."""
        return state_text(state)

    def num_requests(self, questions: list[Question], joint: bool = False) -> int:
        """The engine requests a decision takes."""
        return len(questions)

    async def read_joint(
        self,
        questions: list[Question],
        instructions: str | None,
        state: str,
        *,
        request_id: str,
        chat_template_kwargs: dict[str, Any] | None,
        lora_request: LoRARequest | None,
        priority: int,
        cache_salt: str | None = None,
        trace_headers: Mapping[str, str] | None = None,
        media: Mapping[str, Sequence[str]] | None = None,
    ) -> list[QuestionRead]:
        """Like read, with questions sharing canvases. Only if joint_reads."""
        raise NotImplementedError

    @abstractmethod
    async def read(
        self,
        questions: list[Question],
        instructions: str | None,
        state: str,
        *,
        request_id: str,
        chat_template_kwargs: dict[str, Any] | None,
        lora_request: LoRARequest | None,
        priority: int,
        cache_salt: str | None = None,
        trace_headers: Mapping[str, str] | None = None,
        media: Mapping[str, Sequence[str]] | None = None,
    ) -> list[QuestionRead]:
        """One read per question, in order."""


def prompt_tail(tokenizer: TokenizerLike, prompt_ids: Sequence[int]) -> list[int]:
    """The prompt's ids after its last added token."""
    added = set(tokenizer.get_added_vocab().values())
    start = max((i + 1 for i, t in enumerate(prompt_ids) if t in added), default=0)
    return list(prompt_ids[start:])


def reply_label_ids(
    tokenizer: TokenizerLike,
    prompt_ids: Sequence[int],
    labels: Sequence[str] = LABELS,
) -> tuple[list[int], list[int]]:
    """The prompt's ids after its last added token, and the token each of
    ``labels`` adds after them as the first token of the reply. Raises
    ValueError if a label is not one distinct token there."""
    tail = prompt_tail(tokenizer, prompt_ids)
    text = tokenizer.decode(tail)
    ids: list[int] = []
    for label in labels:
        extended = tokenizer.encode(text + label, add_special_tokens=False)
        if extended[:-1] != tail or extended[-1] in ids:
            raise ValueError(f"label {label!r} is not one distinct token after this model's chat prompt")
        ids.append(extended[-1])
    return tail, ids


def type_label_ids(tokenizer: TokenizerLike, prompt_ids: Sequence[int]) -> dict[str, int]:
    """The reply's first token for each label of every registered question
    type. Raises ValueError as reply_label_ids does."""
    label_ids: dict[str, int] = {}
    for qtype in QUESTION_TYPES.values():
        _, ids = reply_label_ids(tokenizer, prompt_ids, qtype.label_set)
        label_ids.update(zip(qtype.label_set, ids))
    return label_ids


def cap_pixels(image: Image.Image, max_pixels: int) -> Image.Image:
    """Shrink ``image`` to at most ``max_pixels`` pixels (bicubic)."""
    w, h = image.size
    if w * h <= max_pixels:
        return image
    scale = math.sqrt(max_pixels / (w * h))
    size = (max(1, int(w * scale)), max(1, int(h * scale)))
    return image.resize(size, Image.Resampling.BICUBIC)


class NextTokenStrategy(ReadStrategy):
    """Autoregressive models. Each question is one request: the state, then the
    question with its labeled options, and the logprobs of the label tokens as
    the reply's first token. The requests share the state. With prefix caching,
    it is prefilled once."""

    #: Images are shrunk to at most this many pixels before the processor.
    max_image_pixels: ClassVar[int | None] = None

    def __init__(self, context: ReadContext):
        super().__init__(context)
        # Every prompt ends with the same generation prompt, so a label's token
        # is the same for every question.
        prompt = self._generation_prompt()
        self.tail = prompt_tail(self._tokenizer(), prompt)
        self.label_ids = type_label_ids(self._tokenizer(), prompt)

    def _tokenizer(self) -> TokenizerLike:
        return self.context.online_renderer.renderer.get_tokenizer()

    def _generation_prompt(self) -> list[int]:
        """A chat prompt's ids up to the generation prompt, with thinking off."""
        context, tokenizer = self.context, self._tokenizer()
        probe = tokenizer.apply_chat_template(
            [{"role": "user", "content": "x"}],
            **{
                **context.default_chat_template_kwargs,
                "chat_template": context.chat_template,
                "add_generation_prompt": True,
                "tokenize": True,
                "return_dict": False,
                "enable_thinking": False,
            },
        )
        if isinstance(probe, str):
            probe = tokenizer.encode(probe, add_special_tokens=False)
        return list(probe)

    def _sampling_params(self, label_ids: list[int], prompt_ids: list[int]) -> SamplingParams:
        return SamplingParams(max_tokens=1, temperature=0.0, logprob_token_ids=label_ids)

    def _read_input(self, engine_input: EngineInput, prompt_ids: list[int]) -> EngineInput:
        """The engine input for a read of the rendered prompt."""
        return engine_input

    def _label_groups(self, question: Question) -> list[list[int]]:
        """The tokens that count for each label. A label scores its best one."""
        return [[self.label_ids[label]] for label in question.labels]

    def _content(self, state: str, question: Question) -> str:
        """The user turn: the state, then the question."""
        return f"{state}\n\n{question.type.prompt(question)}"

    async def _image_parts(self, urls: Sequence[str]) -> list[dict[str, Any]]:
        """The images as chat content parts, fetched once so that every
        question reads the same pictures."""
        if not urls:
            return []
        try:
            images = await asyncio.gather(*map(self._connector.fetch_image_async, urls))
        except ValueError as e:
            raise StructuredDecisionError(f"media: {e}") from e
        if self.max_image_pixels is not None:
            images = [cap_pixels(image, self.max_image_pixels) for image in images]
        return [{"type": "image_pil", "image_pil": image} for image in images]

    def limits(self) -> DecisionLimits:
        return DecisionLimits(max_questions=64, max_options=len(LABELS))

    def _read_request(
        self, chat_template_kwargs: dict[str, Any] | None, cache_salt: str | None = None
    ) -> ReadPromptRequest:
        if (chat_template_kwargs or {}).get("enable_thinking"):
            raise StructuredDecisionError("a read is the reply's first token, so thinking must be off")
        return ReadPromptRequest(chat_template_kwargs=chat_template_kwargs, cache_salt=cache_salt)

    async def _render(
        self, read_request: ReadPromptRequest, messages: list[dict[str, Any]]
    ) -> tuple[EngineInput, list[int]]:
        ctx = self.context
        _, (engine_input,) = await ctx.online_renderer.preprocess_chat(
            read_request,
            messages,
            default_template=ctx.chat_template,
            default_template_content_format=ctx.chat_template_content_format,
            default_template_kwargs=ctx.default_chat_template_kwargs,
            # A read that fails before it reaches the engine must not leave the
            # sender cache believing the engine has the images.
            skip_mm_cache=True,
        )
        prompt_ids = extract_prompt_components(ctx.engine_client.model_config, engine_input).token_ids
        return engine_input, list(prompt_ids or [])

    async def _warm_shared_prefix(
        self,
        prompts: Sequence[Sequence[int]],
        request_id: str,
        *,
        lora_request: LoRARequest | None,
        priority: int,
        cache_salt: str | None,
        trace_headers: Mapping[str, str] | None,
    ) -> None:
        """Prefill the tokens every question starts with, before the reads.

        The reads share the state only through the prefix cache. Reads that
        start together cannot reuse each other's recurrent state, so on hybrid
        models each of them would prefill the state again. Attention-only
        models share the blocks within a step and skip this.
        """
        engine_client = self.context.engine_client
        if (
            len(prompts) < 2
            or not engine_client.model_config.is_hybrid
            or not engine_client.vllm_config.cache_config.enable_prefix_caching
        ):
            return
        shared = common_prefix_len(prompts)
        if shared < MIN_WARM_PREFIX_TOKENS:
            return
        generator = engine_client.generate(
            tokens_input(list(prompts[0][:shared]), cache_salt=cache_salt),
            SamplingParams(max_tokens=1, temperature=0.0),
            f"{request_id}-prefix",
            lora_request=lora_request,
            trace_headers=trace_headers,
            priority=priority,
        )
        async for _ in generator:
            pass

    async def _prompts(
        self,
        read_request: ReadPromptRequest,
        contents: list[str],
        instructions: str | None,
        image_parts: list[dict[str, Any]],
    ) -> list[tuple[EngineInput, list[int]]]:
        """Each content rendered as a chat prompt, with the prompt's ids."""
        prompts = []
        for text in contents:
            content: Any = text
            if image_parts:
                content = [*image_parts, {"type": "text", "text": text}]
            messages = [{"role": "user", "content": content}]
            if instructions:
                messages.insert(0, {"role": "system", "content": instructions})
            engine_input, prompt_ids = await self._render(read_request, messages)
            if prompt_ids[len(prompt_ids) - len(self.tail) :] != self.tail:
                raise StructuredDecisionError(
                    "these chat options end the prompt differently, so the labels' tokens are unknown"
                )
            prompts.append((engine_input, prompt_ids))
        return prompts

    @staticmethod
    def _question_read(
        result: RequestOutput,
        groups: list[list[int]],
        logprobs: list[float],
        position: int = 0,
        usage: bool = True,
    ) -> QuestionRead:
        """The answer at ``position``. Only a read with ``usage`` counts tokens."""
        token_ids = result.outputs[0].token_ids
        remaining = iter(logprobs)
        read = QuestionRead(
            probs=label_softmax([max(next(remaining) for _ in g) for g in groups]),
            label_mass=sum(math.exp(lp) for lp in logprobs),
            argmax_is_label=len(token_ids) > position and any(token_ids[position] in g for g in groups),
        )
        if usage:
            read.input_tokens = len(result.prompt_token_ids or ())
            read.output_tokens = len(token_ids)
            read.cached_tokens = result.num_cached_tokens or 0
            read.cache_write_tokens = result.num_cache_creation_tokens or 0
        return read

    async def read(
        self,
        questions: list[Question],
        instructions: str | None,
        state: str,
        *,
        request_id: str,
        chat_template_kwargs: dict[str, Any] | None,
        lora_request: LoRARequest | None,
        priority: int,
        cache_salt: str | None = None,
        trace_headers: Mapping[str, str] | None = None,
        media: Mapping[str, Sequence[str]] | None = None,
    ) -> list[QuestionRead]:
        ctx = self.context
        read_request = self._read_request(chat_template_kwargs, cache_salt)
        image_parts = await self._image_parts((media or {}).get("image", ()))

        contents = [self._content(state, q) for q in questions]
        rendered = await self._prompts(read_request, contents, instructions, image_parts)

        slots, engine_inputs, params = [], [], []
        for q, (engine_input, prompt_ids) in zip(questions, rendered):
            slots.append(self._label_groups(q))
            engine_inputs.append(self._read_input(engine_input, prompt_ids))
            params.append(self._sampling_params([i for ids in slots[-1] for i in ids], prompt_ids))

        await self._warm_shared_prefix(
            [prompt_ids for _, prompt_ids in rendered],
            request_id,
            lora_request=lora_request,
            priority=priority,
            cache_salt=cache_salt,
            trace_headers=trace_headers,
        )
        label_reads = await next_token_label_reads(
            ctx.engine_client,
            engine_inputs,
            params,
            request_id,
            lora_request=lora_request,
            trace_headers=trace_headers,
            priority=priority,
        )

        return [self._question_read(r.result, groups, r.logprobs) for groups, r in zip(slots, label_reads)]


READ_STRATEGIES = MappingProxyType(
    {
        "Qwen3ForCausalLM": NextTokenStrategy,
        "Qwen3_5ForConditionalGeneration": NextTokenStrategy,
        "Qwen3_5MoeForConditionalGeneration": NextTokenStrategy,
    }
)


# label_mass sums the labels' full-vocabulary probabilities. The logprobs modes
# return those probabilities. The logits modes return raw logits instead.
LOGPROBS_MODES = frozenset({"raw_logprobs", "processed_logprobs"})


def select_read_strategy(model_config: ModelConfig, supported_tasks: Sequence[SupportedTask]) -> type[ReadStrategy]:
    """Raise ValueError when the model cannot serve structured decisions."""
    strategy = READ_STRATEGIES.get(model_config.architecture)
    if strategy is None:
        raise ValueError(
            "The structured decisions API does not support "
            f"{model_config.architecture}. Supported architectures: "
            f"{sorted(READ_STRATEGIES)}"
        )
    if strategy.task not in supported_tasks:
        raise ValueError(f"Structured decisions on {model_config.architecture} need the {strategy.task} task")
    if strategy.task == "generate" and model_config.logprobs_mode not in LOGPROBS_MODES:
        raise ValueError(
            "Structured decisions need --logprobs-mode raw_logprobs or "
            f"processed_logprobs, not {model_config.logprobs_mode}"
        )
    return strategy
