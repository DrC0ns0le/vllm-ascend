# SPDX-License-Identifier: Apache-2.0
"""Startup-only native model graph capture with bounded shape adaptation.

No custom attention/recurrence kernels, graph task updates, serving-time graph
admission, or per-layer Python callbacks. Graphs own all mutable replay inputs.
"""

import copy
from contextlib import ExitStack, contextmanager
from types import SimpleNamespace

import torch
from vllm.config import CUDAGraphMode
from vllm.forward_context import BatchDescriptor
from vllm.logger import logger

from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.compilation.native_full_layout import (
    DEFAULT_NATIVE_FULL_MAX_CONTEXT,
    NativeGraphPlan,
    cached_attention_mask,
    padded_token_indices,
)
from vllm_ascend.compilation.native_lora import NativeGraphLoRA, native_lora_wrappers
from vllm_ascend.ops.gdn_attn_builder import (
    GDNCausalConv1dMetadata,
    GDNDecodeMetadata,
    GDNPrefillMetadata,
    GDNSpecCausalConv1dMetadata,
    GDNSpecDecodeMetadata,
    _build_non_spec_chunked_prefill_metadata,
)
from vllm_ascend.ops.pto_chunk_gdn.eligibility import CHUNK_SIZE
from vllm_ascend.ops.pto_chunk_gdn.workspace import MegaGDNGraphWorkspace


def native_full_requested(config):
    text = getattr(config.model_config, "hf_text_config", None)
    hf = getattr(config.model_config, "hf_config", None)
    qwen = any(getattr(c, "model_type", "") in ("qwen3_5", "qwen3_5_text") for c in (text, hf))
    return (
        qwen
        and not config.model_config.enforce_eager
        and config.compilation_config.cudagraph_mode in (CUDAGraphMode.FULL, CUDAGraphMode.FULL_AND_PIECEWISE)
        and (config.additional_config or {}).get("native_full_graph", True)
    )


def native_full_speculative_tokens(config):
    spec = config.speculative_config
    if spec is None:
        return 0
    method = getattr(spec, "method", None)
    count = getattr(spec, "num_speculative_tokens", None)
    # recurrent_gated_delta_rule.h reserves MAX_MTP=16 checkpoints,
    # including the target token. MTP's supported rollout is 1/2/3 drafts.
    max_drafts = 15 if method in ("ngram", "ngram_gpu") else 3
    if (
        method not in ("mtp", "qwen3_5_mtp", "ngram", "ngram_gpu")
        or type(count) is not int
        or not 1 <= count <= max_drafts
        or getattr(spec, "parallel_drafting", False)
    ):
        raise ValueError(
            "Native Qwen3.5 FULL supports autoregressive MTP with num_speculative_tokens=1/2/3 "
            "or ngram/ngram_gpu with 1-15 speculative tokens"
        )
    return count


@contextmanager
def native_context(context, metadata, tokens, mode=CUDAGraphMode.FULL, lora=()):
    fields = dict(
        attn_metadata=metadata,
        cudagraph_runtime_mode=mode,
        capturing=False,
        batch_descriptor=BatchDescriptor(num_tokens=tokens, has_lora=bool(lora)),
    )
    for name in ("num_tokens", "padded_num_tokens", "num_actual_tokens"):
        if hasattr(context, name):
            fields[name] = tokens
    saved = {name: getattr(context, name) for name in fields}
    try:
        for name, value in fields.items():
            setattr(context, name, value)
        with ExitStack() as stack:
            for mapping in lora:
                stack.enter_context(mapping.bind())
            yield
    finally:
        for name, value in saved.items():
            setattr(context, name, value)


class NativeFullEntry:
    def __init__(self, owner, shape, context, inputs):
        self.shape = shape
        self.backend = owner.backend
        self.megagdn_workspace = owner.megagdn_workspace
        self.uses_speculation = owner.num_speculative_tokens > 0
        self.graph = None
        self.output = None
        self.replays = 0
        device = inputs["positions"].device
        count, width = shape.requests, shape.width
        tokens = shape.tokens
        self.lora = tuple(NativeGraphLoRA(wrapper, shape) for wrapper in owner.lora_wrappers)
        self.input_ids = torch.zeros(tokens, dtype=inputs["input_ids"].dtype, device=device)
        self.positions = torch.zeros(
            (*inputs["positions"].shape[:-1], tokens), dtype=inputs["positions"].dtype, device=device
        )
        self.lengths = torch.full((count,), width if shape.fresh else 0, dtype=torch.int64, device=device)
        self.contexts = torch.zeros(count, dtype=torch.int64, device=device)
        self.valid = torch.ones(tokens, dtype=torch.bool, device=device)
        self.has_initial_state = torch.zeros(count, dtype=torch.bool, device=device)
        self.query_start = torch.arange(count + 1, dtype=torch.int32, device=device) * width
        self.metadata = {}
        self.capacity = owner.plan.max_context
        self.mask = None
        chunk = None
        if owner.requires_gdn:
            chunk = _build_non_spec_chunked_prefill_metadata(
                SimpleNamespace(vllm_config=owner.config), self.query_start.cpu(), device
            )
        memo = {}
        for name, source in context.attn_metadata.items():
            if id(source) in memo:
                self.metadata[name] = memo[id(source)]
                continue
            target = copy.copy(source)
            memo[id(source)] = target
            target.native_full = self
            target.num_actual_tokens = tokens
            target.num_prefills = count
            target.num_decodes = 0
            target.num_decode_tokens = 0
            if type(source).__name__ == "GDNAttentionMetadata":
                indices = torch.full((count,), -1, dtype=torch.int32, device=device)
                target.non_spec_state_indices_tensor = indices
                target.native_conv_indices = torch.arange(count, dtype=torch.int32, device=device)
                target.non_spec_query_start_loc = self.query_start
                target.has_initial_state = self.has_initial_state
                target.prefill_query_start_loc = self.query_start
                target.prefill_state_indices = indices
                target.prefill_has_initial_state = self.has_initial_state
                target.num_spec_decodes = 0
                target.num_spec_decode_tokens = 0
                target.num_prefill_tokens = tokens
                target.spec_sequence_masks = None
                target.spec_token_indx = None
                target.non_spec_token_indx = None
                target.spec_state_indices_tensor = None
                target.spec_decode_metadata = None
                conv = GDNCausalConv1dMetadata(self.query_start, indices, self.has_initial_state)
                target.non_spec_prefill_metadata = GDNPrefillMetadata(conv, chunk)
                target.non_spec_decode_metadata = GDNDecodeMetadata(
                    conv, torch.cat((self.query_start[:1], torch.ones_like(self.query_start[1:])))
                )
                if width == 1 and not shape.fresh:
                    target.num_prefills = 0
                    target.num_prefill_tokens = 0
                    target.num_decodes = count
                    target.num_decode_tokens = tokens
                    target.non_spec_prefill_metadata = None
                elif not shape.fresh:
                    target.num_prefills = 0
                    target.num_prefill_tokens = 0
                    target.num_spec_decodes = count
                    target.num_spec_decode_tokens = tokens
                    target.non_spec_prefill_metadata = None
                    target.spec_state_indices_tensor = torch.full((count, width), -1, dtype=torch.int32, device=device)
                    target.num_accepted_tokens = torch.ones(count, dtype=torch.int32, device=device)
                    target.spec_decode_metadata = GDNSpecDecodeMetadata(
                        GDNSpecCausalConv1dMetadata(
                            self.query_start, target.spec_state_indices_tensor, target.num_accepted_tokens
                        ),
                        torch.zeros(count + 1, dtype=torch.int32, device=device),
                    )
            elif type(source).__name__ == "AscendMetadata":
                target.attn_state = AscendAttentionState.PrefillNoCache
                target.query_start_loc = self.query_start
                target.actual_seq_lengths_q = [width * (i + 1) for i in range(count)]
                target.seq_lens_list = [width] * count
                target.seq_lens = self.lengths
                target.max_query_len = width
                target.slot_mapping = torch.full((tokens,), -1, dtype=source.slot_mapping.dtype, device=device)
                target.block_tables = torch.zeros(
                    (count, source.block_tables.shape[1]), dtype=source.block_tables.dtype, device=device
                )
            else:
                raise ValueError(f"Native FULL cannot capture metadata {type(source).__name__}")
            self.metadata[name] = target

    def prepare(self):
        self.valid.copy_(
            (torch.arange(self.shape.width, device=self.lengths.device)[None, :] < self.lengths[:, None]).reshape(-1)
        )
        if not self.shape.fresh:
            for metadata in {id(m): m for m in self.metadata.values()}.values():
                if type(metadata).__name__ == "GDNAttentionMetadata":
                    if self.shape.width == 1:
                        metadata.non_spec_decode_metadata.actual_seq_lengths[1:].copy_(self.lengths)
                    else:
                        # Native recurrence consumes packed row lengths. Keep
                        # every live row at the capture width (including short
                        # final verifications); only trailing dummy rows are 0.
                        metadata.spec_decode_metadata.actual_seq_lengths[1:].copy_(
                            torch.where(self.lengths > 0, self.shape.width, 0)
                        )
            self.mask = cached_attention_mask(self.lengths, self.contexts, self.shape.width, self.capacity)

    def execute(self, runnable):
        self.prepare()
        return runnable(
            input_ids=self.input_ids, positions=self.positions, intermediate_tensors=None, inputs_embeds=None
        )

    def update(self, context, inputs, items):
        device = self.lengths.device
        # Fresh source allocations are retained by PyTorch's transfer allocator;
        # never overwrite pinned host storage while a preceding DMA is in flight.
        count, width = self.shape.requests, self.shape.width
        output_sources = [row * width + offset for row, item in enumerate(items) for offset in range(item.length)]
        output_targets = [item.start + offset for item in items for offset in range(item.length)]
        real_tokens = len(output_sources)
        items = (*items, *((None,) * (count - len(items))))
        host = torch.tensor(
            [0 if item is None else item.length for item in items]
            + [0 if item is None else item.context for item in items]
            + [0 if item is None else item.request for item in items]
            + padded_token_indices(items, width)
            + output_sources
            + output_targets,
            dtype=torch.int64,
            pin_memory=device.type != "cpu",
        )
        packed = host.to(device=device, non_blocking=True)
        lengths, contexts, rows, tokens, output_sources, output_targets = packed.split(
            [count, count, count, self.shape.tokens, real_tokens, real_tokens]
        )
        self.lengths.copy_(lengths)
        self.contexts.copy_(contexts)
        self.input_ids.copy_(inputs["input_ids"].index_select(0, tokens))
        self.positions.copy_(inputs["positions"].index_select(-1, tokens))
        valid = (torch.arange(self.shape.width, device=device)[None, :] < lengths[:, None]).reshape(-1)
        for mapping in self.lora:
            mapping.update(tokens, valid)
        seen = set()
        for name, target in self.metadata.items():
            if id(target) in seen:
                continue
            seen.add(id(target))
            source = context.attn_metadata[name]
            if type(target).__name__ == "GDNAttentionMetadata":
                all_states = getattr(source, "native_state_indices", None)
                slots = (
                    all_states[:, 0] if all_states is not None else source.non_spec_state_indices_tensor
                ).index_select(0, rows)
                target.non_spec_state_indices_tensor.copy_(torch.where(lengths > 0, slots, -1))
                if not self.shape.fresh and width > 1:
                    if all_states is None or all_states.shape[1] < width:
                        raise ValueError(f"Native verification requires {width} recurrent checkpoints per request")
                    target.spec_state_indices_tensor.copy_(
                        torch.where(lengths[:, None] > 0, all_states[:, :width].index_select(0, rows), -1)
                    )
                    accepted = source.native_num_accepted_tokens
                    if accepted is None:
                        raise ValueError("Native verification requires scheduler accepted-token counts")
                    target.num_accepted_tokens.copy_(torch.where(lengths > 0, accepted.index_select(0, rows), 1))
            else:
                # Async speculation may leave optimistic CPU sequence lengths.
                # Attention must use the authoritative device lengths after
                # rejection, without a device-to-host synchronization.
                if self.uses_speculation:
                    self.contexts.copy_(
                        torch.where(lengths > 0, source.native_seq_lens.index_select(0, rows) - lengths, 0)
                    )
                target.block_tables.copy_(source.block_tables.index_select(0, rows))
                slots = source.slot_mapping.index_select(0, tokens)
                target.slot_mapping.copy_(torch.where(valid, slots, -1))
        return output_sources, output_targets


class NativeFullGraphCache:
    entry_type = NativeFullEntry
    requires_gdn = True
    registry_name = "Native FULL"

    def __init__(self, config):
        self.config = config
        self.enabled = native_full_requested(config)
        if self.enabled and type((config.additional_config or {}).get("native_full_graph", True)) is not bool:
            raise ValueError("native_full_graph must be a boolean")
        self.ready = False
        self.warming = False
        self.entries = {}
        self.compatibility_steps = 0
        self.megagdn_workspace = None
        self.lora_wrappers = ()
        if not self.enabled:
            return
        self.num_speculative_tokens = native_full_speculative_tokens(config)
        self.backend = (config.additional_config or {}).get("native_full_graph_backend", "ascendc")
        if self.backend not in ("ascendc", "megagdn"):
            raise ValueError("native_full_graph_backend must be ascendc or megagdn")
        parallel = config.parallel_config
        if (
            any(
                getattr(parallel, field, 1) != 1
                for field in (
                    "tensor_parallel_size",
                    "pipeline_parallel_size",
                    "data_parallel_size",
                    "prefill_context_parallel_size",
                    "decode_context_parallel_size",
                )
            )
            or config.cache_config.mamba_cache_mode != "none"
            or config.kv_transfer_config is not None
            or getattr(config.quant_config, "enable_c8_quant", False)
        ):
            raise ValueError(
                "Native Qwen3.5 FULL requires single-rank execution, mamba_cache_mode=none, and no KV transfer or C8"
            )
        if config.lora_config is not None and (
            config.lora_config.max_lora_rank not in (8, 16, 32, 64) or config.lora_config.fully_sharded_loras
        ):
            raise ValueError("Native FULL LoRA requires unsharded adapters with max_lora_rank=8/16/32/64")
        if config.lora_config is not None:
            lora_dtype = getattr(config.lora_config, "lora_dtype", "auto")
            model_dtype = getattr(config.model_config, "dtype", None)
            if lora_dtype != "auto" and model_dtype is not None and lora_dtype != model_dtype:
                raise ValueError("Native FULL LoRA weights must use the model dtype; use lora_dtype=auto")
        capacity = (config.additional_config or {}).get(
            "native_full_graph_max_context", min(config.model_config.max_model_len, DEFAULT_NATIVE_FULL_MAX_CONTEXT)
        )
        if type(capacity) is not int or not 0 < capacity <= config.model_config.max_model_len:
            raise ValueError("native_full_graph_max_context must be a positive integer <= max_model_len")
        self.plan = NativeGraphPlan(
            config.compilation_config.cudagraph_capture_sizes or (),
            config.scheduler_config.max_num_batched_tokens,
            config.scheduler_config.max_num_seqs,
            capacity,
            (config.additional_config or {}).get("native_full_graph_request_counts"),
            self.num_speculative_tokens,
        )
        shared_workspace = (config.additional_config or {}).get("native_full_graph_megagdn_shared_workspace", True)
        if type(shared_workspace) is not bool:
            raise ValueError("native_full_graph_megagdn_shared_workspace must be a boolean")
        if self.backend == "megagdn" and shared_workspace:
            fresh_shapes = [shape for shape in self.plan.shapes if shape.fresh]
            # Max chunks can come from a different rectangle than max tokens:
            # every request, including short padded rows, has its own last chunk.
            self.megagdn_workspace = MegaGDNGraphWorkspace(
                max(shape.tokens for shape in fresh_shapes),
                max(shape.requests * ((shape.width + CHUNK_SIZE - 1) // CHUNK_SIZE) for shape in fresh_shapes),
            )

    def warmup(self, context, inputs, runnable):
        if "910B" not in torch.npu.get_device_name():
            raise ValueError("Native Qwen3.5 FULL currently targets Ascend 910B")
        if self.ready or self.entries:
            raise RuntimeError("Native FULL warmup must run once, before serving")
        if self.requires_gdn and not any(
            type(m).__name__ == "GDNAttentionMetadata" for m in context.attn_metadata.values()
        ):
            raise ValueError("Native FULL warmup requires Qwen3.5 GDN metadata")
        if self.requires_gdn and self.num_speculative_tokens:
            for metadata in context.attn_metadata.values():
                if type(metadata).__name__ == "GDNAttentionMetadata":
                    states = getattr(metadata, "native_state_indices", None)
                    if states is None or states.ndim != 2 or states.shape[1] < self.plan.decode_width:
                        raise ValueError(
                            f"Native capture requires {self.plan.decode_width} scheduler-owned recurrent checkpoints"
                        )
        if self.config.lora_config is not None:
            model = runnable if self.requires_gdn else runnable.model
            self.lora_wrappers = native_lora_wrappers(model, required=self.requires_gdn)
        # Warmup uses -1 write slots and zero-length decode rows. It never
        # reads/writes scheduler-owned recurrent state.
        stream = torch.npu.Stream()
        current = torch.npu.current_stream()
        stream.wait_stream(current)
        pool = None
        try:
            with torch.npu.stream(stream):
                # Largest first permits later graphs to reuse the allocator pool.
                for shape in sorted(self.plan.shapes, key=lambda s: s.tokens, reverse=True):
                    entry = self.entry_type(self, shape, context, inputs)
                    with native_context(context, entry.metadata, shape.tokens, lora=entry.lora):
                        entry.execute(runnable)
                        stream.synchronize()
                        entry.graph = torch.npu.NPUGraph()
                        with torch.npu.graph(entry.graph, pool=pool, stream=stream):
                            entry.output = entry.execute(runnable)
                        if not isinstance(entry.output, torch.Tensor):
                            raise ValueError("Native FULL requires a tensor model output")
                        if pool is None:
                            pool = entry.graph.pool()
                    self.entries[shape] = entry
            current.wait_stream(stream)
            current.synchronize()
        except BaseException:
            current.wait_stream(stream)
            raise
        self.ready = True
        if self.megagdn_workspace is not None:
            self.megagdn_workspace.sealed = True
            logger.info(
                "Native FULL MegaGDN shared scratch: bytes=%d geometries=%d (outside graph pools)",
                self.megagdn_workspace.reserved_bytes,
                len(self.megagdn_workspace.buffers),
            )
        logger.info(
            "%s registry sealed: entries=%d widths=%s max_requests=%d max_context=%d backend=%s requests=%s",
            self.registry_name,
            len(self.entries),
            self.plan.widths,
            self.plan.max_requests,
            self.plan.max_context,
            self.backend,
            self.plan.request_counts,
        )
        return next(iter(self.entries.values())).output[: inputs["positions"].shape[-1]].clone()

    def run(self, context, inputs, runnable=None):
        if not self.ready:
            raise RuntimeError("Native FULL serving started before startup capture completed")
        if (
            inputs.get("input_ids") is None
            or inputs.get("inputs_embeds") is not None
            or inputs.get("intermediate_tensors") is not None
        ):
            raise ValueError("Native FULL currently supports text token IDs on a single pipeline rank")
        if any(
            value is not None
            for name, value in inputs.items()
            if name not in ("input_ids", "positions", "inputs_embeds", "intermediate_tensors")
        ):
            raise ValueError("Native FULL received unsupported model inputs")
        attention = next((m for m in context.attn_metadata.values() if type(m).__name__ == "AscendMetadata"), None)
        if attention is None:
            raise ValueError("Native FULL requires standard attention metadata")
        # The runner's live count excludes its optional padding request.
        actual = attention.num_actual_tokens
        boundaries = [0] + [end for end in attention.actual_seq_lengths_q if end <= actual]
        if boundaries[-1] != actual:
            raise ValueError("Native FULL attention boundaries do not cover live tokens")
        lengths = attention.seq_lens_list[: len(boundaries) - 1]
        is_prefilling = None
        if self.num_speculative_tokens:
            gdn = next(m for m in context.attn_metadata.values() if type(m).__name__ == "GDNAttentionMetadata")
            flags = gdn.native_is_prefilling
            if flags is None or flags.device.type != "cpu":
                raise ValueError("Native verification requires scheduler CPU prefill/decode classification")
            is_prefilling = flags[: len(lengths)].tolist()
        batches = self.plan.batches(boundaries, lengths, is_prefilling)
        if batches is None:
            self.compatibility_steps += 1
            logger.debug("Native FULL compatibility step: non-fresh prefill or capture capacity exceeded")
            with native_context(context, context.attn_metadata, actual, mode=CUDAGraphMode.NONE):
                return runnable(**inputs)
        # Validate the entire step before writing any state.
        if any(shape not in self.entries for shape, _ in batches):
            raise RuntimeError("Native FULL registry is incomplete; eager/piecewise fallback is disabled")
        result = None
        for shape, items in batches:
            entry = self.entries[shape]
            output_sources, output_targets = entry.update(context, inputs, items)
            entry.graph.replay()
            entry.replays += 1
            if result is None:
                result = entry.output.new_empty((inputs["positions"].shape[-1], *entry.output.shape[1:]))
            # One gather/scatter per group, including at c64. Complete this
            # before another graph reuses the shared pool's output storage.
            result.index_copy_(0, output_targets, entry.output.index_select(0, output_sources))
            logger.debug(
                "Native FULL replay hit: requests=%d width=%d fresh=%s", shape.requests, shape.width, shape.fresh
            )
        return result
