# SPDX-License-Identifier: Apache-2.0
"""Startup-only native model graph capture with bounded shape adaptation.

No custom attention/recurrence kernels, graph task updates, serving-time graph
admission, or per-layer Python callbacks. Graphs own all mutable replay inputs.
"""

import copy
from contextlib import contextmanager
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
from vllm_ascend.ops.gdn_attn_builder import (
    GDNCausalConv1dMetadata,
    GDNDecodeMetadata,
    GDNPrefillMetadata,
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


@contextmanager
def native_context(context, metadata, tokens, mode=CUDAGraphMode.FULL):
    fields = dict(
        attn_metadata=metadata,
        cudagraph_runtime_mode=mode,
        capturing=False,
        batch_descriptor=BatchDescriptor(num_tokens=tokens),
    )
    for name in ("num_tokens", "padded_num_tokens", "num_actual_tokens"):
        if hasattr(context, name):
            fields[name] = tokens
    saved = {name: getattr(context, name) for name in fields}
    try:
        for name, value in fields.items():
            setattr(context, name, value)
        yield
    finally:
        for name, value in saved.items():
            setattr(context, name, value)


class NativeFullEntry:
    def __init__(self, owner, shape, context, inputs):
        self.shape = shape
        self.backend = owner.backend
        self.megagdn_workspace = owner.megagdn_workspace
        self.graph = None
        self.output = None
        self.replays = 0
        device = inputs["positions"].device
        count, width = shape.requests, shape.width
        tokens = shape.tokens
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

    def execute(self, runnable):
        self.valid.copy_(
            (torch.arange(self.shape.width, device=self.lengths.device)[None, :] < self.lengths[:, None]).reshape(-1)
        )
        if not self.shape.fresh:
            for metadata in {id(m): m for m in self.metadata.values()}.values():
                if type(metadata).__name__ == "GDNAttentionMetadata":
                    metadata.non_spec_decode_metadata.actual_seq_lengths[1:].copy_(self.lengths)
            self.mask = cached_attention_mask(self.lengths, self.contexts, self.shape.width, self.capacity)
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
        seen = set()
        for name, target in self.metadata.items():
            if id(target) in seen:
                continue
            seen.add(id(target))
            source = context.attn_metadata[name]
            if type(target).__name__ == "GDNAttentionMetadata":
                target.non_spec_state_indices_tensor.copy_(
                    torch.where(lengths > 0, source.non_spec_state_indices_tensor.index_select(0, rows), -1)
                )
            else:
                target.block_tables.copy_(source.block_tables.index_select(0, rows))
                slots = source.slot_mapping.index_select(0, tokens)
                target.slot_mapping.copy_(torch.where(valid, slots, -1))
        return output_sources, output_targets


class NativeFullGraphCache:
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
        if not self.enabled:
            return
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
            or config.speculative_config is not None
            or config.kv_transfer_config is not None
            or config.lora_config is not None
            or getattr(config.quant_config, "enable_c8_quant", False)
        ):
            raise ValueError(
                "Native Qwen3.5 FULL requires single-rank execution, mamba_cache_mode=none, "
                "and no speculation, LoRA, KV transfer or C8"
            )
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
        if not any(type(m).__name__ == "GDNAttentionMetadata" for m in context.attn_metadata.values()):
            raise ValueError("Native FULL warmup requires Qwen3.5 GDN metadata")
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
                    entry = NativeFullEntry(self, shape, context, inputs)
                    with native_context(context, entry.metadata, shape.tokens):
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
            "Native FULL registry sealed: entries=%d widths=%s max_requests=%d max_context=%d backend=%s requests=%s",
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
        batches = self.plan.batches(boundaries, lengths)
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
