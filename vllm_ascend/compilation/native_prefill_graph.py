"""Startup-only AscendC prefill graphs; decode stays with the standard runner."""

import copy
from types import SimpleNamespace

import torch
from vllm.config import CUDAGraphMode
from vllm.forward_context import BatchDescriptor
from vllm.logger import logger

from vllm_ascend.ascend_forward_context import set_ascend_forward_context
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.compilation.native_prefill_layout import (
    DEFAULT_NATIVE_PREFILL_MAX_CONTEXT,
    NativePrefillPlan,
    padded_token_indices,
)
from vllm_ascend.ops.gdn_attn_builder import _build_non_spec_chunked_prefill_metadata
from vllm_ascend.ops.rotary_embedding import update_cos_sin


class NativePrefillEntry:
    def __init__(self, owner, shape, context, inputs):
        self.shape = shape
        self.graph = None
        self.output = None
        self.replays = 0
        device = inputs["positions"].device
        self.inputs = {
            name: None if inputs.get(name) is None else inputs[name].new_zeros((shape.tokens, *inputs[name].shape[1:]))
            for name in ("input_ids", "inputs_embeds")
        }
        self.inputs["positions"] = inputs["positions"].new_zeros((*inputs["positions"].shape[:-1], shape.tokens))
        self.inputs["intermediate_tensors"] = None
        self.lengths = torch.full((shape.requests,), shape.width, dtype=torch.int64, device=device)
        self.valid = torch.ones(shape.tokens, dtype=torch.bool, device=device)
        query_start_cpu = torch.arange(shape.requests + 1, dtype=torch.int32) * shape.width
        self.query_start = query_start_cpu.to(device)
        self.has_initial_state = torch.zeros(shape.requests, dtype=torch.bool, device=device)
        self.conv_indices = torch.arange(1, shape.requests + 1, dtype=torch.int32, device=device)
        self.chunk = _build_non_spec_chunked_prefill_metadata(
            SimpleNamespace(vllm_config=owner.config), query_start_cpu, device
        )
        self.metadata = {}
        memo = {}
        for name, source in context.attn_metadata.items():
            if id(source) in memo:
                self.metadata[name] = memo[id(source)]
                continue
            target = copy.copy(source)
            memo[id(source)] = target
            target.native_prefill = self
            target.num_actual_tokens = shape.tokens
            target.num_prefills = shape.requests
            target.num_decodes = 0
            target.num_decode_tokens = 0
            if type(source).__name__ == "GDNAttentionMetadata":
                target.non_spec_state_indices_tensor = torch.full(
                    (shape.requests,), -1, dtype=torch.int32, device=device
                )
            elif type(source).__name__ == "AscendMetadata":
                target.attn_state = AscendAttentionState.PrefillNoCache
                target.query_start_loc = self.query_start
                target.query_start_loc_gpu = self.query_start
                target.actual_seq_lengths_q = query_start_cpu[1:].tolist()
                target.seq_lens_list = [shape.width] * shape.requests
                target.seq_lens = torch.full((shape.requests,), shape.width, dtype=torch.int32, device=device)
                target.seq_lens_gpu = target.seq_lens
                target.seq_lens_cpu = torch.full((shape.requests,), shape.width, dtype=torch.int32)
                target.max_query_len = shape.width
                target.slot_mapping = source.slot_mapping.new_full((shape.tokens,), -1)
                target.block_tables = source.block_tables.new_zeros((shape.requests, source.block_tables.shape[1]))
            else:
                raise ValueError(f"Native prefill cannot capture metadata {type(source).__name__}")
            self.metadata[name] = target

    def execute(self, runnable):
        self.valid.copy_(
            (torch.arange(self.shape.width, device=self.lengths.device)[None, :] < self.lengths[:, None]).reshape(-1)
        )
        update_cos_sin(self.inputs["positions"])
        return runnable(**self.inputs)

    def accepts(self, inputs):
        if inputs.get("intermediate_tensors") is not None:
            return False
        if any(value is not None for name, value in inputs.items() if name not in self.inputs):
            return False
        for name in ("input_ids", "inputs_embeds", "positions"):
            source, target = inputs.get(name), self.inputs[name]
            if (source is None) != (target is None):
                return False
            if source is not None:
                axis = -1 if name == "positions" else 0
                source_shape, target_shape = list(source.shape), list(target.shape)
                source_shape[axis] = target_shape[axis] = 0
                if source_shape != target_shape or source.dtype != target.dtype or source.device != target.device:
                    return False
        return True

    def update(self, context, inputs, items):
        count, width = self.shape.requests, self.shape.width
        output_sources = [row * width + offset for row, item in enumerate(items) for offset in range(item.length)]
        output_targets = [item.start + offset for item in items for offset in range(item.length)]
        real_tokens = len(output_sources)
        items = (*items, *((None,) * (count - len(items))))
        host = torch.tensor(
            [0 if item is None else item.length for item in items]
            + [0 if item is None else item.request for item in items]
            + padded_token_indices(items, width)
            + output_sources
            + output_targets,
            dtype=torch.int64,
            pin_memory=self.lengths.device.type != "cpu",
        )
        packed = host.to(device=self.lengths.device, non_blocking=True)
        lengths, rows, tokens, output_sources, output_targets = packed.split(
            [count, count, self.shape.tokens, real_tokens, real_tokens]
        )
        self.lengths.copy_(lengths)
        for name in ("input_ids", "inputs_embeds", "positions"):
            if self.inputs[name] is not None:
                axis = -1 if name == "positions" else 0
                self.inputs[name].copy_(inputs[name].index_select(axis, tokens))
        valid = (torch.arange(width, device=lengths.device)[None, :] < lengths[:, None]).reshape(-1)
        seen = set()
        for name, target in self.metadata.items():
            if id(target) in seen:
                continue
            seen.add(id(target))
            source = context.attn_metadata[name]
            if type(target).__name__ == "GDNAttentionMetadata":
                indices = source.non_spec_state_indices_tensor
                if indices.ndim == 2:
                    indices = indices[:, 0]
                target.non_spec_state_indices_tensor.copy_(torch.where(lengths > 0, indices.index_select(0, rows), -1))
            else:
                slots = source.slot_mapping.index_select(0, tokens)
                target.slot_mapping.copy_(torch.where(valid, slots, -1))
        return output_sources, output_targets


class NativePrefillGraphCache:
    def __init__(self, config):
        self.config = config
        options = config.additional_config or {}
        self.enabled = options.get("native_prefill_graph", False)
        if type(self.enabled) is not bool:
            raise ValueError("native_prefill_graph must be a boolean")
        self.ready = False
        self.warming = False
        self.entries = {}
        if not self.enabled:
            return
        text_config = config.model_config.hf_text_config
        # vLLM 0.30 downgrades FULL to PIECEWISE for pooling models. Clef's
        # inherited Qwen3.5 forward returns all token hidden states; its
        # schema-dependent pooler runs later in the runner, outside this
        # registry. Its backbone capture therefore does not need the standard
        # dispatcher to support FULL (or to capture any graphs at all).
        clef_pooling = (
            getattr(config.model_config, "architecture", None) == "ClefFlashForJointSchema"
            and getattr(config.model_config, "runner_type", None) == "pooling"
        )
        if (
            getattr(text_config, "model_type", "") not in ("qwen3_5", "qwen3_5_text")
            or config.model_config.enforce_eager
            or (not clef_pooling and not config.compilation_config.cudagraph_mode.has_full_cudagraphs())
        ):
            raise ValueError(
                "Native prefill graphs require non-eager dense Qwen3.5 with FULL graph support enabled, "
                "or ClefFlashForJointSchema with runner=pooling (backbone capture only). "
                f"Got model_type={getattr(text_config, 'model_type', None)}, "
                f"runner={getattr(config.model_config, 'runner_type', None)}, "
                f"cudagraph_mode={config.compilation_config.cudagraph_mode}, "
                f"enforce_eager={config.model_config.enforce_eager}"
            )
        if (
            any(
                getattr(config.parallel_config, field, 1) != 1
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
            or config.lora_config is not None
            or config.kv_transfer_config is not None
            or config.quant_config is not None
        ):
            raise ValueError(
                "Native prefill graphs require single-rank unquantized execution, mamba_cache_mode=none, "
                "and no speculation, LoRA or KV transfer"
            )
        capacity = options.get(
            "native_prefill_graph_max_context",
            min(config.model_config.max_model_len, DEFAULT_NATIVE_PREFILL_MAX_CONTEXT),
        )
        if type(capacity) is not int or not 0 < capacity <= config.model_config.max_model_len:
            raise ValueError("native_prefill_graph_max_context must be a positive integer <= max_model_len")
        self.plan = NativePrefillPlan(
            config.compilation_config.cudagraph_capture_sizes or (),
            config.scheduler_config.max_num_batched_tokens,
            config.scheduler_config.max_num_seqs,
            capacity,
            options.get("native_prefill_graph_request_counts"),
        )

    def warmup(self, context, inputs, runnable):
        if "910B" not in torch.npu.get_device_name():
            raise ValueError("Native prefill graphs currently target Ascend 910B")
        if self.ready or self.entries:
            raise RuntimeError("Native prefill capture must run once, before serving")
        if not any(type(metadata).__name__ == "GDNAttentionMetadata" for metadata in context.attn_metadata.values()):
            raise ValueError("Native prefill capture requires GDN metadata")
        stream = torch.npu.Stream()
        current = torch.npu.current_stream()
        stream.wait_stream(current)
        pool = None
        try:
            with torch.npu.stream(stream):
                for shape in sorted(self.plan.shapes, key=lambda shape: shape.tokens, reverse=True):
                    entry = NativePrefillEntry(self, shape, context, inputs)
                    with set_ascend_forward_context(
                        entry.metadata,
                        self.config,
                        num_tokens=shape.tokens,
                        num_actual_tokens=shape.tokens,
                        aclgraph_runtime_mode=CUDAGraphMode.FULL,
                        batch_descriptor=BatchDescriptor(num_tokens=shape.tokens),
                        model_instance=runnable,
                        skip_compiled=True,
                    ):
                        entry.execute(runnable)
                        stream.synchronize()
                        entry.graph = torch.npu.NPUGraph()
                        with torch.npu.graph(entry.graph, pool=pool, stream=stream):
                            entry.output = entry.execute(runnable)
                        if not isinstance(entry.output, torch.Tensor):
                            raise ValueError("Native prefill graphs require a tensor model output")
                        if pool is None:
                            pool = entry.graph.pool()
                    self.entries[shape] = entry
            current.wait_stream(stream)
            current.synchronize()
        except BaseException:
            current.wait_stream(stream)
            raise
        self.ready = True
        logger.info(
            "Native prefill registry sealed: entries=%d widths=%s requests=%s backend=ascendc",
            len(self.entries),
            self.plan.widths,
            self.plan.request_counts,
        )
        return next(iter(self.entries.values())).output[: inputs["positions"].shape[-1]].clone()

    def replay(self, context, inputs):
        """Return None before any mutation when the standard path must run."""
        if not self.ready or context.attn_metadata is None or context.capturing:
            return None
        attention = next(
            (metadata for metadata in context.attn_metadata.values() if type(metadata).__name__ == "AscendMetadata"),
            None,
        )
        if (
            attention is None
            or attention.attn_state != AscendAttentionState.PrefillNoCache
            or attention.num_actual_tokens == 0
        ):
            return None
        actual = attention.num_actual_tokens
        boundaries = [0]
        for end in attention.actual_seq_lengths_q:
            if end > actual or boundaries[-1] == actual:
                break
            boundaries.append(end)
        if boundaries[-1] != actual:
            return None
        batches = self.plan.batches(boundaries, attention.seq_lens_list[: len(boundaries) - 1])
        if not batches:
            return None
        if any(shape not in self.entries for shape, _ in batches):
            raise RuntimeError("Native prefill startup registry is incomplete")
        if any(not self.entries[shape].accepts(inputs) for shape, _ in batches):
            return None
        result = None
        for shape, items in batches:
            entry = self.entries[shape]
            output_sources, output_targets = entry.update(context, inputs, items)
            entry.graph.replay()
            entry.replays += 1
            if result is None:
                result = entry.output.new_zeros((inputs["positions"].shape[-1], *entry.output.shape[1:]))
            result.index_copy_(0, output_targets, entry.output.index_select(0, output_sources))
            logger.debug("Native prefill replay hit: requests=%d width=%d", shape.requests, shape.width)
        return result
