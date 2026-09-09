# SPDX-License-Identifier: Apache-2.0
"""Native FULL prefill/mixed entries for the breakable ACLGraph wrapper.

The capacity route reads request lengths and state flags on device. Other
geometries retain a bounded layout-specialized fallback. Both routes refresh
graph-owned metadata and never borrow capture-time request buffers.
"""

import copy
from dataclasses import dataclass, is_dataclass
from enum import Enum
from typing import Any

import torch
from vllm.config import CUDAGraphMode
from vllm.logger import logger

from vllm_ascend.attention.attention_v1 import AscendAttentionBackendImpl
from vllm_ascend.compilation.acl_graph import _new_graph_params
from vllm_ascend.compilation.full_graph_metadata import FullGraphMetadataAdapter, capacity_metadata_signature

DEFAULT_FULL_PREFILL_GRAPH_MAX_ENTRIES = 8


def _flatten(value, tensors, memo, *, update_attention=False, host_values=True):
    if value is None or isinstance(value, (bool, int, float, str, Enum)):
        return (type(value), value), value
    identity = id(value)
    if identity in memo:
        index, static = memo[identity]
        return ("ref", index), static
    index = len(memo)
    if isinstance(value, torch.Tensor):
        # Preserve alias relationships, including metadata shared by layers.
        # Cloning is deferred until a cache miss has been admitted.
        memo[identity] = index, value
        tensors.append(value)
        values = tuple(value.reshape(-1).tolist()) if value.device.type == "cpu" and host_values else None
        return ("tensor", tuple(value.shape), value.dtype, value.device, tuple(value.stride()), values), value
    if isinstance(value, (dict, list, tuple)) or is_dataclass(value):
        memo[identity] = index, value
        if isinstance(value, dict):
            return (
                type(value),
                tuple(
                    (key, _flatten(item, tensors, memo, update_attention=update_attention)[0])
                    for key, item in value.items()
                ),
            ), value
        if isinstance(value, (list, tuple)):
            return (
                type(value),
                tuple(_flatten(item, tensors, memo, update_attention=update_attention)[0] for item in value),
            ), value
        # GDN attaches Ascend-specific fields after dataclass construction.
        fields = []
        for name, item in vars(value).items():
            dynamic_length = update_attention and type(value).__name__ == "AscendMetadata"
            if type(value).__name__ == "GDNChunkedPrefillMetadata" and name == "fresh_prefill":
                # This proof gates MegaGDN, which is excluded from FULL.
                # Baseline conv/GDN consume device initial-state flags, so a
                # fresh request and its continuation can share this graph.
                signature = (bool,)
            elif dynamic_length and name == "seq_lens_list":
                # Keep alias topology even though length values are dynamic:
                # two layers that shared a list at capture must not replay
                # with different per-layer values through that same buffer.
                if id(item) in memo:
                    signature = ("ref", memo[id(item)][0])
                else:
                    memo[id(item)] = len(memo), item
                    signature = (list, len(item))
            else:
                signature, _ = _flatten(
                    item,
                    tensors,
                    memo,
                    update_attention=update_attention,
                    host_values=not (dynamic_length and name == "seq_lens_cpu"),
                )
            fields.append((name, signature))
        return (type(value), tuple(fields)), value
    raise TypeError(f"Unsupported FULL prefill metadata: {type(value).__name__}")


def _clone(value, memo):
    if value is None or isinstance(value, (bool, int, float, str, Enum)):
        return value
    if id(value) in memo:
        return memo[id(value)]
    if isinstance(value, torch.Tensor):
        result = value.clone()
    elif isinstance(value, dict):
        result = {key: _clone(item, memo) for key, item in value.items()}
    elif isinstance(value, list):
        result = [_clone(item, memo) for item in value]
    elif isinstance(value, tuple):
        result = tuple(_clone(item, memo) for item in value)
    else:
        result = copy.copy(value)
        result.__dict__ = {key: _clone(item, memo) for key, item in vars(value).items()}
    memo[id(value)] = result
    return result


@dataclass
class FullPrefillEntry:
    batch_descriptor: Any
    inputs: Any
    tensors: list[torch.Tensor]
    capture: Any = None
    output: Any = None
    input_addresses: Any = None
    graph_params: Any = None
    metadata_adapter: Any = None


def _stateful_gdn_layers(context, attn_metadata=None):
    for name, metadata in (context.attn_metadata if attn_metadata is None else attn_metadata).items():
        if type(metadata).__name__ == "GDNAttentionMetadata" and (
            metadata.num_decodes > 0 or not metadata.non_spec_prefill_metadata.chunk.fresh_prefill
        ):
            yield name, metadata


def _snapshot_recurrent_states(context, attn_metadata=None):
    """Save only active cache rows on device, only when admitting a new graph."""
    snapshots = []
    seen = set()
    for name, metadata in _stateful_gdn_layers(context, attn_metadata):
        layer = context.no_compile_layers[name]
        # With mamba_cache_mode=none, conv and recurrent state use the same
        # block index. PIECEWISE input metadata contains live, unpadded rows.
        # Metadata builders use int32; index_copy_ requires int64 indices.
        source_indices = metadata.non_spec_state_indices_tensor
        indices = source_indices.to(dtype=torch.int64)
        for state in layer.kv_cache:
            identity = id(state), id(source_indices)
            if identity not in seen:
                seen.add(identity)
                snapshots.append((state, indices, state.index_select(0, indices)))
    return snapshots


def _restore_recurrent_states(snapshots):
    for state, indices, saved in snapshots:
        state.index_copy_(0, indices, saved)


class FullPrefillGraphCache:
    """Native graph cache with startup admission in FULL-only mode.

    FULL_AND_PIECEWISE permits disabling the cache or falling back when its
    entry limit is reached. FULL seals warmed capacities before serving and
    rejects misses without evicting graphs or changing execution mode.
    """

    def __init__(self, config):
        options = config.additional_config or {}
        self.max_entries = options.get("full_prefill_graph_max_entries", DEFAULT_FULL_PREFILL_GRAPH_MAX_ENTRIES)
        if type(self.max_entries) is not int or self.max_entries < 0:
            raise ValueError("full_prefill_graph_max_entries must be a non-negative integer")
        parallel = config.parallel_config
        self.enabled = (
            self.max_entries > 0
            and parallel.tensor_parallel_size == parallel.pipeline_parallel_size == parallel.data_parallel_size == 1
            and parallel.prefill_context_parallel_size == parallel.decode_context_parallel_size == 1
            and config.cache_config.mamba_cache_mode == "none"
            and config.speculative_config is None
            and config.kv_transfer_config is None
            and config.lora_config is None
            and not config.model_config.enforce_eager
            and not getattr(getattr(config, "quant_config", None), "enable_c8_quant", False)
        )
        if self.enabled:
            self.enabled = "910B" in torch.npu.get_device_name()
        self.entries = {}
        self.config = config
        self.full_only = (
            getattr(getattr(config, "compilation_config", None), "cudagraph_mode", None) == CUDAGraphMode.FULL
        )
        self.sealed = False
        self.update_stream = None
        self.capture_stream = None
        # Capacity graphs need at most one entry per configured token bucket.
        # Retain the explicit user limit, but do not stop the default c64
        # route at eight arbitrary layouts before all token buckets exist.
        sizes = getattr(getattr(config, "compilation_config", None), "cudagraph_capture_sizes", ()) or ()
        self.capacity_max_entries = options.get("full_prefill_graph_max_entries", max(self.max_entries, len(sizes)))

    def clear(self):
        if self.entries:
            # Complete queued replays before releasing their private metadata.
            torch.npu.current_stream().synchronize()
            self.entries.clear()
        self.sealed = self.full_only

    def begin_capture(self):
        self.sealed = False

    def seal(self):
        """Finish startup admission; serving may only replay existing graphs."""
        if self.full_only and not self.entries:
            raise RuntimeError("FULL startup did not capture any native prefill graphs")
        self.sealed = True

    def _eligible(self, context):
        allowed_mode = context.cudagraph_runtime_mode == CUDAGraphMode.PIECEWISE or (
            self.full_only and context.cudagraph_runtime_mode == CUDAGraphMode.FULL
        )
        if not self.enabled or not allowed_mode:
            return False
        if context.batch_descriptor is None or context.batch_descriptor.has_lora:
            return False
        if getattr(context, "is_draft_model", False) or getattr(context, "sinks", False):
            return False
        metadata = context.attn_metadata
        if not isinstance(metadata, dict) or not metadata:
            return False
        found_gdn = False
        for item in metadata.values():
            if type(item).__name__ not in ("GDNAttentionMetadata", "AscendMetadata"):
                return False
            if type(item).__name__ == "GDNAttentionMetadata":
                if item.num_prefills <= 0:
                    return False
                prefill = getattr(item, "non_spec_prefill_metadata", None)
                if (
                    item.num_spec_decodes != 0
                    or getattr(item, "spec_sequence_masks", None) is not None
                    or prefill is None
                ):
                    return False
                # The current Ascend kernels implement cache-mode=none's
                # single state anchor. Do not silently consume upstream
                # all-mode metadata before checkpoint/dual-anchor kernels land.
                if getattr(item, "all_state_indices_tensor", None) is not None:
                    return False
                boundaries = prefill.chunk.cu_seqlens_host
                if (
                    boundaries is None
                    or len(boundaries) != item.num_prefills + 1
                    or boundaries[0] != 0
                    or any(end <= start for start, end in zip(boundaries, boundaries[1:]))
                    or getattr(prefill.chunk, "keep_meta", None) is not None
                ):
                    return False
                found_gdn = True
        # Mixed and continuing-prefill captures must be able to rewind all
        # affected recurrent caches. Unknown layer ownership fails closed.
        layers = getattr(context, "no_compile_layers", {})
        for name, item in _stateful_gdn_layers(context):
            layer = layers.get(name)
            states = getattr(layer, "kv_cache", ())
            if len(states) != 2 or not all(isinstance(state, torch.Tensor) for state in states):
                return False
            indices = getattr(item, "non_spec_state_indices_tensor", None)
            if not isinstance(indices, torch.Tensor) or indices.shape != (item.num_decodes + item.num_prefills,):
                return False
        return found_gdn

    def _updates_attention(self, metadata):
        attention = [item for item in metadata.values() if type(item).__name__ == "AscendMetadata"]
        # PrefillNoCache consumes packed K/V rather than paged KV lengths.
        # Preserve Phase 1 for that path; task updates below use paged FIA.
        query_length_ids = {id(item.actual_seq_lengths_q) for item in attention}
        return bool(attention) and all(
            item.attn_state.name in ("ChunkedPrefill", "PrefillCacheHit")
            and id(item.seq_lens_list) not in query_length_ids
            for item in attention
        )

    def run(self, context, args, kwargs, *, runnable, capture, replay):
        if getattr(context, "in_profile_run", False) or (
            self.full_only and not self.sealed and context.cudagraph_runtime_mode == CUDAGraphMode.NONE
        ):
            return False, None
        required = (
            self.full_only
            and isinstance(context.attn_metadata, dict)
            and any(
                type(item).__name__ == "GDNAttentionMetadata" and item.num_prefills > 0
                for item in context.attn_metadata.values()
            )
        )
        if not self._eligible(context):
            if required:
                raise RuntimeError("FULL prefill rejected unsupported topology, cache mode, or GDN metadata")
            return False, None
        capacity_signature = capacity_metadata_signature(context, self.config)
        if required and capacity_signature is None:
            raise RuntimeError("FULL requires capacity-based GDN/FIA metadata; layout-specialized fallback is disabled")
        if capacity_signature is None and any(item.num_prefills <= 0 for item in context.attn_metadata.values()):
            return False, None
        inputs = (args, kwargs) if capacity_signature is not None else (args, kwargs, context.attn_metadata)
        update_attention = capacity_signature is not None or self._updates_attention(context.attn_metadata)
        sources = []
        try:
            signature, _ = _flatten(inputs, sources, {}, update_attention=update_attention)
        except TypeError as exc:
            if required:
                raise RuntimeError("FULL model inputs cannot be captured") from exc
            logger.debug("FULL prefill metadata not supported: %s", exc)
            return False, None
        # Capacity entries exclude live request metadata from this key. The
        # legacy route still specializes every host launch dimension.
        key = context.batch_descriptor, capacity_signature, update_attention, signature
        entry = self.entries.get(key)
        if required and entry is None and self.sealed:
            raise RuntimeError(
                f"FULL serving encountered an input signature not captured at startup: {context.batch_descriptor}"
            )
        limit = self.max_entries if capacity_signature is None else self.capacity_max_entries
        if not required and entry is None and len(self.entries) >= limit:
            logger.debug("FULL prefill graph cache at capacity: entries=%d", len(self.entries))
            return False, None

        if entry is None:
            static = _clone(inputs, {})
            targets = []
            _flatten(static, targets, {}, update_attention=update_attention)
            adapter = None
            if capacity_signature is not None:
                adapter = FullGraphMetadataAdapter(context, capacity_signature)
                static = (*static, adapter.metadata)
            entry = FullPrefillEntry(context.batch_descriptor, static, targets, metadata_adapter=adapter)
            if update_attention:
                sizes = (
                    {capacity_signature[0]}
                    if capacity_signature is not None
                    else {
                        item.actual_seq_lengths_q[-1]
                        for item in context.attn_metadata.values()
                        if type(item).__name__ == "AscendMetadata"
                    }
                )
                entry.graph_params = _new_graph_params(sorted(sizes))
                if self.update_stream is None:
                    self.update_stream = torch.npu.Stream()
        else:
            if entry.graph_params is not None:
                # Match standard FULL's task-update ordering: finish the
                # previous replay before changing handles or owned buffers.
                torch.npu.current_stream().synchronize()
            for target, source in zip(entry.tensors, sources, strict=True):
                target.copy_(source, non_blocking=True)
            if entry.metadata_adapter is None:
                for name, source in context.attn_metadata.items():
                    target = entry.inputs[2][name]
                    if update_attention and type(source).__name__ == "AscendMetadata":
                        target.seq_lens_list[:] = source.seq_lens_list
                    elif type(source).__name__ == "GDNAttentionMetadata":
                        target.non_spec_prefill_metadata.chunk.fresh_prefill = (
                            source.non_spec_prefill_metadata.chunk.fresh_prefill
                        )
        if entry.metadata_adapter is not None:
            entry.metadata_adapter.update(context.attn_metadata)

        old_mode, old_metadata = context.cudagraph_runtime_mode, context.attn_metadata
        old_capturing = context.capturing
        old_prefill = getattr(context, "full_prefill_graph", False)
        old_params = getattr(context, "full_prefill_graph_params", None)
        static_args, static_kwargs, context.attn_metadata = entry.inputs
        context.cudagraph_runtime_mode = CUDAGraphMode.FULL
        context.full_prefill_graph = True
        context.full_prefill_graph_params = entry.graph_params
        # Warmup uses ordinary FIA. The wrapper enables task-group capture
        # only when this entry owns attention parameters.
        context.capturing = False
        try:
            if entry.capture is None:
                # Warmup and capture may execute device work. Rewind live
                # conv/SSM rows after each so the serving replay advances each
                # request exactly once, including on the first graph use.
                snapshots = _snapshot_recurrent_states(context, old_metadata)
                try:
                    try:
                        runnable(*static_args, **static_kwargs)
                    finally:
                        _restore_recurrent_states(snapshots)
                    torch.npu.current_stream().synchronize()
                    # Unlike boot-time graph_capture, lazy admission starts
                    # on the request stream. NPU capture requires a separate
                    # non-default stream, reused across admitted entries.
                    if self.capture_stream is None:
                        self.capture_stream = torch.npu.Stream(device=torch.npu.current_device())
                    request_stream = torch.npu.current_stream()
                    if request_stream != self.capture_stream:
                        self.capture_stream.wait_stream(request_stream)
                    try:
                        with torch.npu.stream(self.capture_stream):
                            capture(entry, static_args, static_kwargs)
                    finally:
                        # Rewind state on the request stream only after all
                        # capture work completes, including on failure.
                        if request_stream != self.capture_stream:
                            request_stream.wait_stream(self.capture_stream)
                finally:
                    _restore_recurrent_states(snapshots)
                del snapshots
                if entry.capture.num_graphs != 1 or entry.capture.num_eager_breaks != 0:
                    raise RuntimeError("FULL prefill must capture the entire model without eager breaks")
                self.entries[key] = entry
                logger.info(
                    "FULL prefill graph captured: bucket=%s entries=%d", entry.batch_descriptor, len(self.entries)
                )
            # Execute even on a miss: serving must return the graph's output,
            # not depend on whether stream capture executes device work.
            result = replay(entry, static_args, static_kwargs)
            if entry.graph_params is not None:
                context.capturing = False
                # Replay waits on the captured ExternalEvents. Updates must
                # be submitted on a separate stream, after replay submission,
                # before returning to MRV1.
                for num_tokens in entry.graph_params.attn_params:
                    AscendAttentionBackendImpl.update_graph_params(
                        self.update_stream,
                        context,
                        num_tokens,
                        self.config,
                    )
            return True, result
        finally:
            context.cudagraph_runtime_mode, context.attn_metadata = old_mode, old_metadata
            context.capturing = old_capturing
            context.full_prefill_graph = old_prefill
            context.full_prefill_graph_params = old_params
