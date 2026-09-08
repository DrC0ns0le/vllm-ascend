# SPDX-License-Identifier: Apache-2.0
"""Adapt runner metadata to fixed-capacity, graph-owned target-model inputs."""

import copy
import math

import torch

from vllm_ascend.ops.gdn_graph_metadata import GDN_GRAPH_HEAD_DIM, MAX_GDN_GRAPH_HEADS
from vllm_ascend.ops.triton.fla.graph import allocate_graph_metadata, update_graph_metadata
from vllm_ascend.ops.triton.graph_metadata import refresh_attention_metadata

MAX_GRAPH_REQUESTS = 64
FIA_CACHE_BLOCK_SIZE = 128


def capacity_metadata_signature(context, config):
    """Return a capacity signature only for the implemented BF16 GDN/FIA route."""
    layers = getattr(context, "no_compile_layers", {})
    scheduler = getattr(config, "scheduler_config", None)
    if scheduler is None:
        return None
    tokens = context.batch_descriptor.num_tokens
    requests = min(scheduler.max_num_seqs, tokens)
    if requests > MAX_GRAPH_REQUESTS:
        return None
    aliases = {}
    result = []
    found_attention = False
    for name, metadata in context.attn_metadata.items():
        count = metadata.num_decodes + metadata.num_prefills
        if count > requests or metadata.num_actual_tokens > tokens:
            return None
        if id(metadata) not in aliases:
            aliases[id(metadata)] = len(aliases)
        if type(metadata).__name__ == "GDNAttentionMetadata":
            layer = layers.get(name)
            states = getattr(layer, "kv_cache", ())
            if len(states) != 2 or not all(isinstance(state, torch.Tensor) for state in states):
                return None
            if states[1].ndim != 4 or states[1].shape[-2:] != (GDN_GRAPH_HEAD_DIM, GDN_GRAPH_HEAD_DIM):
                return None
            heads = states[1].shape[1]
            if heads > MAX_GDN_GRAPH_HEADS or heads < 1 or heads & (heads - 1):
                return None
            conv = metadata.non_spec_prefill_metadata.causal_conv1d
            if conv.query_start_loc.shape != (count + 1,) or conv.initial_state_mode is None:
                return None
            # Cache-mode none's builder passes the single-column block
            # table to conv1d, while recurrent metadata uses a vector view.
            if conv.initial_state_mode.shape != (count,) or conv.cache_indices.shape not in ((count,), (count, 1)):
                return None
            result.append((name, "gdn", aliases[id(metadata)], states[1].shape[1]))
        else:
            if metadata.attn_state.name not in ("PrefillNoCache", "PrefillCacheHit", "ChunkedPrefill", "DecodeOnly"):
                return None
            if not metadata.causal or metadata.model_runner_type == "pooling":
                return None
            if getattr(metadata, "reshape_cache_event", None) is not None:
                return None
            if not isinstance(metadata.actual_seq_lengths_q, list) or not isinstance(metadata.seq_lens_list, list):
                return None
            if (
                len(metadata.actual_seq_lengths_q) != count
                or metadata.actual_seq_lengths_q[-1] != metadata.num_actual_tokens
            ):
                return None
            if len(metadata.seq_lens_list) != count or metadata.block_tables.shape[0] < count:
                return None
            mask = metadata.attn_mask
            mask_layout = None if mask is None else (tuple(mask.shape), mask.dtype, mask.device)
            block_size = config.cache_config.block_size or FIA_CACHE_BLOCK_SIZE
            columns = max(metadata.block_tables.shape[1], math.ceil(tokens / block_size))
            result.append((name, "fia", aliases[id(metadata)], columns, metadata.block_tables.dtype, mask_layout))
            found_attention = True
    return (tokens, requests, tuple(result)) if found_attention else None


class FullGraphMetadataAdapter:
    def __init__(self, context, signature):
        self.tokens, self.requests, layout = signature
        self.metadata = {}
        self.heads = {}
        shared = {}
        for spec in layout:
            name, backend, alias = spec[:3]
            source = context.attn_metadata[name]
            # GDN layers sharing runner metadata can have different head
            # geometry. Include geometry in the ownership key.
            ownership = backend, alias, spec[3:]
            if ownership in shared:
                self.metadata[name] = shared[ownership]
                if backend == "gdn":
                    self.heads[name] = spec[3]
                continue
            if backend == "gdn":
                heads = spec[3]
                target = allocate_graph_metadata(
                    self.tokens, self.requests, heads, source.non_spec_state_indices_tensor.device
                )
                self.heads[name] = heads
            else:
                target = copy.copy(source)
                # Fresh prefill also uses the just-written paged KV cache.
                target.attn_state = type(source.attn_state).ChunkedPrefill
                target.full_graph_token_capacity = self.tokens
                rows = self.requests + 1
                target.block_tables = source.block_tables.new_zeros((rows, spec[3]))
                target.slot_mapping = source.slot_mapping.new_full((self.tokens,), -1)
                target.seq_lens = source.seq_lens.new_zeros(rows)
                # The standard Ascend builder supplies host sequence lengths.
                # Share their graph-owned buffer instead of constructing and
                # copying a second CPU tensor on every metadata refresh.
                target.seq_lens_cpu = (
                    target.seq_lens
                    if target.seq_lens.device.type == "cpu"
                    else torch.zeros(rows, dtype=source.seq_lens.dtype)
                )
                target.query_start_loc = source.query_start_loc.new_zeros(rows + 1)
                target.seq_lens_list = [0] * rows
                target.actual_seq_lengths_q = [0] * rows
                target.attn_mask = None if source.attn_mask is None else source.attn_mask.clone()
            self.metadata[name] = target
            shared[ownership] = target

    def update(self, sources):
        seen = set()
        for name, target in self.metadata.items():
            if id(target) in seen:
                continue
            seen.add(id(target))
            source = sources[name]
            count = source.num_decodes + source.num_prefills
            if name in self.heads:
                conv = source.non_spec_prefill_metadata.causal_conv1d
                update_graph_metadata(
                    target,
                    conv.query_start_loc,
                    conv.cache_indices,
                    conv.initial_state_mode,
                    self.heads[name],
                    skip_single_token=True,
                )
                continue
            actual = source.num_actual_tokens
            padding = self.tokens - actual
            # Empty request rows keep the same terminal offset. One dummy
            # attention sequence consumes token padding, keeping FIA's T
            # fixed even when the total scheduled token count changes.
            target.actual_seq_lengths_q[:] = (
                source.actual_seq_lengths_q + [actual] * (self.requests - count) + [self.tokens]
            )
            target.seq_lens_list[:] = source.seq_lens_list + [0] * (self.requests - count) + [padding]
            target.seq_lens.zero_()
            target.seq_lens[:count].copy_(source.seq_lens[:count])
            target.seq_lens[-1:].fill_(padding)
            if target.seq_lens_cpu is not target.seq_lens:
                target.seq_lens_cpu.copy_(torch.tensor(target.seq_lens_list, dtype=target.seq_lens_cpu.dtype))
            refresh_attention_metadata(target, source, count, actual)
            # Dummy queries read valid allocated block 0, possibly repeated;
            # their outputs are discarded and their KV slots are all -1.
            if target.attn_mask is not None:
                target.attn_mask.copy_(source.attn_mask)
            target.num_actual_tokens = actual
            target.num_decodes, target.num_prefills = source.num_decodes, source.num_prefills
            target.num_decode_tokens = source.num_decode_tokens
            target.max_query_len = source.max_query_len
