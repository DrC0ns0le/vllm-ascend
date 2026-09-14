# SPDX-License-Identifier: Apache-2.0
"""Static Qwen3.5 MTP graphs: one padded first pass and subsequent skinny passes.

The model passes, LM heads, greedy draft selection and draft KV updates
are captured together. Target rejection sampling remains owned by vLLM.
"""

from types import SimpleNamespace

import torch
from vllm.forward_context import get_forward_context

from vllm_ascend.compilation.native_full_graph import NativeFullEntry, NativeFullGraphCache, native_context
from vllm_ascend.compilation.native_full_layout import NativeGraphShape, padded_token_indices


class NativeMTPEntry(NativeFullEntry):
    def __init__(self, owner, shape, context, inputs):
        super().__init__(owner, shape, context, inputs)
        count, tokens = shape.requests, shape.tokens
        self.hidden_states = inputs["hidden_states"].new_zeros((tokens, inputs["hidden_states"].shape[-1]))
        self.sample_indices = torch.zeros(count, dtype=torch.int64, device=self.lengths.device)
        self.active = torch.zeros(count, dtype=torch.bool, device=self.lengths.device)
        self.block_size = owner.block_size
        self.max_model_len = owner.config.model_config.max_model_len
        self.next_steps = [
            NativeFullEntry(owner, NativeGraphShape(count, 1, False), context, inputs)
            for _ in range(owner.num_speculative_tokens - 1)
        ]
        for step in self.next_steps:
            for name, metadata in step.metadata.items():
                metadata.block_tables = self.metadata[name].block_tables

    def execute(self, proposer):
        self.prepare()
        context = get_forward_context()
        with native_context(context, self.metadata, self.shape.tokens, lora=self.lora):
            hidden = proposer.model(
                input_ids=self.input_ids,
                positions=self.positions,
                hidden_states=self.hidden_states,
                inputs_embeds=None,
            )
            hidden = hidden.index_select(0, self.sample_indices)
            positions = self.positions.index_select(-1, self.sample_indices)
            token_ids = proposer.compute_draft_token_ids(hidden)
        drafts = [token_ids]
        for index, step in enumerate(self.next_steps):
            # Rejected first-pass tokens never extend the draft context. The
            # first pass's length ends at the sampler's last accepted token.
            step.contexts.copy_(torch.where(self.active, self.contexts + self.lengths + index, 0))
            step.lengths.copy_(self.active)
            for mapping, first in zip(step.lora, self.lora, strict=True):
                mapping.next_draft(first, self.active)
            step.prepare()
            for metadata in {id(m): m for m in step.metadata.values()}.values():
                logical_positions = step.contexts
                columns = (logical_positions // self.block_size).clamp(max=metadata.block_tables.shape[1] - 1)
                blocks = metadata.block_tables.gather(1, columns[:, None]).squeeze(1)
                slots = blocks.to(torch.int64) * self.block_size + logical_positions % self.block_size
                metadata.slot_mapping.copy_(torch.where(self.active, slots, -1))
            positions = positions + 1
            positions = torch.where(positions < self.max_model_len, positions, 0)
            with native_context(context, step.metadata, step.shape.tokens, lora=step.lora):
                hidden = proposer.model(
                    input_ids=token_ids, positions=positions, hidden_states=hidden, inputs_embeds=None
                )
                token_ids = proposer.compute_draft_token_ids(hidden)
            drafts.append(token_ids)
        return torch.stack(drafts, dim=1)

    def update_draft(self, context, inputs, sample_indices, items):
        super().update(context, inputs, items)
        count, width = self.shape.requests, self.shape.width
        items = (*items, *((None,) * (count - len(items))))
        host = torch.tensor(
            [0 if item is None else item.request for item in items]
            + [0 if item is None else item.start for item in items]
            + padded_token_indices(items, width),
            dtype=torch.int64,
            pin_memory=self.lengths.device.type != "cpu",
        )
        rows, starts, tokens = host.to(self.lengths.device, non_blocking=True).split([count, count, self.shape.tokens])
        # Padded proposer inputs retain rejected tokens. Exclude those device
        # positions from KV writes and subsequent draft passes without a D2H.
        selected = sample_indices.index_select(0, rows) - starts
        self.active.copy_((self.lengths > 0) & (selected >= 0) & (selected < self.lengths))
        self.lengths.copy_(torch.where(self.active, selected + 1, 0))
        self.contexts.copy_(torch.where(self.active, self.contexts, 0))
        self.sample_indices.copy_(
            torch.arange(count, device=rows.device) * width + torch.where(self.active, selected, 0)
        )
        self.hidden_states.copy_(inputs["hidden_states"].index_select(0, tokens))
        valid = (torch.arange(width, device=rows.device)[None, :] < self.lengths[:, None]).reshape(-1)
        for mapping in self.lora:
            mapping.mask_padding(valid)
        for metadata in {id(m): m for m in self.metadata.values()}.values():
            metadata.slot_mapping.masked_fill_(~valid, -1)
        return rows[: sum(item is not None for item in items)]


class NativeMTPGraphCache(NativeFullGraphCache):
    entry_type = NativeMTPEntry
    requires_gdn = False
    registry_name = "Native MTP"

    def __init__(self, config, proposer):
        super().__init__(config)
        self.proposer = proposer
        if (
            config.speculative_config.method not in ("mtp", "qwen3_5_mtp")
            or self.num_speculative_tokens not in (1, 2, 3)
            or proposer.model_returns_tuple()
            or not proposer.pass_hidden_states_to_model
        ):
            raise ValueError("Native MTP graphs require the Qwen3.5 hidden-state MTP interface and 1/2/3 draft tokens")
        # Draft layers are full attention; MegaGDN remains a target-prefill option.
        self.backend = "ascendc"
        self.megagdn_workspace = None

    def warmup(self, context, inputs, runnable):
        # Attention groups are assigned after model loading, once cache specs
        # are available. Resolve their block size at capture, not construction.
        if len(self.proposer.draft_attn_groups) != 1 or any(
            type(metadata).__name__ != "AscendMetadata" for metadata in context.attn_metadata.values()
        ):
            raise ValueError("Native Qwen3.5 MTP capture requires one standard full-attention cache group")
        self.block_size = self.proposer.draft_attn_groups[0].kv_cache_spec.block_size
        return super().warmup(context, inputs, runnable)

    def propose(self, common, num_tokens, sample_indices):
        if not self.ready:
            raise RuntimeError("Native MTP serving started before draft graph warmup")
        boundaries = common.query_start_loc_cpu[: common.num_reqs + 1].tolist()
        lengths_cpu = common.seq_lens_cpu_upper_bound
        if lengths_cpu is None:
            lengths_cpu = common.seq_lens_cpu
        if (
            lengths_cpu is None
            or common.is_prefilling is None
            or lengths_cpu.device.type != "cpu"
            or common.is_prefilling.device.type != "cpu"
        ):
            raise ValueError("Native MTP needs scheduler CPU request boundaries and prefill flags")
        lengths = lengths_cpu[: common.num_reqs].tolist()
        # Reserve room for the remaining draft KV writes in the captured
        # dense attention capacity. Outside it, use the existing proposer.
        if any(total + self.num_speculative_tokens - 1 > self.plan.max_context for total in lengths):
            self.compatibility_steps += 1
            return None
        batches = self.plan.batches(boundaries, lengths, common.is_prefilling[: common.num_reqs].tolist())
        if batches is None:
            self.compatibility_steps += 1
            return None
        if any(shape not in self.entries for shape, _ in batches):
            raise RuntimeError("Native MTP registry is incomplete")
        inputs = dict(
            input_ids=self.proposer.input_ids[:num_tokens],
            positions=self.proposer._get_positions(num_tokens),
            hidden_states=self.proposer.hidden_states[:num_tokens],
        )
        source = SimpleNamespace(
            block_tables=common.block_table_tensor, slot_mapping=common.slot_mapping, native_seq_lens=common.seq_lens
        )
        result = torch.empty(
            (common.num_reqs, self.num_speculative_tokens), dtype=torch.int64, device=sample_indices.device
        )
        for shape, items in batches:
            entry = self.entries[shape]
            context = SimpleNamespace(attn_metadata={name: source for name in entry.metadata})
            rows = entry.update_draft(context, inputs, sample_indices, items)
            entry.graph.replay()
            entry.replays += 1
            # Graphs share their pool; preserve this group's outputs before
            # another group reuses it. Proposer input buffers are read-only.
            result.index_copy_(0, rows, entry.output[: len(items)])
        return result
