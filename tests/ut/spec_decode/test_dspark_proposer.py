#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
"""Unit tests for the dspark speculative-decoding proposer."""

from __future__ import annotations

import inspect
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from vllm.config import CUDAGraphMode
from vllm.v1.worker.utils import AttentionGroup

import vllm_ascend.spec_decode.dspark_proposer as dspark_proposer_module
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.dsa_v1 import AscendDSAMetadataBuilder
from vllm_ascend.spec_decode.dflash_proposer import AscendDflashProposer
from vllm_ascend.spec_decode.dspark_proposer import AscendDSparkProposer
from vllm_ascend.spec_decode.llm_base_proposer import AscendSpecDecodeBaseProposer

# 0 = single-DP (no padding); >0 = multi-DP where num_input_tokens >
# num_query_total, the out-of-bounds regime.
MULTI_DP_PADDING_SIZES = [0, 8, 32]
_NUM_SPECULATIVE_TOKENS = 3
_MAX_BATCH_SIZE = 2
_MAX_NUM_TOKENS = 8
_HIDDEN_SIZE = 16


class _DSparkProposerTestBase:
    """Shared helpers for ``AscendDSparkProposer`` tests."""

    @staticmethod
    def _make_vllm_config(hf_config: SimpleNamespace) -> SimpleNamespace:
        """Build the minimal config consumed by the DSpark initializer."""
        draft_model_config = SimpleNamespace(hf_config=hf_config, get_hidden_size=lambda: _HIDDEN_SIZE)
        return SimpleNamespace(
            speculative_config=SimpleNamespace(draft_sample_method="greedy", draft_model_config=draft_model_config),
            model_config=SimpleNamespace(hf_text_config=SimpleNamespace(model_type="deepseek_v4")),
        )

    @classmethod
    def _make_proposer(
        cls,
        *,
        max_num_tokens: int,
        num_reqs: int,
        block_size: int,
        hf_config: SimpleNamespace | None = None,
        draft_attn_causal: bool | None = None,
    ):
        device = torch.device("cpu")
        vllm_config = cls._make_vllm_config(hf_config or SimpleNamespace())

        def mock_parent_init(
            proposer: AscendDSparkProposer,
            vllm_config: SimpleNamespace,
            device: torch.device,
            runner: object | None = None,
        ) -> None:
            del runner
            proposer.draft_model_config = vllm_config.speculative_config.draft_model_config
            proposer.num_speculative_tokens = block_size
            proposer.max_batch_size = num_reqs
            proposer.max_num_tokens = max_num_tokens
            proposer.dtype = torch.float32
            proposer.device = device
            proposer.hidden_size = _HIDDEN_SIZE
            proposer.hidden_states = torch.empty(0)
            proposer._dflash_hidden_states = torch.empty(0)
            proposer.use_cuda_graph = False
            proposer.model = (
                SimpleNamespace(get_draft_attn_causal=lambda: [draft_attn_causal])
                if draft_attn_causal is not None
                else SimpleNamespace()
            )

        with patch.object(AscendDSparkProposer.__base__, "__init__", mock_parent_init):
            proposer = AscendDSparkProposer(vllm_config, device)
        num_query_total = num_reqs * proposer.num_query_per_req
        proposer.positions = torch.zeros(max_num_tokens, dtype=torch.int32, device=device)
        proposer.positions[:num_query_total] = torch.arange(num_query_total, dtype=torch.int32)
        proposer.parallel_drafting_token_id = 0
        proposer.kv_cache_gid = 0
        proposer._dflash_num_context = 0

        proposer.input_ids = torch.zeros(max_num_tokens, dtype=torch.int64, device=device)
        proposer._context_positions_buffer = torch.zeros(max_num_tokens, dtype=torch.int32, device=device)
        proposer._slot_mapping_buffer = torch.zeros(max_num_tokens, dtype=torch.int32, device=device)
        proposer._dspark_seed_buffer = torch.zeros(max_num_tokens, dtype=torch.int64, device=device)
        proposer._dflash_hidden_states = torch.zeros((max_num_tokens, 8), dtype=torch.float32, device=device)
        proposer.arange_dflash = torch.arange(max_num_tokens + 1, dtype=torch.int32, device=device)
        proposer.token_arange_np = np.arange(max_num_tokens + 1, dtype=np.int32)

        gid = 0
        proposer.draft_attn_groups = [
            SimpleNamespace(
                kv_cache_group_id=gid,
                kv_cache_spec=SimpleNamespace(block_size=block_size),
                layer_names=["L0"],
            )
        ]
        proposer._layer_group_idx = [gid]
        block_table = torch.zeros((num_reqs, 16), dtype=torch.int32, device=device)
        proposer._per_group_block_tables = {gid: block_table}
        proposer._per_group_block_table_buffers = {gid: block_table}
        slot = torch.zeros(max_num_tokens, dtype=torch.int32, device=device)
        proposer._per_group_slot_mappings = {gid: slot}
        proposer._per_group_kernel_block_sizes = {gid: block_size}
        proposer._per_group_query_slot_mapping_buffers = {gid: slot.clone()}
        proposer._per_group_context_slot_mapping_buffers = {gid: slot.clone()}
        return proposer

    # fmt: off
    @staticmethod
    def _invoke_set_inputs_first_pass(
        proposer,
        *,
        num_reqs,
        block_size,
        seq_len=128,
        context=None,
        num_rejected=None,
        with_optional_attrs=False,
    ):
        """Drive ``set_inputs_first_pass`` with a configurable cad.

        ``context`` sets ``query_start_loc_cpu[num_reqs]`` so the proposer
        copies ``context`` rows of target hidden states (0 by default).
        Returns ``(num_query_total, token_indices, cad, extra,
        next_token_ids, target_hidden_states)``.
        """
        next_token_ids = torch.arange(1, num_reqs + 1, dtype=torch.int64)
        target_hidden_states = torch.arange(
            num_reqs * 8, dtype=torch.float32
        ).reshape(num_reqs, 8)
        query_start_loc_cpu = torch.zeros(num_reqs + 1, dtype=torch.int32)
        if context is not None:
            query_start_loc_cpu[num_reqs] = context
        cad = SimpleNamespace(
            num_reqs=num_reqs,
            query_start_loc=torch.arange(num_reqs + 1, dtype=torch.int32) * block_size,
            query_start_loc_cpu=query_start_loc_cpu,
            seq_lens=torch.full((num_reqs,), seq_len, dtype=torch.int32),
            max_seq_len=seq_len,
        )
        if with_optional_attrs:
            cad.actual_seq_lengths_q = [0] * num_reqs
            cad.decode_token_per_req = 0
        num_query_total, token_indices, cad, extra = proposer.set_inputs_first_pass(
            target_token_ids=torch.zeros(num_reqs, dtype=torch.int64),
            next_token_ids=next_token_ids,
            target_positions=torch.zeros(num_reqs, dtype=torch.int32),
            target_hidden_states=target_hidden_states,
            token_indices_to_sample=None,
            cad=cad,
            num_rejected_tokens_gpu=num_rejected,
        )
        return num_query_total, token_indices, cad, extra, next_token_ids, target_hidden_states


# fmt: on


class TestDSparkPositionsFullUnderMultiDp(_DSparkProposerTestBase):
    """Guard: under multi-DP the dspark draft proposer must hand DSA attention a
    full-length positions buffer so ``positions[:num_input_tokens]`` never reads
    out of bounds (the slice is DP-padded and may exceed the local query size)."""

    @staticmethod
    def _call_set_inputs_first_pass(proposer, *, num_reqs, block_size):
        # query_start_loc_cpu[num_reqs] is 0 so _dflash_num_context becomes 0.
        cad = SimpleNamespace(
            num_reqs=num_reqs,
            query_start_loc=torch.arange(num_reqs + 1, dtype=torch.int32) * block_size,
            query_start_loc_cpu=torch.zeros(num_reqs + 1, dtype=torch.int32),
            seq_lens=torch.full((num_reqs,), 128, dtype=torch.int32),
            max_seq_len=128,
        )
        proposer.set_inputs_first_pass(
            target_token_ids=torch.zeros(num_reqs, dtype=torch.int64),
            next_token_ids=torch.zeros(num_reqs, dtype=torch.int64),
            target_positions=torch.zeros(num_reqs, dtype=torch.int32),
            target_hidden_states=torch.zeros((num_reqs, 8), dtype=torch.float32),
            token_indices_to_sample=None,
            cad=cad,
            num_rejected_tokens_gpu=None,
        )
        return cad

    @pytest.mark.parametrize("dp_padding", MULTI_DP_PADDING_SIZES)
    def test_positions_not_pre_sliced(self, monkeypatch, dp_padding):
        """``cad.positions`` must be the full buffer, not ``[:num_query_total]``."""
        monkeypatch.setattr(
            "vllm_ascend.spec_decode.dspark_proposer.copy_and_expand_dflash_and_dspark_inputs_kernel_single_grid",
            MagicMock(),
        )
        num_reqs, block_size, max_num_tokens = 4, 5, 256
        num_query_total = num_reqs * block_size
        num_input_tokens = num_query_total + dp_padding

        proposer = self._make_proposer(max_num_tokens=max_num_tokens, num_reqs=num_reqs, block_size=block_size)
        cad = self._call_set_inputs_first_pass(proposer, num_reqs=num_reqs, block_size=block_size)

        # DSA attention slices positions[:num_input_tokens] (DP-padded); a
        # pre-slice to num_query_total reads out of bounds under multi-DP.
        assert cad.positions.shape[0] == max_num_tokens
        assert cad.positions[:num_input_tokens].shape[0] == num_input_tokens

    @pytest.mark.parametrize("dp_padding", [8, 32])
    def test_positions_full_and_padded_for_dsa(self, monkeypatch, dp_padding):
        """After set_inputs_first_pass + _pad_draft_buffers, positions[:num_input]
        is full-length and zero-padded in the DP region."""
        monkeypatch.setattr(
            "vllm_ascend.spec_decode.dspark_proposer.copy_and_expand_dflash_and_dspark_inputs_kernel_single_grid",
            MagicMock(),
        )
        num_reqs, block_size, max_num_tokens = 4, 5, 256
        num_query_total = num_reqs * block_size
        num_input_tokens = num_query_total + dp_padding

        proposer = self._make_proposer(max_num_tokens=max_num_tokens, num_reqs=num_reqs, block_size=block_size)
        proposer.positions[num_query_total:num_input_tokens] = -999
        cad = self._call_set_inputs_first_pass(proposer, num_reqs=num_reqs, block_size=block_size)
        proposer._pad_draft_buffers(num_query_total, num_input_tokens)

        dsa_slice = cad.positions[:num_input_tokens]
        assert dsa_slice.shape[0] == num_input_tokens
        assert torch.all(dsa_slice[num_query_total:] == 0)


class TestPadDraftBuffersBeforeBuild(_DSparkProposerTestBase):
    """Guard: ``_pad_draft_buffers`` must zero the DP-padding region of positions
    and run before ``build_draft_attn_metadata``, so the attention backend reads
    valid (zero) padding instead of stale values."""

    def test_zeros_dp_padding_region(self):
        """``_pad_draft_buffers`` zeros positions / input_ids / slot_mapping in
        the DP-padding region."""
        num_reqs, block_size, max_num_tokens = 4, 5, 256
        num_actual = num_reqs * block_size
        num_input = num_actual + 16

        proposer = self._make_proposer(max_num_tokens=max_num_tokens, num_reqs=num_reqs, block_size=block_size)
        proposer.positions[num_actual:num_input] = -999
        proposer.input_ids[num_actual:num_input] = -999
        proposer._slot_mapping_buffer[num_actual:num_input] = -999
        for buf in proposer._per_group_query_slot_mapping_buffers.values():
            buf[num_actual:num_input] = -999

        proposer._pad_draft_buffers(num_actual, num_input)

        assert torch.all(proposer.positions[num_actual:num_input] == 0)
        assert torch.all(proposer.input_ids[num_actual:num_input] == proposer.parallel_drafting_token_id)
        assert torch.all(proposer._slot_mapping_buffer[num_actual:num_input] == -1)
        for buf in proposer._per_group_query_slot_mapping_buffers.values():
            assert torch.all(buf[num_actual:num_input] == -1)
        assert torch.all(proposer.positions[:num_actual] != -999)

    def test_noop_without_dp_padding(self):
        """Single-DP (num_input <= num_actual) leaves buffers untouched."""
        num_reqs, block_size, max_num_tokens = 4, 5, 256
        num_actual = num_reqs * block_size

        proposer = self._make_proposer(max_num_tokens=max_num_tokens, num_reqs=num_reqs, block_size=block_size)
        snapshot = proposer.positions.clone()
        proposer._pad_draft_buffers(num_actual, num_actual)
        assert torch.equal(proposer.positions, snapshot)

    def test_must_precede_build(self):
        """build_draft_attn_metadata reads positions but does not zero it, so
        _pad_draft_buffers must run first."""
        num_reqs, block_size, max_num_tokens = 4, 5, 256
        num_actual = num_reqs * block_size
        num_input = num_actual + 16

        def capture_build():
            captured = {}

            def fake_build(common_attn_metadata, num_input_tokens, num_actual_tokens):
                captured["region"] = common_attn_metadata.positions[num_actual:num_input].clone()
                return None, common_attn_metadata

            return captured, fake_build

        ok = self._make_proposer(max_num_tokens=max_num_tokens, num_reqs=num_reqs, block_size=block_size)
        ok.positions[num_actual:num_input] = -999
        cap_ok, build_ok = capture_build()
        ok.build_draft_attn_metadata = build_ok
        ok._pad_draft_buffers(num_actual, num_input)
        ok.build_draft_attn_metadata(SimpleNamespace(positions=ok.positions), num_input, num_actual)
        assert torch.all(cap_ok["region"] == 0)

        bug = self._make_proposer(max_num_tokens=max_num_tokens, num_reqs=num_reqs, block_size=block_size)
        bug.positions[num_actual:num_input] = -999
        cap_bug, build_bug = capture_build()
        bug.build_draft_attn_metadata = build_bug
        bug.build_draft_attn_metadata(SimpleNamespace(positions=bug.positions), num_input, num_actual)
        bug._pad_draft_buffers(num_actual, num_input)
        assert torch.all(cap_bug["region"] == -999)

    def test_called_before_build_in_propose(self):
        """In ``_propose`` the ``_pad_draft_buffers`` call must precede
        ``build_draft_attn_metadata``."""
        src = inspect.getsource(AscendSpecDecodeBaseProposer._propose)
        pad_idx = src.find("self._pad_draft_buffers(")
        build_idx = src.find("self.build_draft_attn_metadata(")
        # Only assert when both calls live directly in _propose; a refactor that
        # extracts them elsewhere leaves this guard inert rather than brittle.
        if pad_idx != -1 and build_idx != -1:
            assert pad_idx < build_idx, (
                "_pad_draft_buffers must be called before build_draft_attn_metadata "
                "in _propose, otherwise the attention backend reads un-zeroed "
                "positions in the DP-padding region."
            )


class TestDSparkInitialization(_DSparkProposerTestBase):
    """Tests for DSpark initialization configuration."""

    @pytest.mark.parametrize(
        ("additional_config", "draft_sample_method", "message"),
        [
            pytest.param(
                {"enable_reduce_sample": True},
                "greedy",
                "does not support enable_reduce_sample",
                id="reduce-sample-bypasses-markov-head",
            ),
            pytest.param(
                {},
                "probabilistic",
                "probabilistic draft sampling is not supported",
                id="probabilistic-draft-sampling",
            ),
        ],
    )
    def test_rejects_invalid_config_before_parent_initialization(
        self,
        additional_config: dict,
        draft_sample_method: str,
        message: str,
    ) -> None:
        vllm_config = self._make_vllm_config(SimpleNamespace())
        vllm_config.additional_config = additional_config
        vllm_config.speculative_config.draft_sample_method = draft_sample_method
        parent_init = MagicMock()

        with (
            pytest.raises(ValueError, match=message),
            patch.object(
                AscendDSparkProposer.__base__,
                "__init__",
                parent_init,
            ),
        ):
            AscendDSparkProposer(vllm_config, torch.device("cpu"))

        parent_init.assert_not_called()

    @pytest.mark.parametrize(
        ("hf_config", "expected_sample_from_anchor", "expected_num_query_per_req"),
        [
            pytest.param(SimpleNamespace(), True, _NUM_SPECULATIVE_TOKENS),
            pytest.param(SimpleNamespace(sample_from_anchor=False), False, 1 + _NUM_SPECULATIVE_TOKENS),
        ],
    )
    def test_configures_anchor_sampling(
        self,
        hf_config: SimpleNamespace,
        expected_sample_from_anchor: bool,
        expected_num_query_per_req: int,
    ) -> None:
        """Verify the anchor-sampling setting selects the expected query layout."""
        proposer = self._make_proposer(
            max_num_tokens=_MAX_NUM_TOKENS,
            num_reqs=_MAX_BATCH_SIZE,
            block_size=_NUM_SPECULATIVE_TOKENS,
            hf_config=hf_config,
        )
        expected_max_query_tokens = _MAX_BATCH_SIZE * expected_num_query_per_req
        assert proposer.sample_from_anchor is expected_sample_from_anchor
        assert proposer.num_query_per_req == expected_num_query_per_req
        assert proposer.max_query_tokens == expected_max_query_tokens


# fmt: off
class TestSetPerGroupAttnMetadata(_DSparkProposerTestBase):
    """``set_per_group_attn_metadata`` stores the runner-provided per-group
    block table / slot mapping into the read-only dicts the proposer consults
    during ``set_inputs_first_pass``."""

    def test_stores_block_table_and_slot_mapping(self):
        num_reqs, block_size, max_num_tokens = 4, 5, 256
        proposer = self._make_proposer(
            max_num_tokens=max_num_tokens, num_reqs=num_reqs, block_size=block_size
        )
        # a gid not pre-populated by _make_proposer (which only seeds gid=0)
        gid = 7
        block_table = torch.zeros((num_reqs, 16), dtype=torch.int32)
        slot_mapping = torch.full((max_num_tokens,), 42, dtype=torch.int32)

        proposer.set_per_group_attn_metadata(gid, block_table, slot_mapping)

        assert proposer._per_group_block_tables[gid] is block_table
        assert proposer._per_group_slot_mappings[gid] is slot_mapping

    def test_overwrites_existing_gid(self):
        num_reqs, block_size, max_num_tokens = 2, 5, 256
        proposer = self._make_proposer(
            max_num_tokens=max_num_tokens, num_reqs=num_reqs, block_size=block_size
        )
        gid = 0  # already populated by _make_proposer
        old_block_table = proposer._per_group_block_tables[gid]
        new_block_table = torch.ones((num_reqs, 16), dtype=torch.int32)
        new_slot_mapping = torch.ones(max_num_tokens, dtype=torch.int32)

        proposer.set_per_group_attn_metadata(gid, new_block_table, new_slot_mapping)

        assert proposer._per_group_block_tables[gid] is new_block_table
        assert proposer._per_group_slot_mappings[gid] is new_slot_mapping
        assert proposer._per_group_block_tables[gid] is not old_block_table


class TestDSparkInitValidation:
    """``AscendDSparkProposer.__init__`` rejects probabilistic draft sampling
    (unsupported on the v1 model runner) and, for the greedy path, allocates
    the DSpark-specific draft/seed buffers and overrides the DFlash
    query-token / cudagraph defaults."""

    @staticmethod
    def _make_vllm_config(
        *,
        num_speculative_tokens,
        max_batch_size,
        max_num_tokens,
        draft_sample_method,
        hidden_size=8,
        target_model_type="qwen3",
        draft_architecture="Qwen3DSparkModel",
    ):
        speculative_config = SimpleNamespace(
            num_speculative_tokens=num_speculative_tokens,
            draft_sample_method=draft_sample_method,
            draft_model_config=SimpleNamespace(
                hf_config=SimpleNamespace(architectures=[draft_architecture]),
                get_hidden_size=lambda: hidden_size
            ),
        )
        return SimpleNamespace(
            speculative_config=speculative_config,
            model_config=SimpleNamespace(
                hf_text_config=SimpleNamespace(model_type=target_model_type)
            ),
        )

    @staticmethod
    def _stub_dflash_init(
        monkeypatch,
        *,
        num_speculative_tokens,
        max_batch_size,
        max_num_tokens,
        dtype,
        device,
    ):
        """Replace the heavy DFlash/Eagle base init with a stub that only sets
        the attributes DSpark's ``__init__`` subsequently reads."""

        def _stub(self, vllm_config, device, runner=None):
            self.num_speculative_tokens = num_speculative_tokens
            self.max_batch_size = max_batch_size
            self.max_num_tokens = max_num_tokens
            self.dtype = dtype
            self.device = device
            self.draft_model_config = vllm_config.speculative_config.draft_model_config
            # present so the ``del`` in DSpark.__init__ succeeds
            self.hidden_size = 0
            self.hidden_states = None
            self._dflash_hidden_states = None
            self.use_cuda_graph = True

        monkeypatch.setattr(AscendDflashProposer, "__init__", _stub)

    def test_probabilistic_rejected(self, monkeypatch):
        device = torch.device("cpu")
        self._stub_dflash_init(
            monkeypatch,
            num_speculative_tokens=5,
            max_batch_size=16,
            max_num_tokens=256,
            dtype=torch.float32,
            device=device,
        )
        vllm_config = self._make_vllm_config(
            num_speculative_tokens=5,
            max_batch_size=16,
            max_num_tokens=256,
            draft_sample_method="probabilistic",
        )
        with pytest.raises(ValueError, match="probabilistic"):
            AscendDSparkProposer(vllm_config, device)

    def test_greedy_allocates_dspark_buffers(self, monkeypatch):
        device = torch.device("cpu")
        num_spec, max_batch, max_num_tokens, hidden = 5, 16, 256, 8
        self._stub_dflash_init(
            monkeypatch,
            num_speculative_tokens=num_spec,
            max_batch_size=max_batch,
            max_num_tokens=max_num_tokens,
            dtype=torch.float32,
            device=device,
        )
        vllm_config = self._make_vllm_config(
            num_speculative_tokens=num_spec,
            max_batch_size=max_batch,
            max_num_tokens=max_num_tokens,
            draft_sample_method="greedy",
            hidden_size=hidden,
        )
        proposer = AscendDSparkProposer(vllm_config, device)

        blk = 1 + num_spec
        max_query_tokens = max_batch * num_spec
        # DSpark-specific draft / seed buffers.
        assert proposer._dspark_draft_buffer.shape == (max_batch, blk)
        assert proposer._dspark_draft_buffer.dtype == torch.int64
        assert proposer._dspark_seed_buffer.shape == (max_batch,)
        assert proposer._dspark_seed_buffer.dtype == torch.int64
        # hidden_size / hidden states come from the draft model config.
        assert proposer.hidden_size == hidden
        assert proposer.hidden_states.shape == (max_num_tokens, hidden)
        assert proposer._dflash_hidden_states.shape == (max_num_tokens, hidden)
        # Non-DSV4 DSpark implementations retain the eager fallback.
        assert proposer.use_cuda_graph is False
        # anchor-first: N query tokens per request, no bonus token (unlike
        # DFlash's 1+N).
        assert proposer.max_query_tokens == max_query_tokens
        assert proposer.positions.shape == (max_query_tokens,)
        assert proposer.positions.dtype == torch.int32
        assert proposer._slot_mapping_buffer.shape == (max_query_tokens,)
        # per-group bookkeeping dicts start empty / None.
        assert proposer._per_group_block_tables == {}
        assert proposer._per_group_slot_mappings == {}
        assert proposer._context_slot_mapping_buffers is None

    def test_dsv4_static_greedy_retains_graph_enablement(self, monkeypatch):
        device = torch.device("cpu")
        self._stub_dflash_init(
            monkeypatch,
            num_speculative_tokens=7,
            max_batch_size=16,
            max_num_tokens=256,
            dtype=torch.float32,
            device=device,
        )
        vllm_config = self._make_vllm_config(
            num_speculative_tokens=7,
            max_batch_size=16,
            max_num_tokens=256,
            draft_sample_method="greedy",
            target_model_type="deepseek_v4",
            draft_architecture="DSparkDraftModel",
        )

        proposer = AscendDSparkProposer(vllm_config, device)

        assert proposer.supports_dsv4_aclgraph is True
        assert proposer.use_cuda_graph is True

    def test_dspark_draft_architecture_on_non_dsv4_target_stays_eager(self, monkeypatch):
        device = torch.device("cpu")
        self._stub_dflash_init(
            monkeypatch,
            num_speculative_tokens=7,
            max_batch_size=16,
            max_num_tokens=256,
            dtype=torch.float32,
            device=device,
        )
        vllm_config = self._make_vllm_config(
            num_speculative_tokens=7,
            max_batch_size=16,
            max_num_tokens=256,
            draft_sample_method="greedy",
            target_model_type="qwen3",
            draft_architecture="DSparkDraftModel",
        )

        proposer = AscendDSparkProposer(vllm_config, device)

        assert proposer.supports_dsv4_aclgraph is False
        assert proposer.use_cuda_graph is False

    def test_dsv4_dynamic_width_keeps_eager_fallback(self, monkeypatch):
        device = torch.device("cpu")
        self._stub_dflash_init(
            monkeypatch,
            num_speculative_tokens=7,
            max_batch_size=16,
            max_num_tokens=256,
            dtype=torch.float32,
            device=device,
        )
        vllm_config = self._make_vllm_config(
            num_speculative_tokens=7,
            max_batch_size=16,
            max_num_tokens=256,
            draft_sample_method="greedy",
            target_model_type="deepseek_v4",
            draft_architecture="DSparkDraftModel",
        )
        vllm_config.speculative_config.uses_dynamic_speculative_decoding = lambda: True

        proposer = AscendDSparkProposer(vllm_config, device)

        assert proposer.supports_dsv4_aclgraph is True
        assert proposer.use_cuda_graph is False


class TestDSparkGraphGeometry(_DSparkProposerTestBase):
    @pytest.mark.parametrize("draft_width", [5, 7])
    def test_target_descriptor_maps_to_native_draft_width(self, draft_width):
        proposer = AscendDSparkProposer.__new__(AscendDSparkProposer)
        proposer.num_query_per_req = draft_width
        proposer.num_speculative_tokens = draft_width
        descriptor = SimpleNamespace(
            uniform=True,
            num_reqs=4,
            num_tokens=4 * (draft_width + 1),
        )

        assert proposer.get_graph_num_input_tokens(descriptor) == 4 * draft_width

    @pytest.mark.parametrize(
        ("draft_width", "num_reqs"),
        [(5, 4), (5, 8), (5, 16), (5, 32), (7, 4), (7, 8), (7, 16), (7, 32)],
    )
    def test_native_width_dispatches_with_target_graph_key(self, draft_width, num_reqs):
        proposer = AscendDSparkProposer.__new__(AscendDSparkProposer)
        proposer.num_query_per_req = draft_width
        proposer.num_speculative_tokens = draft_width

        dispatch_tokens = proposer.get_graph_dispatch_num_tokens(draft_width * num_reqs)

        assert dispatch_tokens == (draft_width + 1) * num_reqs

    def test_graph_dispatch_rejects_partial_draft_rows(self):
        proposer = AscendDSparkProposer.__new__(AscendDSparkProposer)
        proposer.num_query_per_req = 7
        proposer.num_speculative_tokens = 7

        with pytest.raises(ValueError, match="complete request rows"):
            proposer.get_graph_dispatch_num_tokens(15)

    def test_nonuniform_fallback_keeps_native_width(self):
        proposer = AscendDSparkProposer.__new__(AscendDSparkProposer)
        proposer.num_query_per_req = 7
        proposer.num_speculative_tokens = 7

        assert proposer.get_graph_dispatch_num_tokens(7, uniform_decode=False) == 7
        assert proposer.get_graph_dispatch_num_tokens(14, uniform_decode=False) == 14

    def test_eager_fallback_does_not_adopt_graph_padding(self):
        proposer = AscendDSparkProposer.__new__(AscendDSparkProposer)
        proposer.num_query_per_req = 7
        descriptor = SimpleNamespace(uniform=True, num_reqs=4, num_tokens=32)

        assert (
            proposer.get_graph_num_input_tokens_for_runtime(
                descriptor,
                CUDAGraphMode.FULL,
                actual_num_tokens=13,
            )
            == 28
        )
        assert (
            proposer.get_graph_num_input_tokens_for_runtime(
                descriptor,
                CUDAGraphMode.NONE,
                actual_num_tokens=13,
            )
            == 13
        )

    def test_spec_state_is_limited_to_full_graph_runtime(self):
        proposer = AscendDSparkProposer.__new__(AscendDSparkProposer)
        proposer.supports_dsv4_aclgraph = True
        proposer.use_cuda_graph = True

        assert proposer._get_draft_attn_state(CUDAGraphMode.FULL) == AscendAttentionState.SpecDecoding
        assert proposer._get_draft_attn_state(CUDAGraphMode.NONE) == AscendAttentionState.ChunkedPrefill

    def test_non_anchor_width_still_uses_target_one_plus_n_key(self):
        proposer = AscendDSparkProposer.__new__(AscendDSparkProposer)
        proposer.num_speculative_tokens = 7
        proposer.num_query_per_req = 8

        assert proposer.get_graph_dispatch_num_tokens(4 * 8) == 4 * 8

    def test_query_start_loc_pads_complete_rows_through_cpu_buffer(self):
        proposer = AscendDSparkProposer.__new__(AscendDSparkProposer)
        proposer.num_query_per_req = 7
        cpu = torch.zeros(5, dtype=torch.int32)
        gpu = torch.zeros(5, dtype=torch.int32)
        cpu[:3] = torch.tensor([0, 7, 14], dtype=torch.int32)
        dual = SimpleNamespace(cpu=cpu, gpu=gpu)
        dual.copy_to_gpu = lambda: dual.gpu.copy_(dual.cpu)

        num_reqs = proposer.pad_query_start_loc_for_graph(dual, 28, 2, 4)

        assert num_reqs == 4
        assert dual.cpu.tolist() == [0, 7, 14, 21, 28]
        assert dual.gpu.tolist() == [0, 7, 14, 21, 28]

    def test_padding_makes_every_group_inactive_and_keeps_addresses(self):
        proposer = AscendDSparkProposer.__new__(AscendDSparkProposer)
        proposer.num_query_per_req = 5
        block_tables = {
            0: torch.full((4, 3), 11, dtype=torch.int32),
            2: torch.full((4, 3), 22, dtype=torch.int32),
        }
        slot_mappings = {
            0: torch.arange(20, dtype=torch.int32),
            2: torch.arange(20, dtype=torch.int32) + 100,
        }
        proposer._per_group_block_table_buffers = block_tables
        proposer._per_group_query_slot_mapping_buffers = slot_mappings
        addresses = {
            "block": [tensor.data_ptr() for tensor in block_tables.values()],
            "slot": [tensor.data_ptr() for tensor in slot_mappings.values()],
        }

        proposer.pad_attention_inputs_for_graph(real_num_reqs=2, graph_num_reqs=4)

        for tensor in block_tables.values():
            assert torch.all(tensor[:2] != 0)
            assert torch.all(tensor[2:] == 0)
        for tensor in slot_mappings.values():
            assert torch.all(tensor[10:] == -1)
        assert addresses["block"] == [tensor.data_ptr() for tensor in block_tables.values()]
        assert addresses["slot"] == [tensor.data_ptr() for tensor in slot_mappings.values()]

    def test_padded_request_has_positive_kv_length(self):
        proposer = AscendSpecDecodeBaseProposer.__new__(AscendSpecDecodeBaseProposer)
        proposer.method = "dspark"

        padded = proposer._adjust_parallel_draft_seq_lens_for_graph(
            torch.tensor([17, 23], dtype=torch.int32),
            4,
        )

        assert padded.tolist() == [17, 23, 1, 1]

    def test_missing_causal_hook_defaults_dsv4_capture_to_noncausal(self):
        proposer = AscendDSparkProposer.__new__(AscendDSparkProposer)
        proposer.model = SimpleNamespace()

        assert proposer._get_draft_attn_causal() is False

        proposer.model.get_draft_attn_causal = lambda: [True]
        assert proposer._get_draft_attn_causal() is True

    def test_context_kv_is_eager_and_graph_output_is_sliced(self):
        propose_source = inspect.getsource(AscendSpecDecodeBaseProposer._propose)
        merged_source = inspect.getsource(AscendSpecDecodeBaseProposer._run_merged_draft)

        context_idx = propose_source.index("self.build_model_inputs_first_pass(")
        graph_context_idx = propose_source.index("with set_ascend_forward_context(")
        assert context_idx < graph_context_idx
        assert 'aclgraph_runtime_mode == CUDAGraphMode.FULL' in propose_source
        assert 'forward_context.cudagraph_runtime_mode != CUDAGraphMode.FULL' in merged_source
        assert 'draft_token_ids = draft_token_ids[:batch_size]' in propose_source
        update_branch = propose_source.index('if self.enable_enpu or self.method == "dspark":')
        update_call = propose_source.index("self._update_full_graph_params_if_needed(", update_branch)
        replay_call = propose_source.index("draft_token_ids = run_draft()", update_branch)
        assert update_call < replay_call

    def test_dummy_capture_uses_all_groups_and_warmup_fallback(self):
        source = inspect.getsource(AscendDSparkProposer.dummy_run)
        padding_source = inspect.getsource(AscendDSparkProposer.pad_query_start_loc_for_graph)

        assert "for attn_group in self.draft_attn_groups:" in source
        assert "self._get_graph_capture_block_table(gid)" in source
        assert "causal = self._get_draft_attn_causal()" in source
        assert ".np" not in source + padding_source
        assert "query_start_loc.cpu" in padding_source

    def test_dummy_capture_uses_drafting_builder_and_spec_state(self, monkeypatch):
        """Capture must build the same DSV4 sparse metadata used at replay."""
        proposer = AscendDSparkProposer.__new__(AscendDSparkProposer)
        proposer.num_query_per_req = 7
        proposer.num_speculative_tokens = 7
        proposer.max_query_tokens = 28
        proposer.max_num_tokens = 32
        proposer.use_cuda_graph = True
        proposer.supports_dsv4_aclgraph = True
        proposer.device = torch.device("cpu")
        proposer.vllm_config = SimpleNamespace()
        proposer.positions = torch.zeros(28, dtype=torch.int32)
        proposer.input_ids = torch.zeros(28, dtype=torch.int64)
        proposer.hidden_states = torch.zeros((32, 8), dtype=torch.float32)
        proposer._context_positions_buffer = torch.zeros(32, dtype=torch.int32)
        proposer.token_indices_to_sample = torch.zeros(28, dtype=torch.int32)
        proposer.arange = torch.arange(28, dtype=torch.int32)
        proposer.token_arange_np = np.arange(29, dtype=np.int32)

        query_start_loc_cpu = torch.zeros(5, dtype=torch.int32)
        query_start_loc_gpu = torch.zeros(5, dtype=torch.int32)
        proposer.query_start_loc = SimpleNamespace(
            cpu=query_start_loc_cpu,
            gpu=query_start_loc_gpu,
            copy_to_gpu=lambda: query_start_loc_gpu.copy_(query_start_loc_cpu),
        )
        proposer.query_start_loc_group = [torch.zeros(5, dtype=torch.int32)]
        proposer.seq_lens_group = [torch.ones(4, dtype=torch.int32)]

        block_table = torch.ones((4, 8), dtype=torch.int32)
        second_block_table = torch.full((4, 8), 2, dtype=torch.int32)
        proposer._per_group_block_tables = {0: block_table, 1: second_block_table}
        proposer._per_group_block_table_buffers = {}
        proposer._per_group_query_slot_mapping_buffers = {
            0: torch.full((28,), -1, dtype=torch.int32),
            1: torch.full((28,), -1, dtype=torch.int32),
        }
        proposer._per_group_kernel_block_sizes = {0: 128, 1: 128}

        captured_states = []

        def build_for_drafting(common_attn_metadata, *, draft_index, block_size):
            captured_states.append(common_attn_metadata.attn_state)
            assert common_attn_metadata.attn_state == AscendAttentionState.SpecDecoding
            assert common_attn_metadata.query_start_loc.data_ptr() == proposer.query_start_loc_group[0].data_ptr()
            assert draft_index == 1
            assert block_size == 128
            return SimpleNamespace(
                attn_mask=None,
                attn_state=common_attn_metadata.attn_state,
                decode=SimpleNamespace(dspark_swa_indices=torch.ones((14, 1, 8))),
            )

        builder = SimpleNamespace(
            build_for_drafting=MagicMock(side_effect=build_for_drafting),
            build_for_graph_capture=MagicMock(
                side_effect=AssertionError("DSpark capture must use build_for_drafting")
            ),
        )
        proposer.draft_attn_groups = [
            SimpleNamespace(
                kv_cache_group_id=gid,
                kv_cache_spec=SimpleNamespace(block_size=384),
                layer_names=[f"draft.dsa.{gid}"],
                get_metadata_builder=lambda: builder,
            )
            for gid in (0, 1)
        ]
        proposer.model = SimpleNamespace(get_draft_attn_causal=lambda: [False])
        proposer.runner = SimpleNamespace(
            seq_lens=torch.ones(4, dtype=torch.int32),
            optimistic_seq_lens_cpu=torch.ones(4, dtype=torch.int32),
            _sync_metadata_across_dp=lambda num_tokens, is_draft_model: (
                num_tokens,
                None,
                None,
            ),
        )
        proposer._pad_draft_buffers = MagicMock()
        proposer._runnable = MagicMock()
        proposer._get_positions = lambda num_tokens: proposer.positions[:num_tokens]

        monkeypatch.setattr(
            dspark_proposer_module,
            "set_ascend_forward_context",
            lambda *args, **kwargs: nullcontext(),
        )
        monkeypatch.setattr(
            dspark_proposer_module,
            "get_forward_context",
            lambda: SimpleNamespace(cudagraph_runtime_mode=CUDAGraphMode.NONE),
        )

        proposer.dummy_run(
            num_tokens=16,
            num_reqs=2,
            aclgraph_runtime_mode=CUDAGraphMode.FULL,
            batch_descriptor=SimpleNamespace(
                uniform=True,
                num_reqs=2,
                num_tokens=16,
            ),
        )

        assert captured_states == [
            AscendAttentionState.SpecDecoding,
            AscendAttentionState.SpecDecoding,
        ]
        assert builder.build_for_drafting.call_count == 2
        builder.build_for_graph_capture.assert_not_called()
        graph_metadata = proposer._runnable.call_args.kwargs["multi_steps_attn_metadata"]
        for gid in (0, 1):
            assert graph_metadata[0][f"draft.dsa.{gid}"].decode.dspark_swa_indices is not None

    def test_runtime_metadata_uses_logical_kernel_block_size(self):
        proposer = AscendDSparkProposer.__new__(AscendDSparkProposer)
        proposer.method = "dspark"
        proposer.supports_dsv4_aclgraph = True
        proposer.use_compress = False
        proposer._per_group_kernel_block_sizes = {0: 128}
        proposer._per_group_block_table_buffers = {
            0: torch.ones((2, 8), dtype=torch.int32)
        }
        proposer._per_group_query_slot_mapping_buffers = {
            0: torch.zeros(14, dtype=torch.int32)
        }

        metadata = SimpleNamespace(attn_mask=None, causal=False)
        builder = SimpleNamespace(build_for_drafting=MagicMock(return_value=metadata))
        proposer.draft_attn_groups = [
            SimpleNamespace(
                kv_cache_group_id=0,
                kv_cache_spec=SimpleNamespace(block_size=384),
                layer_names=["draft.dsa"],
                get_metadata_builder=lambda: builder,
            )
        ]
        common_attn_metadata = SimpleNamespace(
            num_reqs=2,
            block_table_tensor=torch.ones((2, 8), dtype=torch.int32),
            slot_mapping=torch.zeros(14, dtype=torch.int32),
        )

        proposer.build_draft_attn_metadata(common_attn_metadata, 14, 14)

        assert builder.build_for_drafting.call_args.kwargs == {
            "draft_index": 1,
            "block_size": 128,
        }

    def test_warmup_block_table_falls_back_to_runner_dummy_table(self):
        proposer = AscendDSparkProposer.__new__(AscendDSparkProposer)
        proposer._per_group_block_tables = {}
        dummy_table = torch.ones((4, 8), dtype=torch.int32)
        table = SimpleNamespace(get_device_tensor=lambda: dummy_table)
        proposer.runner = SimpleNamespace(
            input_batch=SimpleNamespace(block_table={3: table})
        )

        assert proposer._get_graph_capture_block_table(3) is dummy_table

        runtime_table = torch.full((4, 8), 9, dtype=torch.int32)
        proposer._per_group_block_tables[3] = runtime_table
        assert proposer._get_graph_capture_block_table(3) is runtime_table

    def test_graph_block_table_keeps_address_while_refreshing_rows(self):
        class MultiGroupBlockTable:
            """Minimal production-shaped table: indexable, but has no len()."""

            def __init__(self, tables):
                self.tables = tables

            def __getitem__(self, gid):
                return self.tables[gid]

        proposer = AscendDSparkProposer.__new__(AscendDSparkProposer)
        proposer.supports_dsv4_aclgraph = True
        proposer.use_cuda_graph = True
        proposer._per_group_graph_block_table_buffers = {}
        capacity_table = torch.zeros((4, 8), dtype=torch.int32)
        proposer.runner = SimpleNamespace(
            input_batch=SimpleNamespace(
                block_table=MultiGroupBlockTable(
                    {0: SimpleNamespace(get_device_tensor=lambda: capacity_table)}
                )
            )
        )

        first = proposer._publish_graph_block_table(
            0,
            torch.ones((2, 8), dtype=torch.int32),
        )
        address = first.data_ptr()
        second = proposer._publish_graph_block_table(
            0,
            torch.full((3, 8), 9, dtype=torch.int32),
        )

        assert second.data_ptr() == address
        assert torch.all(second[:3] == 9)
        assert torch.all(second[3:] == 0)


class TestDSparkSparseIndexGraphBuffer:
    @staticmethod
    def _make_builder(capacity=32):
        builder = AscendDSAMetadataBuilder.__new__(AscendDSAMetadataBuilder)
        builder.slot_mapping_shape = (capacity, 2)
        builder.dspark_swa_indices_buffer = None
        return builder

    def test_indices_keep_address_and_refresh_contents_across_replay(self):
        builder = self._make_builder()
        first = torch.arange(4 * 8, dtype=torch.int32).reshape(4, 1, 8)
        second = torch.full((8, 1, 8), 37, dtype=torch.int32)

        stored_first = builder._store_dspark_swa_indices(first, 4)
        address = stored_first.data_ptr()
        stored_second = builder._store_dspark_swa_indices(second, 8)

        assert stored_second.data_ptr() == address
        assert stored_second.shape == (8, 1, 8)
        assert torch.equal(stored_second, second)

    def test_sparse_index_buffer_rejects_capacity_overflow(self):
        builder = self._make_builder(capacity=4)
        indices = torch.zeros((5, 1, 8), dtype=torch.int32)

        with pytest.raises(ValueError, match="capacity"):
            builder._store_dspark_swa_indices(indices, 5)


class TestSetInputsFirstPassOutputs(_DSparkProposerTestBase):
    """``set_inputs_first_pass`` returns the anchor-first query budget and
    rewrites the common attention metadata into the DSpark cross-attention
    shape (N query tokens per request, non-causal, chunked-prefill state)."""

    @pytest.fixture(autouse=True)
    def _mock_kernel(self, monkeypatch):
        monkeypatch.setattr(
            "vllm_ascend.spec_decode.dspark_proposer."
            "copy_and_expand_dflash_and_dspark_inputs_kernel_single_grid",
            MagicMock(),
        )

    def test_return_value_and_token_indices(self):
        num_reqs, block_size, max_num_tokens = 4, 5, 256
        proposer = self._make_proposer(
            max_num_tokens=max_num_tokens, num_reqs=num_reqs, block_size=block_size
        )
        num_query_total, token_indices, _cad, extra = (
            self._invoke_set_inputs_first_pass(
                proposer, num_reqs=num_reqs, block_size=block_size
            )[:4]
        )
        assert num_query_total == num_reqs * block_size
        assert token_indices.shape == (num_reqs * block_size,)
        assert token_indices.dtype == torch.int32
        # 4th return slot is unused (no per-group attn metadata tuple here).
        assert extra is None

    def test_seed_buffer_copied_from_next_tokens(self):
        num_reqs, block_size, max_num_tokens = 4, 5, 256
        proposer = self._make_proposer(
            max_num_tokens=max_num_tokens, num_reqs=num_reqs, block_size=block_size
        )
        self._invoke_set_inputs_first_pass(
            proposer, num_reqs=num_reqs, block_size=block_size
        )
        expected = torch.arange(1, num_reqs + 1, dtype=torch.int64)
        assert torch.equal(proposer._dspark_seed_buffer[:num_reqs], expected)
        assert torch.all(proposer._dspark_seed_buffer[num_reqs:] == 0)

    def test_context_hidden_states_copied(self):
        num_reqs, block_size, max_num_tokens = 4, 5, 256
        proposer = self._make_proposer(
            max_num_tokens=max_num_tokens, num_reqs=num_reqs, block_size=block_size
        )
        self._invoke_set_inputs_first_pass(
            proposer, num_reqs=num_reqs, block_size=block_size, context=num_reqs
        )
        assert proposer._dflash_num_context == num_reqs
        expected = torch.arange(num_reqs * 8, dtype=torch.float32).reshape(num_reqs, 8)
        assert torch.equal(proposer._dflash_hidden_states[:num_reqs], expected)

    def test_query_slot_kernel_uses_logical_block_size(self):
        num_reqs, num_speculative_tokens, max_num_tokens = 1, 7, 32
        proposer = self._make_proposer(
            max_num_tokens=max_num_tokens,
            num_reqs=num_reqs,
            block_size=num_speculative_tokens,
        )
        proposer.draft_attn_groups[0].kv_cache_spec.block_size = 384
        proposer._per_group_kernel_block_sizes[0] = 128

        self._invoke_set_inputs_first_pass(
            proposer,
            num_reqs=num_reqs,
            block_size=num_speculative_tokens,
            seq_len=720,
        )

        kernel = dspark_proposer_module.copy_and_expand_dflash_and_dspark_inputs_kernel_single_grid
        kwargs = kernel[1,].call_args.kwargs
        assert proposer.draft_attn_groups[0].kv_cache_spec.block_size == 384
        assert kwargs["block_size"] == 128

    def test_cad_rewritten_to_cross_attention_shape(self):
        num_reqs, block_size, max_num_tokens = 4, 5, 256
        proposer = self._make_proposer(
            max_num_tokens=max_num_tokens, num_reqs=num_reqs, block_size=block_size
        )
        num_query_total, _, cad, _ = self._invoke_set_inputs_first_pass(
            proposer, num_reqs=num_reqs, block_size=block_size, with_optional_attrs=True
        )[:4]
        # token budgets reflect anchor-first (N per request, no bonus).
        assert cad.num_actual_tokens == num_query_total
        assert cad.num_input_tokens == num_query_total
        assert cad.max_query_len == block_size
        assert cad.max_seq_len == 128 + block_size
        # attention is non-causal cross-attention over the draft query block.
        assert cad.causal is False
        assert cad.attn_mask is None
        assert cad.attn_state == AscendAttentionState.ChunkedPrefill
        # positions is the full buffer (DSA slices it), not a pre-slice.
        assert cad.positions is proposer.positions
        # slot mapping is a slice of the primary group's query buffer (shares
        # storage from offset 0); a fresh slice is not identity-equal, so check
        # the underlying storage and length instead.
        assert (
            cad.slot_mapping.data_ptr()
            == proposer._per_group_query_slot_mapping_buffers[0].data_ptr()
        )
        assert cad.slot_mapping.shape[0] == num_query_total
        # optional attrs the proposer rewrites when present.
        assert cad.actual_seq_lengths_q == [block_size] * num_reqs
        assert cad.decode_token_per_req == block_size

    def test_cad_uses_model_reported_causality(self):
        num_reqs, block_size, max_num_tokens = 4, 5, 256
        proposer = self._make_proposer(
            max_num_tokens=max_num_tokens,
            num_reqs=num_reqs,
            block_size=block_size,
            draft_attn_causal=True,
        )
        _, _, cad, _ = self._invoke_set_inputs_first_pass(
            proposer, num_reqs=num_reqs, block_size=block_size
        )[:4]

        assert cad.causal is True

    def test_cad_query_start_loc_and_seq_lens(self):
        num_reqs, block_size, max_num_tokens = 4, 5, 256
        proposer = self._make_proposer(
            max_num_tokens=max_num_tokens, num_reqs=num_reqs, block_size=block_size
        )
        _nqt, _ti, cad, _extra = self._invoke_set_inputs_first_pass(
            proposer, num_reqs=num_reqs, block_size=block_size
        )[:4]
        expected_qsl = torch.arange(num_reqs + 1, dtype=torch.int32) * block_size
        assert torch.equal(cad.query_start_loc, expected_qsl)
        assert torch.equal(cad.query_start_loc_cpu, expected_qsl)
        # seq_lens grow by block_size when no tokens were rejected.
        assert torch.equal(cad.seq_lens, torch.full((num_reqs,), 128 + block_size, dtype=torch.int32))


class TestSetInputsFirstPassRejectedTokens(_DSparkProposerTestBase):
    """The ``has_num_rejected`` branch must shrink ``seq_lens`` by the rejected
    token count before adding the draft block size, and flag the kernel."""

    def test_seq_lens_subtracts_rejected(self, monkeypatch):
        monkeypatch.setattr(
            "vllm_ascend.spec_decode.dspark_proposer."
            "copy_and_expand_dflash_and_dspark_inputs_kernel_single_grid",
            MagicMock(),
        )
        num_reqs, block_size, max_num_tokens = 4, 5, 256
        proposer = self._make_proposer(
            max_num_tokens=max_num_tokens, num_reqs=num_reqs, block_size=block_size
        )
        rejected = torch.full((num_reqs,), 2, dtype=torch.int32)
        _nqt, _ti, cad, _extra = self._invoke_set_inputs_first_pass(
            proposer, num_reqs=num_reqs, block_size=block_size, num_rejected=rejected
        )[:4]
        # effective = seq_lens(128) - rejected(2) = 126; then + block_size(5) = 131.
        assert torch.equal(
            cad.seq_lens, torch.full((num_reqs,), 128 - 2 + block_size, dtype=torch.int32)
        )

    def test_kernel_called_with_has_num_rejected(self, monkeypatch):
        kernel = MagicMock()
        monkeypatch.setattr(
            "vllm_ascend.spec_decode.dspark_proposer."
            "copy_and_expand_dflash_and_dspark_inputs_kernel_single_grid",
            kernel,
        )
        num_reqs, block_size, max_num_tokens = 4, 5, 256
        proposer = self._make_proposer(
            max_num_tokens=max_num_tokens, num_reqs=num_reqs, block_size=block_size
        )
        rejected = torch.full((num_reqs,), 2, dtype=torch.int32)
        self._invoke_set_inputs_first_pass(
            proposer, num_reqs=num_reqs, block_size=block_size, num_rejected=rejected
        )
        # The proposer calls the kernel as ``kernel[1,](...)`` (Triton-style
        # grid indexing), so the call lands on the indexed sub-mock.
        sub = kernel[1,]
        assert sub.called
        kwargs = sub.call_args.kwargs
        assert kwargs["HAS_NUM_REJECTED"] is True
        assert kwargs["num_rejected_tokens_ptr"] is rejected
        assert kwargs["SAMPLE_FROM_ANCHOR"] is True


class TestInitializeAttnBackendErrors(_DSparkProposerTestBase):
    """``initialize_attn_backend`` raises clearly when the draft model does not
    expose the DSpark layer-name API, or when no draft attention groups can be
    built from the kv-cache groups."""

    @staticmethod
    def _make_proposer_for_init():
        proposer = AscendDSparkProposer.__new__(AscendDSparkProposer)
        proposer.vllm_config = SimpleNamespace()
        # The real proposer constructor always sets this field.  These
        # lightweight initializer tests bypass __init__, so preserve that
        # production invariant with a generic non-K3 draft config.
        proposer.draft_model_config = SimpleNamespace(hf_config=object())
        proposer.device = torch.device("cpu")
        return proposer

    def test_model_without_draft_layer_names_raises(self, monkeypatch):
        # get_layers_from_vllm_config is called first; stub it so the model
        # check is what actually fails.
        monkeypatch.setattr(
            "vllm_ascend.spec_decode.dspark_proposer.get_layers_from_vllm_config",
            lambda *a, **k: {},
        )
        proposer = self._make_proposer_for_init()
        # model lacks get_draft_kv_cache_layer_names entirely.
        proposer.model = SimpleNamespace()

        kv_cache_config = SimpleNamespace(kv_cache_groups=[])
        with pytest.raises(RuntimeError, match="get_draft_kv_cache_layer_names"):
            proposer.initialize_attn_backend(kv_cache_config)

    def test_no_draft_attn_groups_raises(self, monkeypatch):
        monkeypatch.setattr(
            "vllm_ascend.spec_decode.dspark_proposer.get_layers_from_vllm_config",
            lambda *a, **k: {},
        )
        proposer = self._make_proposer_for_init()
        # draft layer names exist, but no kv-cache group names overlap them.
        proposer.model = SimpleNamespace(get_draft_kv_cache_layer_names=lambda: {"L0"})

        non_overlapping_group = SimpleNamespace(layer_names=["OTHER_LAYER"])
        kv_cache_config = SimpleNamespace(kv_cache_groups=[non_overlapping_group])
        with pytest.raises(RuntimeError, match="registered draft attention groups"):
            proposer.initialize_attn_backend(kv_cache_config)

    def test_initialization_tracks_logical_block_size_per_gid(self, monkeypatch):
        manager_specs = [MagicMock(), MagicMock()]
        for spec in manager_specs:
            spec.block_size = 384

        backend = MagicMock()
        backend.full_cls_name.return_value = "fake.backend"
        layers = {}
        for gid in range(2):
            layer = MagicMock()
            layer.get_attn_backend.return_value = backend
            layers[f"L{gid}"] = layer
        monkeypatch.setattr(
            "vllm_ascend.spec_decode.dspark_proposer.get_layers_from_vllm_config",
            lambda *a, **k: layers,
        )

        proposer = self._make_proposer_for_init()
        proposer.model = SimpleNamespace(
            get_draft_kv_cache_layer_names=lambda: {"L0", "L1"}
        )
        assert not isinstance(
            proposer.draft_model_config.hf_config,
            dspark_proposer_module.K3DSparkConfig,
        )
        proposer.max_query_tokens = 8
        proposer.max_num_tokens = 16
        kv_cache_config = SimpleNamespace(
            kv_cache_groups=[
                SimpleNamespace(
                    layer_names=[f"L{gid}"],
                    kv_cache_spec=manager_specs[gid],
                )
                for gid in range(2)
            ],
        )

        with patch.object(AttentionGroup, "create_metadata_builders") as create_builders:
            proposer.initialize_attn_backend(
                kv_cache_config,
                kernel_block_sizes=[128, 64],
            )

        assert [spec.block_size for spec in manager_specs] == [384, 384]
        assert proposer._per_group_kernel_block_sizes == {0: 128, 1: 64}
        assert [g.kv_cache_group_id for g in proposer.draft_attn_groups] == [0, 1]
        assert set(proposer._per_group_query_slot_mapping_buffers) == {0, 1}
        assert set(proposer._per_group_context_slot_mapping_buffers) == {0, 1}
        assert proposer.kernel_block_size == 128
        assert [
            call.kwargs["kernel_block_size"]
            for call in create_builders.call_args_list
        ] == [128, 64]

    def test_k3_initialization_enables_rope_on_mla_builders(self, monkeypatch):
        class FakeK3Config:
            pass

        class FakeMLABuilder:
            def __init__(self):
                self.use_mla_rope = False

        fake_builder = FakeMLABuilder()

        def create_metadata_builders(group, *args, **kwargs):
            del args, kwargs
            group.metadata_builders = [fake_builder]

        backend = MagicMock()
        backend.full_cls_name.return_value = "fake.backend"
        layer = MagicMock()
        layer.get_attn_backend.return_value = backend
        monkeypatch.setattr(
            dspark_proposer_module,
            "get_layers_from_vllm_config",
            lambda *args, **kwargs: {"L0": layer},
        )
        monkeypatch.setattr(dspark_proposer_module, "K3DSparkConfig", FakeK3Config)
        monkeypatch.setattr(
            dspark_proposer_module,
            "AscendMLAMetadataBuilder",
            FakeMLABuilder,
        )
        monkeypatch.setattr(
            AttentionGroup,
            "create_metadata_builders",
            create_metadata_builders,
        )

        proposer = self._make_proposer_for_init()
        proposer.draft_model_config = SimpleNamespace(hf_config=FakeK3Config())
        proposer.model = SimpleNamespace(
            get_draft_kv_cache_layer_names=lambda: {"L0"}
        )
        proposer.max_query_tokens = 8
        proposer.max_num_tokens = 16
        manager_spec = MagicMock()
        manager_spec.block_size = 128
        kv_cache_config = SimpleNamespace(
            kv_cache_groups=[
                SimpleNamespace(
                    layer_names=["L0"],
                    kv_cache_spec=manager_spec,
                )
            ]
        )

        proposer.initialize_attn_backend(kv_cache_config)

        assert fake_builder.use_mla_rope is dspark_proposer_module.K3_DSPARK_USE_MLA_ROPE

    def test_kernel_block_size_falls_back_to_cache_spec(self):
        proposer = self._make_proposer_for_init()

        assert (
            proposer._resolve_kernel_block_size(
                0,
                SimpleNamespace(block_size=384),
                None,
            )
            == 384
        )
# fmt: on
