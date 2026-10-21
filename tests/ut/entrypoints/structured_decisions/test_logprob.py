# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test runner-v2 score orchestration, replacing only accelerator kernels."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch


class Kernel:
    def __init__(self, fn):
        self.fn = fn

    def __getitem__(self, grid):
        return self.fn


def fill_ids(
    out,
    out_stride,
    mask,
    mask_stride,
    sampled,
    topk,
    topk_stride,
    mapping,
    counts,
    custom,
    custom_stride,
    NUM_TOPK,
    PADDED_COLS,
):
    out[:, 0] = sampled
    mask[:, 0] = True
    for row, index in enumerate(mapping.tolist()):
        count = int(counts[index])
        ids = custom[index, :count] if count else topk[row, :NUM_TOPK]
        out[row, 1 : 1 + len(ids)] = ids
        mask[row, 1 : 1 + len(ids)] = True


def ranks(out, logits, stride, sampled, vocab_size, batch_size, rows_per_core, **kwargs):
    out.copy_((logits > logits.gather(-1, sampled[:, None])).sum(-1))


@pytest.fixture
def cpu_kernels(monkeypatch):
    # Import only the scoring module; avoid loading unrelated fused-MoE NPU
    # registrations on CPU. Production math and vLLM output types remain real.
    device = ModuleType("vllm_ascend.ops.triton.triton_utils")
    device.get_vectorcore_num = lambda: 2
    fill = ModuleType("vllm_ascend.ops.triton.v2.sample.fill_logprob_token_idx")
    fill._fill_logprob_token_ids_kernel = Kernel(fill_ids)
    monkeypatch.setitem(sys.modules, device.__name__, device)
    monkeypatch.setitem(sys.modules, fill.__name__, fill)
    path = Path(__file__).resolve().parents[4] / "vllm_ascend/worker/v2/sample/logprob.py"
    spec = importlib.util.spec_from_file_location("decision_test_logprob", path)
    logprob = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(logprob)
    monkeypatch.setattr(logprob, "_fill_logprob_token_ids_kernel", Kernel(fill_ids))
    monkeypatch.setattr(logprob, "_ranks_kernel", Kernel(ranks))
    monkeypatch.setattr(logprob, "get_vectorcore_num", lambda: 2)
    monkeypatch.setattr(logprob.triton, "next_power_of_2", lambda n: 1 << (n - 1).bit_length(), raising=False)
    monkeypatch.setattr(
        logprob, "compute_token_logprobs", lambda logits, ids: logits.float().log_softmax(-1).gather(-1, ids.long())
    )

    return logprob


@pytest.mark.parametrize("num_logprobs", [0, 2, 4])
@pytest.mark.parametrize("logits_mode", [False, True])
@pytest.mark.parametrize("custom_ids", [False, True])
@pytest.mark.parametrize("tensor_boundaries", [False, True])
def test_mixed_label_reads(cpu_kernels, num_logprobs, logits_mode, custom_ids, tensor_boundaries):
    logprob = cpu_kernels
    logits = torch.tensor(
        [
            [1.0, 4.0, 0.0, 3.0, -5.0, 2.0],
            [2.0, 1.0, 3.0, 0.0, -1.0, 4.0],
            [5.0, 3.0, 2.0, 0.0, 1.0, -2.0],
            [-2.0, 0.0, 3.0, 1.0, 4.0, 2.0],
        ]
    )
    sampled = logits.argmax(-1)
    state = SimpleNamespace(
        num_token_ids=SimpleNamespace(gpu=torch.tensor([3, 0, 1], dtype=torch.int32)),
        token_ids=SimpleNamespace(gpu=torch.tensor([[4, 2, 0], [0, 0, 0], [5, 0, 0]], dtype=torch.int32)),
    )
    mapping = torch.tensor([2, 0, 1, 0], dtype=torch.int32)
    boundaries = torch.tensor([0, 1, 2, 4]) if tensor_boundaries else [0, 1, 2, 4]
    out = logprob.compute_topk_scores(
        logits,
        num_logprobs,
        sampled,
        boundaries,
        logprob_token_ids_state=state if custom_ids else None,
        expanded_idx_mapping=mapping if custom_ids else None,
        max_per_req_token_ids=3 if custom_ids else 0,
        logits_mode=logits_mode,
    )
    scores = logits if logits_mode else logits.log_softmax(-1)
    for row, idx in enumerate(mapping.tolist()):
        count = int(state.num_token_ids.gpu[idx]) if custom_ids else 0
        ids = state.token_ids.gpu[idx, :count] if count else logits[row].topk(num_logprobs).indices
        expected = torch.cat((sampled[row : row + 1], ids.long()))
        assert torch.equal(out.logprob_token_ids[row, : len(expected)], expected)
        torch.testing.assert_close(out.logprobs[row, : len(expected)], scores[row, expected])
        assert torch.isneginf(out.logprobs[row, len(expected) :]).all()
    assert (
        (out.cu_num_generated_tokens_tensor is boundaries)
        if tensor_boundaries
        else (out.cu_num_generated_tokens == boundaries)
    )


def test_custom_ids_require_request_state(cpu_kernels):
    logprob = cpu_kernels
    with pytest.raises(AssertionError):
        logprob.compute_topk_scores(torch.ones(1, 4), 0, torch.tensor([0]), max_per_req_token_ids=2)
