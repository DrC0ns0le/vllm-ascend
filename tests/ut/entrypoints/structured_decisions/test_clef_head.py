# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The shared Clef pooler against a port of the checkpoint's joint schema head,
which loops over questions where the pooler batches them."""

import math
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F
from vllm.pooling_params import PoolingParams
from vllm.v1.pool.metadata import PoolingMetadata, PoolingStates

from vllm_ascend.pooling.clef import (
    JointSchemaHead,
    JointSchemaPooler,
    embed_lexical,
    parse_joint_schema,
)

DEVICE_TYPE = "cpu"
HIDDEN, VOCAB = 64, 500
TYPES = [0, 1, 1, 2]
COUNTS = [2, 3, 1, 5]


def reference_logits(head, hidden, token_ids, embedding, question_spans, spans):
    """``JointSchemaHead.forward`` of the checkpoint's joint_schema_model.py
    for one record, with ``spans`` the option spans of each question."""
    hidden = head.hidden_norm(hidden)
    memory = head.memory_projection(hidden).unsqueeze(0)
    global_vector = hidden[-1]
    questions = torch.stack([hidden[s:e].mean(0) for s, e in question_spans])
    contexts = [torch.stack([hidden[s:e].mean(0) for s, e in q]) for q in spans]
    lexicals = [torch.stack([embedding[token_ids[s:e]].mean(0) for s, e in q]) for q in spans]
    options = torch.cat(
        [
            head.option_context_projection(c)
            + head.option_lexical_projection(lex)
            + head.option_question_projection(questions[i]).unsqueeze(0)
            for i, (c, lex) in enumerate(zip(contexts, lexicals))
        ]
    ).unsqueeze(0)
    for layer in head.evidence_layers:
        options = layer(options, memory)
    routed = torch.split(options[0], [len(q) for q in spans])

    base = head.question_projection(questions)
    summaries = []
    for field, opts in zip(base, routed):
        weights = torch.softmax(opts @ field / math.sqrt(opts.shape[-1]), dim=0)
        summaries.append((weights[:, None] * opts).sum(0))
    fields = (
        base
        + head.option_summary_norm(torch.stack(summaries))
        + head.global_projection(global_vector)
        + head.type_embedding(torch.tensor(TYPES, device=hidden.device))
    ).unsqueeze(0)
    for layer in head.layers:
        fields = layer(fields, memory)
    fields = head.field_norm(fields[0])

    logits = []
    for i, (field, lexical, opts) in enumerate(zip(fields, lexicals, routed)):
        anchor = F.normalize(questions[i] + global_vector, dim=-1)
        prior_scale = head.prior_logit_scale.clamp(max=math.log(100.0)).exp()
        prior = prior_scale * (F.normalize(lexical, dim=-1) @ anchor)
        opts = head.option_norm(opts)
        field = field.expand_as(opts)
        features = torch.cat([field, opts, field * opts, (field - opts).abs()], -1)
        joint_scale = head.joint_logit_scale.clamp(max=math.log(100.0)).exp()
        joint = joint_scale * F.cosine_similarity(field, opts, dim=-1)
        joint = joint + head.residual_scorer(features).squeeze(-1)
        logits.append(prior + torch.sigmoid(head.residual_gate) * joint)
    return logits


@pytest.fixture(autouse=True)
def cpu_memory(monkeypatch):
    # The accelerator-oriented metadata defaults to pinned allocations.
    monkeypatch.setattr("vllm.v1.pool.metadata.PIN_MEMORY", False)
    monkeypatch.setattr("vllm.utils.torch_utils.PIN_MEMORY", False)


@pytest.fixture
def pooler():
    torch.manual_seed(0)
    head = JointSchemaHead(
        hidden_size=HIDDEN,
        width=32,
        routing_layers=2,
        layers=2,
        heads=4,
        feedforward=48,
    ).eval()
    with torch.no_grad():
        for p in head.parameters():
            p.add_(torch.randn_like(p) * 0.2)
    head.to(DEVICE_TYPE)
    embedding = torch.randn(VOCAB, HIDDEN, device=DEVICE_TYPE)
    config = SimpleNamespace(scheduler_config=SimpleNamespace(enable_chunked_prefill=False, async_scheduling=False))
    with patch(
        "vllm.model_executor.layers.pooler.tokwise.methods.get_current_vllm_config",
        return_value=config,
    ):
        pooler = JointSchemaPooler(head, lambda ids: embedding[ids])
    return pooler, embedding


def schema(length: int):
    """Questions after a 20-token state, with options of 1 to 4 tokens."""
    pos = 20
    question_spans: list[tuple[int, int]] = []
    spans: list[list[tuple[int, int]]] = []
    for count in COUNTS:
        question_spans.append((pos, pos + 4))
        pos += 6
        spans.append([])
        for k in range(count):
            n = 1 + (k * 3) % 4
            spans[-1].append((pos, pos + n))
            pos += n + 1
    assert pos < length
    extra = {
        "question_types": TYPES,
        "question_spans": [[s - length, e - length] for s, e in question_spans],
        "option_spans": [[s - length, e - length] for q in spans for s, e in q],
        "option_counts": COUNTS,
    }
    return {"joint_schema": extra}, question_spans, spans


@torch.inference_mode()
def test_pooler_matches_reference(pooler):
    pooler, embedding = pooler
    lengths = [120, 90, 100]
    token_ids = [torch.randint(0, VOCAB, (n,)) for n in lengths]
    hidden = torch.randn(sum(lengths), HIDDEN, device=DEVICE_TYPE)
    first, first_questions, first_spans = schema(lengths[0])
    last, last_questions, last_spans = schema(lengths[2])
    params = [
        PoolingParams(task="token_classify", use_activation=True, extra_kwargs=first),
        # A prompt without a schema, like the profiling run's.
        PoolingParams(task="token_classify", use_activation=True),
        PoolingParams(task="token_classify", use_activation=False, extra_kwargs=last),
    ]
    padded = torch.zeros(len(lengths), max(lengths), dtype=torch.long)
    for i, ids in enumerate(token_ids):
        padded[i, : len(ids)] = ids
    metadata = PoolingMetadata(
        prompt_lens=torch.tensor(lengths),
        prompt_token_ids=None,
        prompt_token_ids_cpu=padded,
        pooling_params=params,
        pooling_states=[PoolingStates() for _ in lengths],
    )
    metadata.build_pooling_cursor(
        torch.tensor(lengths).numpy(),
        seq_lens_cpu=torch.tensor(lengths),
        device=hidden.device,
    )
    probs, empty, logits = pooler(hidden, metadata)

    first_hidden, _, last_hidden = torch.split(hidden, lengths)
    expected = reference_logits(pooler.head, first_hidden, token_ids[0], embedding, first_questions, first_spans)
    expected_probs = torch.cat([q.softmax(-1) for q in expected])
    torch.testing.assert_close(probs, expected_probs, atol=1e-5, rtol=1e-4)
    assert empty.numel() == 0
    expected = reference_logits(pooler.head, last_hidden, token_ids[2], embedding, last_questions, last_spans)
    torch.testing.assert_close(logits, torch.cat(expected), atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize(
    "update",
    [
        {"question_types": []},
        {"question_types": [3, 1, 1, 2]},
        {"question_types": [float("inf"), 1, 1, 2]},
        {"option_counts": [0, 3, 1, 5]},
        {"option_counts": [2]},
        {"option_spans": [[-200, -1]]},
        {"question_spans": [[-5, -5]]},
        {"question_spans": [[-1, 1]] * 4},
        {"option_spans": None},
    ],
)
def test_invalid_schema(update):
    extra, _, _ = schema(120)
    extra["joint_schema"].update(update)
    assert parse_joint_schema(extra, 120) is None


@pytest.mark.parametrize("extra", [None, {}, {"joint_schema": []}, {"joint_schema": {}}])
def test_missing_schema(extra):
    assert parse_joint_schema(extra, 120) is None


@torch.inference_mode()
def test_chunked_prefill_matches_full_prompt(pooler):
    pooler, embedding = pooler
    pooler.pooling.enable_chunked_prefill = True
    pooler.pooling.clone_finished = True
    length, first_chunk = 120, 40
    token_ids = torch.randint(0, VOCAB, (length,))
    hidden = torch.randn(length, HIDDEN)
    extra, questions, options = schema(length)
    metadata = PoolingMetadata(
        prompt_lens=torch.tensor([length]),
        prompt_token_ids=None,
        prompt_token_ids_cpu=token_ids[None],
        pooling_params=[PoolingParams(task="token_classify", extra_kwargs=extra)],
        pooling_states=[PoolingStates()],
    )
    metadata.build_pooling_cursor(torch.tensor([first_chunk]).numpy(), torch.tensor([first_chunk]), hidden.device)
    scratch = hidden[:first_chunk].clone()
    assert pooler(scratch, metadata) == [None]
    scratch.zero_()  # A later model step may reuse the runner's hidden-state buffer.
    metadata.build_pooling_cursor(torch.tensor([length - first_chunk]).numpy(), torch.tensor([length]), hidden.device)
    (actual,) = pooler(hidden[first_chunk:], metadata)
    expected = reference_logits(pooler.head, hidden, token_ids, embedding, questions, options)
    torch.testing.assert_close(actual, torch.cat([q.softmax(-1) for q in expected]))
    assert metadata.pooling_states[0].hidden_states_cache == []
    assert pooler.get_supported_tasks() == {"token_classify"}
    assert pooler.get_pooling_updates("token_classify").requires_token_ids


@pytest.mark.parametrize("shards", [1, 2, 4])
def test_lexical_output_embeddings_across_tp_shards(shards):
    # Include padding between base vocabulary and added-token rows.
    base_vocab, padded_base, added_vocab, width = 13, 16, 4, 8
    weight = torch.randn(base_vocab + added_vocab, width)
    token_ids = torch.tensor([0, 4, 8, 12, 13, 14, 16, 4])
    contributions = []
    for rank in range(shards):
        begin = rank * (padded_base // shards)
        end = min(begin + padded_base // shards, base_vocab)
        added_begin = base_vocab + rank * (added_vocab // shards)
        added_end = added_begin + added_vocab // shards
        local = (
            torch.cat(
                [
                    weight[begin:end],
                    torch.zeros(begin + padded_base // shards - end, width),
                    weight[added_begin:added_end],
                ]
            )
            if shards > 1
            else weight
        )
        head = SimpleNamespace(
            tp_size=shards,
            weight=local,
            shard_indices=SimpleNamespace(
                org_vocab_start_index=begin,
                org_vocab_end_index=end,
                added_vocab_start_index=added_begin,
                added_vocab_end_index=added_end,
                num_org_vocab_padding=begin + padded_base // shards - end,
            ),
            quant_method=SimpleNamespace(embedding=lambda head, ids: F.embedding(ids, head.weight)),
            output_reduce_group=SimpleNamespace(all_reduce=lambda tensor: tensor),
        )
        # A simulated sum collects exactly the per-rank inputs to HCCL all_reduce.
        contributions.append(embed_lexical(head, token_ids))
    torch.testing.assert_close(torch.stack(contributions).sum(0), weight[token_ids])
