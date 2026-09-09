# SPDX-License-Identifier: Apache-2.0
"""FULL-only coverage and startup lifecycle without an Ascend runtime."""

import ast
from bisect import bisect_left
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from enum import Enum
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from test_capacity_metadata import ROOT, load_source


@pytest.fixture
def policy_class():
    return load_source(
        "vllm_ascend/compilation/full_graph_policy.py",
        {"FullGraphCapacityPolicy"},
        dict(dataclass=dataclass, bisect_left=bisect_left, __name__=__name__),
    )["FullGraphCapacityPolicy"]


CAPTURE_SIZES = [1, 64, 128, 196, 256, 384, 512, 768, 1024, 1536, 2048, 3172, 4096]


@pytest.mark.parametrize("maximum", [1536, 2048, 4096, 8192])
@pytest.mark.parametrize("entries", [None, 1, 2, 8])
def test_every_token_total_has_a_covering_full_capacity(policy_class, maximum, entries):
    policy = policy_class.build(maximum, 64, CAPTURE_SIZES, entries)
    assert policy.capacities[-1] == maximum
    if entries is not None:
        assert len(policy.capacities) <= entries
    for tokens in range(1, maximum + 1):
        capacity = policy.capacity(tokens, min(tokens, 64))
        assert tokens <= capacity <= maximum
        # Request count and partition do not create additional capacities.
        assert policy.capacity(tokens, 1) == capacity
        lengths = policy.dummy_lengths(capacity, 768)
        assert sum(lengths) == capacity and max(lengths) <= 768
        assert 1 <= len(lengths) <= 64


@pytest.mark.parametrize("limit", [0, -1, True, "8"])
def test_full_cannot_disable_graphs_with_entry_limit(policy_class, limit):
    with pytest.raises(ValueError, match="must be positive"):
        policy_class.build(4096, 64, CAPTURE_SIZES, limit)


class Mode(Enum):
    NONE = 0
    PIECEWISE = 1
    FULL = 2
    FULL_DECODE_ONLY = 3

    @classmethod
    def valid_runtime_modes(cls):
        return set(cls)


@dataclass(frozen=True)
class Descriptor:
    num_tokens: int
    num_reqs: int | None = None
    uniform: bool = False
    has_lora: bool = False


def runner_method(name, namespace):
    path = ROOT / "vllm_ascend/worker/model_runner_v1.py"
    cls = next(
        node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.ClassDef) and node.name == "NPUModelRunner"
    )
    function = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == name)
    function.decorator_list = []
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), function],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[name]


def test_runner_routes_ragged_batches_above_dispatcher_limit_to_full(policy_class):
    namespace = dict(
        CUDAGraphMode=Mode,
        BatchDescriptor=Descriptor,
        np=SimpleNamespace(all=all),
        enable_sp=lambda *args: False,
        breakable_cudagraph=SimpleNamespace(is_breakable_cudagraph_enabled=lambda: True),
        logger=Mock(),
    )
    determine = runner_method("_determine_batch_execution_and_padding", namespace)
    runner = SimpleNamespace(
        _pad_for_sequence_parallelism=lambda n: n,
        input_batch=SimpleNamespace(num_computed_tokens_cpu=torch.ones(64), lora_id_to_lora_request={}),
        speculative_config=None,
        uniform_decode_query_len=1,
        model_config=SimpleNamespace(is_encoder_decoder=False),
        full_graph_policy=policy_class.build(8192, 64, [1, 64, 128], 2),
        vllm_config=SimpleNamespace(
            parallel_config=SimpleNamespace(data_parallel_size=1),
            observability_config=SimpleNamespace(cudagraph_metrics=False),
        ),
        cudagraph_dispatcher=SimpleNamespace(dispatch=Mock(return_value=(Mode.FULL, Descriptor(64, 64, True)))),
    )
    for lengths in ([137], [1, 1, 193], [63, 65, 127], [767] * 10, [1] * 63 + [768]):
        result = determine(runner, sum(lengths), len(lengths), lengths, max(lengths), False)
        assert result[:2] == (Mode.FULL, Descriptor(runner.full_graph_policy.capacity(sum(lengths), len(lengths))))
    runner.cudagraph_dispatcher.dispatch.assert_not_called()
    assert determine(runner, 17, 17, [1] * 17, 1, False)[0] == Mode.FULL
    runner.cudagraph_dispatcher.dispatch.assert_called_once()
    for kwargs in (dict(force_eager=True), dict(use_cascade_attn=True)):
        with pytest.raises(RuntimeError, match="FULL-only"):
            determine(runner, 137, 1, [137], 137, **dict(use_cascade_attn=False) | kwargs)


@pytest.mark.parametrize("actual", [1, 5, 64])
def test_full_only_preserves_uniform_decode_request_padding(actual):
    query = SimpleNamespace(np=torch.zeros(66, dtype=torch.int32), copy_to_gpu=Mock())
    query.np[: actual + 1] = torch.arange(actual + 1)
    runner = SimpleNamespace(
        full_graph_policy=object(),
        compilation_config=SimpleNamespace(cudagraph_mode=Mode.FULL),
        uniform_decode_query_len=1,
        arange_np=torch.arange(66),
    )
    pad = runner_method("_pad_query_start_loc_for_fia", dict(CUDAGraphMode=Mode))
    assert pad(runner, query, 64, 64, actual, Mode.FULL, 64) == 64
    assert query.np[:65].tolist() == list(range(65))
    query.copy_to_gpu.assert_called_once()


@pytest.mark.parametrize("outcome", ["success", "downgrade", "error"])
def test_standard_capture_initializes_decode_only_and_preserves_full_contract(outcome):
    initialize = Mock()
    graph_params = Mock()

    def resolve(**kwargs):
        assert config.cudagraph_mode == Mode.FULL_DECODE_ONLY
        if outcome == "error":
            raise RuntimeError("resolution failed")
        return Mode.PIECEWISE if outcome == "downgrade" else Mode.FULL_DECODE_ONLY

    config = SimpleNamespace(cudagraph_mode=Mode.FULL, resolve_cudagraph_mode_and_sizes=resolve)
    runner = SimpleNamespace(
        full_graph_policy=object(),
        compilation_config=config,
        uniform_decode_query_len=1,
        parallel_config=SimpleNamespace(tensor_parallel_size=1),
        kv_cache_config=object(),
        max_num_reqs=64,
        cudagraph_dispatcher=SimpleNamespace(
            initialize_cudagraph_keys=initialize,
            get_capture_descs=lambda: [(Mode.FULL, [Descriptor(1, 1, True), Descriptor(64, 64, True)])],
        ),
        speculative_config=None,
        use_aclgraph=True,
    )
    check = runner_method(
        "_check_and_update_cudagraph_mode",
        dict(
            CUDAGraphMode=Mode,
            AttentionCGSupport=SimpleNamespace(ALWAYS=object()),
            update_pass_config=lambda runner: nullcontext(),
            set_graph_params=graph_params,
        ),
    )
    if outcome == "success":
        check(runner, [], [])
        initialize.assert_called_once_with(Mode.FULL_DECODE_ONLY, 1)
        graph_params.assert_called_once_with([1, 64])
    else:
        with pytest.raises((ValueError, RuntimeError)):
            check(runner, [], [])
        initialize.assert_not_called()
    assert config.cudagraph_mode == Mode.FULL


@pytest.mark.parametrize("fail", [False, True])
def test_native_startup_captures_all_capacities_and_restores_dummy_state(policy_class, fail):
    class Wrapper:
        full_prefill = SimpleNamespace(seal=Mock(), begin_capture=Mock())

    class Mamba:
        pass

    policy = policy_class.build(4096, 64, CAPTURE_SIZES, 3)
    rows = len(policy.dummy_lengths(4096, 768))
    states = (torch.randn(64, 8, 4), torch.randn(64, 2, 4, 4))
    before = [state.clone() for state in states]
    table = SimpleNamespace(cpu=torch.full((64, 1), 23, dtype=torch.int32), copy_to_gpu=Mock())
    computed = torch.full((64,), 31, dtype=torch.int32)
    prompts = torch.full((64,), 57, dtype=torch.int32)
    calls = []

    def dummy(capacity, **kwargs):
        assert kwargs["native_full_graph"] and kwargs["cudagraph_runtime_mode"] == Mode.FULL
        assert table.cpu[:rows, 0].tolist() == list(range(rows))
        states[0][:rows].zero_()
        states[1][:rows].zero_()
        computed[:rows].zero_()
        prompts[:rows].zero_()
        calls.append((capacity, kwargs["native_input_ids"]))
        if fail:
            raise RuntimeError("capture failed")

    runner = SimpleNamespace(
        full_graph_policy=policy,
        model=Wrapper(),
        max_model_len=768,
        compilation_config=SimpleNamespace(static_forward_context={"gdn": SimpleNamespace(kv_cache=states)}),
        input_batch=SimpleNamespace(
            num_computed_tokens_cpu_tensor=computed,
            num_prompt_tokens_cpu_tensor=prompts,
            block_table=[SimpleNamespace(block_table=table)],
        ),
        kv_cache_config=SimpleNamespace(kv_cache_groups=[SimpleNamespace(kv_cache_spec=Mamba())]),
        supports_mm_inputs=True,
        enable_prompt_embeds=False,
        _dummy_run=dummy,
    )
    capture = runner_method(
        "_capture_native_full_graphs",
        dict(
            torch=SimpleNamespace(
                Tensor=torch.Tensor,
                arange=torch.arange,
                npu=SimpleNamespace(current_stream=lambda: Mock(), synchronize=Mock()),
            ),
            BreakableACLGraphWrapper=Wrapper,
            MambaSpec=Mamba,
            CUDAGraphMode=Mode,
            logger=Mock(),
        ),
    )
    if fail:
        with pytest.raises(RuntimeError, match="capture failed"):
            capture(runner)
        runner.model.full_prefill.seal.assert_not_called()
    else:
        capture(runner)
        assert calls == [(size, ids) for size in reversed(policy.capacities) for ids in (True, False)]
        runner.model.full_prefill.seal.assert_called_once()
    for actual, expected in zip(states, before, strict=True):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert computed.eq(31).all() and prompts.eq(57).all() and table.cpu.eq(23).all()


@pytest.mark.parametrize("capacity", [1, 196, 4096])
@pytest.mark.parametrize("input_ids", [False, True])
def test_real_dummy_path_builds_fresh_metadata_before_full_capture(policy_class, capacity, input_ids):
    policy = policy_class.build(4096, 64, CAPTURE_SIZES)
    lengths = torch.tensor(policy.dummy_lengths(capacity, 768), dtype=torch.int32)
    rows = len(lengths)

    def buffer(shape):
        cpu = torch.zeros(shape, dtype=torch.int32)
        gpu = cpu.clone()
        return SimpleNamespace(np=cpu, gpu=gpu, copy_to_gpu=lambda: gpu.copy_(cpu))

    class Tables(list):
        def commit_block_table(self, count):
            assert count == rows

    tables = Tables([SimpleNamespace(slot_mapping=SimpleNamespace(gpu=torch.zeros(4096, dtype=torch.int64)))])
    runner = SimpleNamespace(
        full_graph_policy=policy,
        scheduler_config=SimpleNamespace(max_num_batched_tokens=4096, max_num_seqs=64),
        max_model_len=768,
        max_num_tokens=4096,
        uniform_decode_query_len=1,
        dynamic_eplb=False,
        use_dcp=False,
        speculative_config=None,
        _has_gdn=True,
        _has_sinks=False,
        use_compress=False,
        use_aux_hidden_state_outputs=False,
        drafter=None,
        lora_config=None,
        supports_mm_inputs=True,
        enable_prompt_embeds=True,
        uses_mrope=False,
        uses_xdrope_dim=0,
        model_config=SimpleNamespace(is_encoder_decoder=False),
        model=Mock(),
        device="cpu",
        vllm_config=SimpleNamespace(),
        query_start_loc=buffer(66),
        gdn_query_start_loc=buffer(66),
        query_pos=buffer(4096),
        seq_lens=torch.zeros(64, dtype=torch.int32),
        optimistic_seq_lens_cpu=torch.zeros(64, dtype=torch.int32),
        input_batch=SimpleNamespace(
            num_computed_tokens_cpu_tensor=torch.full((64,), 17, dtype=torch.int32),
            num_prompt_tokens_cpu_tensor=torch.full((64,), 31, dtype=torch.int32),
            block_table=tables,
        ),
        kv_cache_config=SimpleNamespace(kv_cache_groups=[object()]),
        input_ids=buffer(4096),
        inputs_embeds=buffer((4096, 4)),
        positions=torch.zeros(4096, dtype=torch.int32),
        synchronize_input_prep=nullcontext,
        maybe_dummy_run_with_lora=lambda *args, **kwargs: nullcontext(),
        _get_cumsum_and_arange=lambda tokens, pos: tokens.cumsum(0),
        _should_build_dummy_attn_metadata=lambda *args: True,
        _determine_batch_execution_and_padding=Mock(side_effect=AssertionError("native startup must bypass dispatch")),
        _pad_query_start_loc_for_fia=Mock(side_effect=AssertionError("native adapter owns padding")),
        _finalize_dump_data=Mock(),
    )
    seen = []

    def build(**kwargs):
        assert not kwargs["for_cudagraph_capture"]
        assert kwargs["num_reqs"] == kwargs["num_reqs_padded"] == rows
        assert runner.input_batch.num_computed_tokens_cpu_tensor[:rows].eq(0).all()
        torch.testing.assert_close(runner.input_batch.num_prompt_tokens_cpu_tensor[:rows], lengths)
        torch.testing.assert_close(runner.seq_lens[:rows], lengths)
        assert runner.gdn_query_start_loc.gpu[: rows + 1].tolist() == [0, *lengths.cumsum(0).tolist()]
        assert tables[0].slot_mapping.gpu.eq(-1).all()
        metadata = type("AscendMetadata", (), {})()
        metadata.seq_lens = runner.optimistic_seq_lens_cpu[:rows]
        metadata.seq_lens_list = lengths.tolist()
        return {"attention": metadata}, None

    @contextmanager
    def forward_context(metadata, config, **kwargs):
        assert kwargs["aclgraph_runtime_mode"] == Mode.FULL
        assert kwargs["batch_descriptor"] == Descriptor(capacity)
        assert metadata["attention"].seq_lens_list == [768] * rows
        seen.append(capacity)
        yield

    def forward(tokens, ids, positions, intermediate, embeds):
        assert tokens == capacity and intermediate is None
        assert (ids is not None) == input_ids
        assert (embeds is not None) != input_ids
        return torch.zeros(capacity, 4)

    runner._build_attention_metadata = build
    runner._model_forward = forward
    dummy = runner_method(
        "_dummy_run",
        dict(
            CUDAGraphMode=Mode,
            BatchDescriptor=Descriptor,
            AscendAttentionState=SimpleNamespace(PrefillNoCache="prefill", DecodeOnly="decode"),
            torch=SimpleNamespace(from_numpy=lambda x: x, zeros=torch.zeros, int32=torch.int32),
            np=SimpleNamespace(array=torch.tensor, ones=torch.ones, int32=torch.int32),
            get_pp_group=lambda: SimpleNamespace(is_first_rank=True),
            lmhead_tp_enable=lambda: False,
            update_cos_sin=Mock(),
            set_ascend_forward_context=forward_context,
        ),
    )
    output, _ = dummy(
        runner,
        capacity,
        cudagraph_runtime_mode=Mode.FULL,
        force_attention=True,
        native_full_graph=True,
        native_input_ids=input_ids,
    )
    assert output.shape == (capacity, 4) and seen == [capacity]
