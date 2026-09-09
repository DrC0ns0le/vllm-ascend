# SPDX-License-Identifier: Apache-2.0
"""CPU configuration, dispatch and startup contracts for native piecewise GDN."""

import ast
from bisect import bisect_left
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

ROOT = Path(__file__).resolve().parents[4]
SIZES = [1, 64, 128, 196, 256, 384, 512, 768, 1024, 1536, 2048, 3172, 4096]


class Mode(Enum):
    NONE = 0
    PIECEWISE = 1
    FULL = 2
    FULL_AND_PIECEWISE = 3
    FULL_DECODE_ONLY = 4

    @classmethod
    def valid_runtime_modes(cls):
        return (cls.NONE, cls.PIECEWISE, cls.FULL)

    def mixed_mode(self):
        return Mode.PIECEWISE if self == Mode.FULL_AND_PIECEWISE else self


@dataclass(frozen=True)
class Descriptor:
    num_tokens: int
    num_reqs: int | None = None
    uniform: bool = False
    has_lora: bool = False


def load_function(path, name, namespace):
    source = ROOT / path
    tree = ast.parse(source.read_text())
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
    node.decorator_list = []
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)
    return namespace[name]


def config(mode=Mode.FULL_AND_PIECEWISE, maximum=4096):
    return SimpleNamespace(
        compilation_config=SimpleNamespace(
            cudagraph_mode=mode, cudagraph_capture_sizes=SIZES.copy(), max_cudagraph_capture_size=4096
        ),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=maximum),
        model_config=SimpleNamespace(enforce_eager=False),
    )


def configure(config, *, breakable=True, gdn=True):
    namespace = dict(
        CUDAGraphMode=Mode,
        logger=Mock(),
        envs_vllm=SimpleNamespace(VLLM_USE_BREAKABLE_CUDAGRAPH=breakable),
        check_gdn_layer=lambda _: gdn,
    )
    fn = load_function("vllm_ascend/compilation/piecewise_graph.py", "configure_gdn_piecewise_graphs", namespace)
    fn(config)


@pytest.mark.parametrize("mode", [Mode.FULL, Mode.FULL_AND_PIECEWISE, Mode.PIECEWISE])
@pytest.mark.parametrize("maximum", [256, 4096, 8192])
def test_config_covers_every_scheduled_total_with_existing_buckets_and_ceiling(mode, maximum):
    cfg = config(mode, maximum)
    configure(cfg)
    expected = sorted({n for n in SIZES if n <= maximum} | {maximum})
    assert cfg.compilation_config.cudagraph_capture_sizes == expected
    assert cfg.compilation_config.max_cudagraph_capture_size == maximum
    assert cfg.compilation_config.cudagraph_mode == (Mode.FULL_AND_PIECEWISE if mode == Mode.FULL else mode)
    for tokens in range(1, maximum + 1):
        bucket = expected[bisect_left(expected, tokens)]
        assert tokens <= bucket <= maximum
    configure(cfg)
    assert cfg.compilation_config.cudagraph_capture_sizes == expected


@pytest.mark.parametrize("reason", ["eager", "no_breakable", "other_model", "none", "decode_only"])
def test_config_does_not_enable_graphs_outside_requested_scope(reason):
    cfg = config()
    if reason in ("none", "decode_only"):
        cfg.compilation_config.cudagraph_mode = Mode.NONE if reason == "none" else Mode.FULL_DECODE_ONLY
    cfg.model_config.enforce_eager = reason == "eager"
    before = vars(cfg.compilation_config).copy()
    configure(cfg, breakable=reason != "no_breakable", gdn=reason != "other_model")
    assert vars(cfg.compilation_config) == before


def test_runner_passes_mixed_layouts_to_standard_padded_dispatcher():
    calls = []

    def dispatch(*, num_tokens, uniform_decode, **kwargs):
        calls.append((num_tokens, uniform_decode))
        if uniform_decode:
            return Mode.FULL, Descriptor(64, 64, True)
        return Mode.PIECEWISE, Descriptor(SIZES[bisect_left(SIZES, num_tokens)])

    determine = load_function(
        "vllm_ascend/worker/model_runner_v1.py",
        "_determine_batch_execution_and_padding",
        dict(
            CUDAGraphMode=Mode,
            BatchDescriptor=Descriptor,
            logger=Mock(),
            np=SimpleNamespace(all=all),
            enable_sp=lambda *a: False,
            breakable_cudagraph=SimpleNamespace(is_breakable_cudagraph_enabled=lambda: True),
        ),
    )
    runner = SimpleNamespace(
        _pad_for_sequence_parallelism=lambda n: n,
        input_batch=SimpleNamespace(num_computed_tokens_cpu=torch.ones(64), lora_id_to_lora_request={}),
        speculative_config=None,
        uniform_decode_query_len=1,
        model_config=SimpleNamespace(is_encoder_decoder=False),
        vllm_config=SimpleNamespace(
            parallel_config=SimpleNamespace(data_parallel_size=1),
            observability_config=SimpleNamespace(cudagraph_metrics=False),
        ),
        cudagraph_dispatcher=SimpleNamespace(dispatch=dispatch),
    )
    for lengths in ([61], [17, 44], [1, 1, 59]):
        result = determine(runner, sum(lengths), len(lengths), lengths, max(lengths), False)
        assert result[:2] == (Mode.PIECEWISE, Descriptor(64))
    for lengths in ([1] * 63 + [137], [1, 767], [767] * 4):
        tokens = sum(lengths)
        result = determine(runner, tokens, len(lengths), lengths, max(lengths), False)
        assert result[:2] == (Mode.PIECEWISE, Descriptor(SIZES[bisect_left(SIZES, tokens)]))
    assert determine(runner, 17, 17, [1] * 17, 1, False)[0] == Mode.FULL
    # Startup must capture a mixed bucket even when all dummy lengths are one.
    assert determine(runner, 17, 17, [1] * 17, 1, False, force_uniform_decode=False)[0] == Mode.PIECEWISE
    runner.input_batch.num_computed_tokens_cpu.zero_()
    assert determine(runner, 17, 17, [1] * 17, 1, False)[0] == Mode.PIECEWISE


@pytest.mark.parametrize("missing", [None, "descriptor", "entry", "capture"])
def test_startup_checks_every_predefined_piecewise_bucket(missing):
    namespace = dict(CUDAGraphMode=Mode, logger=Mock())
    validate = load_function("vllm_ascend/compilation/breakable_aclgraph.py", "validate_piecewise_capture", namespace)
    descs = [Descriptor(size) for size in SIZES]
    entries = {d: SimpleNamespace(capture=object()) for d in descs}
    if missing == "entry":
        entries.pop(descs[-1])
    elif missing == "capture":
        entries[descs[-1]].capture = None
    elif missing == "descriptor":
        descs.pop()
    wrapper = SimpleNamespace(compilation_config=config().compilation_config, entries=entries)
    if missing:
        with pytest.raises(RuntimeError, match="startup capture is incomplete"):
            validate(wrapper, [(Mode.PIECEWISE, descs)])
    else:
        validate(wrapper, [(Mode.FULL, []), (Mode.PIECEWISE, descs)])
        namespace["logger"].info.assert_called_once()


@pytest.mark.parametrize("tokens", [1, 64, 196, 4096])
def test_real_dummy_run_forces_mixed_capture_even_with_one_token_requests(tokens):
    class ReachedDispatcher(Exception):
        pass

    dispatch = Mock(side_effect=ReachedDispatcher)
    dummy = load_function(
        "vllm_ascend/worker/model_runner_v1.py",
        "_dummy_run",
        dict(
            CUDAGraphMode=Mode,
            np=SimpleNamespace(
                array=lambda values, dtype: torch.tensor(values), int32=torch.int32, ones=lambda n, dtype: torch.ones(n)
            ),
            torch=SimpleNamespace(from_numpy=lambda tensor: tensor),
        ),
    )
    runner = SimpleNamespace(
        uniform_decode_query_len=1,
        scheduler_config=SimpleNamespace(max_num_batched_tokens=4096, max_num_seqs=64),
        dynamic_eplb=False,
        _determine_batch_execution_and_padding=dispatch,
    )
    with pytest.raises(ReachedDispatcher):
        dummy(runner, tokens, cudagraph_runtime_mode=Mode.PIECEWISE, uniform_decode=False)
    assert dispatch.call_args.kwargs["force_uniform_decode"] is False
    assert dispatch.call_args.kwargs["num_tokens"] == tokens


@pytest.mark.parametrize("actual,capacity", [(61, 64), (195, 196), (257, 384)])
def test_native_kv_update_excludes_padded_rows(actual, capacity):
    cache = torch.full((capacity, 2), -123.0)
    slots = torch.arange(capacity - 1, -1, -1)
    inputs = torch.arange(capacity * 2).reshape(capacity, 2).float()
    inputs[actual:] = float("nan")

    def store(*, key, value, key_cache, value_cache, slot_mapping):
        assert key.shape[0] == value.shape[0] == slot_mapping.numel() == actual
        key_cache[slot_mapping] = key

    update = load_function(
        "vllm_ascend/attention/attention_v1.py",
        "reshape_and_cache",
        dict(
            DeviceOperator=SimpleNamespace(reshape_and_cache=store),
            AttentionType=SimpleNamespace(ENCODER_DECODER="cross"),
            notify_kv_cache_written=Mock(),
        ),
    )
    impl = SimpleNamespace(key_cache=cache, value_cache=cache, kv_sharing_target_layer_name=None, attn_type="decoder")
    update(
        impl,
        inputs,
        inputs,
        inputs,
        [cache, cache],
        SimpleNamespace(slot_mapping=slots, num_actual_tokens=actual),
        torch.empty_like(inputs),
    )
    torch.testing.assert_close(cache[slots[:actual]], inputs[:actual])
    assert cache[slots[actual:]].eq(-123).all()
