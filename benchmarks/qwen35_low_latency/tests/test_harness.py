# SPDX-License-Identifier: Apache-2.0
import importlib.util
import json
from pathlib import Path

import pytest


def load(name):
    path = Path(__file__).resolve().parents[1] / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("packed", [True, False])
def test_qwen_patch_wraps_every_present_registration_and_is_idempotent(packed):
    helper = load("patch_qwen")
    source = "from vllm import envs\n"
    names = helper.OP_NAMES if packed else helper.OP_NAMES[:1]
    for name in names:
        source += f'direct_register_custom_op(op_name="{name}", op_func={name}, mutates_args=["output"])\n'
    modified, found = helper.patch_source(source)
    assert found == list(names)
    assert modified.count("op_func=eager_break_during_capture(") == len(names)
    assert helper.patch_source(modified)[0] == modified


def test_qwen_patch_rejects_unknown_op_wrapper():
    helper = load("patch_qwen")
    with pytest.raises(ValueError, match="Unexpected registration"):
        helper.patch_source('direct_register_custom_op(op_name="qwen_gdn_attention_core", op_func=other)')


def test_variants_form_factorial_experiment():
    serve = load("serve")
    for variant, graph, mega in (("A", "0", "0"), ("B", "1", "0"), ("C", "0", "1"), ("D", "1", "1")):
        env, config = serve.configuration(variant, "1536")
        assert env["VLLM_USE_BREAKABLE_CUDAGRAPH"] == graph
        assert env["VLLM_ASCEND_PTO_CHUNK_GDN"] == mega
        assert env["TASK_QUEUE_ENABLE"] == "1"
        assert 196 not in config["cudagraph_capture_sizes"]
    assert serve.configuration("D", "4096")[1]["cudagraph_capture_sizes"][-3:] == [2048, 3072, 4096]


def test_streaming_timing_ignores_empty_initial_event(monkeypatch):
    bench = load("benchmark")

    class Response:
        def __enter__(self):
            return iter(
                [
                    b'data: {"choices":[{"text":""}]}\n',
                    b'data: {"choices":[{"text":"hi","logprobs":{"tokens":["hi"]}}]}\n',
                    b'data: {"choices":[{"text":"!","logprobs":{"tokens":["!"]}}]}\n',
                    b'data: {"choices":[],"usage":{"prompt_tokens":3,"completion_tokens":2}}\n',
                    b"data: [DONE]\n",
                ]
            )

        def __exit__(self, *args):
            pass

    monkeypatch.setattr(bench, "post", lambda *args: Response())
    ticks = iter([1.0, 1.1, 1.108, 1.11])
    monkeypatch.setattr(bench.time, "perf_counter", lambda: next(ticks))
    row = bench.request_one("http://unused", "qwen", [1, 2, 3], 2)
    assert row["ttft_ms"] == pytest.approx(100)
    assert row["tpot_ms"] == pytest.approx(8)
    assert row["e2e_ms"] == pytest.approx(110)
    assert row["text"] == "hi!"
    json.dumps(row)


def test_latency_is_not_divided_by_concurrency():
    bench = load("benchmark")
    report = bench.summary([dict(ttft_ms=100, e2e_ms=124, tpot_ms=8)] * 4, 0.2)
    assert report["ttft_ms_p50"] == 100
    assert report["e2e_ms_p50"] == 124
    assert report["requests_per_second"] == 20


def test_comparison_uses_repeated_baselines_and_reports_missing_prompts():
    compare = load("compare").compare
    key = ((1, 2), 4)
    baseline = {key: [dict(text="a", tokens=["a"]), dict(text="b", tokens=["b"])]}
    candidate = {
        key: [dict(text="b", tokens=["b"]), dict(text="!", tokens=["!"])],
        ((3,), 4): [dict(text="c", tokens=["c"])],
    }
    report = compare(baseline, candidate, "[a-z]")
    assert report["compared_requests"] == 2
    assert report["unmatched_requests"] == 1
    assert report["exact_match_rate"] == 0.5
    assert report["mean_token_sequence_divergence"] == 0.5
    assert report["invalid_format"] == 1


def test_log_evidence_requires_actual_mixed_metadata_and_latest_layer_counters():
    inspect = load("inspect_log").inspect
    assert not inspect("clients overlap\nQwen GDN step: prefills=1 decodes=0")["mixed_gdn_observed"]
    report = inspect(
        "Qwen GDN step: prefills=2 decodes=1\n"
        "Breakable ACLGraph replay: graphs=19 eager_breaks=18\n"
        "MegaGDN counters: layer=layer0 counts={'megagdn': 1}\n"
        "MegaGDN counters: layer=layer0 counts={'megagdn': 2}\n"
        "MegaGDN counters: layer=layer1 counts={'megagdn': 2, 'fallback:hardware': 1}\n"
    )
    assert report["mixed_gdn_observed"]
    assert report["graph_segment_submissions"] == 19
    assert report["eager_break_invocations"] == 18
    assert report["mega_counters"] == {"megagdn": 4, "fallback:hardware": 1}
