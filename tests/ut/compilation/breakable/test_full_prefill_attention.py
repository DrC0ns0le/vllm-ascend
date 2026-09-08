# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for the production FIA task-update and buffer ownership code."""

import ast
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import torch

ROOT = Path(__file__).resolve().parents[4]


def load_functions(relative_path, names, namespace, owner=None):
    path = ROOT / relative_path
    tree = ast.parse(path.read_text())
    nodes = (
        tree.body
        if owner is None
        else next(node.body for node in tree.body if isinstance(node, ast.ClassDef) and node.name == owner)
    )
    functions = [node for node in nodes if isinstance(node, ast.FunctionDef) and node.name in names]
    for node in functions:
        node.decorator_list = []
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *functions],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace


def test_native_prefill_graph_params_override_decode_only_within_context():
    decode_params, prefill_params = object(), object()
    context = SimpleNamespace(full_prefill_graph_params=prefill_params)
    namespace = load_functions(
        "vllm_ascend/compilation/acl_graph.py",
        {"get_graph_params"},
        dict(get_forward_context=lambda: context, _graph_params=decode_params, _select_graph_params=lambda p: p),
    )
    assert namespace["get_graph_params"]() is prefill_params
    context.full_prefill_graph_params = None
    assert namespace["get_graph_params"]() is decode_params


def test_task_updates_resolve_captured_layer_names_and_current_owned_metadata():
    events = []
    params = SimpleNamespace(
        attn_params={193: []}, handles={193: ["h1", "h2"]}, events={193: []}, workspaces={193: "owned workspace"}
    )
    # Captured order differs from metadata insertion order and contains no GDN.
    for name in ("layers.1", "layers.5"):
        params.attn_params[193].append(
            (
                "query",
                "key",
                "value",
                "capture blocks",
                None,
                128,
                [1, 2],
                [1, 193],
                2,
                8,
                0.5,
                "output",
                "lse",
                3,
                2147483647,
                2147483647,
                None,
                None,
                None,
                None,
                name,
            )
        )
        params.events[193].append(SimpleNamespace(record=lambda stream: events.append("record")))
    first = SimpleNamespace(seq_lens_list=[51, 301], actual_seq_lengths_q=[1, 193], block_tables="first owned blocks")
    second = SimpleNamespace(seq_lens_list=[52, 302], actual_seq_lengths_q=[1, 193], block_tables="second owned blocks")
    context = SimpleNamespace(
        full_prefill_graph_params=params, attn_metadata={"gdn": object(), "layers.5": second, "layers.1": first}
    )
    out = Mock(side_effect=lambda **kwargs: events.append((kwargs["actual_seq_lengths_kv"], kwargs["block_table"])))
    namespace = load_functions(
        "vllm_ascend/attention/attention_v1.py",
        {"update_graph_params"},
        dict(
            needs_layer_aware_fia_graph_replay=lambda: False,
            # Even a token count ordinarily routed to PA must update FIA here.
            using_paged_attention=lambda *args: True,
            _EXTRA_CTX=SimpleNamespace(is_draft_model=False, sinks=False),
            get_graph_params=lambda: params,
            PagedAttentionGraphParam=type("PagedAttentionGraphParam", (), {}),
            torch=SimpleNamespace(
                npu=SimpleNamespace(
                    stream=lambda stream: nullcontext(),
                    graph_task_update_begin=lambda stream, handle: events.append(handle),
                    graph_task_update_end=lambda stream: events.append("end"),
                )
            ),
            torch_npu=SimpleNamespace(npu_fused_infer_attention_score=SimpleNamespace(out=out)),
        ),
        owner="AscendAttentionBackendImpl",
    )
    config = SimpleNamespace(model_config=SimpleNamespace(hf_text_config=SimpleNamespace(sliding_window=None)))
    namespace["update_graph_params"]("update stream", context, 193, config)
    assert events == [
        "h1",
        ([51, 301], "first owned blocks"),
        "end",
        "record",
        "h2",
        ([52, 302], "second owned blocks"),
        "end",
        "record",
    ]
    assert all(call.kwargs["workspace"] == "owned workspace" for call in out.call_args_list)
    assert all(call.kwargs["actual_seq_lengths"] == [1, 193] for call in out.call_args_list)


def test_full_prefill_fia_capture_slices_query_and_output_to_live_token_count():
    namespace = load_functions(
        "vllm_ascend/attention/attention_v1.py",
        {"forward_fused_infer_attention"},
        dict(_EXTRA_CTX=SimpleNamespace(capturing=True, full_prefill_graph_params=object())),
        owner="AscendAttentionBackendImpl",
    )
    output = torch.zeros(256, 8, 16)

    def capture(query, key, value, metadata, target):
        assert query.shape == target.shape == (193, 8, 16)
        target.fill_(7)
        return target, 193

    impl = SimpleNamespace(sinks=None, full_graph_fia=Mock(side_effect=capture))
    result = namespace["forward_fused_infer_attention"](
        impl,
        torch.empty_like(output),
        None,
        None,
        SimpleNamespace(actual_seq_lengths_q=[1, 193]),
        output,
    )
    assert result is output
    torch.testing.assert_close(output[:193], torch.full_like(output[:193], 7))
    torch.testing.assert_close(output[193:], torch.zeros_like(output[193:]))


def test_full_prefill_records_layer_name_even_when_decode_uses_ordered_metadata():
    namespace = load_functions(
        "vllm_ascend/attention/attention_v1.py",
        {"forward"},
        dict(_EXTRA_CTX=SimpleNamespace(full_prefill_graph_params=object())),
        owner="AscendAttentionBackendImpl",
    )
    impl = SimpleNamespace(_use_layer_aware_fia_graph_replay=False)
    layer = SimpleNamespace(layer_name="model.layers.3.self_attn", _k_scale_float=1, _v_scale_float=1)
    output = torch.ones(1, 8, 16)
    namespace["forward"](impl, layer, output, None, None, (), None, output)
    assert impl._layer_name == layer.layer_name
