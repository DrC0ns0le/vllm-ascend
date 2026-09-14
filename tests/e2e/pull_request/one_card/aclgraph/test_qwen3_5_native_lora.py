# SPDX-License-Identifier: Apache-2.0
"""910B LoRA capture/replay with adapter swaps and speculative verification."""

import shutil
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file
from vllm import SamplingParams
from vllm.config import LoRAConfig
from vllm.lora.request import LoRARequest

from tests.e2e.conftest import VllmRunner, wait_until_npu_memory_free
from tests.e2e.pull_request.one_card.aclgraph.test_qwen3_5_native_mtp import mtp_registry_status
from vllm_ascend.compilation.native_full_layout import NativeGraphShape
from vllm_ascend.compilation.native_lora import NativeGraphLoRA
from vllm_ascend.lora.punica_npu import PunicaWrapperNPU


def lora_registry_status(worker):
    cache = worker.model_runner.model.native_full
    weights = []
    for module in worker.model_runner.model.runnable.modules():
        for name in ("lora_a_stacked", "lora_b_stacked", "lora_a_packed"):
            stacked = getattr(module, name, None)
            if stacked is not None:
                weights.extend(t.data_ptr() for t in (stacked if isinstance(stacked, tuple) else (stacked,)))
    return dict(
        registries=mtp_registry_status(worker),
        weights=weights,
        mappings=[mapping.tokens.data_ptr() for entry in cache.entries.values() for mapping in entry.lora],
    )


def adapter_arrivals(llm, first, second):
    engine = llm.llm_engine
    finished = {}
    prompt = "Schema: singer(singer_id, name, age, country). Write SQL to count all singers."

    def step():
        for output in engine.step():
            if output.finished:
                finished[output.request_id] = output.outputs[0].token_ids

    for index, adapter in enumerate((first, second, None)):
        engine.add_request(
            str(index),
            prompt,
            SamplingParams(temperature=0, max_tokens=16, ignore_eos=True),
            lora_request=adapter,
        )
    step()
    step()
    assert engine.has_unfinished_requests()
    engine.add_request(
        "late",
        prompt + " Use SELECT.",
        SamplingParams(temperature=0, max_tokens=8, ignore_eos=True),
        lora_request=second,
    )
    for _ in range(128):
        if not engine.has_unfinished_requests():
            break
        step()
    assert len(finished) == 4
    return finished


def adapter_swap_cycles(llm, adapters):
    first, second, replacement = adapters
    assert llm.llm_engine.add_lora(first)
    assert llm.llm_engine.add_lora(second)
    initial = adapter_arrivals(llm, first, second)
    assert llm.llm_engine.remove_lora(first.lora_int_id)
    assert llm.llm_engine.add_lora(replacement)
    swapped = adapter_arrivals(llm, replacement, second)
    assert llm.llm_engine.remove_lora(replacement.lora_int_id)
    assert llm.llm_engine.add_lora(first)
    restored = adapter_arrivals(llm, first, second)
    assert initial == restored
    return initial, swapped, restored


@wait_until_npu_memory_free()
@pytest.mark.parametrize("backend", ["ascendc", "megagdn"])
@pytest.mark.parametrize("method,count", [(None, 0), ("mtp", 1), ("mtp", 2), ("mtp", 3), ("ngram", 3)])
def test_qwen3_5_native_lora_swap_without_recapture(
    monkeypatch, tmp_path, qwen35_text_lora_files, backend, method, count
):
    if not torch.npu.is_available() or "910B" not in torch.npu.get_device_name():
        pytest.skip("Requires 910B, Qwen3.5-4B and the trained SQL adapter fixture")
    monkeypatch.setenv("VLLM_USE_BREAKABLE_CUDAGRAPH", "1")
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "0")
    monkeypatch.setenv("VLLM_ASCEND_PTO_CHUNK_GDN", "0")
    original = Path(qwen35_text_lora_files)
    tensors = load_file(str(original / "adapter_model.safetensors"))
    adapters = [LoRARequest("sql", 1, str(original))]
    # Real adapter, zero-delta adapter, and changed weights in an evicted slot.
    for index, scale in enumerate((0, -1), start=2):
        path = tmp_path / str(index)
        path.mkdir()
        shutil.copyfile(original / "adapter_config.json", path / "adapter_config.json")
        save_file(
            {name: value * scale if "lora_B" in name else value for name, value in tensors.items()},
            str(path / "adapter_model.safetensors"),
        )
        adapters.append(LoRARequest(str(index), index, str(path)))
    options = dict(
        dtype="bfloat16",
        max_model_len=768,
        max_num_seqs=4,
        max_num_batched_tokens=4096,
        enable_lora=True,
        max_loras=2,
        max_cpu_loras=2,
        max_lora_rank=8,
        enable_chunked_prefill=False,
        enable_prefix_caching=False,
        mamba_cache_mode="none",
    )
    with VllmRunner("Qwen/Qwen3.5-4B", enforce_eager=True, **options) as runner:
        expected = adapter_swap_cycles(runner.model, adapters)
        assert expected[0]["1"] == expected[0]["2"], "Zero-delta adapter must match the base model"
        assert any(wave["0"] != wave["2"] for wave in expected), "Fixture must exercise a visible LoRA delta"
    speculation = None if method is None else dict(method=method, num_speculative_tokens=count)
    if method == "ngram":
        speculation.update(prompt_lookup_min=1, prompt_lookup_max=3)
    with VllmRunner(
        "Qwen/Qwen3.5-4B",
        **options,
        speculative_config=speculation,
        additional_config={"native_full_graph_backend": backend, "native_full_graph_request_counts": [1, 2, 4]},
        compilation_config={"cudagraph_mode": "FULL", "cudagraph_capture_sizes": [1, 64, 128, 768]},
    ) as runner:
        before = runner.model.collective_rpc(lora_registry_status)[0]
        assert before["weights"] and before["mappings"]
        assert adapter_swap_cycles(runner.model, adapters) == expected
        after = runner.model.collective_rpc(lora_registry_status)[0]
        assert after["weights"] == before["weights"]
        assert after["mappings"] == before["mappings"]
        for name, initial in before["registries"].items():
            final = after["registries"][name]
            assert initial["ready"] and final["ready"]
            assert final["entries"] == initial["entries"]
            assert final["compatibility_steps"] == 0
            assert final["fresh_replays"] > initial["fresh_replays"]
            assert final["verify_replays"] > initial["verify_replays"]


@pytest.mark.parametrize("rank", [8, 16, 32, 64])
@pytest.mark.parametrize("output_size", [16, 128])
def test_native_lora_graph_replay_reads_changed_slots_weights_and_padding(rank, output_size):
    if not torch.npu.is_available() or "910B" not in torch.npu.get_device_name():
        pytest.skip("Requires 910B native BGMV kernels")
    torch.manual_seed(35)
    count, width, hidden = 4, 4, 128
    tokens = count * width
    lora_config = LoRAConfig(max_loras=2, max_lora_rank=rank, lora_dtype=torch.bfloat16)
    punica = PunicaWrapperNPU(tokens, count, "npu", lora_config=lora_config)
    punica.enable_native_full_graph()
    mapping = NativeGraphLoRA(punica, NativeGraphShape(count, width, True))
    a = torch.randn(2, 1, rank, hidden, device="npu", dtype=torch.bfloat16) * 0.05
    b = torch.randn(2, 1, output_size, rank, device="npu", dtype=torch.bfloat16) * 0.05
    x = torch.randn(tokens, hidden, device="npu", dtype=torch.bfloat16)
    base = torch.randn(tokens, output_size, device="npu", dtype=torch.bfloat16)
    y = torch.empty_like(base)
    pointers = (a.data_ptr(), b.data_ptr(), mapping.tokens.data_ptr())

    def run():
        y.copy_(base)
        punica.add_lora_linear(y, x, (a,), (b,), 1.0, (output_size,))

    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream), mapping.bind():
        run()
        stream.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, stream=stream):
            run()
    torch.npu.current_stream().wait_stream(stream)
    xc, bc = x.float().cpu(), base.float().cpu()
    source_tokens = torch.arange(tokens, device="npu")
    valid = torch.tensor([True] * 11 + [False] * 5, device="npu")
    for index, row_slots in enumerate(([0, 1, -1, 0], [1, -1, 0, 1], [-1] * 4, [0, 0, 1, 1])):
        if index == 1:
            # A smaller-rank replacement must erase the previous rank's tail.
            a[0].zero_()
            b[0].zero_()
            a[0, 0, :8].fill_(0.01)
            b[0, 0, :, :8].fill_(-0.02)
        indices = torch.tensor(row_slots).repeat_interleave(width)
        punica._token_lora_indices.copy_(indices)
        punica._embeddings_indices.zero_()
        mapping.update(source_tokens, valid)
        graph.replay()
        ac, bweights = a.float().cpu(), b.float().cpu()
        expected = bc.clone()
        shrink = torch.zeros(tokens, rank)
        for token, slot in enumerate(indices.tolist()):
            if slot >= 0 and token < 11:
                shrink[token] = xc[token] @ ac[slot, 0].T
                expected[token] += shrink[token] @ bweights[slot, 0].T
        torch.testing.assert_close(y.float().cpu(), expected, rtol=1e-2, atol=1e-2)
        # Startup profiling can still use SGMV before native graph warmup.
        # Its narrow-output rank check must match the BGMV capture path.
        sgmv_output = base.clone()
        torch.ops._C_ascend.sgmv_expand(
            shrink.npu(), b, mapping.tokens, torch.ones_like(mapping.tokens), sgmv_output, 0, output_size
        )
        torch.testing.assert_close(sgmv_output.float().cpu(), expected, rtol=1e-2, atol=1e-2)
        assert pointers == (a.data_ptr(), b.data_ptr(), mapping.tokens.data_ptr())
