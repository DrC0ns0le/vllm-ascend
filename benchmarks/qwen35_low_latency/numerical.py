# SPDX-License-Identifier: Apache-2.0
"""910B numerical gate: baseline vs PTO outputs AND recurrent state."""

import argparse
import itertools
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch


def errors(reference, actual):
    reference, actual = reference.float().cpu(), actual.float().cpu()
    absolute = (actual - reference).abs().flatten()
    relative = absolute / reference.abs().flatten().clamp_min(1e-3)
    report = {"finite": bool(torch.isfinite(actual).all())}
    for label, values in (("absolute", absolute), ("relative_floor_1e-3", relative)):
        report[label] = {"max": values.max().item()}
        for p in (50, 95, 99):
            report[label][f"p{p}"] = torch.quantile(values, p / 100).item()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("/dev/shm/Qwen3_5-2B"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--atol", type=float, default=0.05, help="Provisional screening tolerance; not an E2E quality gate"
    )
    parser.add_argument("--rtol", type=float, default=0.05)
    args = parser.parse_args()
    # Lazy loading for the standalone NPU validation process.
    import torch_npu  # noqa: F401
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import ensure_model_parallel_initialized, init_distributed_environment
    from vllm.forward_context import set_forward_context

    from vllm_ascend.ops.gdn_attn_builder import _build_non_spec_chunked_prefill_metadata
    from vllm_ascend.ops.pto_chunk_gdn.backend import MegaGDNBackend
    from vllm_ascend.ops.triton.fla.chunk import chunk_gated_delta_rule
    from vllm_ascend.utils import adapt_patch

    adapt_patch()
    torch.npu.set_device(0)
    device = torch.device("npu:0")
    model = json.loads((args.model / "config.json").read_text())
    model = model.get("text_config", model)
    heads, key_heads = model["linear_num_value_heads"], model["linear_num_key_heads"]
    dim, value_dim = model["linear_key_head_dim"], model["linear_value_head_dim"]
    if dim != value_dim or dim != 128:
        raise RuntimeError(f"Unsupported loaded GDN dimensions Dk={dim}, Dv={value_dim}")
    config = VllmConfig()
    builder = SimpleNamespace(
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(hf_text_config=SimpleNamespace(**model)),
            parallel_config=SimpleNamespace(tensor_parallel_size=1),
        )
    )
    backend = MegaGDNBackend(topology_supported=True, prefix="numerical")
    backend.prepare(device, heads, key_heads, dim)
    report = dict(
        H=heads,
        Hg=key_heads,
        D=dim,
        chunk_size=128,
        dtype="bfloat16",
        internal_dtype="float16",
        atol=args.atol,
        rtol=args.rtol,
        cases=[],
    )
    lengths = [
        (1,),
        (7,),
        (31,),
        (32,),
        (33,),
        (63,),
        (64,),
        (65,),
        (95,),
        (96,),
        (97,),
        (127,),
        (128,),
        (129,),
        (159,),
        (160,),
        (191,),
        (192,),
        (193,),
        (223,),
        (224,),
        (255,),
        (256,),
        (257,),
        (383,),
        (384,),
        (385,),
        (511,),
        (512,),
        (513,),
        (767,),
        (768,),
        (769,),
        (1023,),
        (1024,),
        (1025,),
        (1535,),
        (1536,),
        (90, 90),
        (127, 128, 129),
        (50, 75, 125, 180),
        (129, 129, 129, 129),
    ]
    failed = False
    with tempfile.TemporaryDirectory() as temporary:
        init_distributed_environment(
            world_size=1, rank=0, local_rank=0, backend="hccl", distributed_init_method=f"file://{temporary}/store"
        )
        ensure_model_parallel_initialized(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
        with set_current_vllm_config(config), set_forward_context(None, config), torch.inference_mode():
            for seed in (1024, 2048, 4096):
                for sizes in lengths:
                    torch.manual_seed(seed)
                    cu_host = (0, *itertools.accumulate(sizes))
                    cu_cpu = torch.tensor(cu_host, dtype=torch.int32)
                    metadata = _build_non_spec_chunked_prefill_metadata(builder, cu_cpu, device)
                    metadata.fresh_prefill = True
                    q = torch.randn(1, sum(sizes), key_heads, dim, device=device, dtype=torch.bfloat16)
                    inputs = dict(
                        q=q,
                        k=torch.randn_like(q),
                        v=torch.randn(1, sum(sizes), heads, dim, device=device, dtype=torch.bfloat16),
                        g=-torch.rand(1, sum(sizes), heads, device=device),
                        beta=torch.rand(1, sum(sizes), heads, device=device, dtype=torch.bfloat16),
                        initial_state=torch.zeros(len(sizes), heads, dim, dim, device=device, dtype=torch.float32),
                        output_final_state=True,
                        cu_seqlens=cu_cpu.to(device),
                        prebuilt_meta=metadata,
                        head_first=False,
                        use_qk_l2norm_in_kernel=True,
                    )
                    reference = chunk_gated_delta_rule(**inputs)
                    actual = backend(**inputs, fresh_prefill=True, fallback=chunk_gated_delta_rule)
                    torch.npu.synchronize()
                    case = dict(lengths=sizes, seed=seed)
                    for label, ref, got in zip(("output", "state"), reference, actual):
                        case[label] = errors(ref, got)
                        case[label]["within_tolerance"] = bool(
                            torch.allclose(ref.float(), got.float(), atol=args.atol, rtol=args.rtol)
                        )
                        failed |= not case[label]["within_tolerance"]
                    report["cases"].append(case)
                    args.output.write_text(json.dumps(report, indent=2) + "\n")
    report["counts"] = dict(backend.counts)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if failed:
        raise SystemExit("Numerical screening failed; inspect output and state errors before serving acceptance")


if __name__ == "__main__":
    main()
