# SPDX-License-Identifier: Apache-2.0
"""910B numerical gate: baseline vs PTO outputs AND recurrent state."""

import argparse
import hashlib
import itertools
import json
import random
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch


def errors(reference, actual):
    reference, actual = reference.float().cpu(), actual.float().cpu()
    absolute = (actual - reference).abs().flatten()
    relative = absolute / reference.abs().flatten().clamp_min(1e-3)
    report = {"finite": bool(torch.isfinite(actual).all() and torch.isfinite(reference).all())}
    for label, values in (("absolute", absolute), ("relative_floor_1e-3", relative)):
        if not report["finite"]:
            report[label] = dict.fromkeys(("max", "mean", "p50", "p95", "p99"))
            continue
        report[label] = {"max": values.max().item(), "mean": values.mean().item()}
        for p in (50, 95, 99):
            report[label][f"p{p}"] = torch.quantile(values, p / 100).item()
    return report


def packed_lengths(seed):
    """Deterministic critical, bounded-count and ragged packed layouts."""
    rng = random.Random(seed)
    return [
        (90, 90),
        (127, 128, 129),
        (50, 75, 125, 180),
        (129,) * 4,
        *((1024 // count,) * count for count in (1, 2, 4, 8, 16, 32, 64)),
        (64, 128),
        (65, 191),
        (64, 127, 192),
        (73, 118, 191, 256),
        tuple(rng.randint(32, 256) for _ in range(8)),
        tuple(rng.randint(16, 128) for _ in range(16)),
    ]


def compare_sequences(reference, actual, lengths, atol, rtol):
    """Report each request independently, including its own final state."""
    rows = []
    start = 0
    for index, length in enumerate(lengths):
        row = dict(sequence=index, tokens=length)
        for label, ref, got in (
            ("output", reference[0][:, start : start + length], actual[0][:, start : start + length]),
            ("state", reference[1][index], actual[1][index]),
        ):
            row[label] = errors(ref, got)
            row[label]["within_tolerance"] = row[label]["finite"] and bool(
                torch.allclose(ref.float(), got.float(), atol=atol, rtol=rtol)
            )
        rows.append(row)
        start += length
    return rows


def continuation(reference_state, actual_state, make_inputs, decode, atol, rtol):
    # Match gdn.py's prefill writeback into the normal [N,H,Dv,Dk] cache.
    states = [state.transpose(-1, -2).contiguous().clone() for state in (reference_state, actual_state)]
    checkpoints = []
    for step in range(1, 9):
        inputs = make_inputs()
        outputs = [decode(**inputs, state=state).unsqueeze(0) for state in states]
        if step in (1, 4, 8):
            rows = compare_sequences(
                (outputs[0], states[0].transpose(-1, -2)),
                (outputs[1], states[1].transpose(-1, -2)),
                (1,) * reference_state.shape[0],
                atol,
                rtol,
            )
            checkpoints.append(dict(decode_steps=step, sequences=rows))
    return checkpoints


def comparisons_pass(rows):
    return bool(rows) and all(row[label]["within_tolerance"] for row in rows for label in ("output", "state"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("/dev/shm/Qwen3_5-2B"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--atol", type=float, default=0.05, help="Provisional screening tolerance; not an E2E quality gate"
    )
    parser.add_argument("--rtol", type=float, default=0.05)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1024, 2048, 4096])
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Lazy loading for the standalone NPU validation process.
    import torch_npu  # noqa: F401
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import ensure_model_parallel_initialized, init_distributed_environment
    from vllm.forward_context import set_forward_context

    from vllm_ascend.ops.gdn_attn_builder import _build_non_spec_chunked_prefill_metadata
    from vllm_ascend.ops.pto_chunk_gdn.backend import MegaGDNBackend
    from vllm_ascend.ops.triton.fla.chunk import chunk_gated_delta_rule
    from vllm_ascend.ops.triton.fla.l2norm import l2norm_fwd
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
    binary = Path(backend.kernel.library._name)
    report = dict(
        binary_path=str(binary),
        binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
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
    failed = False
    with tempfile.TemporaryDirectory() as temporary:
        init_distributed_environment(
            world_size=1, rank=0, local_rank=0, backend="hccl", distributed_init_method=f"file://{temporary}/store"
        )
        ensure_model_parallel_initialized(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
        with set_current_vllm_config(config), set_forward_context(None, config), torch.inference_mode():
            for seed in args.seeds:
                for sizes in packed_lengths(seed) + [
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
                ]:
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
                    before = backend.counts["megagdn"]
                    actual = backend(**inputs, fresh_prefill=True, fallback=chunk_gated_delta_rule)
                    if backend.counts["megagdn"] != before + 1:
                        raise RuntimeError(f"Numerical case silently fell back: {dict(backend.counts)}")
                    torch.npu.synchronize()
                    case = dict(lengths=sizes, seed=seed)
                    case["sequences"] = compare_sequences(reference, actual, sizes, args.atol, args.rtol)
                    count = len(sizes)
                    sequence_lengths = torch.tensor([0] + [1] * count, device=device, dtype=torch.int32)
                    state_indices = torch.arange(count, device=device, dtype=torch.int32)

                    def make_decode_inputs(count=count, sequence_lengths=sequence_lengths, state_indices=state_indices):
                        shape = (1, count, key_heads, dim)
                        return dict(
                            query=l2norm_fwd(torch.randn(shape, device=device, dtype=torch.bfloat16)).squeeze(0),
                            key=l2norm_fwd(torch.randn(shape, device=device, dtype=torch.bfloat16)).squeeze(0),
                            value=torch.randn(count, heads, dim, device=device, dtype=torch.bfloat16),
                            g=-torch.rand(count, heads, device=device),
                            beta=torch.rand(count, heads, device=device, dtype=torch.bfloat16),
                            scale=dim**-0.5,
                            actual_seq_lengths=sequence_lengths,
                            ssm_state_indices=state_indices,
                        )

                    case["continuation"] = continuation(
                        reference[1],
                        actual[1],
                        make_decode_inputs,
                        torch.ops._C_ascend.npu_recurrent_gated_delta_rule,
                        args.atol,
                        args.rtol,
                    )
                    failed |= not comparisons_pass(case["sequences"])
                    failed |= any(not comparisons_pass(check["sequences"]) for check in case["continuation"])
                    report["cases"].append(case)
                    args.output.write_text(json.dumps(report, indent=2) + "\n")
    report["passed"] = not failed
    report["counts"] = dict(backend.counts)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if failed:
        raise SystemExit("Numerical screening failed; inspect output and state errors before serving acceptance")


if __name__ == "__main__":
    main()
