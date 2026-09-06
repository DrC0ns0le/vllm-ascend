# SPDX-License-Identifier: Apache-2.0
"""Generate or execute one of four controlled experiment launch commands."""

import argparse
import json
import os
import shlex
import subprocess
import sys

DECODE_CAPTURE_SIZES = (1, 2, 3, 4, 8, 16, 32, 64)
PREFILL_CAPTURE_SIZES = (128, 192, 256, 384, 512, 768, 1024, 1536, 2048)
DEFAULT_KV_CACHE_BYTES = 15 * 1024**3
MAX_MODEL_LEN = 1536


def configuration(variant, profile=None, *, max_num_seqs=64, max_num_batched_tokens=2048):
    # Retain the old profile spelling for existing experiment commands.
    budget = int(profile) if profile is not None else max_num_batched_tokens
    if not 1 <= max_num_seqs <= 64:
        raise ValueError("max-num-seqs must be in [1, 64]")
    if budget < MAX_MODEL_LEN:
        raise ValueError("max-num-batched-tokens must be >=1536 with chunked prefill disabled")
    graph = variant in ("B", "D")
    mega = variant in ("C", "D")
    # An explicit non-ladder capacity needs its own terminal descriptor because
    # the dispatcher filters FULL capture sizes above max_num_seqs.
    decode = sorted({size for size in DECODE_CAPTURE_SIZES if size <= max_num_seqs} | {max_num_seqs})
    prefill = {size for size in PREFILL_CAPTURE_SIZES + (3072, 4096) if size <= budget} | {budget}
    env = dict(
        VLLM_USE_V2_MODEL_RUNNER="0",
        VLLM_USE_BREAKABLE_CUDAGRAPH=str(int(graph)),
        VLLM_ASCEND_PTO_CHUNK_GDN=str(int(mega)),
        TASK_QUEUE_ENABLE="1",
    )
    compilation = dict(
        cudagraph_mode="FULL_AND_PIECEWISE" if graph else "FULL_DECODE_ONLY",
        cudagraph_capture_sizes=sorted(set(decode) | prefill) if graph else decode,
    )
    if graph:
        compilation["mode"] = 0
    return env, compilation


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variant", choices=list("ABCD"))
    budget = parser.add_mutually_exclusive_group()
    budget.add_argument("--profile", choices=["1536", "4096"], help="Legacy token-budget alias")
    budget.add_argument("--max-num-batched-tokens", type=int, default=2048)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    memory = parser.add_mutually_exclusive_group()
    memory.add_argument("--kv-cache-memory-bytes", type=int, help="Pinned cache bytes; default 15 GiB")
    memory.add_argument("--gpu-memory-utilization", type=float, help="Explicitly use automatic cache sizing instead")
    parser.add_argument("--model", default="/dev/shm/Qwen3_5-2B")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if args.profile is not None:
        args.max_num_batched_tokens = int(args.profile)
    if args.kv_cache_memory_bytes is not None and args.kv_cache_memory_bytes <= 0:
        parser.error("kv-cache-memory-bytes must be positive")
    if args.gpu_memory_utilization is not None and not 0 < args.gpu_memory_utilization <= 1:
        parser.error("gpu-memory-utilization must be in (0, 1]")
    try:
        configuration(args.variant, max_num_seqs=args.max_num_seqs, max_num_batched_tokens=args.max_num_batched_tokens)
    except ValueError as error:
        parser.error(str(error))
    return args


def launch_command(args):
    overrides, compilation = configuration(
        args.variant,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
    )
    command = [
        "vllm",
        "serve",
        args.model,
        "--port",
        str(args.port),
        "--served-model-name",
        "qwen",
        "--dtype",
        "bfloat16",
        "--tensor-parallel-size",
        "1",
        "--data-parallel-size",
        "1",
        "--seed",
        "1024",
        "--max-num-seqs",
        str(args.max_num_seqs),
        "--max-model-len",
        str(MAX_MODEL_LEN),
        "--max-num-batched-tokens",
        str(args.max_num_batched_tokens),
        "--trust-remote-code",
        "--no-enable-prefix-caching",
        "--no-enable-chunked-prefill",
        "--mamba-cache-mode",
        "none",
        "--reasoning-parser",
        "qwen3",
        "--tool-call-parser",
        "qwen3_coder",
        "--enable-auto-tool-choice",
        "--compilation-config",
        json.dumps(compilation, separators=(",", ":")),
        "--additional-config",
        '{"enable_cpu_binding":true}',
        "--async-scheduling",
        "--language-model-only",
    ]
    if args.gpu_memory_utilization is not None:
        command.extend(["--gpu-memory-utilization", str(args.gpu_memory_utilization)])
    else:
        command.extend(["--kv-cache-memory-bytes", str(args.kv_cache_memory_bytes or DEFAULT_KV_CACHE_BYTES)])
    return overrides, command


def main():
    args = arguments()
    overrides, command = launch_command(args)
    print(
        "Requested startup configuration: "
        + json.dumps(
            dict(
                variant=args.variant,
                **overrides,
                max_num_seqs=args.max_num_seqs,
                max_num_batched_tokens=args.max_num_batched_tokens,
                mamba_cache_mode="none",
                prefix_caching=False,
                chunked_prefill=False,
            )
        ),
        file=sys.stderr,
    )
    print(
        "cd /workspace\n"
        + " ".join(f"{key}={shlex.quote(value)}" for key, value in overrides.items())
        + " "
        + shlex.join(command),
        flush=True,
    )
    if args.execute:
        environment = os.environ | overrides
        help_text = subprocess.check_output(["vllm", "serve", "--help=all"], env=environment, text=True)
        flags = [
            "--no-enable-prefix-caching",
            "--no-enable-chunked-prefill",
            "--mamba-cache-mode",
            "--language-model-only",
        ]
        if "--kv-cache-memory-bytes" in command:
            flags.append("--kv-cache-memory-bytes")
        for flag in flags:
            if flag not in help_text:
                raise RuntimeError(f"Installed CLI does not advertise {flag}; inspect CLI before launching")
        os.chdir("/workspace")
        os.execvpe(command[0], command, environment)


if __name__ == "__main__":
    main()
