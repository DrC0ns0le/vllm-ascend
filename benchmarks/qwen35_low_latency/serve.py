# SPDX-License-Identifier: Apache-2.0
"""Generate or execute one of four controlled experiment launch commands."""

import argparse
import json
import os
import shlex
import subprocess

SIZES = (1, 2, 3, 4, 64, 128, 192, 256, 384, 512, 768, 1024, 1536)


def configuration(variant, profile):
    graph = variant in ("B", "D")
    mega = variant in ("C", "D")
    sizes = list(SIZES) + ([2048, 3072, 4096] if profile == "4096" else [])
    env = dict(
        VLLM_USE_V2_MODEL_RUNNER="0",
        VLLM_USE_BREAKABLE_CUDAGRAPH=str(int(graph)),
        VLLM_ASCEND_PTO_CHUNK_GDN=str(int(mega)),
        TASK_QUEUE_ENABLE="1",
    )
    compilation = dict(
        cudagraph_mode="FULL_AND_PIECEWISE" if graph else "FULL_DECODE_ONLY",
        cudagraph_capture_sizes=sizes if graph else [1, 2, 3, 4],
    )
    # Breakable runs without the legacy torch.compile segmentation.
    if graph:
        compilation["mode"] = 0
    return env, compilation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variant", choices=list("ABCD"))
    parser.add_argument("--profile", choices=["1536", "4096"], default="1536")
    parser.add_argument("--model", default="/dev/shm/Qwen3_5-2B")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    overrides, compilation = configuration(args.variant, args.profile)
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
        "4",
        "--max-model-len",
        "1536",
        "--max-num-batched-tokens",
        args.profile,
        "--trust-remote-code",
        "--gpu-memory-utilization",
        "0.95",
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
        for flag in (
            "--no-enable-prefix-caching",
            "--no-enable-chunked-prefill",
            "--mamba-cache-mode",
            "--language-model-only",
        ):
            if flag not in help_text:
                raise RuntimeError(f"Installed CLI does not advertise {flag}; inspect CLI before launching")
        os.chdir("/workspace")
        os.execvpe(command[0], command, environment)


if __name__ == "__main__":
    main()
