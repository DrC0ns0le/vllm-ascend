# SPDX-License-Identifier: Apache-2.0
import importlib.util
from pathlib import Path

import pytest


def load():
    path = Path(__file__).resolve().parents[1] / "serve.py"
    spec = importlib.util.spec_from_file_location("serve", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_c64_default_ladder_and_explicit_capacity_budget_overrides():
    serve = load()
    expected = [1, 2, 3, 4, 8, 16, 32, 64, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048]
    assert serve.configuration("D")[1]["cudagraph_capture_sizes"] == expected
    assert serve.configuration("A")[1]["cudagraph_capture_sizes"] == list(serve.DECODE_CAPTURE_SIZES)
    _, command = serve.launch_command(serve.arguments(["D"]))
    assert command[command.index("--max-num-seqs") + 1] == "64"
    assert command[command.index("--max-num-batched-tokens") + 1] == "2048"
    assert command[command.index("--kv-cache-memory-bytes") + 1] == str(15 * 1024**3)
    args = serve.arguments(
        ["B", "--max-num-seqs", "5", "--max-num-batched-tokens", "3000", "--gpu-memory-utilization", "0.7"]
    )
    _, command = serve.launch_command(args)
    assert "--kv-cache-memory-bytes" not in command
    assert command[command.index("--gpu-memory-utilization") + 1] == "0.7"
    assert serve.configuration("A", max_num_seqs=5)[1]["cudagraph_capture_sizes"] == [1, 2, 3, 4, 5]
    assert serve.configuration("B", max_num_batched_tokens=3000)[1]["cudagraph_capture_sizes"][-1] == 3000
    with pytest.raises(ValueError, match=">=1536"):
        serve.configuration("D", max_num_batched_tokens=1024)
    with pytest.raises(SystemExit):
        serve.arguments(["D", "--kv-cache-memory-bytes", "0"])
