# SPDX-License-Identifier: Apache-2.0
"""Run isolated contracts without importing the unavailable vLLM/NPU runtime."""

import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[2]

if __name__ == "__main__":
    # Bypass only the package initializer (it configures vLLM logging). Tested
    # backend modules are the real files; graph dependencies are explicit fakes
    # in test_wrapper.py. No CPU test is reported as NPU correctness evidence.
    for name in ("vllm_ascend", "vllm_ascend.ops"):
        package = ModuleType(name)
        package.__path__ = [str(ROOT / name.replace(".", "/"))]
        sys.modules[name] = package
    raise SystemExit(
        pytest.main(
            [
                "--noconftest",
                "-q",
                str(ROOT / "tests/ut/ops/pto_chunk_gdn"),
                str(ROOT / "tests/ut/compilation/breakable"),
                str(ROOT / "benchmarks/qwen35_low_latency/tests"),
            ]
        )
    )
