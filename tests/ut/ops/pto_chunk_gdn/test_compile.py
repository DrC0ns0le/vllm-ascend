# SPDX-License-Identifier: Apache-2.0
import subprocess
from pathlib import Path

import pytest

from vllm_ascend.ops.pto_chunk_gdn import compile as compiler


@pytest.fixture
def build_tree(tmp_path, monkeypatch):
    source = tmp_path / "repo/csrc/pto_chunk_gdn"
    source.mkdir(parents=True)
    (source / "mega_kernel.cpp").write_text('#include "chunk_h.cpp"\n')
    included = source / "chunk_h.cpp"
    included.write_text("// original included source\n")
    toolkit = tmp_path / "toolkit"
    headers = toolkit / "include/pto"
    headers.mkdir(parents=True)
    (headers / "pto-inst.hpp").write_text("// installed PTO headers\n")
    executable = toolkit / "compiler/ccec_compiler/bin/bisheng"
    executable.parent.mkdir(parents=True)
    executable.touch()
    monkeypatch.setattr(compiler, "KERNELS_PTO", source)
    monkeypatch.setattr(compiler, "toolkit_path", lambda: toolkit)
    monkeypatch.setattr(compiler.subprocess, "check_output", lambda command: b"bisheng test version")
    calls = []

    def compile_binary(command, **kwargs):
        calls.append(command)
        Path(command[command.index("-o") + 1]).write_bytes(b"test binary")

    monkeypatch.setattr(compiler.subprocess, "run", compile_binary)
    return source, headers, calls


def test_cache_reuses_binary_but_invalidates_included_source_and_headers(build_tree):
    source, headers, calls = build_tree
    shape = dict(num_heads=16, key_heads=8)
    first = compiler.compile_mega_kernel(**shape)
    assert compiler.compile_mega_kernel(**shape) == first
    assert len(calls) == 1
    (source / "chunk_h.cpp").write_text("// changed included source\n")
    second = compiler.compile_mega_kernel(**shape)
    assert second != first
    (headers / "pto-inst.hpp").write_text("// changed dependency header\n")
    assert compiler.compile_mega_kernel(**shape) not in (first, second)
    assert len(calls) == 3


def test_failed_compilation_does_not_publish_cache_entry(build_tree, monkeypatch):
    source, _, _ = build_tree

    def fail(command, **kwargs):
        Path(command[command.index("-o") + 1]).write_bytes(b"partial output")
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(compiler.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        compiler.compile_mega_kernel(num_heads=16, key_heads=8)
    assert not list((source.parent.parent / ".vllm_ascend/pto_chunk_gdn").glob("*.so"))


def test_missing_headers_fail_before_invoking_compiler(build_tree):
    _, headers, calls = build_tree
    (headers / "pto-inst.hpp").unlink()
    with pytest.raises(RuntimeError, match="PTO headers missing"):
        compiler.compile_mega_kernel(num_heads=16, key_heads=8)
    assert calls == []
