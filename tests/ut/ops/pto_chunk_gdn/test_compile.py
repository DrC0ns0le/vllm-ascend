# SPDX-License-Identifier: Apache-2.0
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from vllm_ascend.ops.pto_chunk_gdn import compile as compiler


@pytest.mark.parametrize("locked", [False, True])
def test_queue_bridge_uses_isolated_build_directory_without_removing_lock(tmp_path, monkeypatch, locked):
    import torch.utils.cpp_extension as extension

    root = tmp_path / "extensions"
    root.mkdir()
    lock = root / "lock"
    if locked:
        lock.write_text("another worker's lock")
    calls = []

    def load(**kwargs):
        destination = Path(kwargs["build_directory"])
        assert destination.is_dir() and not (destination / "lock").exists()
        calls.append(destination)
        return "loaded bridge"

    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace(__file__=str(tmp_path / "torch_npu/__init__.py")))
    monkeypatch.setattr(extension, "_get_build_directory", lambda *args, **kwargs: str(root))
    monkeypatch.setattr(extension, "load", load)
    for _ in range(3):
        assert compiler.load_queue_bridge() == "loaded bridge"
    assert len(set(calls)) == 1  # Reuse the same binary across layers.
    if locked:
        assert calls[0].parent == root and calls[0] != root
        assert lock.read_text() == "another worker's lock"
    else:
        assert calls[0] == root and not lock.exists()


def test_recovery_isolates_restarted_and_forked_workers(tmp_path, monkeypatch):
    (tmp_path / "lock").touch()
    first = compiler.queue_bridge_build_directory(tmp_path)
    # Even a stale recovery lock from a previous worker cannot block a restart.
    (first / "lock").touch()
    monkeypatch.setattr(compiler, "BRIDGE_RECOVERY_ID", "next-import")
    restarted = compiler.queue_bridge_build_directory(tmp_path)
    monkeypatch.setattr(compiler.os, "getpid", lambda: 123456789)
    forked = compiler.queue_bridge_build_directory(tmp_path)
    assert len({first, restarted, forked}) == 3
    assert not (restarted / "lock").exists() and not (forked / "lock").exists()
    assert (first / "lock").exists() and (tmp_path / "lock").exists()


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


def test_transpose_source_change_selects_new_binary_without_deleting_old_cache(build_tree):
    source, _, calls = build_tree
    real_source = Path(__file__).resolve().parents[4] / "csrc/pto_chunk_gdn/mega_kernel.cpp"
    fixed = real_source.read_text()
    dependency = "        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);\n        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);\n"
    assert dependency in fixed
    (source / "mega_kernel.cpp").write_text(fixed.replace(dependency, "", 1))
    old = compiler.compile_mega_kernel(num_heads=16, key_heads=8)
    (source / "mega_kernel.cpp").write_text(fixed)
    new = compiler.compile_mega_kernel(num_heads=16, key_heads=8)
    assert new != old
    assert old.is_file() and new.is_file()
    assert compiler.compile_mega_kernel(num_heads=16, key_heads=8) == new
    assert len(calls) == 2
