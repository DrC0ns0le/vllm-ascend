# SPDX-License-Identifier: Apache-2.0
"""Explicit startup compilation for editable Ascend source installations."""

import hashlib
import logging
import os
import shlex
import subprocess
import tempfile
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from filelock import FileLock

from vllm_ascend import envs

logger = logging.getLogger(__name__)
KERNELS_PTO = Path(__file__).resolve().parents[3] / "csrc" / "pto_chunk_gdn"
PTO_COMMIT = "4e27a104f948e883e0bef44670252381bff794c5"
# Immutable process-import nonce; the PID also separates forked workers.
BRIDGE_RECOVERY_ID = uuid4().hex


def toolkit_path() -> Path:
    return Path(envs.ASCEND_HOME_PATH or "/usr/local/Ascend/ascend-toolkit/latest").resolve()


def queue_bridge_build_directory(root: Path) -> Path:
    """Bypass an existing PyTorch build baton without removing another build's lock."""
    if not (root / "lock").exists():
        return root
    # FileBaton has no owner/dead-process detection. Use one private recovery
    # directory across this worker's layers instead of waiting indefinitely or
    # deleting a lock that might still belong to another active builder.
    recovery = root / f"worker-{os.getpid()}-{BRIDGE_RECOVERY_ID}"
    if not recovery.exists():
        recovery.mkdir(parents=True, exist_ok=True)
        logger.warning("MegaGDN build cache is locked; using isolated build directory %s", recovery)
    return recovery


def compile_mega_kernel(*, num_heads: int, key_heads: int, hidden_size: int = 128, chunk_size: int = 128) -> Path:
    toolkit = toolkit_path()
    pto = KERNELS_PTO.parent / "third_party" / "pto-isa"
    # Prefer the exact source dependency if supplied. Otherwise try the installed
    # CANN headers; compilation is the compatibility check, not a version guess.
    include = pto / "include" if (pto / "include/pto/pto-inst.hpp").is_file() else toolkit / "include"
    if not (include / "pto/pto-inst.hpp").is_file():
        raise RuntimeError(f"PTO headers missing. Install pto-isa {PTO_COMMIT} at {pto}, or use CANN PTO headers.")
    compiler = toolkit / "compiler/ccec_compiler/bin/bisheng"
    if not compiler.is_file():
        raise RuntimeError(f"Bisheng compiler missing: {compiler}")
    flags = [
        "-fPIC",
        "-shared",
        "-xcce",
        "-DMEMORY_BASE",
        "-O2",
        "-std=gnu++17",
        "--cce-aicore-arch=dav-c220",
        "-mllvm",
        "-cce-aicore-stack-size=0x8000",
        "-mllvm",
        "-cce-aicore-function-stack-size=0x8000",
        "-mllvm",
        "-cce-aicore-record-overflow=true",
        "-mllvm",
        "-cce-aicore-dcci-insert-for-scalar=false",
        "-Wno-macro-redefined",
        "-Wno-ignored-attributes",
        f"-I{KERNELS_PTO / 'include'}",
        f"-I{include}",
        f"-I{toolkit / 'include'}",
        f"-I{toolkit / 'pkg_inc'}",
        f"-I{toolkit / 'pkg_inc/runtime'}",
        f"-I{toolkit / 'pkg_inc/profiling'}",
        f"-DGDN_H={num_heads}",
        f"-DGDN_HG={key_heads}",
        f"-DGDN_D={hidden_size}",
        f"-DGDN_C={chunk_size}",
    ]
    driver_include = Path("/usr/local/Ascend/driver/kernel/inc")
    if driver_include.is_dir():
        flags.append(f"-I{driver_include}")
    digest = hashlib.sha256(repr(flags).encode())
    digest.update(subprocess.check_output([str(compiler), "--version"]))
    # Hash every included local translation unit and PTO header, not just the
    # top-level mtime. A stale binary must never survive a source backport.
    sources = sorted(KERNELS_PTO.glob("*.cpp")) + sorted((KERNELS_PTO / "include").rglob("*.h"))
    sources += sorted((include / "pto").rglob("*.hpp")) + sorted((include / "pto").rglob("*.h"))
    for source in sources:
        digest.update(str(source).encode())
        digest.update(source.read_bytes())
    cache = KERNELS_PTO.parent.parent / ".vllm_ascend/pto_chunk_gdn"
    cache.mkdir(parents=True, exist_ok=True)
    output = cache / f"mega_H{num_heads}_Hg{key_heads}_D{hidden_size}_C{chunk_size}_{digest.hexdigest()[:20]}.so"
    with FileLock(str(output) + ".lock"):
        cache_hit = output.exists()
        if not cache_hit:
            started = perf_counter()
            with tempfile.TemporaryDirectory(dir=cache) as temporary:
                built = Path(temporary) / "kernel.so"
                command = [str(compiler), *flags, str(KERNELS_PTO / "mega_kernel.cpp"), "-o", str(built)]
                logger.info("MegaGDN compile command: %s", shlex.join(command))
                subprocess.run(
                    command,
                    check=True,
                    timeout=600,
                )
                built.replace(output)
            logger.info(
                "MegaGDN compiled: H=%d Hg=%d D=%d C=%d duration_s=%.3f",
                num_heads,
                key_heads,
                hidden_size,
                chunk_size,
                perf_counter() - started,
            )
        logger.info(
            "MegaGDN binary: path=%s sha256=%s cache_hit=%s",
            output,
            hashlib.sha256(output.read_bytes()).hexdigest(),
            cache_hit,
        )
    return output


def load_queue_bridge():
    # Lazy imports: CPU/offline and disabled-backend paths need no torch_npu.
    import torch_npu
    from torch.utils.cpp_extension import _get_build_directory, load

    npu_package = Path(torch_npu.__file__).resolve().parent
    build_directory = queue_bridge_build_directory(Path(_get_build_directory("ascend_pto_gdn_queue", verbose=False)))
    return load(
        name="ascend_pto_gdn_queue",
        build_directory=str(build_directory),
        sources=[str(KERNELS_PTO / "queue_bridge.cpp")],
        extra_include_paths=[str(npu_package.parent), str(npu_package / "include"), str(toolkit_path() / "include")],
        extra_ldflags=[f"-L{npu_package / 'lib'}", "-ltorch_npu", f"-Wl,-rpath,{npu_package / 'lib'}"],
        verbose=False,
    )
