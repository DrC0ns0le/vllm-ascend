# SPDX-License-Identifier: Apache-2.0
"""Workspace ownership/ABI tests. Device arithmetic is tested on Ascend."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.ops.pto_chunk_gdn.mega_kernel import MegaGDNKernel
from vllm_ascend.ops.pto_chunk_gdn.workspace import MegaGDNGraphWorkspace, workspace_specs


def views(workspace, tokens=64, chunks=1, heads=2):
    return workspace.views(device="cpu", tokens=tokens, heads=heads, hidden_size=128, chunks=chunks, block_dim=2)


def test_all_layers_and_rectangles_reuse_bounded_storage_without_buffer_aliasing():
    workspace = MegaGDNGraphWorkspace(512, 64)
    first = views(workspace)
    addresses = {name: tensor.data_ptr() for name, tensor in first.items()}
    assert len(set(addresses.values())) == len(addresses)
    size = workspace.reserved_bytes
    workspace.sealed = True
    for tokens, chunks in ((64, 64), (512, 4), (129, 2), (1, 1), (512, 4)):
        for _layer in range(18):
            tensors = views(workspace, tokens=tokens, chunks=chunks)
            assert addresses == {name: tensor.data_ptr() for name, tensor in tensors.items()}
            assert workspace.reserved_bytes == size
            assert all(tensor.is_contiguous() for tensor in tensors.values())
    assert len(workspace.buffers) == 1
    # A separate model/registry never shares mutable scratch with this one.
    other = views(MegaGDNGraphWorkspace(512, 64))
    assert not set(addresses.values()) & {tensor.data_ptr() for tensor in other.values()}


def test_reused_workspace_clears_the_same_regions_as_the_validated_launcher():
    workspace = MegaGDNGraphWorkspace(512, 4)
    tensors = views(workspace, tokens=512, chunks=4)
    for tensor in tensors.values():
        tensor.fill_(float("nan"))
    small = views(workspace, tokens=61, chunks=1)
    for spec in workspace_specs(61, 2, 128, 1, 2):
        tensor = small[spec.name]
        assert tensor.shape == spec.shape and tensor.dtype == spec.dtype
        assert tensor.eq(0).all() if spec.zero else tensor.isnan().all()
    # Growing back must clear the entire active extent, including regions that
    # were untouched by the smaller replay and still contain stale NaNs.
    big = views(workspace, tokens=512, chunks=4)
    for spec in workspace_specs(512, 2, 128, 4, 2):
        if spec.zero:
            assert big[spec.name].eq(0).all()


def test_capacity_and_seal_reject_growth_without_replacing_captured_pointers():
    workspace = MegaGDNGraphWorkspace(512, 4)
    first = views(workspace)
    workspace.sealed = True
    with pytest.raises(ValueError, match="capacity"):
        views(workspace, tokens=513)
    with pytest.raises(ValueError, match="capacity"):
        views(workspace, chunks=5)
    with pytest.raises(RuntimeError, match="before graph capture"):
        views(workspace, heads=4)
    assert first["s"].data_ptr() == views(workspace)["s"].data_ptr()
    assert len(workspace.buffers) == 1


def test_different_kernel_geometries_have_separate_storage():
    workspace = MegaGDNGraphWorkspace(512, 4)
    first, second = views(workspace, heads=2), views(workspace, heads=4)
    assert first["A"].data_ptr() != second["A"].data_ptr()
    assert len(workspace.buffers) == 2


def test_launcher_abi_initialization_and_output_lifetime_for_repeated_layers():
    workspace = MegaGDNGraphWorkspace(256, 4)
    captured = []

    def enqueue(address, block_dim, buffers, requests, tokens, matrices):
        assert address == 123 and block_dim == 2 and len(buffers) == 28
        assert matrices == requests * 2  # Each row is shorter than a chunk.
        assert buffers[14] is buffers[15]  # Unused FP32 ABI slot; no allocation.
        assert buffers[20].shape == (requests * 2, 128, 128)
        assert buffers[9].shape == (1, tokens, 2, 128)
        indices = (10, 11, 12, 13, 15, 16, 17, 18, 19, 21, 22, 23, 24, 25, 26, 27)
        for index, spec in zip(indices, workspace_specs(tokens, 2, 128, requests, 2)):
            tensor = buffers[index]
            assert tensor.shape == spec.shape and tensor.dtype == spec.dtype
            if spec.zero:
                assert tensor.eq(0).all()
        assert buffers[20].eq(0).all()
        assert len({buffers[i].data_ptr() for i in (*indices, 9, 20)}) == len(indices) + 2
        # Substitute only device arithmetic. Retain raw buffers as the real
        # queued handler does, so reuse is checked even with pending references.
        captured.append(buffers)
        buffers[9].fill_(len(captured))
        buffers[20].fill_(len(captured) + 10)
        for index in indices:
            buffers[index].fill_(float("nan"))

    kernel = MegaGDNKernel.__new__(MegaGDNKernel)
    kernel.block_dim = 2
    kernel.address = 123
    kernel.bridge = SimpleNamespace(enqueue=enqueue)
    kernel.masks = (torch.ones(128, 128), torch.ones(128, 128), torch.eye(128).half())
    outputs = []
    for shared, count, width in ((True, 1, 61), (True, 4, 64), (False, 1, 61), (True, 1, 61)):
        q = torch.ones(1, count * width, 1, 128, dtype=torch.float16)
        v = torch.ones(1, count * width, 2, 128, dtype=torch.float16)
        cu = tuple(i * width for i in range(count + 1))
        outputs.append(
            kernel.run(
                q,
                q,
                v,
                torch.zeros(1, count * width, 2),
                torch.ones(1, count * width, 2).half(),
                torch.tensor(cu),
                cu_seqlens_host=cu,
                scale=0.5,
                return_final_state=True,
                workspace=workspace if shared else None,
            )
        )
    for index, (output, state) in enumerate(outputs, 1):
        assert output.eq(index * 0.5).all() and state.eq(index + 10).all()
    assert captured[0][13].data_ptr() == captured[1][13].data_ptr() == captured[3][13].data_ptr()
    assert captured[2][13].data_ptr() != captured[0][13].data_ptr()


def test_removed_fp32_inverse_is_an_unused_binary_abi_slot():
    source = (Path(__file__).resolve().parents[4] / "csrc/pto_chunk_gdn/mega_kernel.cpp").read_text()
    # Guard the compatibility alias if the C++ implementation ever changes to
    # consume the reserved pointer: it currently occurs only in the signature.
    assert source.count("A_inv_f32_ptr") == 1
