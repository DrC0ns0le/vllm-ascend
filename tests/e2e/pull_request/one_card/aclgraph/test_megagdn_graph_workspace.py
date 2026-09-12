# SPDX-License-Identifier: Apache-2.0
"""Ascend replay gate for workspace reuse across layers and capture rectangles."""

import pytest
import torch

from vllm_ascend.ops.pto_chunk_gdn.eligibility import total_chunks
from vllm_ascend.ops.pto_chunk_gdn.mega_kernel import MegaGDNKernel
from vllm_ascend.ops.pto_chunk_gdn.workspace import MegaGDNGraphWorkspace


def test_megagdn_shared_scratch_matches_private_launches_across_graphs():
    if not torch.npu.is_available() or "910B" not in torch.npu.get_device_name():
        pytest.skip("Requires Ascend 910B and the PTO toolchain")
    device = torch.device("npu")
    # Irregular/partial chunks, short high-concurrency rows, and alternating
    # sizes exercise overwritten prefixes and previously inactive scratch tails.
    shapes = ((1, 61), (4, 65), (2, 196), (8, 1), (1, 768))
    workspace = MegaGDNGraphWorkspace(768, 8)
    kernels = [MegaGDNKernel(device, 16, 8, 128) for _ in range(2)]
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    entries = []
    pool = None

    def execute(inputs, cu_host, scratch):
        q, k, v, g, beta, cu = inputs
        outputs = []
        for kernel in kernels:
            output, state = kernel.run(
                q,
                k,
                v,
                g,
                beta,
                cu,
                cu_seqlens_host=cu_host,
                scale=128**-0.5,
                return_final_state=True,
                workspace=scratch,
            )
            outputs.append((output, state))
            v = output
        return outputs

    with torch.npu.stream(stream):
        for count, width in shapes:
            tokens = count * width
            q = torch.randn(1, tokens, 8, 128, device=device, dtype=torch.float16) * 0.02
            k = torch.randn_like(q) * 0.02
            v = torch.randn(1, tokens, 16, 128, device=device, dtype=torch.float16) * 0.02
            g = torch.full((1, tokens, 16), -0.1, device=device)
            beta = torch.full((1, tokens, 16), 0.2, device=device, dtype=torch.float16)
            cu_host = tuple(i * width for i in range(count + 1))
            assert total_chunks(cu_host) <= workspace.max_chunks
            cu = torch.tensor(cu_host, dtype=torch.int32, device=device)
            inputs = (q, k, v, g, beta, cu)
            execute(inputs, cu_host, workspace)
            stream.synchronize()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph, pool=pool, stream=stream):
                output = execute(inputs, cu_host, workspace)
            if pool is None:
                pool = graph.pool()
            entries.append((graph, inputs, cu_host, output))
    torch.npu.current_stream().wait_stream(stream)
    workspace.sealed = True
    scratch_bytes = workspace.reserved_bytes
    addresses = tuple(t.data_ptr() for buffers in workspace.buffers.values() for t in buffers.values())
    assert len(workspace.buffers) == 1
    for index in (0, 4, 3, 1, 2, 0, 4, 2, 3):
        graph, inputs, cu_host, outputs = entries[index]
        inputs[0].mul_(0.9)
        inputs[2].add_(0.001)
        expected = execute(inputs, cu_host, None)
        # Queue handler references are released after submission. Churn freed
        # temporaries on the same stream before replay to catch unsafe reuse.
        for _ in range(3):
            torch.empty_like(inputs[2]).fill_(float("nan"))
        # Poison every scratch region. The existing zero-fills and stage writes
        # must make each replay independent of all prior rectangles/layers.
        for buffers in workspace.buffers.values():
            for tensor in buffers.values():
                tensor.fill_(float("nan"))
        graph.replay()
        for actual_layer, expected_layer in zip(outputs, expected):
            for actual, reference in zip(actual_layer, expected_layer):
                assert actual.isfinite().all()
                torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        assert workspace.reserved_bytes == scratch_bytes
        assert tuple(t.data_ptr() for buffers in workspace.buffers.values() for t in buffers.values()) == addresses
