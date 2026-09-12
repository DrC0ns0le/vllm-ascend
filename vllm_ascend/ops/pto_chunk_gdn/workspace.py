# SPDX-License-Identifier: Apache-2.0
"""Scratch storage for serial MegaGDN launches in the native FULL registry.

The registry owns this storage outside the graph allocator pool. All layers and
rectangles with the same kernel geometry reuse it on the execution stream. Each
buffer remains distinct within a launch; no intra-kernel lifetime aliasing or
changes to PTO arithmetic are required. Returned outputs/states are not scratch.
"""

from dataclasses import dataclass
from math import prod

import torch

from vllm_ascend.ops.pto_chunk_gdn.eligibility import CHUNK_SIZE


@dataclass(frozen=True)
class WorkspaceSpec:
    name: str
    shape: tuple[int, ...]
    dtype: torch.dtype = torch.float16
    zero: bool = True

    @property
    def numel(self):
        return prod(self.shape)


def workspace_specs(tokens, heads, hidden_size, chunks, block_dim):
    c = CHUNK_SIZE
    return (
        WorkspaceSpec("g_sum", (1, tokens, heads), torch.float32, False),
        WorkspaceSpec("g_t", (heads, tokens), torch.float32, False),
        WorkspaceSpec("beta_t", (heads, tokens), zero=False),
        WorkspaceSpec("A", (1, tokens, heads, c)),
        WorkspaceSpec("A_inv", (1, tokens, heads, c)),
        WorkspaceSpec("w", (1, tokens, heads, hidden_size), zero=False),
        WorkspaceSpec("u", (1, tokens, heads, hidden_size), zero=False),
        WorkspaceSpec("s", (chunks * heads, hidden_size, hidden_size)),
        WorkspaceSpec("v_new", (1, tokens, heads, hidden_size), zero=False),
        WorkspaceSpec("kkt_ws", (block_dim * 2, c, c)),
        WorkspaceSpec("wy_ws_a1", (block_dim, c, c)),
        WorkspaceSpec("wy_ws_a2", (block_dim, c, c)),
        WorkspaceSpec("h_ws", (block_dim * 4, hidden_size, hidden_size)),
        WorkspaceSpec("o_ws_qk", (block_dim, c, c)),
        WorkspaceSpec("o_ws_qs", (block_dim, c, hidden_size)),
        WorkspaceSpec("o_ws_gated", (block_dim, c, c)),
    )


def allocate_workspace(specs, device):
    return {
        spec.name: (torch.zeros if spec.zero else torch.empty)(spec.shape, dtype=spec.dtype, device=device)
        for spec in specs
    }


class MegaGDNGraphWorkspace:
    """One bounded workspace per registry, never shared with eager fallbacks.

    Allocate all capacity buffers on the first uncaptured warmup for a geometry.
    Never grow them: previously captured pointers must remain valid. Warmup and
    serving already have a stream handoff; serving serializes graph replays and
    consumes each output before the next replay. Parallel streams must use
    separate registries/workspaces, just as they must use separate graph inputs.
    """

    def __init__(self, max_tokens, max_chunks):
        if max_tokens <= 0 or max_chunks <= 0:
            raise ValueError("MegaGDN workspace capacities must be positive")
        self.max_tokens = max_tokens
        self.max_chunks = max_chunks
        self.buffers = {}
        self.sealed = False

    @property
    def reserved_bytes(self):
        return sum(t.numel() * t.element_size() for buffers in self.buffers.values() for t in buffers.values())

    def views(self, *, device, tokens, heads, hidden_size, chunks, block_dim):
        if not 0 < tokens <= self.max_tokens or not 0 < chunks <= self.max_chunks:
            raise ValueError("MegaGDN launch exceeds its graph workspace capacity")
        device = torch.device(device)
        key = (device, heads, hidden_size, block_dim)
        if key not in self.buffers:
            if self.sealed or (device.type == "npu" and torch.npu.is_current_stream_capturing()):
                raise RuntimeError("MegaGDN workspace must be prepared before graph capture")
            capacity = workspace_specs(self.max_tokens, heads, hidden_size, self.max_chunks, block_dim)
            self.buffers[key] = {
                spec.name: torch.empty(spec.numel, dtype=spec.dtype, device=device) for spec in capacity
            }
        buffers = self.buffers[key]
        views = {}
        for spec in workspace_specs(tokens, heads, hidden_size, chunks, block_dim):
            tensor = buffers[spec.name][: spec.numel].view(spec.shape)
            # Keep the validated launcher's initialization, including partial
            # chunk/state padding. These fills are captured on the same stream.
            if spec.zero:
                tensor.zero_()
            views[spec.name] = tensor
        return views
