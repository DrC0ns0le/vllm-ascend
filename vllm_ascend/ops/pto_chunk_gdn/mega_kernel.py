# SPDX-License-Identifier: Apache-2.0
# Launcher layouts and workspace sizes adapted from vllm-ascend PR #8872.
import ctypes

import torch

from vllm_ascend.ops.pto_chunk_gdn.compile import compile_mega_kernel, load_queue_bridge
from vllm_ascend.ops.pto_chunk_gdn.eligibility import CHUNK_SIZE, total_chunks
from vllm_ascend.ops.pto_chunk_gdn.workspace import allocate_workspace, workspace_specs


class MegaGDNKernel:
    def __init__(self, device, num_heads, key_heads, hidden_size):
        name = torch.npu.get_device_name(device)
        if "910B" not in name:
            raise RuntimeError(f"MegaGDN dav-c220 backend requires Ascend 910B, got {name}")
        self.block_dim = int(torch.npu.get_device_properties(device).cube_core_num)
        if self.block_dim <= 0:
            raise RuntimeError("Cannot determine physical AI-Core count for MegaGDN")
        path = compile_mega_kernel(num_heads=num_heads, key_heads=key_heads, hidden_size=hidden_size)
        self.library = ctypes.CDLL(str(path))
        self.address = ctypes.cast(self.library.call_kernel, ctypes.c_void_p).value
        self.bridge = load_queue_bridge()
        lower = torch.tril(torch.ones(CHUNK_SIZE, CHUNK_SIZE, device=device), diagonal=-1).float()
        full = torch.tril(torch.ones(CHUNK_SIZE, CHUNK_SIZE, device=device)).float()
        minus_identity = torch.zeros(CHUNK_SIZE, CHUNK_SIZE, device=device, dtype=torch.float16)
        minus_identity.fill_diagonal_(-1)
        self.masks = lower, full, minus_identity

    def run(
        self,
        q,
        k,
        v,
        g_in,
        beta,
        cu_seqlens,
        *,
        cu_seqlens_host,
        chunk_size=CHUNK_SIZE,
        scale=1.0,
        key_heads=None,
        return_final_state=False,
        workspace=None,
    ):
        dev = q.device
        H, D = v.shape[2], q.shape[3]
        C = chunk_size
        T = q.shape[1]
        N_seq = int(cu_seqlens.numel()) - 1
        bd = self.block_dim
        if C != CHUNK_SIZE:
            raise ValueError(f"MegaGDN requires chunk_size={CHUNK_SIZE}")

        if cu_seqlens.dtype != torch.int32:
            cu_seqlens = cu_seqlens.to(torch.int32)

        msk_lower, msk_full, minus_identity = self.masks

        tc = total_chunks(cu_seqlens_host)
        num_matrices = tc * H

        if workspace is None:
            scratch = allocate_workspace(workspace_specs(T, H, D, tc, bd), dev)
        else:
            scratch = workspace.views(device=dev, tokens=T, heads=H, hidden_size=D, chunks=tc, block_dim=bd)
        # Keep returned tensors private: a later layer/replay may reuse scratch
        # before the caller releases its output or final-state reference.
        fs = torch.zeros(N_seq * H, D, D, device=dev, dtype=torch.float16)
        o_out = torch.empty_like(v)

        buffers = [
            q,
            k,
            v,
            g_in,
            beta,
            msk_lower,
            msk_full,
            minus_identity,
            cu_seqlens,
            o_out,
            scratch["g_sum"],
            scratch["g_t"],
            scratch["beta_t"],
            scratch["A"],
            # Reserved ABI slot: the kernel solves directly into FP16 A_inv
            # and never dereferences A_inv_f32. Avoid its T*H*C*4 allocation
            # and fill while retaining compatibility with compiled binaries.
            scratch["A_inv"],
            scratch["A_inv"],
            scratch["w"],
            scratch["u"],
            scratch["s"],
            scratch["v_new"],
            fs,
            scratch["kkt_ws"],
            scratch["wy_ws_a1"],
            scratch["wy_ws_a2"],
            scratch["h_ws"],
            scratch["o_ws_qk"],
            scratch["o_ws_qs"],
            scratch["o_ws_gated"],
        ]
        self.bridge.enqueue(self.address, bd, buffers, N_seq, T, num_matrices)

        o_scaled = o_out * scale
        if return_final_state:
            return o_scaled, fs.view(N_seq, H, D, D)
        return o_scaled
