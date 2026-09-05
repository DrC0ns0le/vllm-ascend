# SPDX-License-Identifier: Apache-2.0
# Launcher layouts and workspace sizes adapted from vllm-ascend PR #8872.
import ctypes

import torch

from vllm_ascend.ops.pto_chunk_gdn.compile import compile_mega_kernel, load_queue_bridge
from vllm_ascend.ops.pto_chunk_gdn.eligibility import CHUNK_SIZE, total_chunks


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
    ):
        dev = q.device
        H, D = v.shape[2], q.shape[3]
        C = chunk_size
        T = q.shape[1]
        N_seq = int(cu_seqlens.numel()) - 1
        bd = self.block_dim

        if cu_seqlens.dtype != torch.int32:
            cu_seqlens = cu_seqlens.to(torch.int32)

        msk_lower, msk_full, minus_identity = self.masks

        tc = total_chunks(cu_seqlens_host)
        num_matrices = tc * H

        g_sum = torch.empty(1, T, H, device=dev, dtype=torch.float32)
        g_t = torch.empty(H, T, device=dev, dtype=torch.float32)
        beta_t = torch.empty(H, T, device=dev, dtype=torch.float16)
        A = torch.zeros(1, T, H, C, device=dev, dtype=torch.float16)
        A_inv_f32 = torch.zeros(1, T, H, C, device=dev, dtype=torch.float32)
        A_inv = torch.zeros(1, T, H, C, device=dev, dtype=torch.float16)
        w = torch.empty_like(v)
        u = torch.empty_like(v)
        s = torch.zeros(tc * H, D, D, device=dev, dtype=torch.float16)
        v_new = torch.empty_like(v)
        fs = torch.zeros(N_seq * H, D, D, device=dev, dtype=torch.float16)
        kkt_ws = torch.zeros(bd * 2, C, C, device=dev, dtype=torch.float16)
        wy_ws_a1 = torch.zeros(bd, C, C, device=dev, dtype=torch.float16)
        wy_ws_a2 = torch.zeros(bd, C, C, device=dev, dtype=torch.float16)
        h_ws = torch.zeros(bd * 4, D, D, device=dev, dtype=torch.float16)
        o_ws_qk = torch.zeros(bd, C, C, device=dev, dtype=torch.float16)
        o_ws_qs = torch.zeros(bd, C, D, device=dev, dtype=torch.float16)
        o_ws_gated = torch.zeros(bd, C, C, device=dev, dtype=torch.float16)
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
            g_sum,
            g_t,
            beta_t,
            A,
            A_inv_f32,
            A_inv,
            w,
            u,
            s,
            v_new,
            fs,
            kkt_ws,
            wy_ws_a1,
            wy_ws_a2,
            h_ws,
            o_ws_qk,
            o_ws_qs,
            o_ws_gated,
        ]
        self.bridge.enqueue(self.address, bd, buffers, N_seq, T, num_matrices)

        o_scaled = o_out * scale
        if return_final_state:
            return o_scaled, fs.view(N_seq, H, D, D)
        return o_scaled
