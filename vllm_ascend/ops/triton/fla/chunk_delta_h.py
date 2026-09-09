# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# ruff: noqa: E501
# mypy: ignore-errors

import torch
from vllm.triton_utils import tl, triton

from .utils import prepare_chunk_indices, prepare_chunk_offsets, safe_exp

_CONDITIONS = ("seq7168",)


@triton.heuristics(
    {
        "USE_G": lambda args: args["g"] is not None,
        "USE_INITIAL_STATE": lambda args: args["h0"] is not None,
        "STORE_FINAL_STATE": lambda args: args["ht"] is not None,
        "SAVE_NEW_VALUE": lambda args: args["v_new"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.jit(do_not_specialize=["T", "H", "Hg", "K", "V"])
def chunk_gated_delta_rule_fwd_kernel_h_blockdim64(
    k,
    v,
    w,
    v_new,
    g,
    h,
    h0,
    ht,
    cu_seqlens,
    chunk_offsets,
    h_update,
    T,
    H,
    Hg,
    K,
    V,
    BT: tl.constexpr,
    USE_G: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    SAVE_NEW_VALUE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    SKIP_SINGLE_TOKEN: tl.constexpr = False,
    state_read_indices=None,
    state_write_indices=None,
    state_initial_flags=None,
    DIRECT_STATE: tl.constexpr = False,
    STATE_N: tl.constexpr = 0,
    STATE_H: tl.constexpr = 0,
    STATE_V: tl.constexpr = 0,
    STATE_K: tl.constexpr = 0,
    G_TOKEN_STRIDE: tl.constexpr = 1,
    G_HEAD_STRIDE: tl.constexpr = 0,
):
    i_nh = tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H
    T_max = 1 * T
    g_head_stride = G_HEAD_STRIDE if G_HEAD_STRIDE else T_max
    if IS_VARLEN:
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
        if T <= 0 or (SKIP_SINGLE_TOKEN and T == 1):
            return
        NT = tl.cdiv(T, BT)
        boh = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)
        boh = i_n * NT

    stride_v = H * V
    stride_k = Hg * K
    stride_w = H * K

    b_h1_bv1 = tl.zeros([128, 64], dtype=tl.float32)
    b_h1_bv2 = tl.zeros([128, 64], dtype=tl.float32)
    # create b_hupd_bv1 and b_hupd_bv2

    v_start1 = 0
    v_start2 = 64

    offs_k = tl.arange(0, 128)[:, None]
    offs_v1 = v_start1 + tl.arange(0, 64)[None, :]
    offs_v2 = v_start2 + tl.arange(0, 64)[None, :]
    mask_kv1 = (offs_k < K) & (offs_v1 < V)
    mask_kv2 = (offs_k < K) & (offs_v2 < V)

    # load initial state
    if DIRECT_STATE:
        read = tl.load(state_read_indices + i_n)
        write = tl.load(state_write_indices + i_n)
        if (read < 0) | (write < 0):
            return
        initial = tl.load(state_initial_flags + i_n) != 0
        h0_ptr = h0 + read * STATE_N + i_h * STATE_H
        # Cache layout is [slot, head, value, key]. Read the transpose
        # directly into the recurrence's [key, value] accumulators.
        b_h1_bv1 += tl.load(h0_ptr + offs_k * STATE_K + offs_v1 * STATE_V, mask_kv1 & initial, other=0).to(tl.float32)
        b_h1_bv2 += tl.load(h0_ptr + offs_k * STATE_K + offs_v2 * STATE_V, mask_kv2 & initial, other=0).to(tl.float32)
    elif USE_INITIAL_STATE:
        h0_ptr = h0 + i_nh * K * V
        ptr_h0_bv1 = h0_ptr + offs_k * V + offs_v1 * 1
        b_h1_bv1 += tl.load(ptr_h0_bv1, mask=mask_kv1, other=0.0).to(tl.float32)

        ptr_h0_bv2 = h0_ptr + offs_k * V + offs_v2 * 1
        b_h1_bv2 += tl.load(ptr_h0_bv2, mask=mask_kv2, other=0.0).to(tl.float32)

    # main recurrence
    for i_t in range(NT):
        h_base = h + (boh + i_t) * H * K * V + i_h * K * V

        p_h1_bv1 = tl.make_block_ptr(h_base, (K, V), (V, 1), (0, v_start1), (128, 64), (1, 0))
        tl.store(p_h1_bv1, b_h1_bv1.to(p_h1_bv1.dtype.element_ty), boundary_check=(0, 1))

        p_h1_bv2 = tl.make_block_ptr(h_base, (K, V), (V, 1), (0, v_start2), (128, 64), (1, 0))
        tl.store(p_h1_bv2, b_h1_bv2.to(p_h1_bv2.dtype.element_ty), boundary_check=(0, 1))

        offs_t_wv = (i_t * BT + tl.arange(0, BT))[:, None]
        offs_k_wv = tl.arange(0, 128)[None, :]
        mask_w = (offs_t_wv < T) & (offs_k_wv < K)

        w_base = w + bos * H * K + i_h * K
        ptr_w = w_base + offs_t_wv * stride_w + offs_k_wv * 1
        b_w = tl.load(ptr_w, mask=mask_w, other=0.0)

        k_base = k + bos * Hg * K + (i_h // (H // Hg)) * K
        p_k = tl.make_block_ptr(k_base, (K, T), (1, stride_k), (0, i_t * BT), (128, BT), (0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1), padding_option="zero")

        v_new_base = v_new + bos * H * V + i_h * V

        last_idx = min((i_t + 1) * BT, T) - 1
        b_g_last = tl.load(g + (bos + last_idx) * G_TOKEN_STRIDE + i_h * g_head_stride)

        offs_t = i_t * BT + tl.arange(0, BT)
        mask_t = offs_t < T
        g_ptr = g + bos * G_TOKEN_STRIDE + i_h * g_head_stride
        b_g = tl.load(g_ptr + offs_t * G_TOKEN_STRIDE, mask=mask_t, other=0.0)

        b_g = safe_exp(b_g_last - b_g)
        b_g_last = tl.exp(b_g_last)

        offs_t_v = (i_t * BT + tl.arange(0, BT))[:, None]
        mask_v1 = (offs_t_v < T) & (offs_v1 < V)

        v_base = v + bos * H * V + i_h * V
        ptr_v1 = v_base + offs_t_v * stride_v + offs_v1 * 1
        b_v1 = tl.load(ptr_v1, mask=mask_v1, other=0.0)
        b_v_new1 = b_v1.to(tl.float32)
        b_v_new1 -= tl.dot(b_w, b_h1_bv1.to(b_w.dtype))

        if SAVE_NEW_VALUE:
            p_v_new1 = tl.make_block_ptr(v_new_base, (T, V), (stride_v, 1), (i_t * BT, v_start1), (BT, 64), (1, 0))
            tl.store(p_v_new1, b_v_new1.to(p_v_new1.dtype.element_ty), boundary_check=(0, 1))

        if USE_G:
            b_v_new1 = b_v_new1 * b_g[:, None]
            b_h1_bv1 = b_h1_bv1 * b_g_last

        b_v_new1 = b_v_new1.to(k.dtype.element_ty)
        b_h1_bv1 += tl.dot(b_k, b_v_new1)

        mask_v2 = (offs_t_v < T) & (offs_v2 < V)
        ptr_v2 = v_base + offs_t_v * stride_v + offs_v2 * 1
        b_v2 = tl.load(ptr_v2, mask=mask_v2, other=0.0)
        b_v_new2 = b_v2.to(tl.float32)
        b_v_new2 -= tl.dot(b_w, b_h1_bv2.to(b_w.dtype))

        if SAVE_NEW_VALUE:
            p_v_new2 = tl.make_block_ptr(v_new_base, (T, V), (stride_v, 1), (i_t * BT, v_start2), (BT, 64), (1, 0))
            tl.store(p_v_new2, b_v_new2.to(p_v_new2.dtype.element_ty), boundary_check=(0, 1))

        if USE_G:
            b_v_new2 = b_v_new2 * b_g[:, None]
            b_h1_bv2 = b_h1_bv2 * b_g_last

        b_v_new2 = b_v_new2.to(k.dtype.element_ty)
        b_h1_bv2 += tl.dot(b_k, b_v_new2)

    # epilogue
    if DIRECT_STATE:
        ht_ptr = ht + write * STATE_N + i_h * STATE_H
        # Each live request owns disjoint cache rows. Single-token rows
        # have already returned and are updated by single_token_gdn.
        tl.store(ht_ptr + offs_k * STATE_K + offs_v1 * STATE_V, b_h1_bv1, mask_kv1)
        tl.store(ht_ptr + offs_k * STATE_K + offs_v2 * STATE_V, b_h1_bv2, mask_kv2)
    elif STORE_FINAL_STATE:
        ht_ptr = ht + i_nh * K * V

        p_ht1_bv1 = tl.make_block_ptr(ht_ptr, (K, V), (V, 1), (0, v_start1), (128, 64), (1, 0))
        tl.store(p_ht1_bv1, b_h1_bv1.to(p_ht1_bv1.dtype.element_ty), boundary_check=(0, 1))

        p_ht1_bv2 = tl.make_block_ptr(ht_ptr, (K, V), (V, 1), (0, v_start2), (128, 64), (1, 0))
        tl.store(p_ht1_bv2, b_h1_bv2.to(p_ht1_bv2.dtype.element_ty), boundary_check=(0, 1))


def chunk_gated_delta_rule_fwd_h(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    chunk_size: int = 64,  # SY: remove this argument and force chunk size 64?
    save_new_value: bool = True,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.Tensor | None = None,
    chunk_offsets: torch.Tensor | None = None,
    skip_single_token: bool = False,
    state_cache: torch.Tensor | None = None,
    state_metadata=None,
    token_major_g: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    # This kernel is slightly different from fla to support Q/K with different head numbers.
    # In fla, Q/K always have the same head number, so Hg is always equal to H.
    B, T, Hg, K, V = *k.shape, u.shape[-1]
    H = u.shape[-2]
    BT = chunk_size

    if cu_seqlens is not None and chunk_indices is None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    # N: the actual number of sequences in the batch with either equal or variable lengths
    if cu_seqlens is None:
        N, NT, chunk_offsets = B, triton.cdiv(T, BT), None
    else:
        if chunk_offsets is None:
            chunk_offsets = prepare_chunk_offsets(cu_seqlens, BT)
        N, NT, chunk_offsets = (
            len(cu_seqlens) - 1,
            len(chunk_indices),
            chunk_offsets,
        )
    assert K <= 256, "current kernel does not support head dimension larger than 256."

    h = k.new_empty(B, NT, H, K, V)
    direct_state = state_cache is not None
    if direct_state:
        if state_metadata is None or initial_state is not None or not output_final_state or cu_seqlens is None:
            raise ValueError("Direct GDN state requires device metadata, variable lengths, and final-state writeback")
        if state_cache.ndim != 4 or state_cache.shape[1:] != (H, V, K):
            raise ValueError("Direct GDN state requires [slot, head, value, key] cache layout")
        final_state = state_cache
    else:
        final_state = k.new_empty(N, H, K, V, dtype=torch.float32) if output_final_state else None

    v_new = torch.empty_like(u) if save_new_value else None
    if not token_major_g:
        g = g.transpose(1, 2).contiguous()

    def grid(meta):
        return (1, N * H)

    chunk_gated_delta_rule_fwd_kernel_h_blockdim64[grid](
        k=k,
        v=u,
        w=w,
        v_new=v_new,
        g=g,
        h=h,
        h0=state_cache if direct_state else initial_state,
        ht=final_state,
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        h_update=None,
        T=T,
        H=H,
        Hg=Hg,
        K=K,
        V=V,
        BT=BT,
        SKIP_SINGLE_TOKEN=skip_single_token,
        state_read_indices=state_metadata.state_read_indices if direct_state else None,
        state_write_indices=state_metadata.state_write_indices if direct_state else None,
        state_initial_flags=state_metadata.has_initial_state if direct_state else None,
        DIRECT_STATE=direct_state,
        STATE_N=state_cache.stride(0) if direct_state else 0,
        STATE_H=state_cache.stride(1) if direct_state else 0,
        STATE_V=state_cache.stride(2) if direct_state else 0,
        STATE_K=state_cache.stride(3) if direct_state else 0,
        G_TOKEN_STRIDE=g.stride(1) if token_major_g else 1,
        G_HEAD_STRIDE=g.stride(2) if token_major_g else 0,
        num_warps=4,
        num_stages=2,
    )
    return h, v_new, final_state
