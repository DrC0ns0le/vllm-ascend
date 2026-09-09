# SPDX-License-Identifier: Apache-2.0
"""Device-selected recurrent GDN for single-token rows in a mixed graph."""

from vllm.triton_utils import tl, triton

VALUE_TILE = 32
SINGLE_TOKEN_PROGRAMS = 32


@triton.jit
def _single_token_gdn_kernel(
    q,
    k,
    v,
    g,
    beta,
    state,
    cu,
    read_indices,
    write_indices,
    initial_flags,
    output,
    H: tl.constexpr,
    HG: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    STATE_N: tl.constexpr,
    STATE_H: tl.constexpr,
    STATE_V: tl.constexpr,
    STATE_K: tl.constexpr,
    G_T: tl.constexpr,
    G_H: tl.constexpr,
    BETA_T: tl.constexpr,
    BETA_H: tl.constexpr,
    BV: tl.constexpr,
    work=None,
    PROGRAMS: tl.constexpr = 0,
):
    if PROGRAMS:
        limit = tl.load(work) * H * tl.cdiv(V, BV)
        step = PROGRAMS
    else:
        limit = tl.program_id(0) + 1
        step = 1
    for task in range(tl.program_id(0), limit, step):
        if PROGRAMS:
            row = tl.load(work + 1 + task // (H * tl.cdiv(V, BV)))
            head_tile = task % (H * tl.cdiv(V, BV))
        else:
            row = task
            head_tile = tl.program_id(1)
        start = tl.load(cu + row)
        end = tl.load(cu + row + 1)
        if end - start == 1:
            read = tl.load(read_indices + row)
            write = tl.load(write_indices + row)
            if (read >= 0) & (write >= 0):
                head = head_tile // tl.cdiv(V, BV)
                tile = head_tile % tl.cdiv(V, BV)
                key = tl.arange(0, K)
                value = tile * BV + tl.arange(0, BV)
                initial = tl.load(initial_flags + row) != 0
                state_offset = head * STATE_H + key[:, None] * STATE_K + value[None, :] * STATE_V
                h = tl.load(state + read * STATE_N + state_offset, (value[None, :] < V) & initial, other=0).to(
                    tl.float32
                )
                gate = tl.load(g + start * G_T + head * G_H).to(tl.float32)
                strength = tl.load(beta + start * BETA_T + head * BETA_H).to(tl.float32)
                qk_offset = (start * HG + head // (H // HG)) * K + key
                kt = tl.load(k + qk_offset).to(tl.float32)
                qt = tl.load(q + qk_offset).to(tl.float32)
                vt = tl.load(v + (start * H + head) * V + value, value < V, other=0).to(tl.float32)
                h = h * tl.exp(gate)
                delta = (vt - tl.sum(kt[:, None] * h, 0)) * strength
                h = h + kt[:, None] * delta[None, :]
                result = tl.sum(qt[:, None] * h, 0) * (K**-0.5)
                # Disjoint value tiles own both their output and their state
                # columns. No atomics, scratch states, or cross-program barrier.
                tl.store(output + (start * H + head) * V + value, result, value < V)
                tl.store(state + write * STATE_N + state_offset, h, value[None, :] < V)


def single_token_gdn(q, k, v, g, beta, state, metadata, output):
    heads, value_dim = v.shape[-2:]
    work = getattr(metadata, "single_token_work", None)
    grid = (
        (SINGLE_TOKEN_PROGRAMS,)
        if work is not None
        else (metadata.state_read_indices.shape[0], heads * triton.cdiv(value_dim, VALUE_TILE))
    )
    # The persistent loop must not duplicate live FP32 state tiles into
    # Ascend ping-pong buffers. Keep the legacy launch configuration intact.
    compile_options = {"num_stages": 1, "multibuffer": False} if work is not None else {}
    _single_token_gdn_kernel[grid](
        q,
        k,
        v,
        g,
        beta,
        state,
        metadata.query_start_loc,
        metadata.state_read_indices,
        metadata.state_write_indices,
        metadata.has_initial_state,
        output,
        H=heads,
        HG=k.shape[-2],
        K=k.shape[-1],
        V=value_dim,
        STATE_N=state.stride(0),
        STATE_H=state.stride(1),
        STATE_V=state.stride(2),
        STATE_K=state.stride(3),
        G_T=g.stride(1),
        G_H=g.stride(2),
        BETA_T=beta.stride(1),
        BETA_H=beta.stride(2),
        BV=VALUE_TILE,
        work=work,
        PROGRAMS=SINGLE_TOKEN_PROGRAMS if work is not None else 0,
        num_warps=4,
        **compile_options,
    )
