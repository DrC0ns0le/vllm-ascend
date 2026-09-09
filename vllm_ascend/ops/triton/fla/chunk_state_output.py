# SPDX-License-Identifier: Apache-2.0
"""Persistent fused GDN recurrence/output for device-metadata FULL graphs.

Like the PTO MegaGDN pipeline, retain stage intermediates within a launch.
Unlike its fresh-only FP16 interface, read/write the recurrent cache directly
and preserve the baseline BF16 products and FP32 recurrence. A compact device
queue schedules live multi-token requests; single-token rows run separately.
"""

from vllm.triton_utils import tl, triton

from .utils import safe_exp

STATE_OUTPUT_PROGRAMS = 32
STATE_OUTPUT_VALUE_TILE = 32


@triton.jit
def _chunk_state_output_kernel(
    q,
    k,
    w,
    u,
    g,
    state,
    cu,
    reads,
    writes,
    flags,
    work,
    output,
    H: tl.constexpr,
    HG: tl.constexpr,
    D: tl.constexpr,
    SN: tl.constexpr,
    SH: tl.constexpr,
    SV: tl.constexpr,
    SK: tl.constexpr,
    GT: tl.constexpr,
    GH: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    PROGRAMS: tl.constexpr,
):
    tiles = tl.cdiv(D, BV)
    tasks = tl.load(work) * H * tiles
    # Fixed worker count; no capacity-sized grid or host live count.
    for task in range(tl.program_id(0), tasks, PROGRAMS):
        row = tl.load(work + 1 + task // (H * tiles))
        head = task // tiles % H
        tile = task % tiles
        begin = tl.load(cu + row)
        end = tl.load(cu + row + 1)
        read, write = tl.load(reads + row), tl.load(writes + row)
        initial = tl.load(flags + row) != 0
        keys = tl.arange(0, D)
        values = tile * BV + tl.arange(0, BV)
        state_offset = head * SH + keys[:, None] * SK + values[None, :] * SV
        h = tl.load(state + read * SN + state_offset, initial & (values[None, :] < D), other=0).to(tl.float32)
        for chunk in range(tl.cdiv(end - begin, BT)):
            tokens = begin + chunk * BT + tl.arange(0, BT)
            valid = tokens < end
            kv_head = head // (H // HG)
            kt = tl.load(k + (tokens[None, :] * HG + kv_head) * D + keys[:, None], valid[None, :], other=0)
            qt = tl.load(q + (tokens[:, None] * HG + kv_head) * D + keys[None, :], valid[:, None], other=0)
            vt = tl.load(
                u + (tokens[:, None] * H + head) * D + values[None, :],
                valid[:, None] & (values[None, :] < D),
                other=0,
            ).to(tl.float32)
            gate = tl.load(g + tokens * GT + head * GH, valid, other=0)
            # The first fresh chunk has no W@H or Q@H contribution. Keep
            # stale state masked at the load, even though these dots are skipped.
            out = tl.zeros((BT, BV), dtype=tl.float32)
            if initial or chunk > 0:
                wt = tl.load(w + (tokens[:, None] * H + head) * D + keys[None, :], valid[:, None], other=0)
                vt -= tl.dot(wt, h.to(wt.dtype))
                out = tl.dot(qt, h.to(qt.dtype)) * tl.exp(gate)[:, None]
            # Preserve the staged route's BF16 v_new rounding for output,
            # while the state update below uses the unrounded FP32 residual.
            scores = tl.dot(qt, kt) * safe_exp(gate[:, None] - gate[None, :])
            causal = (tokens[:, None] >= tokens[None, :]) & valid[None, :]
            scores = tl.where(causal, scores, 0)
            out = (out + tl.dot(scores.to(u.dtype.element_ty), vt.to(u.dtype.element_ty))) * (D**-0.5)
            tl.store(
                output + (tokens[:, None] * H + head) * D + values[None, :],
                out,
                valid[:, None] & (values[None, :] < D),
            )
            last = tl.minimum(begin + (chunk + 1) * BT, end) - 1
            last_gate = tl.load(g + last * GT + head * GH)
            residual = vt * safe_exp(last_gate - gate)[:, None]
            h = h * tl.exp(last_gate) + tl.dot(kt, residual.to(k.dtype.element_ty))
        tl.store(state + write * SN + state_offset, h, values[None, :] < D)


def chunk_state_output(q, k, w, u, g, state, metadata, output):
    heads, dim = u.shape[-2:]
    if dim != 128 or k.shape[-1] != 128 or state.shape[1:] != (heads, dim, dim):
        raise ValueError("Fused FULL GDN requires K=V=128 and [slot, head, value, key] state")
    _chunk_state_output_kernel[(STATE_OUTPUT_PROGRAMS,)](
        q,
        k,
        w,
        u,
        g,
        state,
        metadata.query_start_loc,
        metadata.state_read_indices,
        metadata.state_write_indices,
        metadata.has_initial_state,
        metadata.recurrent_work,
        output,
        H=heads,
        HG=k.shape[-2],
        D=dim,
        SN=state.stride(0),
        SH=state.stride(1),
        SV=state.stride(2),
        SK=state.stride(3),
        GT=g.stride(1),
        GH=g.stride(2),
        BT=64,
        BV=STATE_OUTPUT_VALUE_TILE,
        PROGRAMS=STATE_OUTPUT_PROGRAMS,
        num_warps=4,
        num_stages=1,
        multibuffer=False,
    )
    return output
