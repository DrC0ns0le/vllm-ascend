# SPDX-License-Identifier: Apache-2.0
"""Host-only eligibility decisions; never inspect recurrent device state."""

CHUNK_SIZE = 128
HEAD_DIM = 128


def fallback_reason(
    *,
    device_type,
    dtype,
    q_shape,
    k_shape,
    v_shape,
    g_shape,
    beta_shape,
    cu_shape,
    cu_host,
    fresh_prefill,
    topology_supported,
    head_first=False,
) -> str | None:
    if not topology_supported:
        return "topology_or_cache_mode"
    if device_type != "npu":
        return "device"
    if dtype not in ("torch.bfloat16", "torch.float16"):
        return "dtype"
    if fresh_prefill is not True:
        return "stateful_or_unknown_prefill"
    if head_first or len(q_shape) != 4 or len(v_shape) != 4:
        return "layout"
    if q_shape != k_shape or q_shape[:2] != v_shape[:2] or q_shape[0] != 1:
        return "shape"
    if q_shape[-1] != HEAD_DIM or v_shape[-1] != HEAD_DIM:
        return "head_dimension"
    if q_shape[2] <= 0 or v_shape[2] % q_shape[2] != 0:
        return "head_count"
    if tuple(g_shape) != tuple(v_shape[:3]) or tuple(beta_shape) != tuple(v_shape[:3]):
        return "gate_shape"
    if cu_host is None or len(cu_host) < 2 or tuple(cu_shape) != (len(cu_host),):
        return "missing_sequence_boundaries"
    if cu_host[0] != 0 or cu_host[-1] != q_shape[1]:
        return "sequence_extent"
    if any(b <= a for a, b in zip(cu_host, cu_host[1:])):
        return "empty_or_invalid_sequence"
    return None


def total_chunks(cu_host: tuple[int, ...]) -> int:
    return sum((b - a + CHUNK_SIZE - 1) // CHUNK_SIZE for a, b in zip(cu_host, cu_host[1:]))
