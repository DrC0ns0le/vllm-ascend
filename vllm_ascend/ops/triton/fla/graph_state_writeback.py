# SPDX-License-Identifier: Apache-2.0
"""Copy fresh final states while excluding padded request rows.

This kernel performs only a contiguous row copy. Recurrence math stays in
AscendC or MegaGDN. A negative slot is tested before any physical-cache access.
"""

from vllm.triton_utils import tl, triton


@triton.jit
def _writeback(source, cache, slots, SIZE: tl.constexpr, STRIDE: tl.constexpr, BLOCK: tl.constexpr):
    row, block = tl.program_id(0), tl.program_id(1)
    slot = tl.load(slots + row)
    if slot < 0:
        return
    offset = block * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(source + row * SIZE + offset, mask=offset < SIZE, other=0)
    tl.store(cache + slot * STRIDE + offset, value.to(cache.dtype.element_ty), mask=offset < SIZE)


def write_fresh_states(source, cache, slots):
    size = source.numel() // source.shape[0]
    _writeback[(source.shape[0], triton.cdiv(size, 4096))](source, cache, slots, size, cache.stride(0), 4096)
