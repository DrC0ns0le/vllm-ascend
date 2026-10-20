"""Write real prefill state rows without accessing padded cache slots."""

from vllm.triton_utils import tl, triton

STATE_COPY_BLOCK_SIZE = 4096


@triton.jit
def _write_fresh_states(source, cache, slots, row_size: tl.constexpr, cache_stride: tl.constexpr, block: tl.constexpr):
    row = tl.program_id(0)
    slot = tl.load(slots + row).to(tl.int64)
    if slot < 0:
        return
    offsets = tl.program_id(1) * block + tl.arange(0, block)
    values = tl.load(source + row * row_size + offsets, mask=offsets < row_size, other=0)
    tl.store(cache + slot * cache_stride + offsets, values.to(cache.dtype.element_ty), mask=offsets < row_size)


def write_fresh_states(source, cache, slots):
    row_size = source.numel() // source.shape[0]
    _write_fresh_states[(source.shape[0], triton.cdiv(row_size, STATE_COPY_BLOCK_SIZE))](
        source, cache, slots, row_size, cache.stride(0), STATE_COPY_BLOCK_SIZE
    )
