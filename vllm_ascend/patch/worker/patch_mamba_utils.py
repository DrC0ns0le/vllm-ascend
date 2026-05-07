# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# vllm_ascend/patch/worker/patch_mamba_utils.py
#
# Root cause analysis:
#
#   Under async scheduling + spec decode, _update_states() sets
#   req_state.num_computed_tokens OPTIMISTICALLY (assumes all spec tokens were
#   accepted), then queues a deferred correction closure to fix it later.
#
#   The correction closure calls _get_valid_sampled_token_count(), which reads
#   from num_accepted_tokens_event. However, _update_states_after_model_execute()
#   only records that event in the non-align branch:
#
#     if mamba_cache_mode == "align":
#         mamba_utils.postprocess_mamba(...)
#         # ← event NEVER recorded
#     else:
#         self.num_accepted_tokens_event.record()  # only here
#
#   So _get_valid_sampled_token_count() always returns empty in align mode,
#   the correction returns early without fixing num_computed_tokens, and
#   preprocess_mamba sees a wildly inflated value that compounds across steps
#   (e.g. dest_block_idx=10922 with block_ids length=4).
#
#   TP workers diverge because each worker's GPU tensor is at a slightly
#   different point when _update_states reads it under async scheduling.
#
# Fix (preprocess):
#   Derive curr_state_idx from the actual block allocation (len(block_ids))
#   instead of from num_computed_tokens arithmetic. Block allocation is the
#   ground truth: the running Mamba state always lives in the last
#   non-speculative block, regardless of how many tokens were computed.
#
# Fix (postprocess):
#   Same idea applied to dest_block_idx. The block that just transitioned
#   from partial-to-full is the block immediately before the running-state
#   block, i.e. len(block_ids) - 2 - num_speculative_blocks. Detect the
#   transition via per-request comparison of the saved running-state slot
#   in mamba_state_idx vs the new running-state slot derived from current
#   allocation: if the new slot moved forward by exactly 1, one block became
#   full this step.
#
#   accept_token_bias is sourced from input_batch.num_accepted_tokens_cpu
#   (the sampler's actual count), not from num_computed_tokens arithmetic.
#
# Fix (defensive):
#   Bounds-check in collect_mamba_copy_meta with a warning. With both primary
#   fixes in place the warning should never fire; if it does, treat as a
#   visible tripwire indicating a new bug.

import itertools
import logging
from typing import Any

from vllm.config import CacheConfig
from vllm.model_executor.layers.mamba.mamba_utils import MambaStateCopyFunc
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker import mamba_utils
from vllm.v1.worker.gpu_input_batch import CachedRequestState

from vllm_ascend.ops.triton.batch_memcpy import batch_memcpy_kernel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Existing Ascend patch: faster batch_memcpy (BLOCK_SIZE 1024 → 8192)
# ---------------------------------------------------------------------------

def batch_memcpy(src_ptrs, dst_ptrs, sizes):
    batch = src_ptrs.shape[0]
    assert dst_ptrs.shape[0] == batch
    assert sizes.shape[0] == batch
    grid = (batch,)
    BLOCK_SIZE = 8192
    batch_memcpy_kernel[grid](src_ptrs, dst_ptrs, sizes, BLOCK_SIZE=BLOCK_SIZE)


mamba_utils.batch_memcpy_kernel = batch_memcpy_kernel
mamba_utils.batch_memcpy = batch_memcpy


# ---------------------------------------------------------------------------
# Defensive: patched collect_mamba_copy_meta — bounds-check before indexing
# block_ids. Uses `continue` not `return` so a bad group does not suppress
# remaining groups. Warning acts as a tripwire for future regressions.
# ---------------------------------------------------------------------------

def _patched_collect_mamba_copy_meta(
    copy_bufs,
    kv_cache_config: KVCacheConfig,
    mamba_state_copy_funcs: tuple[MambaStateCopyFunc, ...],
    mamba_group_ids: list[int],
    src_block_idx: int,
    dest_block_idx: int,
    accept_token_bias: int,
    req_state: CachedRequestState,
    forward_context: dict[str, Any],
) -> None:
    if src_block_idx == dest_block_idx and accept_token_bias == 0:
        return

    src_ptrs_np = copy_bufs.src_ptrs.np
    dst_ptrs_np = copy_bufs.dst_ptrs.np
    sizes_np = copy_bufs.sizes.np
    offset = copy_bufs.offset

    for mamba_group_id in mamba_group_ids:
        block_ids = req_state.block_ids[mamba_group_id]

        if dest_block_idx >= len(block_ids) or src_block_idx >= len(block_ids):
            logger.warning(
                "Mamba state copy skipped: dest_block_idx=%d src_block_idx=%d "
                "exceeds block_ids length=%d for req %s (mamba_group_id=%d). "
                "Possible scheduler/preprocess_mamba mismatch.",
                dest_block_idx,
                src_block_idx,
                len(block_ids),
                getattr(req_state, "req_id", "<unknown>"),
                mamba_group_id,
            )
            continue

        dest_block_id = block_ids[dest_block_idx]
        layer_names = kv_cache_config.kv_cache_groups[mamba_group_id].layer_names
        for layer_name in layer_names:
            attention = forward_context[layer_name]
            kv_caches: list = attention.kv_cache
            for state, state_copy_func in zip(kv_caches, mamba_state_copy_funcs):
                copy_spec = state_copy_func(
                    state, block_ids, src_block_idx, accept_token_bias + 1
                )
                src_ptrs_np[offset] = copy_spec.start_addr
                dst_ptrs_np[offset] = state[dest_block_id].data_ptr()
                sizes_np[offset] = copy_spec.num_elements * state.element_size()
                offset += 1

    copy_bufs.offset = offset


mamba_utils.collect_mamba_copy_meta = _patched_collect_mamba_copy_meta


# ---------------------------------------------------------------------------
# Primary: patched preprocess_mamba — derive curr_state_idx from len(block_ids)
# instead of from num_computed_tokens arithmetic.
# ---------------------------------------------------------------------------

def _patched_preprocess_mamba(
    scheduler_output: SchedulerOutput,
    kv_cache_config: KVCacheConfig,
    cache_config: CacheConfig,
    mamba_state_idx: dict[str, int],
    input_batch,
    requests: dict[str, CachedRequestState],
    forward_context: dict[str, Any],
    mamba_state_copy_funcs: tuple[MambaStateCopyFunc, ...],
    copy_bufs,
):
    """
    Copy the mamba state of previous step to the last
    (1 + num_speculative_blocks) block.

    Ascend patch: derives curr_state_idx from the actual block allocation
    (len(block_ids)) instead of from num_computed_tokens arithmetic, which is
    unreliable under async scheduling + spec decode in mamba_cache_mode=align.
    """
    mamba_group_ids = copy_bufs.mamba_group_ids
    mamba_spec = copy_bufs.mamba_spec
    num_speculative_blocks = mamba_spec.num_speculative_blocks
    assert cache_config.enable_prefix_caching
    block_size = mamba_spec.block_size
    finished_req_ids = scheduler_output.finished_req_ids
    preempted_req_ids = scheduler_output.preempted_req_ids or set()
    resumed_req_ids = scheduler_output.scheduled_cached_reqs.resumed_req_ids
    for req_id in itertools.chain(
        finished_req_ids, preempted_req_ids, resumed_req_ids
    ):
        mamba_state_idx.pop(req_id, None)

    copy_bufs.offset = 0
    primary_group = mamba_group_ids[0]
    for i, req_id in enumerate(input_batch.req_ids):
        if req_id is None:
            continue
        req_state = requests[req_id]
        prev_state_idx = mamba_state_idx.get(req_id)

        if prev_state_idx is None:
            # New or resumed request — no previous state. Match upstream:
            # when num_computed_tokens == 0 this evaluates to -1, the
            # "no previous state" sentinel checked at line 211.
            # Do NOT subtract num_scheduled_tokens here: on a chunked
            # prefill (num_computed_tokens=0, num_scheduled large), that
            # drives prev_state_idx far negative, defeats the `== -1`
            # gate, and produces an out-of-range negative index into
            # block_ids inside collect_mamba_copy_meta.
            prev_state_idx = (req_state.num_computed_tokens - 1) // block_size

        num_allocated_blocks = len(req_state.block_ids[primary_group])
        curr_state_idx = num_allocated_blocks - 1 - num_speculative_blocks
        if curr_state_idx < 0:
            # Insufficient allocation; scheduler will allocate more next step.
            continue

        mamba_state_idx[req_id] = curr_state_idx

        if prev_state_idx == -1 or prev_state_idx == curr_state_idx:
            continue
        # If the loaded prev_state_idx is itself out-of-range
        # (defense-in-depth — should not happen with allocation-derived
        # writes), skip rather than crash.
        if prev_state_idx < 0 or prev_state_idx >= num_allocated_blocks:
            continue

        mamba_utils.collect_mamba_copy_meta(
            copy_bufs,
            kv_cache_config,
            mamba_state_copy_funcs,
            mamba_group_ids,
            prev_state_idx,
            curr_state_idx,
            input_batch.num_accepted_tokens_cpu[i] - 1,
            req_state,
            forward_context,
        )
        input_batch.num_accepted_tokens_cpu[i] = 1

    mamba_utils.do_mamba_copy_block(copy_bufs)


mamba_utils.preprocess_mamba = _patched_preprocess_mamba


# ---------------------------------------------------------------------------
# Primary: patched postprocess_mamba — derive dest_block_idx from
# len(block_ids) instead of from num_computed_tokens arithmetic.
#
# Trigger semantics: postprocess detects "a block transitioned from partial
# to full this step" and commits the running state into that block (so it
# can be cache-hit by future requests).
#
# Allocation-derived detection: the running-state slot is at
# len(block_ids) - 1 - num_speculative_blocks. The block immediately before
# it is the candidate that just became full. We commit ONLY when we have a
# saved src index in mamba_state_idx that's distinct from the new dest idx
# — i.e., the running-state slot moved forward this step.
#
# accept_token_bias comes from input_batch.num_accepted_tokens_cpu (set just
# before postprocess_mamba is called from
# _update_states_after_model_execute, gpu_model_runner.py:1448-1451), which
# is the sampler's actual acceptance count, not corrupted arithmetic.
# ---------------------------------------------------------------------------

def _patched_postprocess_mamba(
    scheduler_output: SchedulerOutput,
    kv_cache_config: KVCacheConfig,
    input_batch,
    requests: dict[str, CachedRequestState],
    mamba_state_idx: dict[str, int],
    forward_context: dict[str, Any],
    mamba_state_copy_funcs: tuple[MambaStateCopyFunc, ...],
    copy_bufs,
):
    """
    If a block is converted from partial block to full block in this step,
    copy the state from the block for running state to the new full block.

    Ascend patch: derives dest_block_idx from the actual block allocation
    (len(block_ids)) instead of from num_computed_tokens arithmetic.
    """
    num_accepted_tokens_cpu = input_batch.num_accepted_tokens_cpu
    mamba_group_ids = copy_bufs.mamba_group_ids
    mamba_spec = copy_bufs.mamba_spec
    num_speculative_blocks = mamba_spec.num_speculative_blocks
    copy_bufs.offset = 0
    primary_group = mamba_group_ids[0]

    for i, req_id in enumerate(input_batch.req_ids):
        if req_id is None:
            continue
        req_state = requests[req_id]
        num_allocated_blocks = len(req_state.block_ids[primary_group])
        new_running_idx = num_allocated_blocks - 1 - num_speculative_blocks
        if new_running_idx < 0:
            continue

        # Block just transitioned to full = block immediately before the
        # current running-state slot.
        dest_block_idx = new_running_idx - 1
        if dest_block_idx < 0:
            # No previously-completed block to commit. First running-state
            # slot for the request, or no transition this step.
            continue

        src_block_idx = mamba_state_idx.get(req_id)
        if src_block_idx is None:
            continue

        # No transition occurred this step.
        if new_running_idx == src_block_idx:
            continue
        # Allocation jumped by more than one block — unexpected per the
        # scheduler's invariant of one new block per fill event. Skip with
        # warning rather than guess which block holds the just-completed state.
        if new_running_idx != src_block_idx + 1:
            logger.warning(
                "postprocess_mamba: running-state slot advanced by %d "
                "(src=%d, new=%d) for req %s; skipping copy.",
                new_running_idx - src_block_idx,
                src_block_idx,
                new_running_idx,
                getattr(req_state, "req_id", "<unknown>"),
            )
            continue

        # Bounds-check src against the current allocation.
        if src_block_idx < 0 or src_block_idx >= num_allocated_blocks:
            continue

        # accept_token_bias from the sampler's actual count.
        # num_accepted_tokens_cpu[i] is set in
        # _update_states_after_model_execute from num_accepted_tokens.gpu,
        # which is computed at gpu_model_runner.py:1428-1445 from the real
        # sampler output. accept_token_bias = num_accepted_tokens - 1, range
        # 0..num_speculative_tokens.
        num_accepted = int(num_accepted_tokens_cpu[i])
        accept_token_bias = max(0, num_accepted - 1)

        mamba_utils.collect_mamba_copy_meta(
            copy_bufs,
            kv_cache_config,
            mamba_state_copy_funcs,
            mamba_group_ids,
            src_block_idx,
            dest_block_idx,
            accept_token_bias,
            req_state,
            forward_context,
        )
        if src_block_idx == dest_block_idx:
            num_accepted_tokens_cpu[i] = 1

    mamba_utils.do_mamba_copy_block(copy_bufs)


mamba_utils.postprocess_mamba = _patched_postprocess_mamba
