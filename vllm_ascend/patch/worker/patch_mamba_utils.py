# mypy: ignore-errors

import logging

from vllm.v1.worker import mamba_utils

from vllm_ascend.ops.triton.batch_memcpy import batch_memcpy_kernel


_diag_logger = logging.getLogger("vllm_ascend.mamba_diag")


def batch_memcpy(src_ptrs, dst_ptrs, sizes):
    batch = src_ptrs.shape[0]
    assert dst_ptrs.shape[0] == batch
    assert sizes.shape[0] == batch

    grid = (batch,)
    # using larger block_size to accelerate copy.
    BLOCK_SIZE = 8192
    batch_memcpy_kernel[grid](src_ptrs, dst_ptrs, sizes, BLOCK_SIZE=BLOCK_SIZE)


_orig_collect_mamba_copy_meta = mamba_utils.collect_mamba_copy_meta


def collect_mamba_copy_meta(
    copy_bufs,
    kv_cache_config,
    mamba_state_copy_funcs,
    mamba_group_ids,
    src_block_idx,
    dest_block_idx,
    accept_token_bias,
    req_state,
    forward_context,
):
    # Diagnostic: log when dest_block_idx (or src) overruns block_ids for
    # any mamba group, before the upstream code IndexErrors. This narrows
    # down whether the bug is `num_computed_tokens` inflation (large
    # dest_block_idx) or unequal/short mamba block allocation (small
    # dest_block_idx but even shorter block_ids).
    try:
        if src_block_idx == dest_block_idx and accept_token_bias == 0:
            return _orig_collect_mamba_copy_meta(
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

        all_group_lens = [len(g) for g in req_state.block_ids]
        for mamba_group_id in mamba_group_ids:
            block_ids = req_state.block_ids[mamba_group_id]
            if dest_block_idx >= len(block_ids) or src_block_idx >= len(block_ids):
                _diag_logger.error(
                    "MAMBA_DIAG out_of_range req=%s mamba_group=%d "
                    "src_idx=%d dest_idx=%d accept_bias=%d "
                    "len_this_group=%d num_groups=%d "
                    "num_computed_tokens=%s "
                    "all_group_lens=%s "
                    "block_ids_this_group=%s",
                    req_state.req_id,
                    mamba_group_id,
                    src_block_idx,
                    dest_block_idx,
                    accept_token_bias,
                    len(block_ids),
                    len(req_state.block_ids),
                    req_state.num_computed_tokens,
                    all_group_lens,
                    block_ids,
                )
    except Exception:
        # Never let diagnostics themselves crash the worker.
        _diag_logger.exception("MAMBA_DIAG diagnostic block raised")

    return _orig_collect_mamba_copy_meta(
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


_orig_preprocess_mamba = mamba_utils.preprocess_mamba


def preprocess_mamba(
    scheduler_output,
    kv_cache_config,
    cache_config,
    mamba_state_idx,
    input_batch,
    requests,
    forward_context,
    mamba_state_copy_funcs,
    copy_bufs,
):
    # Snapshot CPU-side fields the upstream preprocess_mamba math depends on.
    try:
        snap = []
        for i, req_id in enumerate(input_batch.req_ids):
            if req_id is None:
                continue
            r = requests.get(req_id)
            if r is None:
                continue
            snap.append(
                (
                    req_id,
                    r.num_computed_tokens,
                    scheduler_output.num_scheduled_tokens.get(req_id),
                    [len(g) for g in r.block_ids],
                    input_batch.num_accepted_tokens_cpu[i] if i < len(input_batch.num_accepted_tokens_cpu) else None,
                    getattr(r, "prev_num_draft_len", None),
                )
            )
        # Only log when something looks off (num_computed_tokens > 1e7
        # is impossible in practice given max_seq_len=262144) so we don't
        # spam in the healthy case.
        for entry in snap:
            (req_id_, ncomp, nsched, group_lens, naccept, prev_drafts) = entry
            if isinstance(ncomp, int) and ncomp > 10_000_000:
                _diag_logger.error(
                    "MAMBA_DIAG preprocess_entry req=%s "
                    "num_computed_tokens=%s num_scheduled=%s "
                    "all_group_lens=%s num_accepted_cpu=%s prev_num_draft_len=%s",
                    req_id_,
                    ncomp,
                    nsched,
                    group_lens,
                    naccept,
                    prev_drafts,
                )
    except Exception:
        _diag_logger.exception("MAMBA_DIAG preprocess_entry diag raised")

    return _orig_preprocess_mamba(
        scheduler_output,
        kv_cache_config,
        cache_config,
        mamba_state_idx,
        input_batch,
        requests,
        forward_context,
        mamba_state_copy_funcs,
        copy_bufs,
    )


mamba_utils.batch_memcpy_kernel = batch_memcpy_kernel
mamba_utils.batch_memcpy = batch_memcpy
mamba_utils.collect_mamba_copy_meta = collect_mamba_copy_meta
mamba_utils.preprocess_mamba = preprocess_mamba
