# SPDX-License-Identifier: Apache-2.0
"""Configure token buckets for the native GDN breakable graph path."""

import vllm.envs as envs_vllm
from vllm.config import CUDAGraphMode
from vllm.logger import logger

from vllm_ascend.utils import check_gdn_layer


def configure_gdn_piecewise_graphs(config):
    compilation = config.compilation_config
    if (
        not envs_vllm.VLLM_USE_BREAKABLE_CUDAGRAPH
        or not check_gdn_layer(config)
        or config.model_config.enforce_eager
        or compilation.cudagraph_mode
        not in (
            CUDAGraphMode.FULL,
            CUDAGraphMode.FULL_AND_PIECEWISE,
            CUDAGraphMode.PIECEWISE,
        )
    ):
        return
    if compilation.cudagraph_mode == CUDAGraphMode.FULL:
        logger.warning(
            "GDN uses the FULL_AND_PIECEWISE dispatcher policy. Qwen3.5's native FULL registry "
            "overrides model execution when enabled; other paths retain piecewise prefill."
        )
        compilation.cudagraph_mode = CUDAGraphMode.FULL_AND_PIECEWISE
    # Use the scheduler's aggregate budget, not a single request's length or
    # max_num_seqs. This ceiling covers batches above the last supplied bucket.
    maximum = config.scheduler_config.max_num_batched_tokens
    sizes = sorted({size for size in compilation.cudagraph_capture_sizes or () if 0 < size <= maximum} | {maximum})
    compilation.cudagraph_capture_sizes = sizes
    compilation.max_cudagraph_capture_size = maximum
    logger.info("GDN piecewise capture token buckets: %s", sizes)
