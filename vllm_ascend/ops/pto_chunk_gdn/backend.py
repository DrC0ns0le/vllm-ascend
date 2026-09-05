# SPDX-License-Identifier: Apache-2.0
"""Model-owned, opt-in fresh-prefill backend with visible fallbacks."""

import logging
from collections import Counter

import torch

from vllm_ascend.ops.pto_chunk_gdn.eligibility import fallback_reason

logger = logging.getLogger(__name__)


class MegaGDNBackend:
    def __init__(self, *, topology_supported: bool, prefix: str):
        self.topology_supported = topology_supported
        self.prefix = prefix
        self.kernel = None
        self.counts = Counter()

    def prepare(self, device, num_heads, key_heads, hidden_size):
        if self.kernel is None:
            if device.type != "npu" or "910B" not in torch.npu.get_device_name(device):
                return
            from vllm_ascend.ops.pto_chunk_gdn.mega_kernel import MegaGDNKernel

            self.kernel = MegaGDNKernel(device, num_heads, key_heads, hidden_size)

    def __call__(
        self,
        *,
        q,
        k,
        v,
        g,
        beta,
        initial_state,
        output_final_state,
        cu_seqlens,
        prebuilt_meta,
        head_first,
        use_qk_l2norm_in_kernel,
        fresh_prefill,
        fallback,
        scale=None,
    ):
        cu_host = getattr(prebuilt_meta, "cu_seqlens_host", None)
        reason = fallback_reason(
            device_type=q.device.type,
            dtype=str(q.dtype),
            q_shape=q.shape,
            k_shape=k.shape,
            v_shape=v.shape,
            g_shape=g.shape,
            beta_shape=beta.shape,
            cu_shape=() if cu_seqlens is None else cu_seqlens.shape,
            cu_host=cu_host,
            fresh_prefill=fresh_prefill,
            topology_supported=self.topology_supported,
            head_first=head_first,
        )
        if reason is None and (k.dtype != q.dtype or v.dtype != q.dtype):
            reason = "dtype_mismatch"
        if reason is None and any(t.device != q.device for t in (k, v, g, beta, cu_seqlens)):
            reason = "device_mismatch"
        if reason is None and cu_seqlens.dtype not in (torch.int32, torch.int64):
            reason = "sequence_dtype"
        if reason is None:
            # Compile failures propagate. Unsupported hardware falls back before
            # a dav-c220 binary can be launched on another NPU architecture.
            self.prepare(q.device, v.shape[2], q.shape[2], q.shape[3])
            if self.kernel is None:
                reason = "hardware"
        if reason is not None:
            self.counts[f"fallback:{reason}"] += 1
            if self.counts[f"fallback:{reason}"] == 1:
                logger.warning("MegaGDN fallback: layer=%s reason=%s", self.prefix, reason)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("MegaGDN counters: layer=%s counts=%s", self.prefix, dict(self.counts))
            return fallback(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                scale=scale,
                initial_state=initial_state,
                output_final_state=output_final_state,
                cu_seqlens=cu_seqlens,
                prebuilt_meta=prebuilt_meta,
                head_first=head_first,
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            )

        # Compile/load failures are fatal when explicitly enabled: never hide
        # a failed experiment behind an apparently successful fallback benchmark.
        if use_qk_l2norm_in_kernel:
            from vllm_ascend.ops.triton.fla.l2norm import l2norm_fwd

            q, k = l2norm_fwd(q), l2norm_fwd(k)
        self.counts["megagdn"] += 1
        if self.counts["megagdn"] == 1:
            logger.info(
                "MegaGDN selected: layer=%s H=%d Hg=%d D=%d C=128 dtype=%s",
                self.prefix,
                v.shape[2],
                q.shape[2],
                q.shape[3],
                q.dtype,
            )
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("MegaGDN counters: layer=%s counts=%s", self.prefix, dict(self.counts))
        with torch.autograd.profiler.record_function("PTO_MegaGDN_prefill"):
            output, state = self.kernel.run(
                q.to(torch.float16).contiguous(),
                k.to(torch.float16).contiguous(),
                v.to(torch.float16).contiguous(),
                g.float().contiguous(),
                beta.to(torch.float16).contiguous(),
                cu_seqlens.to(torch.int32).contiguous(),
                cu_seqlens_host=cu_host,
                scale=q.shape[-1] ** -0.5 if scale is None else scale,
                return_final_state=True,
            )
        # Retain the baseline state dtype contract. The PTO accumulator itself
        # remains FP16 and must pass the separate numerical accuracy gate.
        state_dtype = q.dtype if initial_state is None else initial_state.dtype
        return output.to(q.dtype), state.to(state_dtype) if output_final_state else None
