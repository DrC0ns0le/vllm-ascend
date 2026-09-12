# SPDX-License-Identifier: Apache-2.0
"""Model-owned, opt-in fresh-prefill backend with visible fallbacks."""

import logging
from collections import Counter

import torch

from vllm_ascend.ops.pto_chunk_gdn.eligibility import fallback_reason, total_chunks

logger = logging.getLogger(__name__)


def is_piecewise_runtime() -> bool:
    # Resolve vLLM in the worker; importing the CPU eligibility helpers does
    # not require initializing the serving runtime.
    from vllm.config import CUDAGraphMode
    from vllm.forward_context import get_forward_context, is_forward_context_available

    if not is_forward_context_available():
        return False
    return get_forward_context().cudagraph_runtime_mode == CUDAGraphMode.PIECEWISE


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
        native_graph=False,
        workspace=None,
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
        if reason is None and not native_graph and not is_piecewise_runtime():
            reason = "runtime_not_piecewise"
        if reason is None:
            # Compile failures propagate. Unsupported hardware falls back before
            # a dav-c220 binary can be launched on another NPU architecture.
            self.prepare(q.device, v.shape[2], q.shape[2], q.shape[3])
            if self.kernel is None:
                reason = "hardware"
        if reason is not None:
            if native_graph:
                raise ValueError(f"MegaGDN fresh graph is unsupported: {reason}")
            self.counts[f"fallback:{reason}"] += 1
            if logger.isEnabledFor(logging.DEBUG):
                self._log_decision(reason, q, v, cu_host)
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
                workspace=workspace,
            )
        if logger.isEnabledFor(logging.DEBUG):
            self._log_decision("megagdn", q, v, cu_host)
        # Retain the baseline state dtype contract. The PTO accumulator itself
        # remains FP16 and must pass the separate numerical accuracy gate.
        state_dtype = q.dtype if initial_state is None else initial_state.dtype
        return output.to(q.dtype), state.to(state_dtype) if output_final_state else None

    def _log_decision(self, reason, q, v, cu_host):
        from vllm.forward_context import get_forward_context, is_forward_context_available

        forward_context = get_forward_context() if is_forward_context_available() else None
        descriptor = getattr(forward_context, "batch_descriptor", None)
        runtime_mode = getattr(forward_context, "cudagraph_runtime_mode", None)
        logger.debug(
            "MegaGDN decision: layer=%s reason=%s mode=%s bucket=%s num_sequences=%s "
            "total_tokens=%s total_chunks=%s H=%s Hg=%s D=%s",
            self.prefix,
            reason,
            getattr(runtime_mode, "name", runtime_mode),
            getattr(descriptor, "num_tokens", None),
            len(cu_host) - 1 if cu_host is not None else None,
            q.shape[1] if q.ndim > 1 else None,
            total_chunks(cu_host) if cu_host is not None else None,
            v.shape[2] if v.ndim > 2 else None,
            q.shape[2] if q.ndim > 2 else None,
            q.shape[3] if q.ndim > 3 else None,
        )
