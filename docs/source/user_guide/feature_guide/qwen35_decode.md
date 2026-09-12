# Qwen3.5 BF16/FP16 decode kernels on Ascend 910B

This opt-in path targets dense Qwen3.5-2B/4B, BF16 or FP16, TP/PP/CP=1, without
quantization or LoRA. NPU compilation, accuracy, and TPOT remain external
validation gates. Recurrent-state math, prefill graph planning, and the
AscendC/MegaGDN prefill kernels are unchanged.

## Configuration

Merge this into the existing `--additional-config`:

```json
{
  "weight_nz_mode": 1,
  "qwen35_decode": {
    "linear_backend": "triton",
    "linear_projections": ["gate_up_proj", "down_proj"],
    "max_batch_size": 8,
    "split_k": 1,
    "fuse_gdn_ba_prepare": true
  }
}
```

Use `--dtype bfloat16` for the BF16 deployment. Kernels preserve the model's
loaded dtype for weights, activations, gates, and outputs; accumulators and
split-K partials remain FP32. FP16 is also supported without a separate kernel
configuration. Inputs and weights must have matching dtypes.

| Option | Meaning |
|---|---|
| `linear_backend=native` | Existing native linear kernels; default. |
| `linear_backend=triton` | AIV vector GEMV at M=1; shared-weight Cube GEMM at M=2..8. |
| `linear_backend=triton_cube` | Cube GEMM for every eligible M, including M=1, for comparison with GEMV. |
| `linear_projections` | Defaults to MLP `gate_up_proj`, `down_proj`. Also accepts `in_proj_qkvz`, `in_proj_ba`, `out_proj`, `qkv_proj`, `o_proj`, `lm_head`. |
| `max_batch_size` | Integer 1..8; applies to both optimizations. |
| `split_k` | 1 (default), 2, or 4. Values above 1 add one FP32 reduction kernel per selected linear. |
| `fuse_gdn_ba_prepare` | Independent boolean, default false. Replaces BA matmul and QKVZ/BA preparation with one vector kernel. |

Selection uses the padded tensor's M dimension, not the live request count.
Larger batches use the existing native path. Very short fresh prompts with M<=8
also use the selected kernels: the projection math is stateless and does not
depend on cache ownership. Ordinary larger prefill batches keep native linears.

Both paths share the loaded ND `[N,K]` weight, with no retained second weight
pack or per-bucket weight copy. `weight_nz_mode=2` is rejected because direct
Triton pointer arithmetic cannot interpret physical FRACTAL_NZ storage. Modes
0 and 1 preserve the existing BF16/FP16 prefill weight layout.

## Implementation

`ops/triton/qwen35_decode_linear.py` implements a vector GEMV with a bounded
16 KiB FP32 accumulator tile and a Cube kernel that shares each weight tile
across all M rows. The Cube tile has 16 rows; unused rows are masked. Programs
are bounded by core count and process remaining tiles on device. There is no
autotuning or new compilation lock. Both kernels fuse optional bias.

Split-K writes disjoint FP32 partials and reduces them deterministically, adding
bias once and rounding to the input dtype. There are no atomics, spin waits, or host
synchronizations. Scratch is `split_k * M * N * 4` bytes per invocation when
split_k>1; split_k=1 needs no partial-output scratch. The extra reduction launch
can outweigh the parallelism benefit, so split-K is disabled by default.

This adapts ideas from [CUDA decode GEMV](https://pytorch.org/blog/accelerating-generative-ai-2/),
[ROCm small-M split-K GEMMs](https://rocm.blogs.amd.com/software-tools-optimization/accelerating-llm-inference-on-amd-gpus-with-low-latency-gemms/README.html),
and [TPU tiled matrix multiplication](https://docs.jax.dev/en/latest/pallas/tpu/matmul.html).
Their published gains do not predict 910B performance; these kernels use Ascend
Triton primitives and do not change checkpoint precision.

`ops/triton/qwen35_gdn_prepare.py` keeps the aligned QKVZ GEMM and fuses the tiny
BA projection with layout preparation. B/A share an activation load and FP32
accumulation, then round to the input dtype. QKV, Z, B, and A are written directly to
contiguous buffers. This removes the separate BA matmul, intermediate BA,
B/A copies, and Z flattening copy. At M=1 some copies were already views, but
BA matmul is still replaced. Fusion takes precedence if `in_proj_ba` is also
selected as a standalone linear. Recurrence and gated norm remain native.

`ops/qwen35_decode.py` registers graph-visible operators with fake tensor
implementations. `ops/linear.py` selects them after weight loading; the GDN
forward calls the fused preparation. `ops/vocab_parallel_embedding.py` has a
separate optional LM-head adapter, preserving embedding lookup behavior.

The BA launch computes power-of-two block sizes in Python and passes them to
the JIT kernel. This avoids calling the host-only `triton.next_power_of_2` with
`tl.constexpr` on older Triton-Ascend versions. The 4B copy span remains 384
elements per head with a separate 512-element masked block.

## Graph coverage

The existing warmup invokes selected kernels before capture. Their launches,
including split-K reduction, are recorded inside ACLGraph. No eager-break
decorator, per-step state staging, or attention metadata update is added.

For `FULL_DECODE_ONLY`, include **1, 2, 4, 8** in
`compilation_config.cudagraph_capture_sizes`. A live M=2 padded to a 64-token
graph uses native kernels. For static native `FULL` / `FULL_AND_PIECEWISE`,
include those counts in `additional_config.native_full_graph_request_counts`
and keep the existing prefill width buckets. Neither capture list is silently
modified.

Selecting `lm_head` changes its kernel but does **not** move logits computation
or sampling into the model graph. Those remain outside model capture in the
current runner.

## Validation and profiling

CPU tests cover loading, configuration, dispatch, fake shapes, and actual
kernel source evaluated with bounds-checked tensor loads/stores and independent
FP32 references. They do not validate Ascend lowering or UB allocation.

```bash
pytest -sv tests/e2e/pull_request/one_card/aclgraph/test_qwen35_decode_kernels.py
pytest -sv tests/e2e/pull_request/one_card/aclgraph/test_qwen35_decode_model.py
```

The NPU tests cover BF16 and FP16 for both models' projection shapes at M=1/2/4/8, bias, split-K,
padded rows, changed inputs, and poisoned outputs across replay. The real-model
test verifies layer selection, FULL_DECODE_ONLY captures without eager breaks,
serving replay hits at each small bucket, and greedy outputs against native
kernels. Also retain the existing native FULL/MegaGDN accuracy gate for the
deployed prefill backend.

Use the existing profiler, at M=1/2/4/8 and the same 32-token context, precision,
capture sizes, and scheduler settings:

1. Native baseline: backend `native`, fusion false.
2. BA fusion alone: backend `native`, fusion true.
3. MLP alone: `triton`, then `triton_cube`, fusion false, split_k=1.
4. Split_k=2/4 on `down_proj` alone before applying it to other projections.
5. Combine measured improvements, then test GDN projections and LM head alone.

Track device time, unprofiled TPOT, graph launch count, and numerical consistency.
Kernel names include `decode_gemv`, `decode_skinny_gemm`, `decode_split_k_reduce`,
and `gdn_ba_prepare`. BA fusion should remove 18 separate BA matmuls per 2B
model step or 24 per 4B step. Other MatMulV2 calls remain unless selected.
No specific latency reduction is claimed before NPU profiling.
