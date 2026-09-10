# Graph Mode Guide

## Overview

This guide explains how graph mode is used in vLLM Ascend.

vLLM already provides the generic graph-mode architecture, mode definitions, and compile integration. For those upstream concepts, see:

- [CUDA Graphs](https://docs.vllm.ai/en/latest/design/cuda_graphs/)
- [torch.compile](https://docs.vllm.ai/en/latest/design/torch_compile/)

This document focuses on the Ascend-specific view: how graph mode works on Ascend, which components are involved, how to configure them, and what constraints users should keep in mind.

## Current Status on Ascend

- Graph mode is currently available only on the **V1 Engine**.
- **ACLGraph** (capture/replay via `torch.npu.NPUGraph`) is the runtime graph execution mechanism used by the default graph path on Ascend.
- **Npugraph_ex** is a compile-time FX graph optimization layer, enabled by default in FULL/FULL_DECODE_ONLY modes. It optimizes the graph before ACLGraph captures it.
- **XliteGraph** is an optional graph path for selected model families and environments.
- In context parallel scenarios, `cudagraph_mode="FULL"` is not sufficiently supported yet.

## Graph Paths on Ascend

vLLM Ascend provides two graph paths:

| Graph Path | Default | Description | Since |
|---|---|---|---|
| ACLGraph (+ Npugraph_ex) | Yes | Compile-time FX optimization (Npugraph_ex) + runtime capture/replay (ACLGraph) | v0.9.0rc1 (Npugraph_ex since v0.15.0rc1) |
| XliteGraph | No | Preconfigured graph path for selected model families. Requires separate installation | v0.11.0 |

## How Graph Mode Works on Ascend

The default graph path on Ascend involves two stages: **compile-time optimization** and **runtime capture/replay**. ACLGraph handles the runtime capture/replay. The compile-time stage differs by `cudagraph_mode`:

- **FULL_AND_PIECEWISE**: Default mode, same as the upstream vLLM strategy. The compile-time path follows PIECEWISE compilation, while the runtime may still use full-graph behavior for uniform decode batches.
- **FULL / FULL_DECODE_ONLY**: Npugraph_ex optimizes the FX graph via npugraph_ex (`force_eager=True`, compile-time only, no capture). The optimized callable is then captured and replayed by ACLGraph at runtime.
- **PIECEWISE**: Npugraph_ex is disabled. Only basic FX fusion passes are applied at compile-time. ACLGraph captures and replays the resulting callable at runtime.
- **NONE**: No compilation or graph capture. The model runs in eager mode.

| `cudagraph_mode` | Compile-time | Runtime | Npugraph_ex |
|---|---|---|---|
| FULL_AND_PIECEWISE | Piecewise compilation path | Mixed: PIECEWISE for mixed batches, FULL-capable for uniform decode batches | Disabled |
| FULL / FULL_DECODE_ONLY | Npugraph_ex FX optimization | ACLGraph capture/replay | Enabled |
| PIECEWISE | Fusion pass only | ACLGraph capture/replay | Disabled |
| NONE | None | Eager execution | Disabled |

Additionally, **XliteGraph** is available as an optional alternative graph path for selected model families (see [Using XliteGraph](#using-xlitegraph)).

## Using ACLGraph

ACLGraph is the runtime graph capture/replay mechanism on Ascend. It is enabled automatically when graph mode is active (i.e., `cudagraph_mode` is not `NONE`), and does not require explicit configuration.

### Basic usage

Offline example:

```python
from vllm import LLM

llm = LLM(model="path/to/Qwen3-0.6B")
outputs = llm.generate("Hello, how are you?")
```

Online example:

```bash
vllm serve Qwen/Qwen3-0.6B
```

### Explicit `cudagraph_mode` configuration

The generic `cudagraph_mode` options come from upstream vLLM. On Ascend, the final effective mode may still be adjusted according to platform and backend support, so the official vLLM CUDA Graphs document remains the canonical reference for mode semantics.

CLI example:

```bash
vllm serve Qwen/Qwen3-0.6B \
  --compilation-config '{"cudagraph_mode": "PIECEWISE"}'
```

Python example:

```python
from vllm import LLM

llm = LLM(
    model="Qwen/Qwen3-0.6B",
    compilation_config={"cudagraph_mode": "PIECEWISE"},
)
```

For the detailed meaning of `NONE`, `PIECEWISE`, `FULL`, `FULL_DECODE_ONLY`, and `FULL_AND_PIECEWISE`, as well as the generic fallback policy, see the upstream [CUDA Graphs](https://docs.vllm.ai/en/latest/design/cuda_graphs/) design doc.

### Attention backend compatibility

Not all attention backends support all graph modes. vLLM checks attention backend compatibility during compatibility checks and, when possible, automatically adjusts `cudagraph_mode` to a more compatible mode instead of failing immediately. In practice, this means a requested full-graph mode may be narrowed to a mixed or piecewise mode, and if the backend cannot support graph execution at all, graph mode may be disabled.

On Ascend, the current attention backend support levels are:

| Attention backend | Declared support | Practical meaning |
|---|---|---|
| `attention_v1` | `ALWAYS` | Supports graph execution for mixed prefill/decode batches |
| `context_parallel/attention_cp` | `ALWAYS` | Supports graph execution for mixed prefill/decode batches |
| `mla_v1` | `UNIFORM_BATCH` | Graph execution is limited to uniform batches; full graph is more restricted |
| `context_parallel/mla_cp` | `UNIFORM_BATCH` | Graph execution is limited to uniform batches; full graph is more restricted |
| `sfa_v1` | `UNIFORM_BATCH` | Graph execution is limited to uniform batches; full graph is more restricted |
| `context_parallel/sfa_cp` | `UNIFORM_BATCH` | Graph execution is limited to uniform batches; full graph is more restricted |

This is why the effective graph mode on Ascend may differ from the mode requested in configuration.

### Troubleshooting capture resource exhaustion

If ACLGraph capture fails because the configured graph sizes exceed the runtime resources available on the current stack, vLLM Ascend now raises a dedicated error with mitigation guidance. In practice, the most useful actions are:

- upgrade to a newer HDK/CANN stack if one is available;
- reduce `cudagraph_capture_sizes` or `max_cudagraph_capture_size`;
- prefer `FULL` or `FULL_DECODE_ONLY` when the workload is mostly uniform decode;
- temporarily disable graph mode to confirm the issue is capture-related.

This is most likely to appear in `PIECEWISE` or `FULL_AND_PIECEWISE` configurations because those paths tend to capture more graphs than uniform full-graph decode.

## Using Npugraph_ex

As introduced in the [RFC](https://github.com/vllm-project/vllm-ascend/issues/4715), Npugraph_ex is a compile-time FX graph optimization layer that works together with ACLGraph. It optimizes the model's FX graph before ACLGraph captures it at runtime. Its performance benefits mainly come from fusing multiple operators into single kernels (e.g., add + rms_norm → npu_add_rms_norm) to reduce kernel launch overhead.

!!! note "Atlas inference products"

    Atlas inference products and Atlas 200I Pro do not support `enable_npugraph_ex`. Set --additional-config '{"ascend_compilation_config": {"enable_npugraph_ex":false}}'.

### Default behavior

Npugraph_ex is **enabled by default** when `cudagraph_mode` is `FULL` or `FULL_DECODE_ONLY`. It is automatically disabled in `PIECEWISE` or `NONE` modes.

This means for most users, Npugraph_ex is active without any explicit configuration:

```python
from vllm import LLM

# Npugraph_ex is enabled by default in FULL/FULL_DECODE_ONLY mode
llm = LLM(model="path/to/Qwen2-7B-Instruct")
outputs = llm.generate("Hello, how are you?")
```

### Explicit configuration

To explicitly control Npugraph_ex:

Offline example:

```python
from vllm import LLM

model = LLM(
    model="path/to/Qwen2-7B-Instruct",
    additional_config={
        "ascend_compilation_config": {
            "enable_npugraph_ex": True,
        }
    }
)
outputs = model.generate("Hello, how are you?")
```

Online example:

```bash
vllm serve Qwen/Qwen2-7B-Instruct \
  --additional-config '{"ascend_compilation_config":{"enable_npugraph_ex":true}}'
```

To disable Npugraph_ex explicitly:

```bash
vllm serve Qwen/Qwen2-7B-Instruct \
  --additional-config '{"ascend_compilation_config":{"enable_npugraph_ex":false}}'
```

### Static kernel compilation

Static kernel compilation is an **optional** feature that pre-compiles operator binaries with fixed shapes at compile time, reducing runtime overhead for networks with static or near-static shapes. It is **disabled by default** and must be explicitly enabled.

!!! note

    Enabling static kernel triggers a compilation pass during the graph capture phase at service startup. This may add **several minutes to tens of minutes** to the startup time depending on the number of operators to compile and model complexity. Once completed, subsequent request processing is not affected.

Offline example:

```python
from vllm import LLM

model = LLM(
    model="path/to/Qwen2-7B-Instruct",
    additional_config={
        "ascend_compilation_config": {
            "enable_npugraph_ex": True,
            "enable_static_kernel": True,
        }
    }
)
outputs = model.generate("Hello, how are you?")
```

Online example:

```bash
vllm serve Qwen/Qwen2-7B-Instruct \
  --additional-config '{"ascend_compilation_config":{"enable_npugraph_ex":true, "enable_static_kernel":true}}'
```

#### Verifying static kernel is active

The recommended way to verify static kernel is in effect is through **Ascend Profiling**:

1. Collect a profiling trace of your running model using [Ascend PyTorch Profiler](https://www.hiascend.com/document/detail/zh/Pytorch/2600/apiref/torchnpuCustomsapi/docs/zh/custom_APIs/torch_npu-profiler/torch_npu-profiler-profile.md) (`torch_npu.profiler`).
2. Open the generated `op_statistic.csv` file.
3. Look for operators whose `op_type` or `name` column contains the keyword **`static_kernel`**. If such entries exist, static kernel compilation has taken effect for those operators.

During the compilation phase, you will see a Python warning (visible by default):

```text
Starting static kernel compilation, the build directory is <path>
```

This confirms that compilation has been triggered. The absence of this message means static kernel was not enabled or the cached result was reused directly.

For more details about Npugraph_ex, see the [npugraph_ex guide](https://www.hiascend.com/document/detail/zh/Pytorch/2600/modthirdparty/torchairuseguide/docs/zh/overview.md).

## Using XliteGraph {: #using-xlitegraph }

XliteGraph is an optional path for Llama, Qwen dense series models, Qwen MoE series models, and Qwen3-VL. It requires Xlite to be installed and configured through `xlite_graph_config`.

Install Xlite first:

```bash
pip install xlite
```

Offline example:

```python
from vllm import LLM

# Xlite supports decode-only mode by default.
# Full mode can be enabled with "full_mode": True.
llm = LLM(
    model="path/to/Qwen3-32B",
    tensor_parallel_size=8,
    additional_config={
        "xlite_graph_config": {
            "enabled": True,
            "full_mode": True,
        }
    },
)
outputs = llm.generate("Hello, how are you?")
```

Online example:

```bash
vllm serve path/to/Qwen3-32B \
  --tensor-parallel-size 8 \
  --additional-config '{"xlite_graph_config": {"enabled": true, "full_mode": true}}'
```

For more details about Xlite, see the [Xlite README](https://atomgit.com/openeuler/GVirt/blob/master/xlite/README.md).

## Qwen3.5 native kernels with piecewise token buckets

Use MRV1 breakable graphs with `FULL_AND_PIECEWISE` to retain native Ascend
attention and GDN operators for fresh, continuation, chunked, and mixed prefill.
Projection, normalization, and MLP regions replay captured graphs. The operator
breaks read the current request boundaries and state slots on every replay.
Uniform decode keeps its standard FULL graph path.

The Ascend GDN core owns its eager break, including when warmup has no attention
metadata. This also supports vLLM versions whose Qwen custom operator does not
declare a break. Replay must execute GDN with current metadata; capturing its
no-metadata warmup return would omit recurrent computation entirely.

```bash
VLLM_USE_V2_MODEL_RUNNER=0 VLLM_USE_BREAKABLE_CUDAGRAPH=1 \
vllm serve Qwen/Qwen3.5-2B \
  --dtype bfloat16 \
  --tensor-parallel-size 1 \
  --max-model-len 768 \
  --max-num-seqs 64 \
  --max-num-batched-tokens 4096 \
  --enable-chunked-prefill \
  --no-enable-prefix-caching \
  --mamba-cache-mode none \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[1,64,128,196,256,384,512,768,1024,1536,2048,3172,4096]}'
```

For breakable GDN models, a request for `FULL` now resolves to
`FULL_AND_PIECEWISE` with a warning. The experimental native FULL-prefill cache,
device attention kernel, and fused GDN state/output route have been removed.
The old `full_prefill_graph_max_entries` setting has no effect. Leave
`VLLM_ASCEND_PTO_CHUNK_GDN` disabled (its default) to use the baseline AscendC
GDN pipeline; MegaGDN remains a separate explicit opt-in.

Startup warms and captures the dispatcher's predefined token buckets, then
checks that all piecewise entries exist. Breakable GDN configuration adds
`max_num_batched_tokens` as a final bucket when needed. This is the aggregate
scheduler token budget, not the maximum length of one request.

At serving time, the dispatcher rounds the total scheduled tokens up to the
smallest covering bucket. For example, `[61]`, `[17,44]`, and `[1,1,59]` reuse
the same 64-token piecewise graphs. Their native operators still receive the
original boundaries. Padding does not add recurrent steps or KV writes, and
only live output rows are consumed. Changing the prefill/decode mixture does
not create a layout-specific graph entry.

Graph coverage is limited by the final effective capture sizes after platform
and parallelism compatibility checks. The configuration above targets one NPU.
Graph warmup does not guarantee every native operator specialization is already
compiled for every possible layout. GDN's native operator sequence is submitted
from Python at each eager break, so token bucket coverage does not eliminate
the host overhead within that sequence. Ascend startup/replay accuracy and mixed
serving latency must still be validated on hardware; CPU contract tests do not
establish either accuracy or performance on NPU.

### Investigating Qwen3.5 submission gaps

Native GDN prefill builds its chunk tables from CPU request boundaries and uploads
the tables together in one aligned allocation. Each step owns its allocation;
an in-flight step never shares writable tables with a later step. Standard
attention reuses the runner's uploaded query offsets. Mixed GDN batches write
their decode and prefill outputs directly into the caller's output slices.
These changes reduce preparation and copying without changing recurrent math
or removing FULL replay synchronization.

Compare against `9ec889975` using the same model, command, request sequence, and
capture sizes. Measure latency with profiling and DEBUG logging off. Verify
output correctness before comparing latency, including repeated requests,
continuation chunks, and arrivals during decode. The changes have CPU contract
coverage; NPU performance remains to be measured.

Two further configuration comparisons can help isolate the remaining delay.
Run them separately, changing only the indicated setting:

| Comparison | Change | What it isolates |
| --- | --- | --- |
| Scheduler | Use `--no-async-scheduling` | Whether asynchronous step coordination contributes to latency. Throughput may decrease. |
| Decode graph | Change `cudagraph_mode` from `FULL_AND_PIECEWISE` to `PIECEWISE` | Avoids the FULL decode update/replay path while preserving piecewise prefill. Decode submission work may increase. |

For a separate short diagnostic profile, set
`VLLM_CUSTOM_SCOPES_FOR_PROFILING=1` before server startup and use the existing
torch NPU profiler configuration. No DEBUG logging is needed. New CPU ranges are:

- `ascend::update_states` and `ascend::build_attention_metadata`
- `ascend::model_execution` and `ascend::update_full_graph_params`
- `ascend::full_replay_sync` for the FULL-only host barrier
- `ascend::PIECEWISE::tokens=64::segment=0::graph` (or `eager`), with the actual
  runtime mode, padded token count, and segment index in each label

The segment ranges are nested inside model execution. Existing `prepare input`,
`forward`, `post process`, and `sample_token` ranges locate them within a runner
step. With custom scopes disabled, recorded segments keep their original
callables. Profiling ranges measure host activity, not NPU completion; use the
profiler's CPU/device correlations to locate device gaps. A kernel CSV's wait
duration alone does not identify the host submission time or the waiting API.

If only a text summary can be returned, include the replay mode and token bucket,
counts and durations of the above ranges, unprofiled latency, and whether the
largest gap is inside model execution, in input preparation, or between runner
steps. This is enough to direct the next change without sharing a full trace.

## Common Limitations and Caveats

- XliteGraph should be treated as an alternative graph path, not as a drop-in replacement for ACLGraph in all scenarios.
- Model and backend coverage is still evolving, so a configuration that works for one model family may not yet be recommended for another.
- Encoder-decoder models currently do not keep `FULL_AND_PIECEWISE`; on Ascend they fall back to `PIECEWISE` or `NONE` depending on compilation support.

## Fallback to Eager Mode

If you encounter issues with graph mode, you can temporarily fall back to eager mode by setting `enforce_eager=True`.

If ACL graph capture fails with the confirmed stream-resource signature in the error text, such as `207008` together with `Stream resources are insufficient` or `Insufficient_Stream_Resources`, vLLM Ascend will re-raise that capture failure with targeted mitigation guidance. In practice, the main levers are: upgrading to a newer HDK/CANN stack, reducing `cudagraph_capture_sizes`, lowering `max_cudagraph_capture_size`, or preferring `FULL` / `FULL_DECODE_ONLY` when the workload is mostly uniform decode.

**Offline example:**

```python
from vllm import LLM

llm = LLM(model="path/to/your/model", enforce_eager=True)
outputs = llm.generate("Hello, how are you?")
```

**Online example:**

```bash
vllm serve path/to/your/model --enforce-eager
```

## References

- [CUDA Graphs](https://docs.vllm.ai/en/latest/design/cuda_graphs/)
- [torch.compile](https://docs.vllm.ai/en/latest/design/torch_compile/)
- [Xlite README](https://atomgit.com/openeuler/GVirt/blob/master/xlite/README.md)
- [Npugraph_ex guide](https://www.hiascend.com/document/detail/zh/Pytorch/2600/modthirdparty/torchairuseguide/docs/zh/overview.md)
- [Npugraph_ex RFC](https://github.com/vllm-project/vllm-ascend/issues/4715)
- [ACL Graph Developer Guide](../../developer_guide/Design_Documents/ACL_Graph.md)
