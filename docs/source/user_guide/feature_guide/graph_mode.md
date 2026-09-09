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

## Qwen3.5 FULL Prefill and Mixed Capture

On Ascend 910B, the MRV1 breakable ACLGraph wrapper can capture fresh prefill,
continuation prefill, and mixed prefill/decode batches, including the baseline
GDN core, as one graph with no eager breaks. New requests can therefore join
ongoing requests through the existing scheduler. MegaGDN remains on the
PIECEWISE path. Decode continues
using its existing FULL graph path. Hardware correctness and performance
validation of this prefill path is still required.

This path requires TP/PP/DP/PCP/DCP size 1, `mamba_cache_mode="none"`, and no
speculative decoding, LoRA, KV transfer, C8 attention, attention sinks, or ENPU.
Use `FULL` for execution without PIECEWISE/eager fallback. The runner pads
each scheduled batch to the next token capacity and captures all retained
capacities at startup. It adds `max_num_batched_tokens` as the final ceiling,
so coverage does not end at the last configured capture size. Standard FULL
decode captures remain separate and cover every request count up to
`max_num_seqs`.

For example:

```bash
VLLM_USE_BREAKABLE_CUDAGRAPH=1 vllm serve /path/to/Qwen3.5-2B \
  --max-num-seqs 64 \
  --max-model-len 1536 \
  --max-num-batched-tokens 4096 \
  --compilation-config '{"cudagraph_mode":"FULL","cudagraph_capture_sizes":[1,64,128,196,256,384,512,768,1024,1536,2048,3172,4096]}'
```

Capacities describe aggregate scheduled tokens, not individual prompt lengths.
Two decodes plus 137-token and 220-token prefills total 359 tokens and use a
384-token capture. The 25 padding tokens cannot write KV or recurrent state.
Larger arrivals are handled by the existing scheduler and chunked prefill;
no extra batching layer is introduced.

Startup warmup uses distinct recurrent slots, restores their state afterward,
and captures both token-ID and embedding inputs when applicable. Captures must
contain exactly one graph with no eager breaks. Serving then only replays:
an unsupported configuration or an uncaptured input signature raises an error
instead of silently changing execution mode. Startup profiling and warmup
execute before serving; they are not request-time eager fallbacks.

`FULL_AND_PIECEWISE` retains the earlier compatibility behavior: the wrapper
selects FULL prefill capture from eligible PIECEWISE dispatches and permits
fallback for unsupported layouts.

For the target 128-dimension GDN model, prefill graphs specialize on the
**token bucket**, with request capacity fixed to `max_num_seqs` (up to 64).
Request arrivals, changing query lengths, fresh/stateful flags, and the
prefill/decode split reuse the same graph within that bucket. GDN processes
all requests through one packed baseline route. Chunk tables and state
read/write indices are device tensors; the Triton state/output stages replace
the AscendC stages that require host query/chunk attributes. Empty sentinel
sequences make unused chunk tasks inert, and padded requests never write state.

Within this graph, device query lengths select a direct recurrent GDN kernel
for single-token requests, including fresh requests. These rows bypass the
prefill chunk tables; longer requests retain the chunked path. The chunk
recurrence reads and writes the strided recurrent cache directly using device
state indices and initial-state predicates. It no longer gathers/scatters
packed initial/final states, and its state/output stages read the original
gate layout without two additional contiguous copies per layer.
Both paths are captured together and write disjoint rows, so
changing the decode/prefill mix does not require a new graph specialization.
The latency effect of these changes still requires NPU measurement.

Both ACL wrappers consult the shared native prefill dispatcher before normal
graph dispatch. An outer ACL wrapper reuses an existing native wrapper's
sealed registry. With debug logging enabled, `FULL prefill replay hit` records
the bucket, sealed status, and entry count for actual replay calls.

Metadata refresh uses one device launch per shared GDN metadata group for
query boundaries, state slots, initial-state flags, and all three chunk tables.
FIA query boundaries, KV slots, and block tables are also refreshed in one
launch per shared attention metadata group. Request count and live token count
are runtime scalars, so arrivals and ragged continuation chunks do not create
new metadata-kernel specializations. This replaces eight tensor operations and
three GDN metadata launches, plus seven FIA buffer operations; it does not
remove FIA's per-layer host task updates or the replay synchronization they
currently require. Latency improvements still need measurement on Ascend.
The standard builder's CPU sequence-length fields share one persistent tensor,
avoiding a duplicate host tensor allocation and copy on each refresh.

With `FULL_AND_PIECEWISE`, the first use of a token bucket still pays
lazy capture cost. Warmup should cover the aggregate batch token buckets,
not just the maximum length of one prompt. In particular, prompts shorter
than 768 tokens can collectively fill a larger bucket under concurrent load.
The default scheduler and aggregate capture sizes are unchanged.

FIA query and KV lengths are refreshed through the existing attention
task-update API. A dummy sequence consumes token padding; its output is
discarded and its KV slots are `-1`. Both real and dummy block-table rows
have graph-owned storage. Each
entry owns its attention handles, events, workspaces, and metadata/table
buffers; entries cannot overwrite decode's graph parameters or another
prefill layout's handles at the same token count.

Replay copies current tokens, positions, KV slots, and recurrent-state indices
into private graph buffers. On a mixed/stateful cache miss, active convolution
and SSM rows are saved on device and restored after warmup and capture. The
first replay advances the existing state exactly once. These temporary copies
are released after capture; steady replay uses the normal GDN state cache.

In `FULL`, `full_prefill_graph_max_entries` limits the number of prefill token
capacities. A smaller limit coarsens the capacities and increases padding;
it never reduces token coverage or evicts a graph during serving. A limit of
one uses the scheduler ceiling for all prefills/mixed batches; zero is rejected.
Without an explicit limit, every configured capacity up to the scheduler
budget is retained, together with that budget's ceiling. Graph memory and
startup time increase with the number of retained capacities and input forms.

In `FULL_AND_PIECEWISE`, the default capacity cache has room for every configured token bucket
(at least 8 entries). Unsupported geometries retain the earlier
layout-specialized path, limited to 8 entries by default. New entries beyond
the configured limit use PIECEWISE without evicting existing graphs.
Set `full_prefill_graph_max_entries` in
`--additional-config` to change this non-negative limit; `0` disables FULL
prefill capture. The first use of an admitted layout includes warmup and
capture, so performance validation should distinguish capture from replay.

This preserves the standard decode-only FULL
dispatch invariant described in [vLLM #55123](https://github.com/vllm-project/vllm/pull/55123).
The next metadata backend should consume the device-side FIA tiling interface
in [Ascend #15336](https://github.com/vllm-project/vllm-ascend/pull/15336), whose
custom operators are not present in this checkout. GDN metadata is already
device-driven; FIA still uses host task updates. Persistent per-entry buffer ownership
follows the same constraint highlighted by
[Ascend #15246](https://github.com/vllm-project/vllm-ascend/pull/15246).

Continuation here uses `mamba_cache_mode="none"` and the existing single state
anchor. All-mode prefix caching, block checkpoints, and separate read/write
anchors from [vLLM #54637](https://github.com/vllm-project/vllm/pull/54637) and
[#26807](https://github.com/vllm-project/vllm/pull/26807) are not implemented;
their metadata is rejected by this capture path until the corresponding
Ascend kernels and builder integration are available. No alternate block or
checkpoint manager is introduced. CPU contract tests cover routing, capture
rollback, stateful GDN slicing/writeback, request arrivals within one captured
bucket, and attention metadata ownership. The Triton state/output kernels
also run under CPU pointer emulation against a recurrent numerical reference
with poisoned padding. This does not validate Triton compilation on Ascend:
NPU arithmetic, capture legality, and performance remain unvalidated.
Single-token recurrence tests also cover 4, 10, and 64 consecutive updates
with reordered requests, cache-slot reuse, grouped heads, and strided state
and gate tensors, including exact preservation of inactive slots.
Ascend compile/run regressions are in
`tests/ut/ops/a2/test_gdn_full_graph.py`; they cover Boolean/byte predicates and
direct cache access against staged recurrence. The real-weight startup test
`tests/e2e/pull_request/one_card/aclgraph/test_qwen3_5_full_startup.py` also
requires serving replay hits with an unchanged sealed registry.

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
