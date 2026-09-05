# Qwen3.5-2B: breakable ACLGraph + MegaGDN on 910B

This is an **offline implementation, pending NPU validation**. Both paths are implemented. No 910B runtime, model weights, production corpus, CANN compiler, or remote working tree was accessible during development. CPU contracts and source inspection do not establish numerical correctness, graph replay correctness, latency improvement, or production readiness.

The target remains BF16, TP/DP=1, MRV1, four concurrent requests, prefix caching off, Mamba cache mode `none`, and chunked prefill off. `TASK_QUEUE_ENABLE=1` stays enabled. No MRV2, speculative decoding, MTP, quantization, or FULL prefill implementation is included.

## Source and implementation

The Ascend base is `4095369c34722c24c694d2461f4075f9bed3cbc0`, branch `perf/dsv4-flash-dspark-aclgraph-v026`. The local tree was clean before changes. The companion vLLM patch was generated against `d02df748bf9efd99022f1a062597dc3cb3808485`; the runtime's actual delta is unknown.

References read:

- [Ascend #12744](https://github.com/vllm-project/vllm-ascend/pull/12744): MRV1 wrapper, workspace ownership, NPU replay ordering and CUDA compatibility mappings. Unrelated models, drafter wrapping and MRV2 changes were excluded.
- [vLLM #54361](https://github.com/vllm-project/vllm/pull/54361): wrap the Qwen custom-op registrations with `eager_break_during_capture`. The exact reference commit has only the standard registration. The local-copy helper supports both standard and packed registrations when present.
- [Ascend #8872](https://github.com/vllm-project/vllm-ascend/pull/8872): experimental PTO device kernels and launcher layouts. Device code is retained; integration is scoped to Ascend GDN prefills, without its global worker monkeypatch.
- [Ascend #11161](https://github.com/vllm-project/vllm-ascend/pull/11161): reviewed as background. This checkout already passes causal-convolution metadata as device tensors; no additional causal-convolution backport was made.

`BreakableACLGraphWrapper` supports FULL decode and PIECEWISE prefill/mixed dispatch without the legacy compilation route. The existing GDN `UNIFORM_BATCH` capability remains unchanged. Fresh one-token prompts are excluded from uniform FULL decode dispatch when breakable graphs are enabled. Successful FULL capture retains the flag needed by MRV1's subsequent attention-update decision; capture failure restores it.

MegaGDN replaces only the already-separated prefill chunk call. The existing recurrent decode kernels, prefill state indexing, transposition and state writeback remain in place. Eligibility uses CPU sequence-length upper bounds and rebased prefill boundaries. Unknown or stateful metadata, unsupported topology, layout, dimensions, dtype and hardware fall back with a reason. Compilation failures propagate when the experimental backend is explicitly enabled.

The model remains BF16. PTO uses FP16 q/k/v/beta and recurrent intermediates, with FP32 gates; its final state is converted back to the baseline state dtype. This conversion does not recover precision lost inside the kernel. Numerical and application accuracy gates remain mandatory.

The new queue bridge uses `OpCommand.SetCustomHandler`, following this repository's existing C++ dispatch pattern. It retains tensor arguments until submission and places the raw PTO launch in the same task queue as preceding casts and following consumers. It obtains the stream with `stream(false)`: the default accessor drains the host queue in [Torch-NPU 2.10's implementation](https://github.com/Ascend/pytorch/blob/v2.10.0/torch_npu/csrc/core/npu/NPUStream.cpp). Actual queue/stream behavior still needs NPU validation.

Kernel handles and small masks are owned by each layer's backend. Workspaces retain the reference implementation's per-call allocations and clears. No unmeasured reusable-workspace optimization was added. Chunk counts use existing host boundaries; no device-to-host boundary copy or recurrent-state scan is introduced.

## Backup and installation workflow

The delivery includes a local backup manifest, pre-change status/diffs, per-file originals, SHA256 hashes, a source map, and separate logical patches under the results directory linked in the task handoff. These are **local/reference-source backups**, not backups of the inaccessible runtime.

Before deploying, record `git status --short`, `git branch --show-current`, `git rev-parse HEAD`, `git diff`, and `git diff --cached` for **both runtime repositories**. Copy every affected existing runtime file into a separate local directory, retain `.original` copies and SHA256 hashes, and adapt the patches there. Review `diff -u` before copying modified files back. Check that the remote original hash has not changed immediately before copying. Preserve untracked files and unrelated changes.

The companion patch is `patches/0002-vllm-qwen-eager-break.patch`. For a different Qwen source revision, generate a patch from its actual local copy:

```bash
python benchmarks/qwen35_low_latency/patch_qwen.py \
  /local/work/qwen_gdn_linear_attn.py /local/work/qwen-patched
```

This writes an original, modified file, manifest and unified patch into a new directory. It refuses to write under `/vllm-workspace`. Copy its modified file back only after reviewing its diff. The helper is idempotent and rejects unfamiliar registration wrappers.

Do not apply the complete upstream PRs or reset either runtime checkout. For restoration, verify that a deployed file still matches the recorded modified hash, then copy its exact original back. Remove a new file only if its hash still matches the delivered version. Reverse the separate patches in reverse dependency order when their application context is unchanged.

## Build prerequisites

Use the existing `/vllm-workspace/vllm` and `/vllm-workspace/vllm-ascend` editable installations with the existing Torch/Torch-NPU 2.10 and CANN environment. Do not upgrade Transformers. This backend currently requires the source tree; building a standalone distributable wheel with bundled PTO sources is not implemented.

Source the installed CANN environment so `ASCEND_HOME_PATH` identifies the toolkit. The compiler is resolved at `compiler/ccec_compiler/bin/bisheng`. The target architecture is `dav-c220` (910B). The physical cube-core count is read from the selected device; no guessed core count is used.

The build first checks for the optional source dependency at `csrc/third_party/pto-isa/include`. Otherwise it tries the installed CANN PTO headers. Successful compilation is the compatibility check; CANN 9.0.1 header compatibility was not tested here. If the installed headers fail, prepare the exact source dependency outside the live checkout and copy it into that path:

```bash
git clone https://gitcode.com/cann/pto-isa.git /local/work/pto-isa
git -C /local/work/pto-isa checkout --detach 4e27a104f948e883e0bef44670252381bff794c5
```

Verify the repository and commit against PR #8872 before installation. The dependency is not vendored or installed by this patch. The utility header's Clear BSD license is retained in `csrc/pto_chunk_gdn/LICENSE.pto-kernels`.

The first model profile pass prepares the exact H/Hg/D specialization and loads the queue bridge. The numerical script also compiles explicitly before running cases. Shared libraries use a locked, atomic source-hash cache in `.vllm_ascend/pto_chunk_gdn`; the key includes local sources, PTO headers, compiler version and compilation flags. PyTorch's extension cache stores the queue bridge. C++ compilation requires the existing compiler toolchain and Ninja. Compilation duration is logged separately. Complete serving warmup before recording latency.

## Launch A/B/C/D

Run these from the Ascend checkout to print the complete commands. Add `--execute` to validate the installed CLI's advertised flags, switch to `/workspace`, and directly execute `vllm serve`.

```bash
python benchmarks/qwen35_low_latency/serve.py A --execute
python benchmarks/qwen35_low_latency/serve.py B --execute
python benchmarks/qwen35_low_latency/serve.py C --execute
python benchmarks/qwen35_low_latency/serve.py D --execute
```

Run one server at a time. Stop only the server started for this experiment before switching variants. The launcher does not terminate existing processes.

| Variant | Graph path | Prefill GDN | Breakable switch | PTO switch |
| --- | --- | --- | --- | --- |
| A | FULL_DECODE_ONLY | Current | 0 | 0 |
| B | FULL_AND_PIECEWISE | Current | 1 | 0 |
| C | FULL_DECODE_ONLY | MegaGDN when eligible | 0 | 1 |
| D | FULL_AND_PIECEWISE | MegaGDN when eligible | 1 | 1 |

All variants explicitly set BF16, TP/DP=1, max-num-seqs=4, max-model-len=1536, no prefix caching, no chunked prefill, `mamba_cache_mode=none`, MRV1 and task queue 1. The breakable environment is set **before process import**. B/D explicitly use compilation mode 0 to avoid legacy FX segmentation.

Profile 1536 uses capture sizes `1,2,3,4,64,128,192,256,384,512,768,1024,1536`. `--profile 4096` sets the scheduling budget to 4096 and adds `2048,3072,4096`. A/C capture only FULL decode sizes 1–4. The vLLM dispatcher may capture both FULL and PIECEWISE entries for the small sizes; count actual descriptors rather than assuming the list length is the entry count.

Inspect the effective startup configuration and `Breakable ACLGraph config` log. Confirm BF16, cache mode `none`, prefix caching false, chunked prefill false, and four request slots. If the branch warns that disabling chunked prefill is unsupported, test its actual scheduler behavior before accepting this launch profile. The metadata-based MegaGDN fallback remains safe for continuing chunks, but this does not establish scheduler correctness.

## Validation commands

Offline contracts (requires pytest, CPU Torch 2.10, NumPy, regex, filelock, a C++ compiler and Ninja; the helper bypasses runtime package initializers):

```bash
python benchmarks/qwen35_low_latency/offline_tests.py
```

NPU registration/segmentation contract, in a fresh process after applying the companion patch:

```bash
VLLM_USE_V2_MODEL_RUNNER=0 VLLM_USE_BREAKABLE_CUDAGRAPH=1 TASK_QUEUE_ENABLE=1 \
  python benchmarks/qwen35_low_latency/graph_contract.py
```

This invokes the actual registered standard Qwen custom op with a synthetic core. It checks mutable context re-entry across `1×192`, `2×96`, `3×64`, `4×48`, `2×90`, P133 and P1 sharing a 192-token capture. It also checks that FULL capture ignores eager breaks. It does not validate real GDN math or full-attention metadata.

Direct numerical gate, without running a server:

```bash
TASK_QUEUE_ENABLE=1 python benchmarks/qwen35_low_latency/numerical.py \
  --model /dev/shm/Qwen3_5-2B --output /workspace/megagdn-numerical.json
```

The script reads H/Hg/D from the checkpoint config, compares output and final recurrent state at every requested boundary, includes multiple partial final chunks and multi-sequence cases, and repeats three random seeds. It reports finite checks, max and p50/p95/p99 absolute/relative errors. Relative error uses a 1e-3 denominator floor. Default 0.05 absolute/relative tolerances are provisional screening limits; passing them is not a task-quality acceptance decision.

Serving matrix; repeat for each variant, and run the primary suite at concurrency 1, 2 and 4:

```bash
python benchmarks/qwen35_low_latency/benchmark.py --variant B --suite primary \
  --concurrency 1 --repetitions 100 --output /workspace/B-c1
python benchmarks/qwen35_low_latency/benchmark.py --variant B --suite boundaries \
  --repetitions 4 --output /workspace/B-boundaries
python benchmarks/qwen35_low_latency/benchmark.py --variant B --suite concurrent \
  --repetitions 25 --output /workspace/B-concurrent
python benchmarks/qwen35_low_latency/benchmark.py --variant B --suite mixed \
  --repetitions 25 --output /workspace/B-mixed
python benchmarks/qwen35_low_latency/benchmark.py --variant D --suite stress \
  --repetitions 1 --output /workspace/D-stress
```

The stress suite issues 1000 requests at concurrency four per repetition. Content alternates even at identical lengths. Synthetic prompts use exact token-ID lengths, and output lengths are fixed for latency comparison. A first-token event releases each new mixed-test request while earlier requests are generating. Longer 32/24/16-token anchor outputs make overlap more likely; the production benchmarks still use O1/O4/O8.

**Client overlap is not proof of a mixed scheduler batch.** In a separate DEBUG run, collect the server log and require current GDN metadata showing both prefill and decode:

```bash
python benchmarks/qwen35_low_latency/inspect_log.py /workspace/B-debug.log --require-mixed
```

Also inspect replay descriptors for 4→3→2→1 transitions. Retry arrival timing/output lengths if the scheduler did not form the required batch. Do not label such a run as covered. Keep DEBUG logging off during latency measurements.

P1536 plus any positive output exceeds max-model-len=1536. The serving harness explicitly reports that case as outside the context budget. P1536 is covered by the direct numerical script; positive-output serving at that boundary requires a separately identified larger-context validation profile. Do not silently shorten the prompt or mislabel its length.

Production corpus JSONL accepts `{"prompt": "fully formatted prompt", "max_tokens": 8}` or token IDs. Corpus mode permits natural EOS and records the actual generated length:

```bash
python benchmarks/qwen35_low_latency/benchmark.py --variant A --suite corpus \
  --corpus /workspace/query-correction.jsonl --repetitions 5 --concurrency 4 \
  --output /workspace/A-corpus
python benchmarks/qwen35_low_latency/compare.py \
  /workspace/A-corpus/requests.jsonl /workspace/D-corpus/requests.jsonl
```

Run repeated A/B/C/D corpus evaluations. The comparison reports held-out baseline repeatability, exact match against repeated baseline outputs, and token-sequence divergence. Supply `--valid-format` for a task-specific full-match regex and separately score correction quality. No production corpus or quality rubric was supplied here.

## Profiling and acceptance

Profile A, B and D at P128/O1 and P128/O8. Use the existing runtime's Torch-NPU/msprof workflow and preserve original trace files. Separate direct eager launches from launches inside graph replay. Collect NPU busy time and wall span, graph submissions, eager-break count, host ACL/API calls and time, tiny kernels, GDN, full-attention, layout/copy and workspace costs. `inspect_log.py` reports logical graph submissions only; it does not manufacture kernel or device-time metrics.

Capture logs report each descriptor's graph/eager-break counts and duration. With only 18 GDN breaks, about 19 graph segments are expected. Existing upstream attention decorators can add breaks; this patch adds no full-attention break. Explain any count other than the observed architecture's expectation, especially zero GDN breaks. Logs also record free memory before/after complete capture for the 1536 and 4096 profiles. Measure process startup separately from graph capture and PTO compilation.

For the 1000-request stress run, sample NPU memory with the runtime's monitoring tools after warmup and throughout the run. Confirm graph-entry, kernel-handle and workspace memory do not grow per request. Per-call workspace allocation is retained and needs profiling. Do not claim a memory-stability pass from bounded Python counters alone.

| Required result | Status in this delivery |
| --- | --- |
| Offline routing, metadata, fallback and measurement contracts | Run locally; see validation log |
| Compiled queue bridge ABI, deferred submission and tensor lifetime with CPU Torch and NPU API test doubles | Passed offline; actual NPU queue unverified |
| Bisheng/PTO and real Torch-NPU bridge build | Not run |
| Real-weight graph replay and c1/c2/c4 correctness | Not run |
| Mixed scheduler batches and state isolation | Not run |
| MegaGDN output/final-state numerical equivalence | Not run |
| Production quality relative to baseline nondeterminism | Not run |
| A/B/C/D TTFT, TPOT, E2E and throughput | Unmeasured |
| Launch/API reduction and remaining TTFT bottleneck | Unmeasured |
| Capture memory/startup and 1000-request memory stability | Unmeasured |

The benchmark summaries report per-request TTFT p50/p95/p99, E2E p50/p95/p99, TPOT p50/p95, and request throughput. They never divide batch duration by request count and label it latency.

The requested engineering conclusion remains open: graph-only savings, MegaGDN-only savings, combined savings and their interaction cannot be quantified without A/B/C/D measurements. Neither concurrency correctness nor sub-100-ms O4 is established. Given the supplied baseline of approximately 8 ms per decode token, the arithmetic target remains TTFT below roughly 76 ms for O4; that is a target, not a result. Next decisions must follow the graph-only profile, then the combined profile. Workspace reuse and further kernel/metadata optimizations are deferred until those profiles justify them.
