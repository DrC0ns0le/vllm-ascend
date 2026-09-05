# Offline validation — 5 September 2026

The implementation targets MRV1 breakable ACLGraph plus optional PTO MegaGDN on Ascend 910B. The available development machine is macOS without Ascend hardware, CANN, Torch-NPU, a live vLLM installation or model weights. The user explicitly authorized implementing from source references without runtime access.

## Executed checks

| Check | Result | Scope |
| --- | --- | --- |
| `python benchmarks/qwen35_low_latency/offline_tests.py` | 61 passed | CPU contracts, including compiled C++ bridge with NPU API test doubles |
| Changed-file Ruff and repository source checks | Passed | Python style, imports, logger, package initializers and structural checks |
| Codespell, typos and Markdown checks | Passed | Repository hooks in an isolated source snapshot |
| `git diff --check` | Passed | Whitespace in tracked changes |
| Logical patch application and reversal | See delivery verification log | Exact local Ascend base and reference vLLM file |
| `bash format.sh ci` | Failed for existing formatting and unavailable host tools | See explanation below |

The offline suite executes the actual extracted MRV1 routing method and GDN mixed-batch core using explicit CPU test doubles. It checks fresh P1 routing, prefill/decode separation, rebased sequence boundaries, initial state clearing, final state writeback and untouched cache slots. It also checks backend eligibility and fallback arguments, dtype conversions, compile-cache invalidation and atomic failure handling, wrapper capture flags and replay ordering, and benchmark timing and repeated-output comparison.

The native test compiles the delivered `queue_bridge.cpp` against CPU Torch 2.10.0 and test implementations of the NPU APIs. It checks all 28 pointer arguments and scalar ABI arguments, deferred submission, tensor retention until submission, and use of the stream accessor that does not drain the queue. It does **not** compile against real Torch-NPU headers or execute a PTO kernel.

The full formatting command ran against a temporary copy with a committed clean source base and only this task's changes staged. Ruff changed six unrelated existing files: `test_dspark_deepseekv4.py`, `test_kv_delivery_preemption.py`, `patch_balance_schedule.py`, `patch_kv_delivery_preemption.py`, `dspark_proposer.py` and `llm_base_proposer.py`. Those changes were not copied into the working repository. The Gitleaks hook's downloaded Linux binary cannot execute on macOS, and ShellCheck is unavailable. No secret-scan or ShellCheck pass is claimed. All other hooks passed; changed task files required no changes in that final full run.

## Unexecuted acceptance gates

| Gate | Status |
| --- | --- |
| Real Torch-NPU queue bridge and Bisheng/PTO build | Not run |
| Real-weight service startup and inference | Not run |
| PIECEWISE segmentation, replay and request-aware eager re-entry | Not run on NPU |
| FULL decode transitions 4→3→2→1 | Not run on NPU |
| Arbitrary-length, c1/c2/c4 and mixed-batch correctness | Not run on NPU |
| MegaGDN output and final recurrent-state errors | Not run |
| Repeated production-corpus accuracy | Not run; corpus unavailable |
| A/B/C/D per-request latency and throughput | Unmeasured |
| Launch count, host API time and NPU profile | Unmeasured |
| Graph capture duration and memory for both bucket profiles | Unmeasured |
| 1000-request memory stability | Unmeasured |

The delivery directory contains CSV/JSON result templates with explicit `not_run` status, launch commands for all four variants and both scheduling budgets, source references, originals and hashes, and raw local check logs. Empty measurement cells represent missing measurements, never zero latency or zero errors.

## Engineering conclusion

Both mechanisms are implemented and can be reviewed and deployed using the separate patches. The implementation retains BF16 model execution, existing recurrent decode and state writeback, and reasoned fallbacks for unsupported MegaGDN prefills. The baseline switches remain available for all four experiment variants.

Breakable-only savings, MegaGDN-only savings, combined savings, composition of benefits and the remaining TTFT bottleneck cannot be quantified from offline tests. Concurrency correctness and sub-100-ms O4 have not been established. The supplied approximately 8 ms/token baseline implies a TTFT target below approximately 76 ms for O4; it is not a measured result of this work.

Run the numerical and graph contracts first on 910B, then the real-weight correctness and A/B/C/D matrix in the [runbook](README.md). Keep the experimental PTO path disabled for production until those gates pass. Workspace reuse and further metadata optimizations require profiling evidence and were not introduced.
