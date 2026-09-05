# SPDX-License-Identifier: Apache-2.0
"""Native ABI/lifetime tests with real CPU Torch and a simulated NPU queue.

This compiles the real bridge against small NPU API test doubles. It proves
argument packing and queued tensor retention, not ACL runtime correctness.
"""

import gc
import shutil
from pathlib import Path

import pytest
import torch
from torch.utils.cpp_extension import is_ninja_available, load


@pytest.fixture(scope="module")
def native_bridge(tmp_path_factory):
    if not is_ninja_available() or not (shutil.which("c++") or shutil.which("clang++")):
        pytest.skip("C++ compiler and Ninja required for native bridge contract")
    root = Path(__file__).resolve().parents[4]
    work = tmp_path_factory.mktemp("pto-queue-native")
    stubs = work / "include/torch_npu/csrc"
    sources = {
        "core/npu/NPUStream.h": """
            #pragma once
            namespace c10_npu {
            struct Stream {
                void* stream() { throw std::runtime_error("unexpected task queue drain"); }
                void* stream(bool empty) {
                    if (empty) throw std::runtime_error("unexpected task queue drain");
                    return reinterpret_cast<void*>(0x1234);
                }
            };
            inline Stream getCurrentNPUStream() { return {}; }
            }
        """,
        "core/npu/NPUGuard.h": """
            #pragma once
            namespace c10_npu { struct NPUGuard { explicit NPUGuard(c10::Device) {} }; }
        """,
        "framework/OpCommand.h": """
            #pragma once
            #include <functional>
            #include <vector>
            namespace test_queue {
            inline std::vector<std::function<int()>> pending;
            inline void drain() {
                auto jobs = std::move(pending);
                pending.clear();
                for (const auto& job : jobs) job();
            }
            }
            namespace at_npu::native {
            struct OpCommand {
                std::function<int()> job;
                void Name(const char*) {}
                template<class F> void SetCustomHandler(F callback) { job = callback; }
                void Run() { test_queue::pending.push_back(job); }
            };
            }
        """,
    }
    for name, content in sources.items():
        path = stubs / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    parameters = ", ".join(f"void* p{i}" for i in range(28))
    probe = f"""
        void probe(uint32_t blocks, void* stream, {parameters},
                   int64_t sequences, int64_t tokens, int64_t total, uint32_t matrices) {{
            TORCH_CHECK(stream == reinterpret_cast<void*>(0x1234));
            TORCH_CHECK(total == tokens);
            *static_cast<float*>(p9) = *static_cast<float*>(p0) + *static_cast<float*>(p27)
                                      + blocks + sequences + tokens + matrices;
        }}
    """
    bridge_source = (root / "csrc/pto_chunk_gdn/queue_bridge.cpp").read_text()
    bridge_source = bridge_source.replace("PYBIND11_MODULE(", probe + "\nPYBIND11_MODULE(")
    bridge_source = bridge_source.replace(
        'module.def("enqueue", &enqueue);',
        """
        module.def("enqueue", &enqueue);
        module.def("drain", &test_queue::drain);
        module.def("pending", []() { return test_queue::pending.size(); });
        module.def("probe_address", []() { return reinterpret_cast<uintptr_t>(&probe); });
    """,
    )
    source = work / "bridge_test.cpp"
    source.write_text(bridge_source)
    return load(
        name="pto_queue_contract",
        sources=[str(source)],
        extra_include_paths=[str(work / "include")],
        build_directory=str(work),
        verbose=False,
    )


def test_native_queue_packs_abi_and_retains_tensors_until_submission(native_bridge):
    buffers = [torch.full((1,), float(i)) for i in range(28)]
    output = buffers[9]
    native_bridge.enqueue(native_bridge.probe_address(), 2, buffers, 3, 180, 32)
    assert native_bridge.pending() == 1
    assert output.item() == 9  # Still queued, not launched directly.
    assert all(tensor._use_count() >= 2 for tensor in buffers)
    del buffers
    gc.collect()
    native_bridge.drain()
    assert output.item() == 0 + 27 + 2 + 3 + 180 + 32
    assert native_bridge.pending() == 0
    assert output._use_count() == 1


def test_native_queue_rejects_incorrect_buffer_count(native_bridge):
    with pytest.raises(RuntimeError, match="28 buffers"):
        native_bridge.enqueue(native_bridge.probe_address(), 1, [torch.zeros(1)] * 27, 1, 1, 1)
    assert native_bridge.pending() == 0
