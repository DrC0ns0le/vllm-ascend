// SPDX-License-Identifier: Apache-2.0
// Queue the PTO launch with surrounding PyTorch NPU operations (TASK_QUEUE=1).
#include <torch/extension.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#include <torch_npu/csrc/core/npu/NPUGuard.h>
#include <array>
#include <cstdint>
#include <utility>
#include <type_traits>
#include <vector>

namespace {
constexpr size_t NUM_BUFFERS = 28;
template <size_t... I>
void launch(uintptr_t address, uint32_t blocks, void* stream,
            const std::array<void*, NUM_BUFFERS>& pointers,
            int64_t sequences, int64_t tokens, uint32_t matrices,
            std::index_sequence<I...>) {
    // Use value pointer arguments (not references) for the Bisheng C ABI.
    using Pointer = void*;
    using Kernel = void (*)(uint32_t, void*,
                           std::conditional_t<true, Pointer, decltype(I)>...,
                           int64_t, int64_t, int64_t, uint32_t);
    reinterpret_cast<Kernel>(address)(blocks, stream, pointers[I]...,
                                     sequences, tokens, tokens, matrices);
}
}

void enqueue(uintptr_t address, uint32_t blocks, std::vector<at::Tensor> buffers,
             int64_t sequences, int64_t tokens, uint32_t matrices) {
    TORCH_CHECK(address != 0 && blocks > 0 && sequences > 0 && tokens > 0);
    TORCH_CHECK(buffers.size() == NUM_BUFFERS, "MegaGDN ABI requires 28 buffers");
    c10_npu::NPUGuard guard(buffers.front().device());
    // stream() drains the host task queue in Torch-NPU 2.10. The launch below
    // is itself enqueued, so capture its handle without draining that queue.
    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    for (const auto& tensor : buffers) {
        TORCH_CHECK(tensor.device() == buffers.front().device() && tensor.is_contiguous());
    }
    at_npu::native::OpCommand command;
    command.Name("PTO_MegaGDN");
    // Retain all input/workspace tensors until the queued launch is submitted.
    command.SetCustomHandler([address, blocks, buffers = std::move(buffers), stream,
                              sequences, tokens, matrices]() -> int {
        std::array<void*, NUM_BUFFERS> pointers;
        for (size_t i = 0; i < NUM_BUFFERS; ++i) pointers[i] = buffers[i].data_ptr();
        launch(address, blocks, stream, pointers, sequences, tokens, matrices,
               std::make_index_sequence<NUM_BUFFERS>{});
        return 0;
    });
    command.Run();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("enqueue", &enqueue);
}
