# SPDX-License-Identifier: Apache-2.0
"""Exercise the real C++ queue ownership helper without requiring Torch-NPU."""

import shutil
import subprocess
from pathlib import Path

import pytest


def test_copied_queue_handlers_release_buffers_after_submission(tmp_path):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("Requires a C++ compiler")
    source = tmp_path / "queue_lifetime.cpp"
    source.write_text(
        r"""
#include "queued_launch.h"
#include <cassert>
#include <functional>
#include <stdexcept>

int main() {
    using Tensor = std::shared_ptr<int>;
    auto input = std::make_shared<int>(7);
    auto output = std::make_shared<int>(0);
    std::weak_ptr<int> input_lifetime = input;
    int submissions = 0;
    std::function<int()> handler = ascend_gdn::make_queued_launch(
        std::vector<Tensor>{input, output}, [&](const auto& tensors) {
            assert(tensors.size() == 2 && !input_lifetime.expired());
            *tensors[1] = *tensors[0] * 3;
            ++submissions;
        });
    auto queue_copy = handler;
    auto release_queue_copy = handler;
    input.reset();
    assert(!input_lifetime.expired());
    assert(queue_copy() == 0 && submissions == 1);
    // All callback objects are still alive, but none retains input storage.
    assert(input_lifetime.expired());
    assert(*output == 21 && output.use_count() == 1);

    auto failed_input = std::make_shared<int>(5);
    std::weak_ptr<int> failed_lifetime = failed_input;
    std::function<int()> failed = ascend_gdn::make_queued_launch(
        std::vector<Tensor>{failed_input}, [](const auto&) {
            throw std::runtime_error("submission failed");
        });
    auto failed_copy = failed;
    failed_input.reset();
    try {
        failed();
        assert(false);
    } catch (const std::runtime_error&) {
        // An unsuccessful submission must not claim its buffers are released.
        assert(!failed_lifetime.expired());
    }
    failed = nullptr;
    assert(!failed_lifetime.expired());
    failed_copy = nullptr;
    assert(failed_lifetime.expired());
}
"""
    )
    include = Path(__file__).resolve().parents[4] / "csrc/pto_chunk_gdn/include"
    binary = tmp_path / "queue_lifetime"
    subprocess.run(
        [compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror", "-I", str(include), str(source), "-o", str(binary)],
        check=True,
        capture_output=True,
        timeout=60,
    )
    subprocess.run([str(binary)], check=True, capture_output=True, timeout=10)
