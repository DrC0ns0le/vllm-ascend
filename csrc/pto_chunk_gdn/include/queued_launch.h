// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <memory>
#include <utility>
#include <vector>

namespace ascend_gdn {

// The task queue copies handlers and may retain those copies after execution.
// Share the ownership container so a successful, single-use submission drops
// tensor references from every copy. Device work remains ordered on the tensors'
// allocation stream; its completion does not require retaining the host handler.
template <typename Tensor, typename Launcher>
auto make_queued_launch(std::vector<Tensor> tensors, Launcher launcher) {
    auto retained = std::make_shared<std::vector<Tensor>>(std::move(tensors));
    return [retained, launcher = std::move(launcher)]() {
        launcher(*retained);
        retained->clear();
        return 0;
    };
}

} // namespace ascend_gdn
