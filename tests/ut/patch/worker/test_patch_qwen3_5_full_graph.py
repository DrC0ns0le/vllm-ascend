# SPDX-License-Identifier: Apache-2.0

import pytest
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import QwenGatedDeltaNetAttention

from vllm_ascend.ops.gdn import AscendGatedDeltaNetAttention
from vllm_ascend.patch.worker import patch_qwen3_5
from vllm_ascend.utils import is_310p


@pytest.mark.skipif(is_310p(), reason="310P uses its own GDN implementation")
def test_real_qwen_layer_exposes_ascend_full_graph_hook():
    assert patch_qwen3_5._GDN_PATCH_TARGET is QwenGatedDeltaNetAttention
    assert QwenGatedDeltaNetAttention._forward_core is AscendGatedDeltaNetAttention._forward_core
    assert QwenGatedDeltaNetAttention._forward_full_graph is AscendGatedDeltaNetAttention._forward_full_graph
    layer = QwenGatedDeltaNetAttention.__new__(QwenGatedDeltaNetAttention)
    assert layer._forward_full_graph.__self__ is layer
