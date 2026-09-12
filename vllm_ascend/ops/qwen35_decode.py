# SPDX-License-Identifier: Apache-2.0
"""Graph-visible boundaries and load-time selection for Qwen3.5 decode kernels."""

import torch
from vllm.utils.torch_utils import direct_register_custom_op

from vllm_ascend.qwen35_decode_config import Qwen35DecodeConfig, is_qwen35_projection


def qwen35_decode_linear(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None, split_k: int, force_cube: bool
) -> torch.Tensor:
    # Lazy import keeps non-Triton worker configurations usable.
    from vllm_ascend.ops.triton.qwen35_decode_linear import decode_linear

    return decode_linear(x, weight, bias, split_k, force_cube)


def qwen35_decode_linear_fake(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None, split_k: int, force_cube: bool
) -> torch.Tensor:
    return x.new_empty((x.shape[0], weight.shape[0]))


def qwen35_gdn_ba_prepare(
    x: torch.Tensor, qkvz: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None, qkv_size: int, head_dim: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    from vllm_ascend.ops.triton.qwen35_gdn_prepare import gdn_ba_prepare

    return gdn_ba_prepare(x, qkvz, weight, bias, qkv_size, head_dim)


def qwen35_gdn_ba_prepare_fake(
    x: torch.Tensor, qkvz: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None, qkv_size: int, head_dim: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    m = x.shape[0]
    heads = weight.shape[0] // 2
    return (
        x.new_empty((m, qkv_size)),
        x.new_empty((m, heads, head_dim)),
        x.new_empty((m, heads)),
        x.new_empty((m, heads)),
    )


direct_register_custom_op(
    op_name="qwen35_decode_linear",
    op_func=qwen35_decode_linear,
    fake_impl=qwen35_decode_linear_fake,
    mutates_args=[],
    dispatch_key="PrivateUse1",
)
direct_register_custom_op(
    op_name="qwen35_gdn_ba_prepare",
    op_func=qwen35_gdn_ba_prepare,
    fake_impl=qwen35_gdn_ba_prepare_fake,
    mutates_args=[],
    dispatch_key="PrivateUse1",
)


def configure_decode_layer(layer: torch.nn.Module) -> None:
    # Called once after loading, never in graph replay. Do not keep a second
    # packed copy of each weight: both the native and Triton paths share ND.
    from vllm_ascend.ascend_config import get_ascend_config
    from vllm_ascend.utils import ACL_FORMAT_FRACTAL_ND, AscendDeviceType, get_ascend_device_type

    layer._ascend_decode_config = None
    config = getattr(get_ascend_config(), "qwen35_decode", None)
    if not isinstance(config, Qwen35DecodeConfig) or not config.enabled:
        return
    if get_ascend_device_type() != AscendDeviceType.A2:
        raise ValueError("qwen35_decode kernels currently target Ascend 910B (A2)")
    if layer.weight.ndim != 2 or not is_qwen35_projection(layer.prefix, *layer.weight.shape):
        return
    if not (config.uses_linear(layer.prefix) or config.uses_ba_prepare(layer.prefix)):
        return
    if layer.weight.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(f"qwen35_decode requires BF16/FP16 weights: {layer.prefix}")
    # Torch reports logical contiguous strides for NZ as well. Resolve the
    # physical layout explicitly once; never rely on stride checks alone.
    import torch_npu

    layer.weight.data = torch_npu.npu_format_cast(layer.weight.data, ACL_FORMAT_FRACTAL_ND).contiguous()
    layer._ascend_decode_config = config


def use_decode_linear(layer: torch.nn.Module, x: torch.Tensor) -> bool:
    config = getattr(layer, "_ascend_decode_config", None)
    return (
        isinstance(config, Qwen35DecodeConfig)
        and config.uses_linear(layer.prefix)
        and x.ndim == 2
        and config.accepts_batch(x.shape[0])
        and x.dtype in (torch.bfloat16, torch.float16)
        and x.stride(1) == 1
    )


def use_gdn_ba_prepare(layer: torch.nn.Module, x: torch.Tensor) -> bool:
    config = getattr(layer, "_ascend_decode_config", None)
    return (
        isinstance(config, Qwen35DecodeConfig)
        and config.uses_ba_prepare(layer.prefix)
        and x.ndim == 2
        and config.accepts_batch(x.shape[0])
        and x.dtype in (torch.bfloat16, torch.float16)
        and x.stride(1) == 1
    )
