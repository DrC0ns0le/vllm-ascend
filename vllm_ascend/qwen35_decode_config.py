# SPDX-License-Identifier: Apache-2.0
"""Opt-in, shape-bounded BF16/FP16 decode optimizations; no device dependencies."""

from dataclasses import dataclass

PROJECTIONS = frozenset(
    ("gate_up_proj", "down_proj", "in_proj_qkvz", "in_proj_ba", "out_proj", "qkv_proj", "o_proj", "lm_head")
)
MAX_DECODE_BATCH = 8


@dataclass(frozen=True)
class Qwen35DecodeConfig:
    linear_backend: str = "native"
    linear_projections: tuple[str, ...] = ("gate_up_proj", "down_proj")
    max_batch_size: int = MAX_DECODE_BATCH
    split_k: int = 1
    fuse_gdn_ba_prepare: bool = False

    @property
    def enabled(self) -> bool:
        return self.linear_backend != "native" or self.fuse_gdn_ba_prepare

    @classmethod
    def from_config(cls, vllm_config, weight_nz_mode: int):
        values = (vllm_config.additional_config or {}).get("qwen35_decode", {})
        if not isinstance(values, dict):
            raise ValueError("qwen35_decode must be a JSON object")
        unknown = set(values) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown qwen35_decode options: {sorted(unknown)}")
        values = dict(values)
        if "linear_projections" in values:
            names = values["linear_projections"]
            if not isinstance(names, (list, tuple)) or any(
                not isinstance(name, str) or name not in PROJECTIONS for name in names
            ):
                raise ValueError(f"qwen35_decode.linear_projections must select from {sorted(PROJECTIONS)}")
            values["linear_projections"] = tuple(names)
        result = cls(**values)
        if result.linear_backend not in ("native", "triton", "triton_cube"):
            raise ValueError("qwen35_decode.linear_backend must be native, triton, or triton_cube")
        if type(result.max_batch_size) is not int or not 1 <= result.max_batch_size <= MAX_DECODE_BATCH:
            raise ValueError("qwen35_decode.max_batch_size must be an integer in [1, 8]")
        if type(result.split_k) is not int or result.split_k not in (1, 2, 4):
            raise ValueError("qwen35_decode.split_k must be 1, 2, or 4")
        if type(result.fuse_gdn_ba_prepare) is not bool:
            raise ValueError("qwen35_decode.fuse_gdn_ba_prepare must be a boolean")
        if not result.enabled:
            return result
        model = vllm_config.model_config
        text = getattr(model, "hf_text_config", None)
        parallel = vllm_config.parallel_config
        if (
            getattr(text, "model_type", "") not in ("qwen3_5", "qwen3_5_text")
            or (getattr(text, "hidden_size", None), getattr(text, "num_hidden_layers", None))
            not in ((2048, 24), (2560, 32))
            or str(model.dtype) not in ("torch.bfloat16", "torch.float16")
            or model.quantization is not None
            or any(
                getattr(parallel, field, 1) != 1
                for field in (
                    "tensor_parallel_size",
                    "pipeline_parallel_size",
                    "prefill_context_parallel_size",
                    "decode_context_parallel_size",
                )
            )
            or getattr(vllm_config, "lora_config", None) is not None
        ):
            raise ValueError(
                "qwen35_decode requires dense Qwen3.5-2B/4B BF16/FP16, TP/PP/CP=1, without quantization or LoRA"
            )
        if weight_nz_mode == 2:
            raise ValueError(
                "qwen35_decode requires ND BF16/FP16 weights: use weight_nz_mode=0 or 1 (no duplicate weights)"
            )
        return result

    def uses_linear(self, prefix: str) -> bool:
        return self.linear_backend != "native" and prefix.rsplit(".", 1)[-1] in self.linear_projections

    def uses_ba_prepare(self, prefix: str) -> bool:
        return self.fuse_gdn_ba_prepare and prefix.endswith(".in_proj_ba")

    def accepts_batch(self, rows: int) -> bool:
        return 1 <= rows <= self.max_batch_size


def is_qwen35_projection(prefix: str, n: int, k: int) -> bool:
    """Exclude vision/projector weights even when they share a suffix."""
    role = prefix.rsplit(".", 1)[-1]
    shapes = {
        "gate_up_proj": ((12288, 2048), (18432, 2560)),
        "down_proj": ((2048, 6144), (2560, 9216)),
        "in_proj_qkvz": ((8192, 2048), (12288, 2560)),
        "in_proj_ba": ((32, 2048), (64, 2560)),
        "out_proj": ((2048, 2048), (2560, 4096)),
        "qkv_proj": ((5120, 2048), (10240, 2560)),
        "o_proj": ((2048, 2048), (2560, 4096)),
        "lm_head": ((248320, 2048), (248320, 2560)),
    }
    return (n, k) in shapes.get(role, ()) and not any(part in prefix for part in ("visual.", "vision_tower."))
