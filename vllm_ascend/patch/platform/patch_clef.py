# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Backport Clef-Flash sidecar detection until vLLM provides it natively."""


def register_clef_config_convertor() -> None:
    # As with DeepSeek-V4, defer config imports until platform registration:
    # global patches can be imported during multiprocessing unpickling.
    from vllm.transformers_utils.model_arch_config_convertor import (
        MODEL_ARCH_CONFIG_CONVERTORS,
        ModelArchConfigConvertorBase,
    )
    from vllm.transformers_utils.repo_utils import file_or_path_exists, get_hf_file_to_dict

    base = MODEL_ARCH_CONFIG_CONVERTORS.get("qwen3_5", ModelArchConfigConvertorBase)
    if getattr(base, "_ascend_clef_convertor", False):
        return

    class AscendClefConfigConvertor(base):
        _ascend_clef_convertor = True

        def __init__(self, hf_config, hf_text_config, revision=None):
            model_path = hf_config.name_or_path
            if (
                hf_config.architectures == ["Qwen3_5ForConditionalGeneration"]
                and model_path
                and file_or_path_exists(model_path, "joint_head_config.json", revision)
            ):
                config = get_hf_file_to_dict("joint_head_config.json", model_path, revision)
                if not isinstance(config, dict):
                    raise ValueError("Clef-Flash joint_head_config.json must contain a JSON object")
                hf_config.architectures = ["ClefFlashForJointSchema"]
                hf_config.joint_head_config = config
            super().__init__(hf_config, hf_text_config, revision)

    MODEL_ARCH_CONFIG_CONVERTORS["qwen3_5"] = AscendClefConfigConvertor
