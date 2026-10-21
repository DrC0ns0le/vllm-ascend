# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Load Clef's separate head without filtering it through the backbone index.

vLLM 0.30 applies model.safetensors.index.json even to explicitly selected
secondary weights. Upstream now skips that filtering for allow-pattern
overrides; limit this compatibility patch to Clef's single sidecar.
"""

from functools import wraps
from pathlib import Path

from vllm.model_executor.model_loader.default_loader import (
    DefaultModelLoader,
    download_weights_from_hf,
    maybe_download_from_modelscope,
)

from vllm_ascend.pooling.clef import JOINT_HEAD_WEIGHTS


def clef_weight_preparer(original):
    @wraps(original)
    def prepare(self, model_name_or_path, subfolder, revision, fall_back_to_pt, allow_patterns_overrides):
        if allow_patterns_overrides != [JOINT_HEAD_WEIGHTS]:
            return original(self, model_name_or_path, subfolder, revision, fall_back_to_pt, allow_patterns_overrides)
        model_path = maybe_download_from_modelscope(model_name_or_path, revision) or model_name_or_path
        if Path(model_path).is_dir():
            folder = model_path
        else:
            folder = download_weights_from_hf(
                model_path,
                self.load_config.download_dir,
                allow_patterns_overrides,
                revision,
                subfolder=subfolder,
                ignore_patterns=self.load_config.ignore_patterns,
            )
        directory = Path(folder) / subfolder if subfolder else Path(folder)
        weight = directory / JOINT_HEAD_WEIGHTS
        if not weight.is_file():
            raise FileNotFoundError(f"Clef-Flash decision head is missing: {weight}")
        return str(directory), [str(weight)], True

    return prepare


DefaultModelLoader._prepare_weights = clef_weight_preparer(DefaultModelLoader._prepare_weights)
