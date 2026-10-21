# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Checkpoint routing and actual v0.30 safetensors loading, without NPU kernels."""

import ast
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from safetensors.torch import save_file
from torch import nn
from transformers import PretrainedConfig
from vllm.config import LoadConfig, VllmConfig
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.model_executor.models.interfaces_base import default_pooling_type
from vllm.model_executor.models.utils import AutoWeightsLoader
from vllm.transformers_utils.model_arch_config_convertor import (
    MODEL_ARCH_CONFIG_CONVERTORS,
    ModelArchConfigConvertorBase,
)

from vllm_ascend.pooling.clef import JOINT_HEAD_WEIGHTS, JointSchemaHead, JointSchemaPooler, embed_lexical

ROOT = Path(__file__).resolve().parents[4]
HEAD_CONFIG = dict(hidden_size=16, width=8, routing_layers=1, layers=1, heads=2, feedforward=24)


def load_patch(path):
    # Import just the patch under test, without the package's unrelated NPU patches.
    spec = importlib.util.spec_from_file_location("clef_test_patch", ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def converter(monkeypatch):
    monkeypatch.setitem(MODEL_ARCH_CONFIG_CONVERTORS, "qwen3_5", ModelArchConfigConvertorBase)
    patch = load_patch("vllm_ascend/patch/platform/patch_clef.py")
    patch.register_clef_config_convertor()
    cls = MODEL_ARCH_CONFIG_CONVERTORS["qwen3_5"]
    patch.register_clef_config_convertor()
    assert MODEL_ARCH_CONFIG_CONVERTORS["qwen3_5"] is cls
    return cls


@pytest.mark.parametrize("with_head", [True, False])
def test_local_checkpoint_detection(converter, tmp_path, with_head):
    config = PretrainedConfig(architectures=["Qwen3_5ForConditionalGeneration"], name_or_path=str(tmp_path))
    if with_head:
        (tmp_path / "joint_head_config.json").write_text(json.dumps(HEAD_CONFIG))
    converted = converter(config, config, "pinned-revision")
    assert converted.revision == "pinned-revision"
    if with_head:
        assert converted.get_architectures() == ["ClefFlashForJointSchema"]
        assert config.joint_head_config == HEAD_CONFIG
    else:
        assert converted.get_architectures() == ["Qwen3_5ForConditionalGeneration"]
        assert not hasattr(config, "joint_head_config")


def test_hub_detection_preserves_revision(converter, monkeypatch):
    exists = Mock(return_value=True)
    read = Mock(return_value=HEAD_CONFIG)
    # The registered converter captures repo helpers at registration time.
    monkeypatch.setattr("vllm.transformers_utils.repo_utils.file_or_path_exists", exists)
    monkeypatch.setattr("vllm.transformers_utils.repo_utils.get_hf_file_to_dict", read)
    monkeypatch.setitem(MODEL_ARCH_CONFIG_CONVERTORS, "qwen3_5", ModelArchConfigConvertorBase)
    load_patch("vllm_ascend/patch/platform/patch_clef.py").register_clef_config_convertor()
    cls = MODEL_ARCH_CONFIG_CONVERTORS["qwen3_5"]
    config = PretrainedConfig(architectures=["Qwen3_5ForConditionalGeneration"], name_or_path="Cloudflare/clef-flash")
    cls(config, config, "commit-sha")
    exists.assert_called_once_with("Cloudflare/clef-flash", "joint_head_config.json", "commit-sha")
    read.assert_called_once_with("joint_head_config.json", "Cloudflare/clef-flash", "commit-sha")


def test_preserves_existing_converter(monkeypatch):
    class Existing(ModelArchConfigConvertorBase):
        def get_hidden_size(self):
            return 123

    monkeypatch.setitem(MODEL_ARCH_CONFIG_CONVERTORS, "qwen3_5", Existing)
    load_patch("vllm_ascend/patch/platform/patch_clef.py").register_clef_config_convertor()
    config = PretrainedConfig(architectures=["Qwen3_5Model"], name_or_path="")
    assert MODEL_ARCH_CONFIG_CONVERTORS["qwen3_5"](config, config).get_hidden_size() == 123


@pytest.fixture
def patched_loader(monkeypatch):
    original = DefaultModelLoader._prepare_weights
    # Record the original for automatic restoration after the import-time patch.
    monkeypatch.setattr(DefaultModelLoader, "_prepare_weights", original)
    module = load_patch("vllm_ascend/patch/worker/patch_clef.py")
    return module, DefaultModelLoader(LoadConfig(load_format="safetensors", use_tqdm_on_load=False))


@pytest.fixture
def adapter(monkeypatch):
    # Execute the actual adapter class with only the expensive backbone replaced.
    # This covers its constructor, pooler, decorators and secondary source wiring.
    class Backbone(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.language_model = nn.Module()
            self.language_model.lm_head = nn.Linear(16, 8, bias=False)
            self.language_model.lm_head.tp_size = 1

    path = ROOT / "vllm_ascend/models/qwen3_5_vl.py"
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "AscendClefFlashForJointSchema")
    namespace = dict(
        __name__=__name__,
        AscendQwen3_5ForConditionalGeneration=Backbone,
        default_pooling_type=default_pooling_type,
        MULTIMODAL_REGISTRY=SimpleNamespace(register_processor=lambda *a, **k: lambda cls: cls),
        AscendQwen3_5VLProcessor=None,
        AscendQwen3_5ProcessingInfo=None,
        Qwen3VLDummyInputsBuilder=None,
        VllmConfig=VllmConfig,
        torch=torch,
        JointSchemaHead=JointSchemaHead,
        JointSchemaPooler=JointSchemaPooler,
        JOINT_HEAD_WEIGHTS=JOINT_HEAD_WEIGHTS,
        DefaultModelLoader=DefaultModelLoader,
        embed_lexical=embed_lexical,
    )
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            model="unused", revision="pinned", hf_config=SimpleNamespace(joint_head_config=HEAD_CONFIG)
        ),
        cache_config=SimpleNamespace(enable_prefix_caching=False),
        parallel_config=SimpleNamespace(pipeline_parallel_size=1, prefill_context_parallel_size=1),
        scheduler_config=SimpleNamespace(enable_chunked_prefill=False, async_scheduling=False),
    )
    monkeypatch.setattr("vllm.model_executor.layers.pooler.tokwise.methods.get_current_vllm_config", lambda: config)
    return namespace["AscendClefFlashForJointSchema"], config


def test_head_and_backbone_load_independently(patched_loader, adapter, tmp_path):
    _, loader = patched_loader
    cls, config = adapter
    config.model_config.model = str(tmp_path)
    model = cls(vllm_config=config)
    assert model.is_pooling_model
    assert model.default_tok_pooling_type == "ALL"
    head_state = {k: torch.randn_like(v) for k, v in model.pooler.head.state_dict().items()}
    backbone = {"language_model.lm_head.weight": torch.randn(8, 16)}
    save_file(backbone, tmp_path / "model-00001-of-00001.safetensors")
    save_file(head_state, tmp_path / JOINT_HEAD_WEIGHTS)
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {k: "model-00001-of-00001.safetensors" for k in backbone}})
    )
    (source,) = model.secondary_weights
    assert source.revision == "pinned"
    assert source.prefix == "pooler.head."
    loaded = AutoWeightsLoader(model).load_weights(loader.get_all_weights(config.model_config, model))
    assert loaded == set(dict(model.named_parameters()))
    for key, tensor in model.pooler.head.state_dict().items():
        torch.testing.assert_close(tensor, head_state[key])
    torch.testing.assert_close(model.language_model.lm_head.weight, backbone["language_model.lm_head.weight"])


def test_missing_head_fails(patched_loader, tmp_path):
    _, loader = patched_loader
    with pytest.raises(FileNotFoundError, match="decision head is missing"):
        loader._prepare_weights(str(tmp_path), None, None, False, [JOINT_HEAD_WEIGHTS])


def test_unrelated_weight_preparation_unchanged(patched_loader):
    module, _ = patched_loader
    original = Mock(return_value=("folder", ["weights"], True))
    wrapped = module.clef_weight_preparer(original)
    obj = object()
    assert wrapped(obj, "model", None, "rev", True, None) == original.return_value
    original.assert_called_once_with(obj, "model", None, "rev", True, None)


@pytest.mark.parametrize("field", ["prefix", "pp", "pcp"])
def test_unsupported_cache_or_parallel_modes(adapter, field):
    cls, config = adapter
    if field == "prefix":
        config.cache_config.enable_prefix_caching = True
    elif field == "pp":
        config.parallel_config.pipeline_parallel_size = 2
    else:
        config.parallel_config.prefill_context_parallel_size = 2
    with pytest.raises(ValueError, match="Clef-Flash requires|Clef-Flash currently requires"):
        cls(vllm_config=config)
