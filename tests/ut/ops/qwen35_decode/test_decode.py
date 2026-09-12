# SPDX-License-Identifier: Apache-2.0
"""CPU dispatch and kernel indexing tests; not an Ascend compiler substitute."""

import ast
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

ROOT = Path(__file__).resolve().parents[4]


def load_config():
    spec = importlib.util.spec_from_file_location(
        "qwen35_decode_config_test", ROOT / "vllm_ascend/qwen35_decode_config.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


config_module = load_config()
Config = config_module.Qwen35DecodeConfig


@pytest.fixture(params=[torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
def dtype(request):
    return request.param


def config(values, **overrides):
    result = SimpleNamespace(
        additional_config={"qwen35_decode": values},
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(model_type="qwen3_5_text", hidden_size=2048, num_hidden_layers=24),
            dtype=torch.bfloat16,
            quantization=None,
        ),
        parallel_config=SimpleNamespace(tensor_parallel_size=1),
        lora_config=None,
    )
    result.__dict__.update(overrides)
    return result


@pytest.mark.parametrize(
    "values",
    [
        {"bad": True},
        {"linear_backend": "cuda"},
        {"linear_projections": "down_proj"},
        {"linear_projections": [["down_proj"]]},
        {"max_batch_size": 64},
        {"max_batch_size": True},
        {"split_k": 3},
        {"split_k": True},
        {"fuse_gdn_ba_prepare": "false"},
    ],
)
def test_invalid_options(values):
    with pytest.raises(ValueError):
        Config.from_config(config(values), 1)


def test_disabled_default_does_not_restrict_other_models():
    assert not Config.from_config(config({}, model_config=None), 2).enabled


@pytest.mark.parametrize(
    "field,value",
    [
        ("dtype", torch.float32),
        ("quantization", "w8a8"),
        ("hf_text_config", SimpleNamespace(model_type="qwen3_5_moe", hidden_size=2048, num_hidden_layers=24)),
    ],
)
def test_incompatible_model(field, value):
    cfg = config({"linear_backend": "triton"})
    setattr(cfg.model_config, field, value)
    with pytest.raises(ValueError, match="dense Qwen"):
        Config.from_config(cfg, 1)


@pytest.mark.parametrize(
    "kwargs", [{"parallel_config": SimpleNamespace(tensor_parallel_size=2)}, {"lora_config": object()}]
)
def test_incompatible_topology(kwargs):
    with pytest.raises(ValueError, match="TP/PP/CP"):
        Config.from_config(config({"fuse_gdn_ba_prepare": True}, **kwargs), 1)


def test_nz_rejected_without_silently_duplicating_weights():
    with pytest.raises(ValueError, match="ND BF16/FP16"):
        Config.from_config(config({"linear_backend": "triton"}), 2)


def test_supported_model_dtype(dtype):
    cfg = config({"linear_backend": "triton", "fuse_gdn_ba_prepare": True})
    cfg.model_config.dtype = dtype
    assert Config.from_config(cfg, 1).enabled


def load_functions(path, names, namespace):
    tree = ast.parse((ROOT / path).read_text())
    funcs = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(funcs) == len(names)
    for node in funcs:
        node.decorator_list = []
    exec(compile(ast.Module(body=funcs, type_ignores=[]), path, "exec"), namespace)
    return SimpleNamespace(**{name: namespace[name] for name in names})


@pytest.mark.parametrize("m", [1, 2, 4, 8, 9, 64])
def test_linear_selection_uses_capture_bucket_not_live_batch(m, dtype):
    functions = load_functions(
        "vllm_ascend/ops/qwen35_decode.py",
        ["use_decode_linear", "use_gdn_ba_prepare"],
        {"torch": torch, "Qwen35DecodeConfig": Config},
    )
    layer = SimpleNamespace(
        prefix="model.layers.0.mlp.down_proj", _ascend_decode_config=Config(linear_backend="triton")
    )
    assert functions.use_decode_linear(layer, torch.empty(m, 128)) is False  # FP32 stays native
    assert functions.use_decode_linear(layer, torch.empty(m, 128, dtype=dtype)) == (m <= 8)
    layer._ascend_decode_config = Config(fuse_gdn_ba_prepare=True)
    layer.prefix = "model.layers.0.linear_attn.in_proj_ba"
    assert functions.use_gdn_ba_prepare(layer, torch.empty(m, 128, dtype=dtype)) == (m <= 8)
    assert not functions.use_decode_linear(layer, torch.empty(m, 128, dtype=dtype))


def test_projection_scope_and_4b_config():
    assert config_module.is_qwen35_projection("model.layers.0.mlp.gate_up_proj", 18432, 2560)
    assert not config_module.is_qwen35_projection("visual.mlp.gate_up_proj", 18432, 2560)
    assert not config_module.is_qwen35_projection("model.layers.0.mlp.down_proj", 2048, 2048)
    cfg = config({"linear_backend": "triton_cube", "linear_projections": ["down_proj"]})
    cfg.model_config.hf_text_config.hidden_size = 2560
    cfg.model_config.hf_text_config.num_hidden_layers = 32
    parsed = Config.from_config(cfg, 1)
    assert parsed.uses_linear("model.layers.0.mlp.down_proj")
    assert not parsed.uses_linear("model.layers.0.linear_attn.in_proj_qkvz")


@pytest.mark.parametrize("enabled", [False, True])
def test_load_time_wiring_and_weight_identity(monkeypatch, enabled, dtype):
    functions = load_functions(
        "vllm_ascend/ops/qwen35_decode.py",
        ["configure_decode_layer"],
        {"torch": torch, "Qwen35DecodeConfig": Config, "is_qwen35_projection": config_module.is_qwen35_projection},
    )
    selected = Config(linear_backend="triton" if enabled else "native")
    monkeypatch.setitem(
        sys.modules,
        "vllm_ascend.ascend_config",
        SimpleNamespace(get_ascend_config=lambda: SimpleNamespace(qwen35_decode=selected)),
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm_ascend.utils",
        SimpleNamespace(
            ACL_FORMAT_FRACTAL_ND=2, AscendDeviceType=SimpleNamespace(A2="A2"), get_ascend_device_type=lambda: "A2"
        ),
    )
    cast = Mock(side_effect=lambda weight, fmt: weight)
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace(npu_format_cast=cast))
    layer = torch.nn.Module()
    layer.prefix = "model.layers.0.mlp.down_proj"
    layer.weight = torch.nn.Parameter(torch.empty(2048, 6144, dtype=dtype))
    address = layer.weight.data_ptr()
    functions.configure_decode_layer(layer)
    assert layer.weight.data_ptr() == address
    assert layer.weight.dtype == dtype
    assert (layer._ascend_decode_config is selected) == enabled
    assert cast.call_count == int(enabled)


def test_fake_shapes_match_gdn_consumers(dtype):
    functions = load_functions(
        "vllm_ascend/ops/qwen35_decode.py",
        ["qwen35_decode_linear_fake", "qwen35_gdn_ba_prepare_fake"],
        {"torch": torch},
    )
    x = torch.empty(4, 2560, device="meta", dtype=dtype)
    w = torch.empty(18432, 2560, device="meta", dtype=dtype)
    linear_output = functions.qwen35_decode_linear_fake(x, w, None, 4, True)
    assert linear_output.shape == (4, 18432) and linear_output.dtype == dtype
    ba_weight = torch.empty(64, 2560, device="meta", dtype=dtype)
    qkvz = torch.empty(4, 12288, device="meta", dtype=dtype)
    outputs = functions.qwen35_gdn_ba_prepare_fake(x, qkvz, ba_weight, None, 8192, 128)
    assert [t.shape for t in outputs] == [(4, 8192), (4, 32, 128), (4, 32), (4, 32)]
    assert all(t.is_contiguous() and t.dtype == dtype for t in outputs)


def test_linear_method_calls_registered_kernel_or_native():
    tree = ast.parse((ROOT / "vllm_ascend/ops/linear.py").read_text())
    cls = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "AscendUnquantizedLinearMethod"
    )
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "apply")
    optimized, native = Mock(return_value="optimized"), Mock(return_value="native")
    ops = SimpleNamespace(vllm=SimpleNamespace(qwen35_decode_linear=optimized, unquantized_gemm=native))
    namespace = {
        "torch": SimpleNamespace(Tensor=torch.Tensor, nn=torch.nn, ops=ops),
        "use_decode_linear": lambda layer, x: x.shape[0] <= 8,
    }
    exec(compile(ast.Module(body=[method], type_ignores=[]), "linear.py", "exec"), namespace)
    layer = SimpleNamespace(
        weight=torch.empty(1), _ascend_decode_config=Config(linear_backend="triton_cube", split_k=2)
    )
    assert namespace["apply"](None, layer, torch.empty(4, 2560)) == "optimized"
    assert optimized.call_args.args[-2:] == (2, True)
    assert namespace["apply"](None, layer, torch.empty(64, 2560)) == "native"
    native.assert_called_once()


@pytest.mark.parametrize("fused", [False, True])
def test_actual_gdn_forward_selects_fusion_without_ba_matmul(fused):
    tree = ast.parse((ROOT / "vllm_ascend/ops/gdn.py").read_text())
    cls = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "AscendGatedDeltaNetAttention"
    )
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "forward")
    m, heads, dim = 4, 2, 128
    qkv_size = 3 * heads * dim
    qkvz = torch.randn(m, 4 * heads * dim)
    ba = torch.randn(m, heads * 2)
    ba_projection = Mock(return_value=(ba, None))
    ba_projection.weight, ba_projection.bias = torch.empty(1), None
    prepared = Mock(
        return_value=(
            qkvz[:, :qkv_size].contiguous(),
            qkvz[:, qkv_size:].reshape(m, heads, dim).contiguous(),
            ba[:, :heads].contiguous(),
            ba[:, heads:].contiguous(),
        )
    )

    def core(qkv, b, a, output, prefix, flag):
        output.copy_(b[:, :, None].expand_as(output))

    torch_ops = SimpleNamespace(vllm=SimpleNamespace(qwen35_gdn_ba_prepare=prepared, qwen_gdn_attention_core=core))
    namespace = {
        "torch": SimpleNamespace(Tensor=torch.Tensor, zeros=torch.zeros, ops=torch_ops),
        "use_gdn_ba_prepare": lambda layer, x: fused,
        "rearrange": lambda value, pattern: value.flatten(1),
    }
    exec(compile(ast.Module(body=[method], type_ignores=[]), "gdn.py", "exec"), namespace)
    layer = SimpleNamespace(
        in_proj_qkvz=lambda x: (qkvz, None),
        in_proj_ba=ba_projection,
        gqa_interleaved_layout=False,
        key_dim=heads * dim,
        value_dim=heads * dim,
        tp_size=1,
        head_v_dim=dim,
        num_v_heads=heads,
        prefix="test",
        _split_ba_for_tp=lambda value: value.chunk(2, -1),
        norm=lambda x, z: x + z,
        out_proj=lambda x: (x, None),
    )
    output = namespace["forward"](layer, torch.empty(m, 128))
    expected = (ba[:, :heads, None] + qkvz[:, qkv_size:].reshape(m, heads, dim)).flatten(1)
    torch.testing.assert_close(output, expected)
    assert ba_projection.call_count == int(not fused)
    assert prepared.call_count == int(fused)


class Pointer:
    """Bounds-checked CPU memory for evaluating the actual Triton source."""

    def __init__(self, tensor, offsets=0):
        self.tensor = tensor.view(-1)
        self.offsets = torch.as_tensor(offsets)

    def __add__(self, value):
        return Pointer(self.tensor, self.offsets + value)


class Language:
    constexpr = int
    float32 = torch.float32
    static_range = range
    arange = staticmethod(torch.arange)
    zeros = staticmethod(lambda shape, dtype: torch.zeros(shape, dtype=dtype))
    sum = staticmethod(torch.sum)
    trans = staticmethod(torch.t)
    maximum = staticmethod(lambda x, y: torch.maximum(x, torch.as_tensor(y)))

    def __init__(self):
        self.pid = 0

    def program_id(self, axis):
        return self.pid

    @staticmethod
    def dot(a, b, acc):
        return a.float() @ b.float() + acc

    @staticmethod
    def load(pointer, mask=True, other=0):
        offsets, mask = torch.broadcast_tensors(pointer.offsets, torch.as_tensor(mask))
        active = offsets[mask]
        assert ((active >= 0) & (active < pointer.tensor.numel())).all(), "active out-of-bounds read"
        values = pointer.tensor[offsets.clamp(0, pointer.tensor.numel() - 1)]
        return torch.where(mask, values, torch.as_tensor(other, dtype=values.dtype))

    @staticmethod
    def store(pointer, value, mask=True):
        offsets, value, mask = torch.broadcast_tensors(pointer.offsets, value, torch.as_tensor(mask))
        active = offsets[mask]
        assert ((active >= 0) & (active < pointer.tensor.numel())).all(), "active out-of-bounds write"
        assert active.unique().numel() == active.numel(), "overlapping active stores"
        pointer.tensor[active] = value[mask].to(pointer.tensor.dtype)


def kernel_functions(path, names):
    language = Language()
    namespace = {
        "tl": language,
        # Host-only next_power_of_2 is deliberately unavailable inside kernels.
        "triton": SimpleNamespace(cdiv=lambda a, b: (a + b - 1) // b),
    }
    return load_functions(path, names, namespace), language


@pytest.mark.parametrize("k,heads,qkv_size,copy_span", [(2048, 16, 6144, 512), (2560, 32, 8192, 384)])
def test_gdn_launcher_computes_plain_int_block_sizes(k, heads, qkv_size, copy_span, dtype):
    launch = Mock()

    class Kernel:
        def __getitem__(self, grid):
            return launch

    def next_power_of_2(value):
        assert type(value) is int, "host helper received a Triton constexpr"
        return 1 << (value - 1).bit_length()

    functions = load_functions(
        "vllm_ascend/ops/triton/qwen35_gdn_prepare.py",
        ["gdn_ba_prepare"],
        {
            "torch": torch,
            "_gdn_ba_prepare": Kernel(),
            "init_device_properties_triton": lambda: None,
            "get_vectorcore_num": lambda: 40,
            "triton": SimpleNamespace(cdiv=lambda a, b: (a + b - 1) // b, next_power_of_2=next_power_of_2),
        },
    )
    x = torch.empty(4, k, dtype=dtype, device="meta")
    qkvz = torch.empty(4, qkv_size + heads * 128, dtype=dtype, device="meta")
    weight = torch.empty(2 * heads, k, dtype=dtype, device="meta")
    outputs = functions.gdn_ba_prepare(x, qkvz, weight, None, qkv_size, 128)
    assert all(t.dtype == dtype for t in outputs)
    assert launch.call_count == 1
    assert launch.call_args.args[-3:] == (next_power_of_2(k), copy_span, 512)
    assert all(type(arg) is int for arg in launch.call_args.args[-3:])


def test_bf16_gemv_preserves_values_outside_fp16_range():
    kernels, language = kernel_functions("vllm_ascend/ops/triton/qwen35_decode_linear.py", ["_decode_gemv"])
    # An implicit FP16 cast would overflow these finite BF16 activations.
    x = torch.full((1, 32), 65536.0, dtype=torch.bfloat16)
    weight = torch.full((8, 32), 0.5, dtype=torch.bfloat16)
    bias = torch.zeros(8, dtype=torch.bfloat16)
    output = torch.full((1, 8), float("nan"), dtype=torch.bfloat16)
    kernels._decode_gemv(*map(Pointer, (x, weight, bias, output)), 8, 32, 1, False, 1, 8, 32)
    torch.testing.assert_close(output, (x.float() @ weight.float().T).bfloat16(), rtol=0, atol=0)


@pytest.mark.parametrize("m", [1, 2, 4, 8])
@pytest.mark.parametrize("split_k", [1, 2, 4])
@pytest.mark.parametrize("has_bias", [False, True])
def test_actual_linear_kernel_indexing_and_fp32_reduction(m, split_k, has_bias, dtype):
    kernels, language = kernel_functions(
        "vllm_ascend/ops/triton/qwen35_decode_linear.py",
        ["_decode_gemv", "_decode_skinny_gemm", "_decode_split_k_reduce"],
    )
    # Irregular N/K exercise both tile tails and an incomplete split-K range.
    torch.manual_seed(123)
    n, k, cores = 37, 289, 3
    x = torch.randn(m, k, dtype=dtype) * 0.1
    w = torch.randn(n, k, dtype=dtype) * 0.1
    bias = torch.randn(n, dtype=dtype) if has_bias else torch.zeros(n, dtype=dtype)
    expected = (x.float() @ w.float().T + bias.float()).to(dtype)
    for vector in [True, False] if m == 1 else [False]:
        y = torch.full((m, n), float("nan"), dtype=dtype)
        partial = y if split_k == 1 else torch.full((split_k, m, n), float("nan"))
        for pid in range(cores):
            language.pid = pid
            if vector:
                kernels._decode_gemv(
                    Pointer(x), Pointer(w), Pointer(bias), Pointer(partial), n, k, split_k, has_bias, cores, 8, 128
                )
            else:
                kernels._decode_skinny_gemm(
                    Pointer(x),
                    Pointer(w),
                    Pointer(bias),
                    Pointer(partial),
                    m,
                    n,
                    k,
                    k,
                    split_k,
                    has_bias,
                    cores,
                    16,
                    16,
                    32,
                )
        if split_k > 1:
            for pid in range((m * n + 63) // 64):
                language.pid = pid
                kernels._decode_split_k_reduce(
                    Pointer(partial), Pointer(bias), Pointer(y), m * n, n, split_k, has_bias, 64
                )
        torch.testing.assert_close(y, expected, rtol=1e-2 if dtype == torch.bfloat16 else 1e-3, atol=5e-4)


@pytest.mark.parametrize("m", [1, 2, 4, 8])
@pytest.mark.parametrize("k,heads,qkv_size", [(2048, 16, 6144), (2560, 32, 8192)])
def test_actual_fused_ba_kernel_layout_and_math(m, k, heads, qkv_size, dtype):
    kernels, language = kernel_functions("vllm_ascend/ops/triton/qwen35_gdn_prepare.py", ["_gdn_ba_prepare"])
    torch.manual_seed(123)
    z_size = heads * 128
    x = torch.randn(m, k, dtype=dtype) * 0.1
    w = torch.randn(2 * heads, k, dtype=dtype) * 0.1
    bias = torch.randn(2 * heads, dtype=dtype)
    qkvz = torch.randn(m, qkv_size + z_size, dtype=dtype)
    qkv = torch.full((m, qkv_size), float("nan"), dtype=dtype)
    z = torch.full((m, z_size), float("nan"), dtype=dtype)
    b, a = (torch.full((m, heads), float("nan"), dtype=dtype) for _ in range(2))
    cores = 7  # Force work-stride iterations and a partial last iteration.
    for pid in range(cores):
        language.pid = pid
        kernels._gdn_ba_prepare(
            *map(Pointer, (x, w, bias, qkvz, qkv, z, b, a)),
            m,
            k,
            heads,
            qkv_size,
            z_size,
            k,
            qkv_size + z_size,
            True,
            cores,
            1 << (k - 1).bit_length(),
            (qkv_size + z_size + heads - 1) // heads,
            512,
        )
    expected_b, expected_a = (x.float() @ w.float().T + bias.float()).to(dtype).chunk(2, -1)
    torch.testing.assert_close(qkv, qkvz[:, :qkv_size], rtol=0, atol=0)
    torch.testing.assert_close(z, qkvz[:, qkv_size:], rtol=0, atol=0)
    rtol = 1e-2 if dtype == torch.bfloat16 else 1e-3
    torch.testing.assert_close(b, expected_b, rtol=rtol, atol=5e-4)
    torch.testing.assert_close(a, expected_a, rtol=rtol, atol=5e-4)
