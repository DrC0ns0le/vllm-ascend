# SPDX-License-Identifier: Apache-2.0
"""CPU-only checks for static-kernel import timing and topology setup."""

import ast
import builtins
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


class TestStaticKernelRuntime(unittest.TestCase):
    def setUp(self):
        self.source = Path(__file__).resolve().parents[3] / "vllm_ascend/compilation/compiler_interface.py"
        self.tree = ast.parse(self.source.read_text())

    def test_no_eager_npugraph_import(self):
        # Other compiler paths must work without importing this optional backend.
        for node in self.tree.body:
            if isinstance(node, ast.ImportFrom):
                self.assertFalse((node.module or "").startswith("npugraph_ex"))
            elif isinstance(node, ast.Import):
                self.assertFalse(any(alias.name.startswith("npugraph_ex") for alias in node.names))

    def test_topology_is_set_before_static_kernel_import(self):
        # Execute the runtime initializer independently of the NPU-only imports.
        initializer = next(
            node
            for node in self.tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "_initialize_static_kernel_runtime"
        )
        namespace = {"os": os, "logger": Mock(), "VllmConfig": object}
        exec(compile(ast.Module(body=[initializer], type_ignores=[]), str(self.source), "exec"), namespace)
        blacklist = Mock()
        original_import = builtins.__import__

        def import_backend(name, *args, **kwargs):
            if name == "npugraph_ex._acl_concrete_graph.static_kernel":
                self.assertEqual(os.environ["LOCAL_WORLD_SIZE"], "4")
                return SimpleNamespace(_set_static_kernel_blacklist=blacklist)
            return original_import(name, *args, **kwargs)

        config = SimpleNamespace(
            parallel_config=SimpleNamespace(
                local_world_size=2, data_parallel_size_local=2, data_parallel_external_lb=False
            )
        )
        with patch.dict(os.environ, {}, clear=True), patch("builtins.__import__", side_effect=import_backend):
            namespace["_initialize_static_kernel_runtime"](config)

        blacklist.assert_called_once_with(
            [
                "TopKV2",
                "GenericBlockSparseAttn",
                "QuantBatchMatmulV3",
                "TransposeQuantBatchMatMul",
                "QuantLightningIndexer",
            ]
        )
