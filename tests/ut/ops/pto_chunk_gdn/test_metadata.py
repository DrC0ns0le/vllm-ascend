# SPDX-License-Identifier: Apache-2.0
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize(
    "upper,starts,decodes,prefills,spec,expected",
    [
        ([90, 90], [0, 90, 180], 0, 2, None, True),
        ([101, 201, 120], [0, 1, 2, 122], 2, 1, None, True),
        ([101, 201, 121], [0, 1, 2, 122], 2, 1, None, False),
        ([1], [0, 1], 0, 1, None, True),
        ([129], [0, 128], 0, 1, None, False),
        (None, [0, 128], 0, 1, None, False),
        ([128], [0, 128], 0, 1, object(), False),
    ],
)
def test_cpu_metadata_proves_freshness_for_only_prefill_slice(upper, starts, decodes, prefills, spec, expected):
    path = Path(__file__).resolve().parents[4] / "vllm_ascend/ops/gdn_attn_builder.py"
    tree = ast.parse(path.read_text())
    block = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If) and "envs.VLLM_ASCEND_PTO_CHUNK_GDN" in ast.unparse(node.test)
    )
    metadata = SimpleNamespace(fresh_prefill=False)
    namespace = dict(
        torch=torch,
        envs=SimpleNamespace(VLLM_ASCEND_PTO_CHUNK_GDN=True),
        m=SimpleNamespace(seq_lens_cpu_upper_bound=None if upper is None else torch.tensor(upper)),
        spec_sequence_masks=spec,
        num_decodes=decodes,
        num_prefills=prefills,
        query_start_loc_cpu=torch.tensor(starts),
        non_spec_chunked_prefill_metadata=metadata,
    )
    exec(compile(ast.Module(body=[block], type_ignores=[]), "gdn_attn_builder.py", "exec"), namespace)
    assert metadata.fresh_prefill is expected
