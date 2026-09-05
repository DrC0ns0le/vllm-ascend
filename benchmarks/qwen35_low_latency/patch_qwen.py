# SPDX-License-Identifier: Apache-2.0
"""Produce a backed-up local Qwen patch. Never writes to /vllm-workspace."""

import argparse
import ast
import difflib
import hashlib
import json
from pathlib import Path

OP_NAMES = ("qwen_gdn_attention_core", "qwen_gdn_attention_core_fused_norm_packed")
IMPORT = "from vllm.compilation.breakable_cudagraph import eager_break_during_capture\n"


def patch_source(source):
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    replacements = []
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        if node.func.id != "direct_register_custom_op":
            continue
        keywords = {kw.arg: kw.value for kw in node.keywords}
        name_node = keywords.get("op_name")
        if not isinstance(name_node, ast.Constant) or name_node.value not in OP_NAMES:
            continue
        name = name_node.value
        found.append(name)
        function = keywords["op_func"]
        if isinstance(function, ast.Call) and isinstance(function.func, ast.Name):
            if function.func.id == "eager_break_during_capture":
                continue
        if not isinstance(function, ast.Name) or function.id != name:
            raise ValueError(f"Unexpected registration for {name}; inspect before patching")
        offset = sum(len(line.encode()) for line in lines[: function.lineno - 1]) + function.col_offset
        replacements.append((offset, len(name), f"eager_break_during_capture({name})"))
    if OP_NAMES[0] not in found:
        raise ValueError("Standard Qwen GDN registration missing")
    data = source.encode()
    for offset, length, value in sorted(replacements, reverse=True):
        data = data[:offset] + value.encode() + data[offset + length :]
    result = data.decode()
    if replacements and IMPORT not in result:
        # Insert immediately before the first vLLM import, after future imports.
        anchor = next(line for line in result.splitlines(True) if line.startswith("from vllm"))
        result = result.replace(anchor, IMPORT + anchor, 1)
    ast.parse(result)
    return result, found


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Local copy of the runtime Qwen source")
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    for path in (args.source, args.output_dir):
        if path.resolve().is_relative_to(Path("/vllm-workspace")):
            parser.error("Copy the runtime file to a separate local work directory first")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    source = args.source.read_text()
    original = args.output_dir / "qwen_gdn_linear_attn.py.original"
    original.write_bytes(args.source.read_bytes())
    modified, registrations = patch_source(source)
    (args.output_dir / "qwen_gdn_linear_attn.py").write_text(modified)
    path = "vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py"
    # Git accepts bare empty context lines. Avoid trailing-space warnings when
    # the patch itself is stored as a new file in the Ascend repository.
    (args.output_dir / "qwen-eager-break.patch").write_text(
        "".join(
            "\n" if line == " \n" else line
            for line in difflib.unified_diff(
                source.splitlines(True), modified.splitlines(True), fromfile="a/" + path, tofile="b/" + path
            )
        )
    )
    (args.output_dir / "manifest.json").write_text(
        json.dumps(
            dict(
                source=str(args.source.resolve()),
                backup=str(original.resolve()),
                sha256=hashlib.sha256(original.read_bytes()).hexdigest(),
                registrations=registrations,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
