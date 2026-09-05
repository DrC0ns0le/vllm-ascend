# SPDX-License-Identifier: Apache-2.0
"""Extract runtime evidence; logical replays are not device kernel counts."""

import argparse
import ast
import collections
import json
from pathlib import Path

import regex as re


def inspect(text):
    captures = re.findall(r"Breakable ACLGraph captured:.*", text)
    replays = re.findall(r"Breakable ACLGraph replay:.*", text)
    mixed = [
        line
        for line in text.splitlines()
        if "Qwen GDN step:" in line and re.search(r"prefills=[1-9]\d* decodes=[1-9]\d*", line)
    ]
    counters = {}
    for match in re.finditer(r"MegaGDN counters: layer=(\S+) counts=(\{.*\})", text):
        counters[match[1]] = ast.literal_eval(match[2])
    totals = collections.Counter()
    for values in counters.values():
        totals.update(values)
    return dict(
        captures=captures,
        logical_replay_calls=len(replays),
        graph_segment_submissions=sum(int(re.search(r"graphs=(\d+)", line)[1]) for line in replays),
        eager_break_invocations=sum(int(re.search(r"eager_breaks=(\d+)", line)[1]) for line in replays),
        mixed_gdn_observed=bool(mixed),
        mixed_examples=mixed[:8],
        mega_counters=dict(totals),
        graph_memory=re.findall(r"ACLGraph capture summary:.*", text),
        effective_config=re.findall(r"Breakable ACLGraph config:.*", text),
        note="Debug evidence only. Excludes baseline, device launches and NPU timing. Use profiler traces for those.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--require-mixed", action="store_true")
    args = parser.parse_args()
    report = inspect(args.log.read_text(errors="replace"))
    print(json.dumps(report, indent=2))
    if args.require_mixed and not report["mixed_gdn_observed"]:
        raise SystemExit("No confirmed mixed GDN batch; overlapping clients do not establish mixed-batch coverage")


if __name__ == "__main__":
    main()
