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
    routes = re.findall(r"Breakable ACLGraph route:.*", text)
    route_counts = collections.Counter()
    for line in routes:
        mode = re.search(r"mode=(\S+)", line)
        bucket = re.search(r"num_tokens=(\d+)", line)
        actual = re.search(r"actual_tokens=(\d+) num_reqs=(\d+)", line)
        if mode and bucket and actual:
            route_counts[(mode[1], int(bucket[1]), int(actual[1]), int(actual[2]))] += 1
    fallback_calls = sum(count for name, count in totals.items() if name.startswith("fallback:"))
    prefill_calls = totals["megagdn"] + fallback_calls
    return dict(
        routes=[
            dict(mode=mode, bucket=bucket, actual_tokens=actual, num_reqs=reqs, calls=count)
            for (mode, bucket, actual, reqs), count in sorted(route_counts.items())
        ],
        mega_selection_fraction=totals["megagdn"] / prefill_calls if prefill_calls else None,
        mega_selection_denominator="Observed layer prefill calls, including ineligible fallbacks; not unique requests",
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
