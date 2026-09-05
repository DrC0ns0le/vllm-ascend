# SPDX-License-Identifier: Apache-2.0
"""Compare repeated outputs against measured baseline nondeterminism."""

import argparse
import collections
import difflib
import json
from pathlib import Path

import regex as re


def read(path):
    groups = collections.defaultdict(list)
    for line in path.read_text().splitlines():
        row = json.loads(line)
        groups[(tuple(row["prompt"]), row.get("output_budget", row["output_tokens"]))].append(row)
    return groups


def compare(baseline, candidate, valid_format=None):
    result = collections.Counter()
    divergence = []
    for key, rows in candidate.items():
        references = baseline.get(key, [])
        if not references:
            result["unmatched_requests"] += len(rows)
            continue
        outputs = {row["text"] for row in references}
        for row in rows:
            result["compared_requests"] += 1
            result["exact_match_any_baseline_repeat"] += row["text"] in outputs
            if valid_format is not None:
                result["invalid_format"] += re.fullmatch(valid_format, row["text"]) is None
            if row["tokens"] and all(ref["tokens"] for ref in references):
                similarity = max(
                    difflib.SequenceMatcher(a=ref["tokens"], b=row["tokens"], autojunk=False).ratio()
                    for ref in references
                )
                divergence.append(1 - similarity)
    total = result["compared_requests"]
    return dict(result) | dict(
        exact_match_rate=result["exact_match_any_baseline_repeat"] / total if total else None,
        mean_token_sequence_divergence=sum(divergence) / len(divergence) if divergence else None,
        formatting_rule=valid_format,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--valid-format", help="Optional full-match regular expression for task formatting")
    args = parser.parse_args()
    baseline, candidate = read(args.baseline), read(args.candidate)
    # Hold out each prompt's last repeat so baseline self-match is not trivially 100%.
    training = {key: rows[:-1] for key, rows in baseline.items() if len(rows) >= 2}
    held_out = {key: rows[-1:] for key, rows in baseline.items() if len(rows) >= 2}
    print(
        json.dumps(
            dict(
                baseline_repeatability=compare(training, held_out, args.valid_format),
                candidate_vs_baseline=compare(baseline, candidate, args.valid_format),
                quality_gate="Requires a representative production corpus and task-specific scoring",
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
