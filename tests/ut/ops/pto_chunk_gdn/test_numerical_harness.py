# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for the NPU validation runner, not numerical kernel evidence."""

import importlib.util
from pathlib import Path

import torch


def load():
    path = Path(__file__).resolve().parents[4] / "benchmarks/qwen35_low_latency/numerical.py"
    spec = importlib.util.spec_from_file_location("numerical", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_packed_matrix_is_bounded_and_reproducible():
    numerical = load()
    cases = numerical.packed_lengths(1024)
    assert cases == numerical.packed_lengths(1024)
    for count in (1, 2, 4, 8, 16, 32, 64):
        assert (1024 // count,) * count in cases
    assert all(sum(lengths) <= 2048 for lengths in cases)
    assert (90, 90) in cases and (127, 128, 129) in cases
    assert (50, 75, 125, 180) in cases and (129,) * 4 in cases


def test_each_sequence_reports_corruption_even_when_last_sequence_is_correct():
    numerical = load()
    output = torch.zeros(1, 5, 1, 2)
    state = torch.zeros(2, 1, 2, 2)
    actual = (output.clone(), state.clone())
    actual[0][:, :2] = 1
    actual[1][0] = float("nan")
    rows = numerical.compare_sequences((output, state), actual, (2, 3), 0.01, 0.01)
    assert not rows[0]["output"]["within_tolerance"]
    assert not rows[0]["state"]["finite"]
    assert rows[1]["output"]["within_tolerance"] and rows[1]["state"]["within_tolerance"]
    assert not numerical.comparisons_pass(rows)


def test_continuation_uses_independent_transposed_states_and_same_decode_inputs():
    numerical = load()
    reference = torch.arange(8).reshape(2, 1, 2, 2).float()
    actual = reference + 0.125
    before = reference.clone()
    calls = []

    def decode(*, value, state):
        if not calls:
            torch.testing.assert_close(state, reference.transpose(-1, -2))
        calls.append((value, state.data_ptr()))
        state.add_(value)
        return state.sum(-1)

    report = numerical.continuation(reference, actual, lambda: dict(value=torch.tensor(1.0)), decode, 0.01, 0)
    assert [row["decode_steps"] for row in report] == [1, 4, 8]
    assert len(calls) == 16
    for left, right in zip(calls[::2], calls[1::2]):
        assert left[0] is right[0]
        assert left[1] != right[1]
    torch.testing.assert_close(reference, before)
    assert all(len(row["sequences"]) == 2 for row in report)
    assert all(not numerical.comparisons_pass(row["sequences"]) for row in report)
