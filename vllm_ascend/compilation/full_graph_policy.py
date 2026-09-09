# SPDX-License-Identifier: Apache-2.0
"""Bounded token capacities covering every legal scheduled batch."""

from bisect import bisect_left
from dataclasses import dataclass


@dataclass(frozen=True)
class FullGraphCapacityPolicy:
    capacities: tuple[int, ...]
    max_tokens: int
    max_requests: int

    @classmethod
    def build(cls, max_tokens, max_requests, capture_sizes, max_entries=None):
        if max_tokens < 1 or not 1 <= max_requests <= 64:
            raise ValueError("Native FULL requires a positive token budget and max_num_seqs in [1, 64]")
        if max_entries is not None and (type(max_entries) is not int or max_entries < 1):
            raise ValueError("FULL cannot disable graph entries: full_prefill_graph_max_entries must be positive")
        # A final ceiling covers batches above the last configured capture.
        # Intermediate sizes affect padding cost, never serving coverage.
        sizes = sorted({size for size in capture_sizes if 0 < size <= max_tokens} | {max_tokens})
        if max_entries is not None and len(sizes) > max_entries:
            # Coarsen capacities before capture instead of falling back when
            # the entry budget fills. Always retain the scheduler ceiling.
            sizes = [sizes[(i * len(sizes) - 1) // max_entries] for i in range(1, max_entries + 1)]
        return cls(tuple(sizes), max_tokens, max_requests)

    def capacity(self, num_tokens, num_requests):
        if not 1 <= num_tokens <= self.max_tokens or not 1 <= num_requests <= self.max_requests:
            raise ValueError("Batch exceeds the configured native FULL token/request capacity")
        if num_requests > num_tokens:
            raise ValueError("Each scheduled request must contain at least one token")
        return self.capacities[bisect_left(self.capacities, num_tokens)]

    def dummy_lengths(self, capacity, max_model_len):
        rows = min(capacity, self.max_requests)
        if capacity > rows * max_model_len:
            raise ValueError("FULL token capacity exceeds max_num_seqs * max_model_len")
        per_request, remainder = divmod(capacity, rows)
        return [per_request + (row < remainder) for row in range(rows)]
