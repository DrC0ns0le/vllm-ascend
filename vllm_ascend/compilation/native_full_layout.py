# SPDX-License-Identifier: Apache-2.0
"""Bounded native graph geometries and state-neutral sequence padding."""

from bisect import bisect_left
from dataclasses import dataclass

import torch

DEFAULT_NATIVE_FULL_MAX_CONTEXT = 1024


@dataclass(frozen=True, order=True)
class NativeGraphShape:
    requests: int
    width: int
    fresh: bool

    @property
    def tokens(self):
        return self.requests * self.width


@dataclass(frozen=True)
class RequestSlice:
    request: int
    start: int
    length: int
    context: int


class NativeGraphPlan:
    """Static fresh-prefill rectangles plus one-token cached decode graphs."""

    def __init__(self, capture_sizes, max_tokens, max_requests, max_context, request_counts=None):
        if min(max_tokens, max_requests, max_context) < 1:
            raise ValueError("Native FULL capacities must be positive")
        if request_counts is None:
            request_counts = [1 << i for i in range((max_requests - 1).bit_length() + 1)]
        if (
            not isinstance(request_counts, (list, tuple))
            or not request_counts
            or any(type(n) is not int or n <= 0 for n in request_counts)
        ):
            raise ValueError("native_full_graph_request_counts must be a nonempty list of positive integers")
        self.request_counts = tuple(sorted(set(request_counts)))
        ceiling = min(max_tokens // self.request_counts[0], max_context)
        if ceiling < 1:
            raise ValueError("Smallest captured request count exceeds the token budget")
        self.widths = tuple(sorted({1, ceiling} | {n for n in capture_sizes if 0 < n <= ceiling}))
        self.max_tokens, self.max_requests, self.max_context = max_tokens, max_requests, max_context
        self.shapes = tuple(
            NativeGraphShape(count, width, fresh)
            for width in self.widths
            for count in self.request_counts
            if count * width <= max_tokens
            for fresh in ((True, False) if width == 1 else (True,))
        )

    def batches(self, boundaries, seq_lengths):
        if len(boundaries) != len(seq_lengths) + 1 or not boundaries or boundaries[0] != 0:
            raise ValueError("Invalid native FULL request boundaries")
        if len(seq_lengths) > self.max_requests:
            raise ValueError("Native FULL request count exceeds startup capacity")
        groups = {}
        for request, (start, end, total) in enumerate(zip(boundaries, boundaries[1:], seq_lengths)):
            length = end - start
            if length <= 0 or total < length:
                raise ValueError("Invalid native FULL query/context length")
            context = total - length
            # No graph specialization or stateful-prefill emulation: caller
            # executes the unchanged compatibility path for this whole step.
            if total > self.max_context or length > self.widths[-1] or (context > 0 and length > 1):
                return None
            width = self.widths[bisect_left(self.widths, length)]
            groups.setdefault((width, context == 0), []).append(RequestSlice(request, start, length, context))
        result = []
        for (width, fresh), items in groups.items():
            counts = [n for n in self.request_counts if n * width <= self.max_tokens]
            while items:
                live = min(len(items), counts[-1])
                padded = counts[bisect_left(counts, live)]
                result.append((NativeGraphShape(padded, width, fresh), tuple(items[:live])))
                items = items[live:]
        return result


def padded_token_indices(items, width):
    """Dummy requests read token zero; write predicates exclude every dummy."""
    return [
        0 if item is None else item.start + min(offset, item.length - 1) for item in items for offset in range(width)
    ]


def fresh_conv_history(x, lengths, history):
    offsets = lengths[:, None] - history + torch.arange(history, device=x.device)[None, :]
    current = x.gather(1, offsets.clamp(min=0)[:, :, None].expand(-1, -1, x.shape[-1]))
    return torch.where(offsets[:, :, None] >= 0, current, 0)


def neutral_gdn_padding(q, k, v, g, beta, valid):
    """g=beta=0 makes each padding step the identity recurrent transition."""
    vector_mask = valid[None, :, None, None]
    gate_mask = valid[None, :, None]
    return (
        torch.where(vector_mask, q, 0),
        torch.where(vector_mask, k, 0),
        torch.where(vector_mask, v, 0),
        torch.where(gate_mask, g, 0),
        torch.where(gate_mask, beta, 0),
    )


def cached_attention_mask(lengths, contexts, width, capacity):
    query = contexts[:, None] + torch.arange(width, device=contexts.device)[None, :]
    key = torch.arange(capacity, device=contexts.device)
    # Padding queries attend the last real prefix. They are discarded by the
    # output gather; avoiding completely masked rows also avoids NaN softmax.
    last = contexts + lengths.clamp(min=1) - 1
    query = torch.minimum(query, last[:, None])
    return (key[None, None, :] > query[:, :, None]).unsqueeze(1)
