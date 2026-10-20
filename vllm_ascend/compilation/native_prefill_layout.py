"""Static fresh-prefill rectangles adapted from the v0.26 padded FULL graphs."""

from bisect import bisect_left
from dataclasses import dataclass

import torch

DEFAULT_NATIVE_PREFILL_MAX_CONTEXT = 1024


@dataclass(frozen=True, order=True)
class PrefillGraphShape:
    requests: int
    width: int

    @property
    def tokens(self):
        return self.requests * self.width


@dataclass(frozen=True)
class PrefillRequest:
    request: int
    start: int
    length: int


class NativePrefillPlan:
    def __init__(self, capture_sizes, max_tokens, max_requests, max_context, request_counts=None):
        if min(max_tokens, max_requests, max_context) < 1:
            raise ValueError("Native prefill capacities must be positive")
        if request_counts is None:
            request_counts = [1 << index for index in range(max_requests.bit_length())]
            request_counts.append(max_requests)
        if (
            not isinstance(request_counts, (list, tuple))
            or not request_counts
            or any(type(count) is not int or not 0 < count <= max_requests for count in request_counts)
        ):
            raise ValueError("native_prefill_graph_request_counts must contain positive integers <= max_num_seqs")
        self.request_counts = tuple(sorted(set(request_counts)))
        ceiling = min(max_tokens // self.request_counts[0], max_context)
        if ceiling < 1:
            raise ValueError("Smallest captured request count exceeds the token budget")
        self.widths = tuple(sorted({1, ceiling} | {size for size in capture_sizes if 0 < size <= ceiling}))
        self.max_tokens = max_tokens
        self.max_requests = max_requests
        self.max_context = max_context
        self.shapes = tuple(
            PrefillGraphShape(count, width)
            for width in self.widths
            for count in self.request_counts
            if count * width <= max_tokens
        )

    def batches(self, boundaries, seq_lengths):
        if not boundaries or boundaries[0] != 0 or len(boundaries) != len(seq_lengths) + 1:
            raise ValueError("Invalid native prefill request boundaries")
        if len(seq_lengths) > self.max_requests or boundaries[-1] > self.max_tokens:
            return None
        groups = {}
        for request, (start, end, total) in enumerate(zip(boundaries, boundaries[1:], seq_lengths)):
            length = end - start
            if length <= 0 or total < length:
                raise ValueError("Invalid native prefill query/context length")
            if total != length or length > self.widths[-1]:
                return None
            width = self.widths[bisect_left(self.widths, length)]
            groups.setdefault(width, []).append(PrefillRequest(request, start, length))
        result = []
        for width, items in groups.items():
            counts = [count for count in self.request_counts if count * width <= self.max_tokens]
            while items:
                live = min(len(items), counts[-1])
                padded = counts[bisect_left(counts, live)]
                result.append((PrefillGraphShape(padded, width), tuple(items[:live])))
                items = items[live:]
        return result


def padded_token_indices(items, width):
    return [
        0 if item is None else item.start + min(offset, item.length - 1) for item in items for offset in range(width)
    ]


def fresh_conv_history(inputs, lengths, history_width):
    offsets = lengths[:, None] - history_width + torch.arange(history_width, device=inputs.device)[None, :]
    history = inputs.gather(1, offsets.clamp(min=0)[:, :, None].expand(-1, -1, inputs.shape[-1]))
    return torch.where(offsets[:, :, None] >= 0, history, 0)


def neutral_gdn_padding(query, key, value, gate, beta, valid):
    """Zero gates and updates make padded steps identity state transitions."""
    vector_mask = valid[None, :, None, None]
    gate_mask = valid[None, :, None]
    return (
        torch.where(vector_mask, query, 0),
        torch.where(vector_mask, key, 0),
        torch.where(vector_mask, value, 0),
        torch.where(gate_mask, gate, 0),
        torch.where(gate_mask, beta, 0),
    )
