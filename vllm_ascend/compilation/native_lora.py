# SPDX-License-Identifier: Apache-2.0
"""Graph-owned adapter selection over the LoRA manager's stable weight slots.

Only metadata is private to a rectangle. Adapter loading/eviction still updates
the existing weight stacks in place, ordered with replay on the worker stream.
The runner's packed mappings are never overwritten, including its logits map.
"""

from contextlib import contextmanager

import torch


def native_lora_wrappers(model, *, required):
    wrappers = {}
    for module in model.modules():
        wrapper = getattr(module, "punica_wrapper", None)
        if wrapper is not None:
            wrappers[id(wrapper)] = wrapper
    if required and not wrappers:
        raise ValueError("Native FULL LoRA capture requires LoRA layers to be installed before warmup")
    for wrapper in wrappers.values():
        wrapper.enable_native_full_graph()
    return tuple(wrappers.values())


class NativeGraphLoRA:
    def __init__(self, wrapper, shape):
        self.wrapper = wrapper
        self.shape = shape
        source = wrapper._token_lora_indices
        # Capture the adapter operations even when startup has no active LoRA.
        # Serving fills -1 for base-model requests and padding; native BGMV
        # skips those rows on device, without a Python no_lora specialization.
        self.tokens = source.new_zeros(shape.tokens)
        self.embeddings = wrapper._embeddings_indices.new_zeros((*wrapper._embeddings_indices.shape[:-1], shape.tokens))
        self.sampler = source.new_zeros(shape.requests)
        self.sampler_padded = source.new_zeros(shape.requests)

    @contextmanager
    def bind(self):
        fields = dict(
            _token_lora_indices=self.tokens,
            _embeddings_indices=self.embeddings,
            _sampler_indices=self.sampler,
            _sampler_indices_padded=self.sampler_padded,
            indices_len=[self.shape.tokens, self.shape.requests, self.shape.requests, self.shape.tokens],
            is_prefill=False,
            no_lora=False,
        )
        saved = {name: getattr(self.wrapper, name) for name in fields}
        try:
            for name, value in fields.items():
                setattr(self.wrapper, name, value)
            yield
        finally:
            for name, value in saved.items():
                setattr(self.wrapper, name, value)

    def update(self, source_tokens, valid):
        self.tokens.copy_(self.wrapper._token_lora_indices.index_select(0, source_tokens))
        self.embeddings.copy_(self.wrapper._embeddings_indices.index_select(-1, source_tokens))
        self.mask_padding(valid)

    def mask_padding(self, valid):
        self.tokens.masked_fill_(~valid, -1)
        self.embeddings.masked_fill_(~valid, 0)
        self.sampler.copy_(self.tokens.view(self.shape.requests, self.shape.width)[:, 0])
        slots = torch.where(self.sampler >= 0, self.sampler, self.wrapper.lora_config.max_loras)
        self.sampler_padded.copy_(torch.arange(self.shape.requests, device=slots.device) + slots * self.shape.requests)

    def next_draft(self, first, active):
        """A skinny draft pass keeps the first pass's adapter per live row."""
        self.tokens.copy_(first.sampler)
        self.embeddings.copy_(
            first.embeddings.view(*first.embeddings.shape[:-1], first.shape.requests, first.shape.width)[..., 0]
        )
        self.mask_padding(active)
