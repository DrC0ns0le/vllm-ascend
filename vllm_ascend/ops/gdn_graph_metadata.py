# SPDX-License-Identifier: Apache-2.0
"""Graph-owned GDN inputs. Request identity and lengths are device values."""

from dataclasses import dataclass

import torch

GDN_GRAPH_HEAD_DIM = 128
MAX_GDN_GRAPH_HEADS = 64


@dataclass
class GDNFullGraphMetadata:
    # Includes an extra empty sequence at the end for unused chunk tasks.
    query_start_loc: torch.Tensor
    state_read_indices: torch.Tensor
    state_write_indices: torch.Tensor
    has_initial_state: torch.Tensor
    chunk_indices: torch.Tensor
    chunk_offsets: torch.Tensor
    solve_indices: torch.Tensor
    cumsum_indices: torch.Tensor
    recurrent_work: torch.Tensor
    single_token_work: torch.Tensor

    @property
    def request_capacity(self):
        return self.state_read_indices.shape[0] - 1
