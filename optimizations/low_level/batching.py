"""Batching strategy wrappers — Direction 2 (Low-Level).

Compares three strategies for handling variable-length irregular sequences:
  - Padding (baseline): pad to max length → up to 80% compute on zeros
  - Nested Tensors: jagged layout, FlashAttention compatible (PyTorch 2.2+)
  - Bucketing: sort by length + group into same-length buckets

Vault reference: Jagged Tensors note — PhysioNet P12 padding overhead ~72%.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import torch
from torch import Tensor
from ..base import OptimizationWrapper


@dataclass
class BatchingStats:
    strategy: str
    padding_waste_fraction: float  # fraction of tokens that are padding


# ---------------------------------------------------------------------------
# Strategies are implemented as DataLoader collate functions, not model
# wrappers, because batching happens before the forward pass. The "wrapper"
# interface here is used only for experiment metadata and stats reporting.
# ---------------------------------------------------------------------------


class PaddingBatching(OptimizationWrapper):
    """Standard zero-padding to max sequence length in the batch (baseline)."""

    def apply(self, model, **kwargs):
        return model  # no model change; batching is in the DataLoader

    def name(self) -> str:
        return "batching_padding"

    def collate_fn(self, samples: Sequence[dict]) -> dict:
        lengths = [s["values"].shape[0] for s in samples]
        max_len = max(lengths)
        D = samples[0]["values"].shape[1]
        B = len(samples)

        values = torch.zeros(B, max_len, D)
        mask = torch.zeros(B, max_len, D)
        times = torch.zeros(B, max_len)

        for i, s in enumerate(samples):
            T = s["values"].shape[0]
            values[i, :T] = s["values"]
            mask[i, :T] = s["mask"]
            times[i, :T] = s["times"]

        total_tokens = B * max_len
        real_tokens = sum(lengths)
        waste = 1.0 - real_tokens / total_tokens

        return {
            "values": values,
            "mask": mask,
            "times": times,
            "labels": torch.stack([s["label"] for s in samples]),
            "_batching_waste": waste,
        }


class NestedTensorBatching(OptimizationWrapper):
    """Jagged layout via PyTorch Nested Tensors (requires PyTorch >= 2.2).

    Eliminates padding overhead entirely for attention-based models.
    FlashAttention automatically dispatches to jagged kernel when input
    is a nested tensor (torch.nested.nested_tensor with layout=torch.jagged).
    """

    def apply(self, model, **kwargs):
        return model

    def name(self) -> str:
        return "batching_nested"

    def collate_fn(self, samples: Sequence[dict]) -> dict:
        values_list = [s["values"] for s in samples]
        mask_list = [s["mask"] for s in samples]
        times_list = [s["times"] for s in samples]

        values_nested = torch.nested.nested_tensor(values_list, layout=torch.jagged)
        mask_nested = torch.nested.nested_tensor(mask_list, layout=torch.jagged)

        lengths = [s["values"].shape[0] for s in samples]
        total = sum(lengths)
        waste = 0.0  # no padding

        return {
            "values": values_nested,
            "mask": mask_nested,
            "times": times_list,
            "labels": torch.stack([s["label"] for s in samples]),
            "_batching_waste": waste,
            "_is_nested": True,
        }


class BucketBatching(OptimizationWrapper):
    """Sort sequences by length and group into buckets of similar sizes.

    Reduces padding waste by 3–5× vs naive padding without requiring
    nested tensor support. Works with all model types.

    Args:
        n_buckets: number of length buckets per batch.
    """

    def __init__(self, n_buckets: int = 4):
        self.n_buckets = n_buckets

    def apply(self, model, **kwargs):
        return model

    def name(self) -> str:
        return f"batching_bucket_{self.n_buckets}"
