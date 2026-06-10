from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor
from torch.utils.data import Dataset


@dataclass
class IrregularBatch:
    """Unified batch format for all irregular MVTS models."""
    times: Tensor    # (B, T) float32 — observation timestamps
    values: Tensor   # (B, T, D) float32 — values (0.0 where missing)
    mask: Tensor     # (B, T, D) bool — True = observed
    labels: Tensor   # (B,) long — target class
    lengths: Optional[Tensor] = None  # (B,) int — actual sequence lengths before padding

    def to(self, device: torch.device) -> "IrregularBatch":
        return IrregularBatch(
            times=self.times.to(device),
            values=self.values.to(device),
            mask=self.mask.to(device),
            labels=self.labels.to(device),
            lengths=self.lengths.to(device) if self.lengths is not None else None,
        )

    @property
    def batch_size(self) -> int:
        return self.labels.shape[0]

    @property
    def seq_len(self) -> int:
        return self.times.shape[1]

    @property
    def n_channels(self) -> int:
        return self.values.shape[2]


def collate_irregular(samples: list[tuple]) -> IrregularBatch:
    """Collate a list of (times, values, mask, label) tuples into an IrregularBatch.

    Pads all sequences to the length of the longest one in the batch.
    """
    times_list, values_list, mask_list, labels_list = zip(*samples)

    lengths = torch.tensor([t.shape[0] for t in times_list], dtype=torch.long)
    max_len = int(lengths.max())
    D = values_list[0].shape[1]

    padded_times = torch.zeros(len(samples), max_len)
    padded_values = torch.zeros(len(samples), max_len, D)
    padded_mask = torch.zeros(len(samples), max_len, D, dtype=torch.bool)

    for i, (t, v, m) in enumerate(zip(times_list, values_list, mask_list)):
        L = t.shape[0]
        padded_times[i, :L] = t
        padded_values[i, :L] = v
        padded_mask[i, :L] = m

    return IrregularBatch(
        times=padded_times,
        values=padded_values,
        mask=padded_mask,
        labels=torch.stack(labels_list),
        lengths=lengths,
    )


class IrregularTSDataset(Dataset):
    """Abstract base for irregular MVTS datasets.

    Subclasses must implement __len__ and __getitem__, returning
    (times, values, mask, label) tuples compatible with collate_irregular.
    """

    def get_class_weights(self) -> Tensor:
        """Returns inverse-frequency class weights for imbalanced datasets."""
        labels = torch.tensor([self[i][3] for i in range(len(self))])
        counts = torch.bincount(labels, minlength=2).float()
        weights = 1.0 / (counts + 1e-6)
        return weights / weights.sum() * len(counts)
