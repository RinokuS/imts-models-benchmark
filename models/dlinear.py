"""DLinear: Are Transformers Effective for Time Series Forecasting? (Zeng et al., 2023).

Adapted for irregular MVTS classification: decompose each channel into trend
and remainder, apply separate linear projections over the time dimension,
then pool and classify.
"""

import torch
import torch.nn as nn
from torch import Tensor

from data.base import IrregularBatch
from models.base import IrregularTSModel


class _MovingAvg(nn.Module):
    """Centered moving average for trend extraction."""

    def __init__(self, kernel_size: int):
        super().__init__()
        # pad to keep same length
        self.k = kernel_size
        self.pad = kernel_size // 2
        self.avg = nn.AvgPool1d(kernel_size, stride=1, padding=0)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, T, D)
        x = x.permute(0, 2, 1)  # (B, D, T)
        # reflect-pad so output has same length T
        x = nn.functional.pad(x, (self.pad, self.pad), mode="reflect")
        trend = self.avg(x)      # (B, D, T)
        return trend.permute(0, 2, 1)  # (B, T, D)


class DLinear(IrregularTSModel):
    """DLinear adapted for irregular MVTS binary classification.

    Strategy: zero-fill missing values, apply trend decomposition per channel,
    run two independent linear layers over the time axis (trend + residual),
    mean-pool across time, concatenate and classify.

    Complexity: O(n * d) time, O(d) space (no activations scale with n).
    """

    def __init__(
        self,
        n_channels: int,
        n_classes: int,
        seq_len: int = 500,
        hidden_dim: int = 64,
        kernel_size: int = 25,
    ):
        super().__init__(n_channels, n_classes)
        self.seq_len = seq_len
        self.moving_avg = _MovingAvg(kernel_size)

        self.trend_proj = nn.Linear(seq_len, hidden_dim)
        self.resid_proj = nn.Linear(seq_len, hidden_dim)

        self.classifier = nn.Sequential(
            nn.Linear(2 * hidden_dim * n_channels, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_classes),
        )

    @property
    def complexity_class(self) -> str:
        return "O(n*d)"

    def forward(self, batch: IrregularBatch) -> Tensor:
        x = batch.values.float()   # (B, T, D)
        B, T, D = x.shape

        # Pad or truncate to self.seq_len
        if T < self.seq_len:
            pad = torch.zeros(B, self.seq_len - T, D, device=x.device)
            x = torch.cat([x, pad], dim=1)
        elif T > self.seq_len:
            x = x[:, :self.seq_len, :]

        trend = self.moving_avg(x)
        residual = x - trend

        # project each channel over time: (B, D, T) -> linear -> (B, D, H)
        trend_t = self.trend_proj(trend.permute(0, 2, 1))    # (B, D, H)
        resid_t = self.resid_proj(residual.permute(0, 2, 1)) # (B, D, H)

        feat = torch.cat([trend_t.flatten(1), resid_t.flatten(1)], dim=-1)  # (B, 2*D*H)
        return self.classifier(feat)
