"""GRU-D: Recurrent Neural Networks for Multivariate Time Series with Missing Values.

Che et al., Scientific Reports 2018.

Key idea: model the temporal decay of missing values and hidden state using
learned decay rates per variable. Handles irregular timestamps natively.
"""

import torch
import torch.nn as nn
from torch import Tensor

from data.base import IrregularBatch
from models.base import IrregularTSModel


class GRUD(IrregularTSModel):
    """GRU-D: GRU with exponential decay for irregular MVTS.

    Complexity: O(n * d^2) time (sequential GRU), O(d) space (state only).
    """

    def __init__(
        self,
        n_channels: int,
        n_classes: int,
        hidden_dim: int = 128,
        n_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__(n_channels, n_classes)
        self.hidden_dim = hidden_dim

        # Input: [x_masked, mask, time_deltas] -> 3 * n_channels
        self.gru = nn.GRU(
            input_size=3 * n_channels,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )

        # Learned decay rates: one per channel (input decay)
        self.gamma_x = nn.Parameter(torch.zeros(n_channels))
        # Hidden state decay: one per hidden unit
        self.gamma_h = nn.Parameter(torch.zeros(hidden_dim))

        # Channel-wise empirical mean (for imputation)
        self.register_buffer("x_mean", torch.zeros(n_channels))

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

    @property
    def complexity_class(self) -> str:
        return "O(n*d^2)"

    def forward(self, batch: IrregularBatch) -> Tensor:
        x = batch.values.float()   # (B, T, D)
        m = batch.mask.float()     # (B, T, D)
        t = batch.times.float()    # (B, T)

        B, T, D = x.shape

        # Compute time deltas since last observation, per channel
        delta = self._compute_deltas(t, m)  # (B, T, D)

        # Input decay: decay toward channel mean where missing
        gamma_x = torch.exp(-torch.relu(self.gamma_x))  # (D,)
        decay_x = torch.exp(-gamma_x * delta)            # (B, T, D)
        x_mean = self.x_mean.unsqueeze(0).unsqueeze(0)   # (1, 1, D)
        x_imputed = m * x + (1 - m) * (decay_x * x + (1 - decay_x) * x_mean)

        inp = torch.cat([x_imputed, m, delta], dim=-1)  # (B, T, 3D)

        # Hidden state decay applied step-by-step — simplified: pass full sequence
        # Full GRU-D requires custom cell; here we use standard GRU on decayed input
        out, _ = self.gru(inp)  # (B, T, H)
        last = out[:, -1, :]   # (B, H)
        return self.classifier(last)

    @staticmethod
    def _compute_deltas(times: Tensor, mask: Tensor) -> Tensor:
        """Time elapsed since the last observed value per channel."""
        B, T, D = mask.shape
        delta = torch.zeros_like(mask)
        for t in range(1, T):
            # delta[t] = 0 where observed; += step size where missing
            step = times[:, t] - times[:, t - 1]  # (B,)
            step = step.unsqueeze(-1).expand(B, D)
            # if observed at t-1: delta[t] = step; else: delta[t] = delta[t-1] + step
            was_observed = mask[:, t - 1, :]      # (B, D)
            delta[:, t, :] = torch.where(
                was_observed.bool(),
                step,
                delta[:, t - 1, :] + step,
            )
        return delta
