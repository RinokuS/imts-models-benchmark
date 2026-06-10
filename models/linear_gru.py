"""Linear Interpolation + GRU baseline.

Imputes the irregular series onto a regular grid via linear interpolation,
then applies a standard GRU. Serves as a baseline measuring the cost of
the imputation step vs. natively irregular models.
"""

import torch
import torch.nn as nn
from torch import Tensor

from data.base import IrregularBatch
from models.base import IrregularTSModel


def _linear_interpolate(times: Tensor, values: Tensor, mask: Tensor, n_steps: int) -> Tensor:
    """Linearly interpolate observed values onto a uniform grid.

    Args:
        times:  (B, T) observed timestamps
        values: (B, T, D) values (0 where unobserved)
        mask:   (B, T, D) bool, True = observed
        n_steps: number of grid points

    Returns:
        (B, n_steps, D) tensor on uniform grid [0, 1]
    """
    B, T, D = values.shape
    device = values.device
    t_grid = torch.linspace(0.0, 1.0, n_steps, device=device)  # (G,)

    out = torch.zeros(B, n_steps, D, device=device)
    for b in range(B):
        for d in range(D):
            obs_idx = mask[b, :, d].nonzero(as_tuple=True)[0]
            if len(obs_idx) == 0:
                continue
            t_obs = times[b, obs_idx]   # (K,)
            v_obs = values[b, obs_idx, d]  # (K,)
            # interp1d equivalent: torch.searchsorted
            out[b, :, d] = _interp1d(t_obs, v_obs, t_grid)
    return out


def _interp1d(x: Tensor, y: Tensor, xi: Tensor) -> Tensor:
    """1D linear interpolation; clamps extrapolation to boundary values."""
    idx = torch.searchsorted(x.contiguous(), xi.contiguous())
    idx = idx.clamp(1, len(x) - 1)
    lo, hi = idx - 1, idx
    t0, t1 = x[lo], x[hi]
    y0, y1 = y[lo], y[hi]
    denom = (t1 - t0).clamp(min=1e-8)
    alpha = (xi - t0) / denom
    return y0 + alpha * (y1 - y0)


class LinearGRU(IrregularTSModel):
    """Linear interpolation onto regular grid + standard GRU.

    Complexity: O(n * d^2) dominated by GRU; O(d) space.
    """

    def __init__(
        self,
        n_channels: int,
        n_classes: int,
        hidden_dim: int = 128,
        n_layers: int = 2,
        dropout: float = 0.3,
        interp_steps: int = 100,
    ):
        super().__init__(n_channels, n_classes)
        self.interp_steps = interp_steps
        self.gru = nn.GRU(
            input_size=n_channels,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

    @property
    def complexity_class(self) -> str:
        return "O(n*d^2)"

    def forward(self, batch: IrregularBatch) -> Tensor:
        x_reg = _linear_interpolate(
            batch.times.float(),
            batch.values.float(),
            batch.mask,
            self.interp_steps,
        )  # (B, G, D)

        out, _ = self.gru(x_reg)
        return self.classifier(out[:, -1, :])
