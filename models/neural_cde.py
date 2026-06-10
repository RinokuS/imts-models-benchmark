"""Neural CDE: Neural Controlled Differential Equations for Irregular Time Series.

Kidger, Morrill, Foster & Lyons, NeurIPS 2020.

Architecture:
  Build a continuous path X(t) via cubic spline through observed values+times
  Integrate CDE: dz = f_θ(z) dX, solved via RK4 with adjoint backprop
  Classifier: MLP from z(t_max) → logits

Key properties:
  - Native irregular time handling via continuous spline path
  - Adjoint method: O(d) memory (constant in sequence length)
  - Time complexity: O(n * s * d)
"""

import torch
import torch.nn as nn
from torch import Tensor

try:
    import torchcde
    _TORCHCDE = True
except ImportError:
    _TORCHCDE = False

from data.base import IrregularBatch
from models.base import IrregularTSModel


class _CDEFunc(nn.Module):
    """Vector field f_θ(z) for Neural CDE: dz/dt = f(z) * dX/dt."""

    def __init__(self, hidden_dim: int, input_channels: int, hidden_hidden_dim: int):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_dim = hidden_dim

        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_hidden_dim, hidden_hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_hidden_dim, hidden_dim * input_channels),
        )

    def forward(self, t: Tensor, z: Tensor) -> Tensor:
        # z: (B, hidden_dim)
        # returns: (B, hidden_dim, input_channels) — the matrix f(z)
        B = z.shape[0]
        out = self.net(z)  # (B, hidden_dim * input_channels)
        return out.view(B, self.hidden_dim, self.input_channels)


class NeuralCDE(IrregularTSModel):
    """Neural CDE for irregular MVTS classification.

    Complexity: O(n * s * d) time, O(d) memory (adjoint).

    Falls back to a simple GRU if torchcde is unavailable.
    """

    def __init__(
        self,
        n_channels: int,
        n_classes: int,
        hidden_dim: int = 64,
        hidden_hidden_dim: int = 256,
        solver: str = "rk4",
        adjoint: bool = True,
        interpolation: str = "cubic",
    ):
        super().__init__(n_channels, n_classes)
        self.hidden_dim = hidden_dim
        self.solver = solver
        self.adjoint = adjoint and _TORCHCDE
        self.interpolation = interpolation

        # +1 for time channel added to the path
        path_channels = n_channels + 1

        self.initial_proj = nn.Linear(path_channels, hidden_dim)
        self.cde_func = _CDEFunc(hidden_dim, path_channels, hidden_hidden_dim)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_classes),
        )

        if not _TORCHCDE:
            # Fallback: plain GRU
            self.fallback_gru = nn.GRU(
                n_channels, hidden_dim, num_layers=2, batch_first=True, dropout=0.1
            )

    @property
    def complexity_class(self) -> str:
        return "O(n*s*d)"

    def forward(self, batch: IrregularBatch) -> Tensor:
        if not _TORCHCDE:
            return self._fallback_forward(batch)

        x = batch.values           # (B, T, D)
        m = batch.mask.to(x.dtype) # (B, T, D)
        t = batch.times            # (B, T)

        B, T, D = x.shape

        # Build path: prepend normalised time index as an additional channel
        # torchcde requires a 1D time vector shared across the batch
        t_grid = torch.linspace(0.0, 1.0, T, device=x.device)  # (T,)
        t_unsq = t_grid.unsqueeze(0).unsqueeze(-1).expand(B, -1, 1)  # (B, T, 1)
        path_data = torch.cat([t_unsq, x], dim=-1)  # (B, T, D+1)

        # Compute cubic spline coefficients
        if self.interpolation == "cubic":
            coeffs = torchcde.natural_cubic_coeffs(path_data, t_grid)
            X = torchcde.CubicSpline(coeffs, t_grid)
        else:
            coeffs = torchcde.linear_interpolation_coeffs(path_data, t_grid)
            X = torchcde.LinearInterpolation(coeffs, t_grid)

        # Initial hidden state from first observation
        z0 = self.initial_proj(X.evaluate(X.interval[0]))  # (B, hidden_dim)

        # Integrate CDE; cdeint returns (B, len(t), hidden_dim)
        z_traj = torchcde.cdeint(
            X=X, func=self.cde_func, z0=z0,
            t=X.interval, method=self.solver,
            adjoint=self.adjoint,
        )
        z_T = z_traj[:, -1, :]  # (B, hidden_dim) — state at final time

        return self.classifier(z_T)

    def _fallback_forward(self, batch: IrregularBatch) -> Tensor:
        x = batch.values
        out, _ = self.fallback_gru(x)
        return self.classifier(out[:, -1, :])
