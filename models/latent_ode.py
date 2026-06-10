"""Latent ODE: Latent Ordinary Differential Equations for Irregularly-Sampled Time Series.

Rubanova, Chen & Duvenaud, NeurIPS 2019.

Architecture:
  Encoder: bidirectional RNN over (reversed) observed sequence → q(z0)
  Latent dynamics: ODE f_θ(z, t) integrated via adjoint (O(d) memory)
  Decoder: MLP from z(t_max) → class logits

Key properties:
  - Handles irregular timestamps natively via ODE time grid
  - Adjoint method: O(d) memory for backprop (constant in sequence length)
  - Time complexity: O(n * s * d) where s = number of ODE solver steps
"""

import torch
import torch.nn as nn
from torch import Tensor

try:
    from torchdiffeq import odeint_adjoint
    _ADJOINT = True
except ImportError:
    from torchdiffeq import odeint
    _ADJOINT = False

from data.base import IrregularBatch
from models.base import IrregularTSModel


class _ODEFunc(nn.Module):
    """Vector field f(z, t) for latent ODE."""

    def __init__(self, latent_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, t: Tensor, z: Tensor) -> Tensor:
        return self.net(z)


class _RecognitionRNN(nn.Module):
    """GRU-based encoder over reversed time sequence → (mu, logvar) of q(z0)."""

    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        # input: value + mask concatenated
        self.gru = nn.GRUCell(input_dim * 2, hidden_dim)
        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)

    def forward(self, times: Tensor, values: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        B, T, D = values.shape
        h = torch.zeros(B, self.gru.hidden_size, device=values.device)

        # Iterate backwards in time
        for t_idx in range(T - 1, -1, -1):
            x = torch.cat([values[:, t_idx, :], mask[:, t_idx, :].to(values.dtype)], dim=-1)
            obs_any = mask[:, t_idx, :].any(dim=-1, keepdim=True).to(values.dtype)
            h_new = self.gru(x, h)
            h = h_new * obs_any + h * (1 - obs_any)  # update only where observed

        mu = self.mu_head(h)
        logvar = self.logvar_head(h)
        return mu, logvar


class LatentODE(IrregularTSModel):
    """Latent ODE for irregular MVTS classification.

    Complexity: O(n * s * d) time, O(d) memory (adjoint backprop).
    """

    def __init__(
        self,
        n_channels: int,
        n_classes: int,
        latent_dim: int = 32,
        hidden_dim: int = 64,
        rec_hidden_dim: int = 64,
        solver: str = "dopri5",
        rtol: float = 1e-3,
        atol: float = 1e-4,
        adjoint: bool = True,
    ):
        super().__init__(n_channels, n_classes)
        self.latent_dim = latent_dim
        self.solver = solver
        self.rtol = rtol
        self.atol = atol
        self.use_adjoint = adjoint and _ADJOINT

        self.encoder = _RecognitionRNN(n_channels, rec_hidden_dim, latent_dim)
        self.ode_func = _ODEFunc(latent_dim, hidden_dim)

        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_classes),
        )

    @property
    def complexity_class(self) -> str:
        return "O(n*s*d)"

    def forward(self, batch: IrregularBatch) -> Tensor:
        x = batch.values           # (B, T, D)
        m = batch.mask.to(x.dtype) # (B, T, D)
        t = batch.times            # (B, T)

        # Encode: q(z0) from reversed sequence
        mu, logvar = self.encoder(t, x, m)

        # Reparameterization trick
        if self.training:
            eps = torch.randn_like(mu)
            z0 = mu + eps * (0.5 * logvar).exp()
        else:
            z0 = mu

        # Integrate ODE from t=0 to t=1 (normalised time range)
        # Use a simple 2-point grid for classification (only need z at t_end)
        t_span = torch.tensor([0.0, 1.0], device=z0.device)

        # ODE solver requires FP32 for step-size numerical stability under FP16 autocast
        z0_fp32 = z0.to(dtype=torch.float32)
        ode_fn = odeint_adjoint if self.use_adjoint else odeint
        z_traj = ode_fn(
            self.ode_func,
            z0_fp32,
            t_span,
            method=self.solver,
            rtol=self.rtol,
            atol=self.atol,
        )
        z_T = z_traj[-1].to(dtype=z0.dtype)  # (B, latent_dim)

        return self.classifier(z_T)
