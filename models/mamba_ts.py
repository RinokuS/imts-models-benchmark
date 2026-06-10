"""Mamba: Linear-Time Sequence Modeling with Selective State Spaces.

Gu & Dao, 2023.

Self-contained pure-PyTorch implementation of the selective scan (SSM with
input-dependent parameters). Optionally uses the official mamba-ssm CUDA
kernels if available.

Adaptation to irregular time series:
  Δt (discretisation step) is parametrized from observed time intervals,
  allowing the model to be aware of variable step sizes.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

try:
    from mamba_ssm import Mamba as MambaOfficial
    _MAMBA_SSM = True
except Exception:
    _MAMBA_SSM = False

from data.base import IrregularBatch
from models.base import IrregularTSModel


class _SelectiveScanPureTorch(nn.Module):
    """Pure-PyTorch selective scan (sequential, no CUDA kernel).

    Implements: h_t = exp(A * Δt) * h_{t-1} + Δt * B_t * u_t
                y_t = C_t * h_t + D * u_t
    """

    def __init__(self, d_inner: int, d_state: int):
        super().__init__()
        self.d_inner = d_inner
        self.d_state = d_state

        # A: (d_inner, d_state) — initialised as -softplus(A_log)
        self.A_log = nn.Parameter(torch.randn(d_inner, d_state))

    def forward(
        self,
        u: Tensor,       # (B, L, d_inner)
        delta: Tensor,   # (B, L, d_inner)
        B: Tensor,       # (B, L, d_state)
        C: Tensor,       # (B, L, d_state)
        D: Tensor,       # (d_inner,)
    ) -> Tensor:
        B_batch, L, d = u.shape
        N = self.d_state

        A = -torch.exp(self.A_log.float())  # (d_inner, N) — stable negative

        # Discretize: Ā = exp(Δ * A),  B̄ = Δ * B
        delta_A = torch.exp(
            delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0)
        )  # (B, L, d, N)
        delta_B = delta.unsqueeze(-1) * B.unsqueeze(2)  # (B, L, d, N) broadcast

        # Sequential scan
        h = torch.zeros(B_batch, d, N, device=u.device, dtype=u.dtype)
        ys = []
        for t in range(L):
            h = delta_A[:, t] * h + delta_B[:, t] * u[:, t].unsqueeze(-1)
            y = (h * C[:, t].unsqueeze(1)).sum(-1)  # (B, d)
            ys.append(y)

        y_seq = torch.stack(ys, dim=1)  # (B, L, d)
        return y_seq + u * D.unsqueeze(0).unsqueeze(0)


class _MambaBlock(nn.Module):
    """Single Mamba block with optional irregular-time Δt conditioning."""

    def __init__(self, d_model: int, d_state: int, d_conv: int, expand: int):
        super().__init__()
        self.d_inner = d_model * expand
        d_inner = self.d_inner

        self.in_proj = nn.Linear(d_model, 2 * d_inner, bias=False)

        self.conv1d = nn.Conv1d(
            in_channels=d_inner, out_channels=d_inner,
            kernel_size=d_conv, padding=d_conv - 1, groups=d_inner
        )

        # SSM parameters
        self.x_proj = nn.Linear(d_inner, d_state + d_state + d_inner, bias=False)
        self.dt_proj = nn.Linear(d_inner, d_inner)

        self.D = nn.Parameter(torch.ones(d_inner))
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)

        # Always create the pure-PyTorch scan; used when _MAMBA_SSM is False
        # or when MambaScanWrapper rebuilds layers for backend='python'.
        self.ssm = _SelectiveScanPureTorch(d_inner, d_state)

        self.d_state = d_state
        self.d_inner_val = d_inner

    def forward(self, x: Tensor, dt_obs: Tensor | None = None) -> Tensor:
        """
        x:      (B, L, d_model)
        dt_obs: (B, L) optional observed time deltas for irregular Δt
        """
        B, L, _ = x.shape
        residual = x

        xz = self.in_proj(x)              # (B, L, 2*d_inner)
        x_inner, z = xz.chunk(2, dim=-1)  # each (B, L, d_inner)

        # Causal convolution
        x_conv = self.conv1d(x_inner.transpose(1, 2))[:, :, :L].transpose(1, 2)
        x_conv = F.silu(x_conv)

        # Project to SSM parameters
        ssm_params = self.x_proj(x_conv)  # (B, L, N+N+d_inner)
        B_ssm = ssm_params[..., :self.d_state]
        C_ssm = ssm_params[..., self.d_state:2*self.d_state]
        dt_raw = ssm_params[..., 2*self.d_state:]

        dt = F.softplus(self.dt_proj(dt_raw))  # (B, L, d_inner)

        # Condition Δt on observed time gaps if provided
        if dt_obs is not None:
            dt_scale = torch.log(dt_obs.clamp(min=1e-4) + 1.0).unsqueeze(-1)
            dt = dt * dt_scale

        # Selective scan
        y = self.ssm(x_conv, dt, B_ssm, C_ssm, self.D)

        y = y * F.silu(z)
        out = self.out_proj(y)
        return self.norm(out + residual)


class MambaTS(IrregularTSModel):
    """Mamba for irregular MVTS classification.

    Uses pure-PyTorch selective scan (or mamba-ssm CUDA kernel if available).
    Δt is conditioned on observed time intervals for irregularity awareness.

    Complexity: O(n * d) time, O(n * d) memory.
    """

    def __init__(
        self,
        n_channels: int,
        n_classes: int,
        d_model: int = 128,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        n_layers: int = 4,
        dropout: float = 0.1,
        use_mamba_ssm: bool = True,
    ):
        super().__init__(n_channels, n_classes)
        self.use_official = use_mamba_ssm and _MAMBA_SSM
        self._mamba_scan_backend = 'cuda' if self.use_official else 'python'

        # Store for MambaScanWrapper layer rebuilding
        self._d_model = d_model
        self._d_state = d_state
        self._d_conv = d_conv
        self._expand = expand
        self._n_layers = n_layers

        self.input_proj = nn.Linear(n_channels * 2, d_model)  # values + mask

        if self.use_official:
            self.layers = nn.ModuleList([
                nn.Sequential(
                    MambaOfficial(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand),
                    nn.LayerNorm(d_model),
                )
                for _ in range(n_layers)
            ])
        else:
            self.layers = nn.ModuleList([
                _MambaBlock(d_model, d_state, d_conv, expand)
                for _ in range(n_layers)
            ])

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, n_classes),
        )

    @property
    def complexity_class(self) -> str:
        return "O(n*d)"

    def forward(self, batch: IrregularBatch) -> Tensor:
        x = batch.values.float()   # (B, T, D)
        m = batch.mask.float()     # (B, T, D)
        t = batch.times.float()    # (B, T)

        # Compute time deltas
        dt = torch.zeros_like(t)
        dt[:, 1:] = (t[:, 1:] - t[:, :-1]).clamp(min=0)

        inp = self.input_proj(torch.cat([x, m], dim=-1))  # (B, T, d_model)

        h = inp
        for layer in self.layers:
            if self.use_official:
                h = layer[1](layer[0](h) + h)  # Sequential(MambaOfficial, LayerNorm)
            else:
                h = layer(h, dt_obs=dt)         # _MambaBlock with pure-PyTorch scan

        # Mean pool over observed steps
        obs = batch.mask.any(dim=-1).float().unsqueeze(-1)
        pooled = (h * obs).sum(1) / obs.sum(1).clamp(min=1)

        return self.classifier(pooled)
