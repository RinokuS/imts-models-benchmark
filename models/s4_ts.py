"""S4: Efficiently Modeling Long Sequences with Structured State Spaces.

Gu, Goel & Ré, ICLR 2022.

This is a self-contained diagonal SSM (S4D variant) implementation.
Key idea: diagonalise A → convolution kernel K → FFT for O(n log n) computation.

Architecture:
  Input projection → stack of S4 blocks → mean pool → classifier
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from data.base import IrregularBatch
from models.base import IrregularTSModel


class _S4DKernel(nn.Module):
    """Diagonal SSM (S4D) convolution kernel.

    State space: x'(t) = Ax(t) + Bu(t), y(t) = Cx(t)
    With diagonal A = diag(Lambda), the kernel K(t) = C * exp(A*t) * B
    can be computed in closed form and FFT-convolved in O(n log n).
    """

    def __init__(self, d_model: int, d_state: int, dt_min: float = 0.001, dt_max: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        # Diagonal eigenvalues: real part negative (stable), imaginary part controls oscillation
        log_dt = torch.rand(d_model) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        self.log_dt = nn.Parameter(log_dt)

        # Complex diagonal A in form: A = -exp(A_log) + i * A_imag
        A_log = torch.randn(d_model, d_state // 2)
        A_imag = torch.randn(d_model, d_state // 2)
        self.A_log = nn.Parameter(A_log)
        self.A_imag = nn.Parameter(A_imag)

        # B and C matrices (complex)
        self.B_re = nn.Parameter(torch.randn(d_model, d_state // 2))
        self.B_im = nn.Parameter(torch.randn(d_model, d_state // 2))
        self.C_re = nn.Parameter(torch.randn(d_model, d_state // 2))
        self.C_im = nn.Parameter(torch.randn(d_model, d_state // 2))

    def forward(self, L: int) -> Tensor:
        """Compute the convolution kernel K of length L.

        Returns: (d_model, L) real kernel.
        """
        dt = self.log_dt.exp()  # (d_model,)

        # Complex A: diag(Lambda) with Lambda_k = -exp(A_log_k) + i * A_imag_k
        A = torch.complex(-self.A_log.exp(), self.A_imag)  # (d_model, N//2)
        B = torch.complex(self.B_re, self.B_im)            # (d_model, N//2)
        C = torch.complex(self.C_re, self.C_im)            # (d_model, N//2)

        # Vandermonde: K_l = sum_k C_k * (exp(A_k * dt))^l * B_k * dt
        dt_A = dt.unsqueeze(-1) * A  # (d_model, N//2)
        l = torch.arange(L, device=dt_A.device).float()

        # K shape: (d_model, L) — sum over state dimension
        # (d_model, N//2, L) = exp(dt_A) ** l
        exp_term = (dt_A.unsqueeze(-1) * l.unsqueeze(0).unsqueeze(0)).exp()  # (d, N//2, L)
        CB = (C * B * dt.unsqueeze(-1)).unsqueeze(-1)  # (d, N//2, 1)
        K = 2 * (CB * exp_term).real.sum(dim=1)  # (d_model, L)

        return K


class _S4Block(nn.Module):
    """Single S4 layer: S4 conv + gating + FFN + LayerNorm."""

    def __init__(self, d_model: int, d_state: int, dropout: float = 0.1):
        super().__init__()
        self.kernel = _S4DKernel(d_model, d_state)
        self.D = nn.Parameter(torch.randn(d_model))  # skip connection

        self.gate = nn.Sequential(nn.Linear(d_model, d_model), nn.Sigmoid())
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, u: Tensor) -> Tensor:
        """u: (B, L, d_model)"""
        B, L, D = u.shape
        K = self.kernel(L)  # (D, L)

        # FFT convolution: O(L log L)
        u_t = u.transpose(1, 2)  # (B, D, L)
        K_fft = torch.fft.rfft(K, n=2 * L, dim=-1)    # (D, L+1) complex
        u_fft = torch.fft.rfft(u_t, n=2 * L, dim=-1)  # (B, D, L+1) complex
        y_fft = K_fft.unsqueeze(0) * u_fft
        y = torch.fft.irfft(y_fft, n=2 * L, dim=-1)[..., :L]  # (B, D, L)

        y = y + self.D.unsqueeze(0).unsqueeze(-1) * u_t  # skip
        y = y.transpose(1, 2)  # (B, L, D)

        y = y * self.gate(y)  # gating
        y = self.dropout(y)
        y = self.norm(u + y)
        y = self.norm(y + self.ffn(y))
        return y


class S4(IrregularTSModel):
    """S4 (diagonal variant) for irregular MVTS classification.

    The model operates on the zero-filled padded sequence.
    Complexity: O(n log n * d) time, O(n * d) memory.
    """

    def __init__(
        self,
        n_channels: int,
        n_classes: int,
        d_model: int = 128,
        d_state: int = 64,
        n_layers: int = 4,
        dropout: float = 0.1,
        prenorm: bool = True,
    ):
        super().__init__(n_channels, n_classes)

        self.input_proj = nn.Linear(n_channels * 2, d_model)  # values + mask

        self.layers = nn.ModuleList([
            _S4Block(d_model, d_state, dropout) for _ in range(n_layers)
        ])

        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, n_classes),
        )

    @property
    def complexity_class(self) -> str:
        return "O(n*log_n*d)"

    def forward(self, batch: IrregularBatch) -> Tensor:
        x = batch.values.float()   # (B, T, D)
        m = batch.mask.float()     # (B, T, D)

        inp = self.input_proj(torch.cat([x, m], dim=-1))  # (B, T, d_model)

        h = inp
        for layer in self.layers:
            h = layer(h)

        # Mean pool over observed time steps
        obs = batch.mask.any(dim=-1).float().unsqueeze(-1)  # (B, T, 1)
        pooled = (h * obs).sum(1) / obs.sum(1).clamp(min=1)

        return self.classifier(pooled)
