"""mTAN: Multi-Time Attention Networks for Irregularly Sampled Time Series.

Shukla & Marlin, ICLR 2021.

Key idea: learn k reference time points; compute cross-attention between
observed times (Q from observations) and reference points (K, V learnable),
resulting in O(n * k) complexity instead of O(n^2).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from data.base import IrregularBatch
from models.base import IrregularTSModel


class _TimeEmbedding(nn.Module):
    """Sinusoidal time embedding with learnable frequencies."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        # learnable frequency + phase
        self.freq = nn.Parameter(torch.randn(embed_dim // 2))
        self.phase = nn.Parameter(torch.zeros(embed_dim // 2))

    def forward(self, t: Tensor) -> Tensor:
        # t: (...,) -> (..., embed_dim)
        t = t.unsqueeze(-1)  # (..., 1)
        freqs = self.freq.unsqueeze(0)  # (1, E/2)
        phases = self.phase.unsqueeze(0)
        angles = t * freqs + phases
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


class _MultiTimeAttention(nn.Module):
    """Cross-attention from observed times to reference time points."""

    def __init__(self, embed_dim: int, n_heads: int, n_ref_points: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        assert embed_dim % n_heads == 0

        # Learnable reference time points in [0, 1]
        self.ref_times = nn.Parameter(torch.linspace(0, 1, n_ref_points))

        self.time_embed = _TimeEmbedding(embed_dim)
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, obs_times: Tensor, obs_values: Tensor) -> Tensor:
        """
        obs_times:  (B, T) observation timestamps
        obs_values: (B, T, d_model) projected observed values

        Returns: (B, k, d_model) — one embedding per reference point
        """
        B, T, _ = obs_values.shape

        t_embed_obs = self.time_embed(obs_times)     # (B, T, E)
        t_embed_ref = self.time_embed(self.ref_times)  # (k, E)
        t_embed_ref = t_embed_ref.unsqueeze(0).expand(B, -1, -1)  # (B, k, E)

        Q = self.W_q(t_embed_ref)   # (B, k, E)
        K = self.W_k(t_embed_obs)   # (B, T, E)
        V = self.W_v(obs_values)    # (B, T, E)

        # Multi-head split
        def split_heads(x):
            B, L, E = x.shape
            return x.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

        Q, K, V = split_heads(Q), split_heads(K), split_heads(V)
        # (B, H, k, T)
        attn = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, V)  # (B, H, k, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, -1, self.n_heads * self.head_dim)
        return self.out_proj(out)  # (B, k, E)


class MTAN(IrregularTSModel):
    """mTAN for irregular MVTS classification.

    Complexity: O(n * k * d) time (cross-attention n obs × k ref points),
                O(n * k) memory (attention matrix n × k).
    """

    def __init__(
        self,
        n_channels: int,
        n_classes: int,
        hidden_dim: int = 128,
        n_heads: int = 8,
        n_ref_points: int = 128,
        n_layers: int = 3,
        dropout: float = 0.3,
        time_embed_dim: int = 64,
    ):
        super().__init__(n_channels, n_classes)
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # Project input channels + mask to hidden_dim
        self.input_proj = nn.Linear(2 * n_channels, hidden_dim)

        self.attn_layers = nn.ModuleList([
            _MultiTimeAttention(hidden_dim, n_heads, n_ref_points)
            for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_classes),
        )

    @property
    def complexity_class(self) -> str:
        return "O(n*k*d)"

    def forward(self, batch: IrregularBatch) -> Tensor:
        x = batch.values.float()   # (B, T, D)
        m = batch.mask.float()     # (B, T, D)
        t = batch.times.float()    # (B, T)

        # Concatenate values + mask as input features
        inp = self.input_proj(torch.cat([x, m], dim=-1))  # (B, T, H)
        inp = self.dropout(inp)

        # Apply stacked multi-time attention
        h = inp
        for attn, norm in zip(self.attn_layers, self.norms):
            h_ref = attn(t, h)  # (B, k, H)
            # take the mean over reference points for residual
            h = norm(h + h_ref.mean(dim=1, keepdim=True).expand_as(h))
            h = self.dropout(h)

        # Pool over observed time steps (masked mean)
        mask_any = batch.mask.any(dim=-1).float()  # (B, T)
        mask_any = mask_any.unsqueeze(-1)
        pooled = (h * mask_any).sum(dim=1) / mask_any.sum(dim=1).clamp(min=1)  # (B, H)

        return self.classifier(pooled)
