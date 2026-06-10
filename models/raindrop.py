"""Raindrop: Graph-Guided Network for Irregularly Sampled Multivariate Time Series.

Zhang, Zeman, Tsiligkaridis & Zitnik, ICLR 2022.

Key idea:
  - Learnable inter-sensor adjacency matrix A (V × V)
  - Per-sensor observation embeddings with temporal encoding
  - Graph attention message passing (2 rounds) over sensor graph
  - Global temporal attention across time steps → sequence embedding → logits

Self-contained pure PyTorch — does not require PyTorch Geometric.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from data.base import IrregularBatch
from models.base import IrregularTSModel


class _SensorEmbedding(nn.Module):
    """Embed each sensor value with time encoding."""

    def __init__(self, d_model: int):
        super().__init__()
        self.value_proj = nn.Linear(1, d_model)
        self.time_proj = nn.Linear(1, d_model)

    def forward(self, values: Tensor, times: Tensor, mask: Tensor) -> Tensor:
        """
        values: (B, T, V) — sensor values (0 where missing)
        times:  (B, T)
        mask:   (B, T, V) bool

        Returns: (B, T, V, d_model)
        """
        B, T, V = values.shape
        v_emb = self.value_proj(values.unsqueeze(-1))   # (B, T, V, d)
        t_emb = self.time_proj(times.unsqueeze(-1).unsqueeze(-1).expand(B, T, V, 1))  # (B, T, V, d)
        emb = v_emb + t_emb
        emb = emb * mask.float().unsqueeze(-1)  # zero out missing
        return emb


class _GraphAttention(nn.Module):
    """One round of graph attention: message passing over sensor adjacency."""

    def __init__(self, d_model: int, n_sensors: int):
        super().__init__()
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        # Learnable adjacency (not masked — softmax normalises)
        self.A = nn.Parameter(torch.zeros(n_sensors, n_sensors))
        nn.init.xavier_uniform_(self.A.unsqueeze(0))

    def forward(self, h: Tensor) -> Tensor:
        """h: (B, V, d) → (B, V, d)"""
        A = F.softmax(self.A, dim=-1)  # (V, V)
        Q = self.W_q(h)  # (B, V, d)
        K = self.W_k(h)
        V = self.W_v(h)

        # Attention scores: (B, V, V)
        scores = torch.bmm(Q, K.transpose(-1, -2)) / math.sqrt(Q.shape[-1])
        scores = scores * A.unsqueeze(0)  # mask by learnable adjacency
        attn = F.softmax(scores, dim=-1)
        return torch.bmm(attn, V)  # (B, V, d)


class Raindrop(IrregularTSModel):
    """Raindrop for irregular MVTS classification.

    Complexity: O(n * V * E) time, O(n * V²) memory.
    V = n_channels (sensors), E = edges ≈ V² worst case.
    """

    def __init__(
        self,
        n_channels: int,
        n_classes: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__(n_channels, n_classes)
        self.d_model = d_model

        self.sensor_embed = _SensorEmbedding(d_model)

        # Stack of graph attention layers
        self.graph_layers = nn.ModuleList([
            _GraphAttention(d_model, n_channels) for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])

        # Temporal self-attention across time steps (on sensor-aggregated features)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model * n_channels,
            nhead=n_heads,
            dim_feedforward=d_model * n_channels * 2,
            dropout=dropout,
            batch_first=True,
        )
        self.temporal_attn = nn.TransformerEncoder(encoder_layer, num_layers=1)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model * n_channels, n_classes),
        )

    @property
    def complexity_class(self) -> str:
        return "O(n*V*E)"

    def forward(self, batch: IrregularBatch) -> Tensor:
        x = batch.values.float()   # (B, T, V)
        m = batch.mask             # (B, T, V)
        t = batch.times.float()    # (B, T)

        B, T, V = x.shape

        # Per-sensor embeddings: (B, T, V, d)
        h = self.sensor_embed(x, t, m)

        # Graph attention per time step
        for graph_attn, norm in zip(self.graph_layers, self.norms):
            h_flat = h.view(B * T, V, self.d_model)
            h_flat = norm(h_flat + graph_attn(h_flat))
            h = h_flat.view(B, T, V, self.d_model)

        # Flatten sensors: (B, T, V*d)
        h_seq = h.view(B, T, V * self.d_model)

        # Temporal attention with padding mask
        pad_mask = ~batch.mask.any(dim=-1)  # (B, T) True = padding
        h_seq = self.temporal_attn(h_seq, src_key_padding_mask=pad_mask)

        # Mean pool over observed time steps
        obs_mask = (~pad_mask).float().unsqueeze(-1)  # (B, T, 1)
        pooled = (h_seq * obs_mask).sum(1) / obs_mask.sum(1).clamp(min=1)  # (B, V*d)

        return self.classifier(pooled)
