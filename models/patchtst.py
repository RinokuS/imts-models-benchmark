"""PatchTST + FlashAttention for irregular MVTS classification.

Nie, Nguyen, Sinthong & Kalagnanam, ICLR 2023.
FlashAttention: Dao et al., NeurIPS 2022.

Key idea:
  1. Patch the imputed/zero-filled sequence into non-overlapping windows
  2. Project each patch into a token embedding
  3. Apply multi-head attention across patches (fewer tokens → faster attention)
  4. torch.nn.functional.scaled_dot_product_attention automatically dispatches
     to FlashAttention 2 when running on CUDA with compatible inputs
  5. Pool and classify

Complexity:
  - FLOP: O((n/p)^2 * d) = O(n^2/p^2 * d) — quadratic in patches
  - HBM bandwidth: O(n/p * d) with FlashAttention (no n×n materialization)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from data.base import IrregularBatch
from models.base import IrregularTSModel


class _PatchEmbedding(nn.Module):
    """Divide sequence into patches and project to d_model."""

    def __init__(self, n_channels: int, patch_len: int, stride: int, d_model: int):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        # Project patch (patch_len * n_channels * 2) → d_model (values + mask)
        self.proj = nn.Linear(patch_len * n_channels * 2, d_model)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        """
        x:    (B, T, D) — zero-filled values
        mask: (B, T, D) — observation mask

        Returns: (B, n_patches, d_model)
        """
        B, T, D = x.shape
        P = self.patch_len
        S = self.stride

        # Pad so T is divisible into complete patches
        n_patches = max(1, (T - P) // S + 1)
        needed = P + (n_patches - 1) * S
        if needed > T:
            pad_len = needed - T
            x = F.pad(x, (0, 0, 0, pad_len))
            mask = F.pad(mask.float(), (0, 0, 0, pad_len)).bool()

        # Unfold: (B, n_patches, P*D)
        xm = torch.cat([x, mask.float()], dim=-1)  # (B, T, 2D)
        patches = xm.unfold(1, P, S)               # (B, n_patches, 2D, P)
        patches = patches.permute(0, 1, 3, 2).contiguous().view(B, -1, P * D * 2)

        return self.proj(patches)  # (B, n_patches, d_model)


class _SDPAAttentionBlock(nn.Module):
    """Transformer block using SDPA (FlashAttention-compatible)."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        assert d_model % n_heads == 0

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        B, L, D = x.shape
        H, Hd = self.n_heads, self.head_dim

        # Self-attention via SDPA (auto-dispatches to FlashAttention on CUDA)
        qkv = self.qkv(self.norm1(x)).view(B, L, 3, H, Hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # each (B, H, L, Hd)

        # scaled_dot_product_attention uses FlashAttention 2 automatically
        # when inputs are contiguous and on CUDA (PyTorch >= 2.0)
        attn_out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, D)
        x = x + self.dropout(self.out_proj(attn_out))

        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class PatchTST(IrregularTSModel):
    """PatchTST with SDPA/FlashAttention for irregular MVTS classification.

    Complexity (FLOP): O((n/p)^2 * d)
    Complexity (HBM):  O(n/p * d) with FlashAttention
    """

    def __init__(
        self,
        n_channels: int,
        n_classes: int,
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 3,
        d_ff: int = 256,
        dropout: float = 0.1,
        use_flash_attn: bool = True,
    ):
        super().__init__(n_channels, n_classes)

        self.patch_embed = _PatchEmbedding(n_channels, patch_len, stride, d_model)

        # Positional encoding
        self.pos_embed = nn.Parameter(torch.zeros(1, 512, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.blocks = nn.ModuleList([
            _SDPAAttentionBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, n_classes),
        )

    @property
    def complexity_class(self) -> str:
        return "O(n^2*d) FLOP / O(n*d) HBM"

    def forward(self, batch: IrregularBatch) -> Tensor:
        x = batch.values.float()   # (B, T, D)
        m = batch.mask             # (B, T, D)

        tokens = self.patch_embed(x, m)  # (B, n_patches, d_model)
        B, N, D = tokens.shape

        # Add positional embedding (truncate or repeat if needed)
        pos = self.pos_embed[:, :N, :]
        tokens = tokens + pos

        for block in self.blocks:
            tokens = block(tokens)

        # CLS-style: mean pool over patches
        pooled = self.norm(tokens).mean(dim=1)  # (B, d_model)
        return self.classifier(pooled)
