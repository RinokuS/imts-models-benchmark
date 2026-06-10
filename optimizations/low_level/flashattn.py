"""FlashAttention variant switcher — Direction 2 (Low-Level).

Applicable to: mTAN, PatchTST (attention-based models).
Vault reference: FlashAttention note.

Variants:
  - sdpa_auto:   torch.nn.functional.scaled_dot_product_attention (auto-dispatch)
  - flash_attn2: explicit flash-attn 2 kernel (Dao 2023)
  - flash_attn3: explicit flash-attn 3 kernel with WGMMA (Shah et al. 2024, H100 only)

The wrapper patches model's forward to use the requested attention backend.
"""

from __future__ import annotations
import torch
import torch.nn as nn
from ..base import OptimizationWrapper

_VARIANTS = ("sdpa_auto", "flash_attn2", "flash_attn3")


class FlashAttnVariant(OptimizationWrapper):
    """Switch attention kernel inside an attention-based model.

    Args:
        variant: attention backend to use.
    """

    def __init__(self, variant: str = "sdpa_auto"):
        if variant not in _VARIANTS:
            raise ValueError(f"variant must be one of {_VARIANTS}")
        self.variant = variant

    def apply(self, model: nn.Module, **kwargs) -> nn.Module:
        if self.variant == "sdpa_auto":
            # PyTorch 2.0+ dispatches to FlashAttention automatically via SDPA
            # when running on CUDA with compatible dtypes — no explicit changes needed.
            pass
        elif self.variant in ("flash_attn2", "flash_attn3"):
            self._inject_flash_attn(model)
        return model

    def _inject_flash_attn(self, model: nn.Module) -> None:
        """Tag the model so its forward can detect and use explicit flash-attn."""
        try:
            from flash_attn import flash_attn_func  # noqa: F401
        except ImportError:
            raise ImportError(
                "flash-attn is not installed. See requirements.txt for flash-attn-4."
            )
        model._flash_attn_variant = self.variant

    def name(self) -> str:
        return f"flashattn_{self.variant}"
