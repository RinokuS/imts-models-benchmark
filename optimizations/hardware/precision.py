"""Mixed-precision wrappers — Direction 3 (Hardware).

Vault reference: Quantization - Overview note.

FP32 → BF16/FP16: ~2× memory, 1.5–2× speed, <0.1% accuracy drop.
No calibration data needed — pure dtype cast.

Uses torch.autocast rather than model.to(dtype) so that models with internal
`.float()` casts (common in sequence models) still work correctly. Autocast
selectively lowers eligible ops to the target dtype while keeping precision
where needed (softmax, normalization, etc.).
"""

from __future__ import annotations
import torch
import torch.nn as nn
from ..base import OptimizationWrapper

_DTYPES = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


class _AutocastWrapper(nn.Module):
    """Thin wrapper that runs forward under torch.autocast."""

    def __init__(self, model: nn.Module, dtype: torch.dtype):
        super().__init__()
        self._model = model
        self._dtype = dtype
        try:
            self._device_type = next(model.parameters()).device.type
        except StopIteration:
            self._device_type = "cpu"

    def forward(self, batch):
        with torch.autocast(device_type=self._device_type, dtype=self._dtype):
            return self._model(batch)


class PrecisionWrapper(OptimizationWrapper):
    """Wrap model forward with torch.autocast for mixed-precision inference.

    BF16 is preferred over FP16 for training stability (larger exponent range).
    FP16 is typically faster on older Ampere cards without BF16 Tensor Cores.
    Note: torch.autocast on CPU supports only bfloat16 (not float16).

    Args:
        dtype: 'fp32' (no-op), 'fp16', or 'bf16'.
    """

    def __init__(self, dtype: str = "bf16"):
        if dtype not in _DTYPES:
            raise ValueError(f"dtype must be one of {list(_DTYPES)}")
        self.dtype_name = dtype
        self.dtype = _DTYPES[dtype]

    def apply(self, model: nn.Module, **kwargs) -> nn.Module:
        if self.dtype_name == "fp32":
            return model
        return _AutocastWrapper(model, self.dtype)

    def name(self) -> str:
        return f"precision_{self.dtype_name}"
