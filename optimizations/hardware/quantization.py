"""Post-training quantization wrappers — Direction 3 (Hardware).

Two stacks as decided:
  - PTQInt8Wrapper:  torch.ao quantization (INT8, native PyTorch)
  - BNBInt4Wrapper:  bitsandbytes NF4/INT4 (simpler API, also works for TS)

Vault reference: Quantization - Overview note, Adaptive Quantization note.
Expected: INT8 → 4× memory, 2–4× speedup, ~0.5–1% AUROC drop.
          INT4 → 8× memory, 3–6× speedup, ~1–3% AUROC drop.
"""

from __future__ import annotations
from typing import Iterable
import torch
import torch.nn as nn
from ..base import OptimizationWrapper


class PTQInt8Wrapper(OptimizationWrapper):
    """Post-Training INT8 Quantization — uses torchao when available, else
    falls back to the legacy torch.ao eager-mode API.

    Dynamic quantization calibrates per-tensor scales at runtime — no
    calibration dataset required. Quantizes Linear (and optionally GRU) layers.
    Yields 2–4× speedup on CPU; for GPU static quant use torch.ao prepare/convert
    with an observer.

    Args:
        layer_types: module types to quantize (default: Linear only).
    """

    def __init__(self, layer_types: tuple = (nn.Linear,)):
        self.layer_types = layer_types

    def apply(self, model: nn.Module, **kwargs) -> nn.Module:
        try:
            import torchao
            from torchao.quantization import quantize_, int8_dynamic_activation_int8_weight
            quantize_(model, int8_dynamic_activation_int8_weight())
            return model
        except (ImportError, Exception):
            # Legacy fallback — torch.ao dynamic quantization is CPU-only.
            # Move model to CPU so quantized ops can run; device detection in
            # run_benchmark.py will see the CPU device and re-run baseline on CPU.
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                quantized = torch.quantization.quantize_dynamic(
                    model.cpu(),
                    qconfig_spec=set(self.layer_types),
                    dtype=torch.qint8,
                )
            quantized._ptq_on_cpu = True
            return quantized

    def name(self) -> str:
        return "quant_int8_ptq_torchao"


class BNBInt4Wrapper(OptimizationWrapper):
    """INT4 / NF4 quantization via bitsandbytes.

    Uses NormalFloat4 (NF4) quantization — optimal for weights that follow
    a normal distribution (per vault: Non-Uniform Quantization note).
    Requires bitsandbytes >= 0.41.

    Args:
        use_nf4: True → NF4 (better accuracy), False → pure INT4.
        double_quant: quantize the quantization constants themselves (saves
                      additional ~0.5 bits/param).
    """

    def __init__(self, use_nf4: bool = True, double_quant: bool = True):
        self.use_nf4 = use_nf4
        self.double_quant = double_quant

    def apply(self, model: nn.Module, **kwargs) -> nn.Module:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError("bitsandbytes is not installed. Run: pip install bitsandbytes")

        quant_type = "nf4" if self.use_nf4 else "int4"
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and module.weight.numel() > 256:
                parent = self._get_parent(model, name)
                child_name = name.split(".")[-1]
                new_layer = bnb.nn.Linear4bit(
                    module.in_features,
                    module.out_features,
                    bias=module.bias is not None,
                    quant_type=quant_type,
                    compress_statistics=self.double_quant,
                )
                setattr(parent, child_name, new_layer)
        return model

    @staticmethod
    def _get_parent(model: nn.Module, name: str) -> nn.Module:
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        return parent

    def name(self) -> str:
        suffix = "nf4" if self.use_nf4 else "int4"
        dq = "_dq" if self.double_quant else ""
        return f"quant_{suffix}_bnb{dq}"
