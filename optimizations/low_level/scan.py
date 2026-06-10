"""Mamba selective scan backend comparison — Direction 2 (Low-Level).

Applicable to: Mamba only.
Vault reference: Selective Scan (Mamba) note.

Backends:
  - python: pure PyTorch sequential scan (baseline, slow)
  - cuda:   compiled CUDA C++ kernel from mamba-ssm (Blelloch parallel prefix scan)
"""

from __future__ import annotations
import torch.nn as nn
from ..base import OptimizationWrapper

_BACKENDS = ("python", "cuda")


class MambaScanWrapper(OptimizationWrapper):
    """Switch selective scan backend in a Mamba model.

    The official mamba-ssm package uses the CUDA kernel by default.
    Setting backend='python' forces the pure PyTorch fallback — useful
    to quantify how much the CUDA kernel contributes (vault: ~5× throughput
    vs Transformer, largely from the parallel prefix scan).

    Args:
        backend: 'python' or 'cuda'.
    """

    def __init__(self, backend: str = "cuda"):
        if backend not in _BACKENDS:
            raise ValueError(f"backend must be one of {_BACKENDS}")
        self.backend = backend

    def apply(self, model: nn.Module, **kwargs) -> nn.Module:
        if not hasattr(model, "_mamba_scan_backend"):
            raise AttributeError(
                f"{type(model).__name__} has no '_mamba_scan_backend' attribute; "
                "patch the Mamba model wrapper to expose this flag."
            )
        if self.backend == "python" and getattr(model, "use_official", False):
            # Replace Sequential(MambaOfficial, LayerNorm) → _MambaBlock so that
            # forward() exercises the pure-PyTorch selective scan, not the CUDA kernel.
            from models.mamba_ts import _MambaBlock
            orig_device = model.input_proj.weight.device
            model.layers = nn.ModuleList([
                _MambaBlock(model._d_model, model._d_state, model._d_conv, model._expand)
                for _ in range(model._n_layers)
            ]).to(orig_device)
            model.use_official = False
        model._mamba_scan_backend = self.backend
        return model

    def name(self) -> str:
        return f"mamba_scan_{self.backend}"
