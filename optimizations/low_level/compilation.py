"""torch.compile wrapper — Direction 2 (Low-Level)."""

import time
import torch
import torch.nn as nn
from ..base import OptimizationWrapper

_VALID_MODES = ("default", "reduce-overhead", "max-autotune")


class CompileWrapper(OptimizationWrapper):
    """Wraps a model with torch.compile.

    Applies kernel fusion, loop unrolling, and auto-generated Triton kernels.
    Expected speedup: 1.5–4× on inference (vault: Compiler Optimizations note).
    No accuracy change — purely a runtime transformation.

    Args:
        mode: TorchInductor mode. "reduce-overhead" uses CUDA Graphs (best for
              fixed-shape inference). "max-autotune" searches tiling parameters
              (high compile time, best throughput).
    """

    def __init__(self, mode: str = "reduce-overhead"):
        if mode not in _VALID_MODES:
            raise ValueError(f"mode must be one of {_VALID_MODES}, got {mode!r}")
        self.mode = mode
        self._compile_time_s: float | None = None

    def apply(self, model: nn.Module, **kwargs) -> nn.Module:
        t0 = time.perf_counter()
        compiled = torch.compile(model, mode=self.mode)
        self._compile_time_s = time.perf_counter() - t0
        return compiled

    def name(self) -> str:
        return f"compile_{self.mode.replace('-', '_')}"

    def metadata(self) -> dict:
        return {
            "optimization": self.name(),
            "compile_mode": self.mode,
            "compile_time_s": self._compile_time_s,
        }
