from abc import ABC, abstractmethod
from typing import Any
import torch.nn as nn


class OptimizationWrapper(ABC):
    """Wraps an IrregularTSModel with a system/hardware optimization.

    Direction 2 (Low-Level): batching, torch.compile, FlashAttention variants,
                             ODE solvers, selective scan backends
    Direction 3 (Hardware):  precision, quantization, structured pruning,
                             knowledge distillation
    """

    @abstractmethod
    def apply(self, model: nn.Module, **kwargs) -> nn.Module:
        """Apply optimization and return modified model."""

    @abstractmethod
    def name(self) -> str:
        """Human-readable label for reports (e.g. 'int8_ptq', 'compile_max_autotune')."""

    def metadata(self) -> dict[str, Any]:
        """Config dict attached to every experiment result row."""
        return {"optimization": self.name()}
