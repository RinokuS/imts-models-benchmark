"""ODE solver variant comparison — Direction 2 (Low-Level).

Applicable to: Latent ODE, Neural CDE.
Vault reference: ODE Solver Acceleration note.

Strategies:
  - dopri5_adaptive: Dormand-Prince adaptive (baseline, best accuracy)
  - rk4_fixed:       Fixed-step RK4 + torch.compile → CUDA Graph
  - torchode:        Batch-parallel JIT solver (Lienen & Günnemann 2022)
"""

from __future__ import annotations
import torch.nn as nn
from ..base import OptimizationWrapper

_SOLVER_BACKENDS = ("dopri5_adaptive", "rk4_fixed", "torchode")


class ODESolverWrapper(OptimizationWrapper):
    """Swap the ODE solver used inside a Neural ODE / CDE model.

    The model must expose a `solver` attribute (string) and optionally an
    `ode_func` attribute (callable) so we can rewire it.

    Args:
        backend: one of 'dopri5_adaptive', 'rk4_fixed', 'torchode'.
        n_fixed_steps: only for rk4_fixed — number of uniform steps.
    """

    def __init__(self, backend: str = "dopri5_adaptive", n_fixed_steps: int = 20):
        if backend not in _SOLVER_BACKENDS:
            raise ValueError(f"backend must be one of {_SOLVER_BACKENDS}")
        self.backend = backend
        self.n_fixed_steps = n_fixed_steps

    def apply(self, model: nn.Module, **kwargs) -> nn.Module:
        if not hasattr(model, "solver"):
            raise AttributeError(
                f"{type(model).__name__} has no 'solver' attribute; "
                "ODESolverWrapper requires a Neural ODE/CDE model."
            )
        if self.backend == "dopri5_adaptive":
            model.solver = "dopri5"
        elif self.backend == "rk4_fixed":
            model.solver = "rk4"
            model.options = {"step_size": 1.0 / self.n_fixed_steps}
        elif self.backend == "torchode":
            self._patch_torchode(model)
        return model

    def _patch_torchode(self, model: nn.Module) -> None:
        try:
            import torchode as to  # noqa: F401
        except ImportError:
            raise ImportError(
                "torchode is not installed. Run: pip install torchode"
            )
        model._use_torchode = True

    def name(self) -> str:
        if self.backend == "rk4_fixed":
            return f"ode_{self.backend}_{self.n_fixed_steps}steps"
        return f"ode_{self.backend}"

    def metadata(self) -> dict:
        return {
            "optimization": self.name(),
            "ode_backend": self.backend,
            "n_fixed_steps": self.n_fixed_steps if self.backend == "rk4_fixed" else None,
        }
