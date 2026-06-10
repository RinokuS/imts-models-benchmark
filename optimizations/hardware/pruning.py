"""Structured pruning wrapper — Direction 3 (Hardware).

Vault reference: Structured Pruning note, Model Pruning - Overview note.

Unstructured pruning (sparse weights) typically gives no GPU speedup without
dedicated sparse hardware. Structured pruning removes entire channels/heads,
yielding dense smaller matrices that are natively faster.

Pipeline: after training → prune → fine-tune (10–20% of original epochs).
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from ..base import OptimizationWrapper
from .shrink import shrink_sequential_linears


class StructuredPruningWrapper(OptimizationWrapper):
    """Structured L1-norm pruning on Linear and Conv1d layers.

    Removes `amount` fraction of output channels per layer based on
    L1-norm of their weight rows (magnitude criterion).

    Args:
        amount: fraction of channels to prune (0.0–1.0).  0.5 = 50% sparsity.
        layer_types: which layer types to prune.
        make_permanent: remove pruning reparameterization after application so
                        the pruned model can be exported / quantized.
    """

    def __init__(
        self,
        amount: float = 0.5,
        layer_types: tuple = (nn.Linear, nn.Conv1d),
        make_permanent: bool = True,
    ):
        if not 0.0 < amount < 1.0:
            raise ValueError(f"amount must be in (0, 1), got {amount}")
        self.amount = amount
        self.layer_types = layer_types
        self.make_permanent = make_permanent

    def apply(self, model: nn.Module, **kwargs) -> nn.Module:
        for module in model.modules():
            if isinstance(module, self.layer_types):
                prune.ln_structured(module, name="weight", amount=self.amount, n=1, dim=0)
                if self.make_permanent:
                    prune.remove(module, "weight")
        n_shrunk = shrink_sequential_linears(model)
        if n_shrunk:
            print(f"  shrink_sequential_linears: {n_shrunk} pair(s) physically reduced")
        return model

    def name(self) -> str:
        pct = int(self.amount * 100)
        return f"structured_pruning_{pct}pct"

    def metadata(self) -> dict:
        return {
            "optimization": self.name(),
            "pruning_amount": self.amount,
            "pruning_type": "structured_l1_channel",
        }

    @staticmethod
    def compression_ratio(original: nn.Module, pruned: nn.Module) -> float:
        """Ratio of non-zero params before vs after pruning."""
        orig_params = sum(p.numel() for p in original.parameters())
        pruned_params = sum(
            (p != 0).sum().item() for p in pruned.parameters()
        )
        return orig_params / max(pruned_params, 1)
