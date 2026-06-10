from abc import ABC, abstractmethod

import torch.nn as nn
from torch import Tensor

from data.base import IrregularBatch


class IrregularTSModel(nn.Module, ABC):
    """Abstract base for all benchmark models.

    All models receive an IrregularBatch and return class logits (B, n_classes).
    """

    def __init__(self, n_channels: int, n_classes: int):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

    @abstractmethod
    def forward(self, batch: IrregularBatch) -> Tensor:
        """Returns logits of shape (B, n_classes)."""
        ...

    @property
    @abstractmethod
    def complexity_class(self) -> str:
        """Theoretical time complexity string, e.g. 'O(n*d)'."""
        ...

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
