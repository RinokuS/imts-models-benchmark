"""Knowledge distillation trainer — Direction 3 (Hardware).

Pair: Raindrop (teacher) → DLinear (student).
Vault reference: Knowledge Distillation note (DistilTS 2026, TimeDistill 2024).

Method: Response-based KD (Hinton et al., 2015).
  L = α · CE(student_logits, hard_labels)
      + (1-α) · KL(student_logits/T || teacher_logits/T)
  T = temperature (softens teacher probabilities to expose dark knowledge).

The teacher is frozen; only student weights are updated.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from ..base import OptimizationWrapper


class KnowledgeDistillationWrapper(OptimizationWrapper):
    """Wraps the student model and provides the KD training objective.

    This wrapper does NOT run training — it provides the `distillation_loss`
    method consumed by the training loop in scripts/run_optimizations.py.

    Args:
        teacher: frozen teacher model (Raindrop or any IrregularTSModel).
        temperature: KD temperature T; higher T → softer distributions.
        alpha: weight for the hard-label CE loss (1-alpha for KD loss).
    """

    def __init__(
        self,
        teacher: nn.Module,
        temperature: float = 4.0,
        alpha: float = 0.3,
    ):
        self.teacher = teacher
        self.temperature = temperature
        self.alpha = alpha

        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()

    def apply(self, student: nn.Module, **kwargs) -> nn.Module:
        """Attach teacher reference to student for access during training."""
        student._kd_teacher = self.teacher
        student._kd_temperature = self.temperature
        student._kd_alpha = self.alpha
        return student

    def distillation_loss(
        self,
        student_logits: Tensor,
        teacher_logits: Tensor,
        labels: Tensor,
    ) -> Tensor:
        """Compute combined KD + hard-label loss.

        Args:
            student_logits: raw logits from student (B, C).
            teacher_logits: raw logits from teacher (B, C), already detached.
            labels: ground-truth integer class labels (B,).
        """
        T = self.temperature
        alpha = self.alpha

        hard_loss = F.cross_entropy(student_logits, labels)
        soft_loss = F.kl_div(
            F.log_softmax(student_logits / T, dim=-1),
            F.softmax(teacher_logits / T, dim=-1),
            reduction="batchmean",
        ) * (T ** 2)

        return alpha * hard_loss + (1.0 - alpha) * soft_loss

    def name(self) -> str:
        teacher_name = type(self.teacher).__name__.lower()
        return f"kd_{teacher_name}_T{self.temperature}_a{self.alpha}"

    def metadata(self) -> dict:
        return {
            "optimization": self.name(),
            "teacher": type(self.teacher).__name__,
            "kd_temperature": self.temperature,
            "kd_alpha": self.alpha,
        }
