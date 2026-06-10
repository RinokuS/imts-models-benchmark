"""Quality metrics for binary classification on irregular MVTS."""

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)


def compute_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    threshold: float | None = None,
) -> dict[str, float]:
    """Compute AUROC, AUPRC, F1-macro and Brier Score from raw logits.

    Args:
        logits: (N, 2) float — raw model outputs (before softmax)
        labels: (N,) long — ground truth binary labels
        threshold: decision threshold for F1; if None, uses 0.5

    Returns:
        dict with keys: auroc, auprc, f1, brier
    """
    probs = torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy()
    y_true = labels.cpu().numpy()

    if len(np.unique(y_true)) < 2:
        return {"auroc": float("nan"), "auprc": float("nan"), "f1": float("nan"), "brier": float("nan")}

    auroc = float(roc_auc_score(y_true, probs))
    auprc = float(average_precision_score(y_true, probs))
    brier = float(brier_score_loss(y_true, probs))

    thr = threshold if threshold is not None else 0.5
    preds = (probs >= thr).astype(int)
    f1 = float(f1_score(y_true, preds, average="macro", zero_division=0))

    return {"auroc": auroc, "auprc": auprc, "f1": f1, "brier": brier}


def find_best_threshold(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Find threshold that maximises F1 on the given split."""
    probs = torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy()
    y_true = labels.cpu().numpy()

    best_thr, best_f1 = 0.5, 0.0
    for thr in np.linspace(0.1, 0.9, 81):
        preds = (probs >= thr).astype(int)
        f1 = f1_score(y_true, preds, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
    return float(best_thr)
