"""Training and evaluation loop shared by all benchmark models."""

import time
from typing import Callable

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from data.base import IrregularBatch, collate_irregular
from evaluation.metrics import compute_metrics, find_best_threshold


def make_loader(dataset, batch_size: int, shuffle: bool, num_workers: int = 0) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_irregular,
        num_workers=num_workers,
        pin_memory=True,
    )


def train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable,
    device: torch.device,
    grad_clip: float = 1.0,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    n_batches = 0
    t0 = time.perf_counter()

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch)
        loss = loss_fn(logits, batch.labels)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1

    return {
        "loss": total_loss / max(n_batches, 1),
        "epoch_time_s": time.perf_counter() - t0,
    }


@torch.no_grad()
def eval_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_fn: Callable,
    device: torch.device,
) -> tuple[dict[str, float], Tensor, Tensor]:
    """Returns (metrics_dict, all_logits, all_labels)."""
    model.eval()
    all_logits, all_labels = [], []
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        batch = batch.to(device)
        logits = model(batch)
        loss = loss_fn(logits, batch.labels)
        all_logits.append(logits.cpu())
        all_labels.append(batch.labels.cpu())
        total_loss += loss.item()
        n_batches += 1

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    metrics = compute_metrics(all_logits, all_labels)
    metrics["loss"] = total_loss / max(n_batches, 1)
    return metrics, all_logits, all_labels


def fit(
    model: torch.nn.Module,
    train_ds,
    val_ds,
    *,
    device: torch.device,
    batch_size: int = 64,
    epochs: int = 100,
    patience: int = 10,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    lr_patience: int = 5,
    class_weights: Tensor | None = None,
    grad_clip: float = 1.0,
    verbose: bool = True,
) -> dict:
    """Full training loop with early stopping on val AUROC.

    Returns a dict with training history and best checkpoint state_dict.
    """
    train_loader = make_loader(train_ds, batch_size, shuffle=True)
    val_loader = make_loader(val_ds, batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=lr_patience, factor=0.5
    )

    if class_weights is not None:
        class_weights = class_weights.to(device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

    best_auroc = -1.0
    best_state = None
    no_improve = 0
    history: list[dict] = []

    for epoch in range(1, epochs + 1):
        train_stats = train_epoch(model, train_loader, optimizer, loss_fn, device, grad_clip)
        val_metrics, _, _ = eval_epoch(model, val_loader, loss_fn, device)
        scheduler.step(val_metrics["auroc"])

        record = {"epoch": epoch, **train_stats, **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(record)

        if verbose:
            print(
                f"Epoch {epoch:3d} | loss={train_stats['loss']:.4f} "
                f"| val AUROC={val_metrics['auroc']:.4f} "
                f"| val AUPRC={val_metrics['auprc']:.4f}"
            )

        if val_metrics["auroc"] > best_auroc:
            best_auroc = val_metrics["auroc"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch} (patience={patience})")
                break

    return {"history": history, "best_auroc": best_auroc, "best_state": best_state}
