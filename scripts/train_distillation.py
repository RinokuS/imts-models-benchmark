"""Raindrop → DLinear knowledge distillation training.

Trains a teacher model, then distils it into a lighter student via response-based
KD (Hinton 2015).  Records AUROC, speedup, and parameter ratio.

Usage:
    python scripts/train_distillation.py --dataset p12 --seeds 42 123 2024
    python scripts/train_distillation.py --dataset p19 --teacher raindrop \
        --student dlinear --epochs 100 --device cuda

Outputs:
    results/checkpoints/<model>_<dataset>_seed<seed>.pt   teacher checkpoints
    results/tables/distillation_<dataset>_seed<seed>.json per-run metrics
    results/benchmark_inference.csv                        shared inference CSV
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
import yaml

from data.base import collate_irregular
from data.synthetic import SyntheticDataset
from evaluation.runner import make_loader, eval_epoch, fit
from evaluation.profiler import profile_model
from optimizations.hardware.distillation import KnowledgeDistillationWrapper

RESULTS_DIR = PROJECT_ROOT / "results"
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"
TABLES_DIR = RESULTS_DIR / "tables"
INFERENCE_CSV = RESULTS_DIR / "benchmark_inference.csv"

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True, warn_only=True)


# ---------------------------------------------------------------------------
# Model & dataset builders (mirrors train.py)
# ---------------------------------------------------------------------------

def _filter(cfg: dict, keys: list) -> dict:
    return {k: cfg[k] for k in keys if k in cfg}


def load_config(config_dir: Path, name: str) -> dict:
    path = config_dir / "models" / f"{name}.yaml"
    if not path.exists():
        return {}
    with open(path) as f:
        return yaml.safe_load(f)


def build_model(name: str, cfg: dict, n_channels: int, n_classes: int, seq_len: int = 500):
    if name == "dlinear":
        from models.dlinear import DLinear
        return DLinear(n_channels, n_classes, seq_len=seq_len,
                       **_filter(cfg, ["hidden_dim", "kernel_size"]))
    elif name == "gru_d":
        from models.gru_d import GRUD
        return GRUD(n_channels, n_classes,
                    **_filter(cfg, ["hidden_dim", "n_layers", "dropout"]))
    elif name == "raindrop":
        from models.raindrop import Raindrop
        return Raindrop(n_channels, n_classes,
                        **_filter(cfg, ["d_model", "n_heads", "n_layers", "dropout"]))
    elif name == "mtan":
        from models.mtan import MTAN
        return MTAN(n_channels, n_classes,
                    **_filter(cfg, ["hidden_dim", "n_heads", "n_ref_points",
                                    "n_layers", "dropout", "time_embed_dim"]))
    else:
        raise ValueError(f"Unknown model for distillation: {name}")


def load_dataset(name: str, ds_cfg: dict, split: str, norm_stats=None):
    if name in ("synthetic", "syn"):
        cfg = ds_cfg.get("synthetic", {})
        ds = SyntheticDataset(
            n_samples=cfg.get("n_samples", 5000),
            seq_len=cfg.get("seq_len", 500),
            n_channels=cfg.get("n_channels", 10),
            missing_frac=cfg.get("missing_frac", 0.40),
            seed=cfg.get("seed", 0),
        )
        n = len(ds)
        tr_end = int(n * 0.7)
        va_end = int(n * 0.85)
        idx = torch.randperm(n, generator=torch.Generator().manual_seed(0))
        split_idx = {"train": idx[:tr_end], "val": idx[tr_end:va_end], "test": idx[va_end:]}
        from torch.utils.data import Subset
        return Subset(ds, split_idx[split].tolist()), None, ds.n_channels, ds.seq_len
    elif name == "p12":
        from data.physionet_p12 import PhysioNetP12Dataset
        ds_dir = ds_cfg.get("physionet_p12", {}).get("data_dir", "data/raw/physionet2012")
        kwargs = {"_stats": norm_stats} if norm_stats else {}
        ds = PhysioNetP12Dataset(ds_dir, split=split, **kwargs)
        return ds, getattr(ds, "_stats", None), 36, 215
    elif name == "p19":
        from data.physionet_p19 import PhysioNetP19Dataset
        ds_dir = ds_cfg.get("physionet_p19", {}).get("data_dir", "data/raw/physionet2019")
        kwargs = {"_stats": norm_stats} if norm_stats else {}
        ds = PhysioNetP19Dataset(ds_dir, split=split, **kwargs)
        return ds, getattr(ds, "_stats", None), 34, 336
    else:
        raise ValueError(f"Unknown dataset: {name}")


# ---------------------------------------------------------------------------
# KD training loop
# ---------------------------------------------------------------------------

def train_epoch_kd(
    student: nn.Module,
    teacher: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    wrapper: KnowledgeDistillationWrapper,
    device: torch.device,
    grad_clip: float = 1.0,
) -> dict:
    student.train()
    teacher.eval()
    total_loss = 0.0
    n_batches = 0
    t0 = time.perf_counter()

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)

        logits_s = student(batch)
        with torch.no_grad():
            logits_t = teacher(batch)

        # Teacher (Raindrop) can produce NaN for specific P19 sequences due
        # to numerical overflow in time-embedding exponentials. Skip the batch
        # to avoid poisoning the student's gradients.
        if torch.isnan(logits_t).any():
            continue

        loss = wrapper.distillation_loss(logits_s, logits_t, batch.labels)
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(student.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1

    return {
        "loss": total_loss / max(n_batches, 1),
        "epoch_time_s": time.perf_counter() - t0,
    }


def fit_student_kd(
    student: nn.Module,
    teacher: nn.Module,
    train_ds,
    val_ds,
    wrapper: KnowledgeDistillationWrapper,
    *,
    device: torch.device,
    batch_size: int = 64,
    epochs: int = 100,
    patience: int = 10,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    lr_patience: int = 5,
    class_weights=None,
    verbose: bool = True,
) -> dict:
    train_loader = make_loader(train_ds, batch_size, shuffle=True)
    val_loader = make_loader(val_ds, batch_size, shuffle=False)

    optimizer = torch.optim.Adam(student.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=lr_patience, factor=0.5
    )
    loss_fn = torch.nn.CrossEntropyLoss(
        weight=class_weights.to(device) if class_weights is not None else None
    )

    best_auroc = -1.0
    best_state = None
    no_improve = 0
    history = []

    for epoch in range(1, epochs + 1):
        train_stats = train_epoch_kd(
            student, teacher, train_loader, optimizer, wrapper, device
        )
        val_metrics, _, _ = eval_epoch(student, val_loader, loss_fn, device)
        scheduler.step(val_metrics["auroc"])

        record = {
            "epoch": epoch,
            **train_stats,
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(record)

        if verbose:
            print(
                f"  Epoch {epoch:3d} | kd_loss={train_stats['loss']:.4f}"
                f" | val AUROC={val_metrics['auroc']:.4f}"
                f" | val AUPRC={val_metrics['auprc']:.4f}"
            )

        if val_metrics["auroc"] > best_auroc:
            best_auroc = val_metrics["auroc"]
            best_state = {k: v.cpu().clone() for k, v in student.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                if verbose:
                    print(f"  Early stopping at epoch {epoch}")
                break

    return {"history": history, "best_auroc": best_auroc, "best_state": best_state}


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def append_row_csv(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ---------------------------------------------------------------------------
# Per-seed pipeline
# ---------------------------------------------------------------------------

def run_seed(
    seed: int,
    teacher_name: str,
    student_name: str,
    dataset: str,
    ds_cfg: dict,
    config_dir: Path,
    device: torch.device,
    epochs: int,
    batch_size: int,
    patience: int,
    temperature: float,
    alpha: float,
    verbose: bool,
) -> None:
    set_seeds(seed)
    print(f"\n{'='*60}")
    print(f"  Dataset={dataset}  Teacher={teacher_name}  Student={student_name}  Seed={seed}")
    print(f"{'='*60}")

    # --- Data ---
    print("Loading datasets...")
    train_ds, norm_stats, n_channels, seq_len = load_dataset(dataset, ds_cfg, "train")
    val_ds, _, _, _ = load_dataset(dataset, ds_cfg, "val", norm_stats)
    test_ds, _, _, _ = load_dataset(dataset, ds_cfg, "test", norm_stats)

    class_weights = None
    try:
        class_weights = train_ds.get_class_weights()
    except AttributeError:
        try:
            class_weights = train_ds.dataset.get_class_weights()
        except Exception:
            pass

    teacher_cfg = load_config(config_dir, teacher_name)
    student_cfg = load_config(config_dir, student_name)

    # --- Teacher: train or load checkpoint ---
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    teacher_ckpt = CHECKPOINT_DIR / f"{teacher_name}_{dataset}_seed{seed}.pt"

    teacher = build_model(teacher_name, teacher_cfg, n_channels, 2, seq_len).to(device)

    if teacher_ckpt.exists():
        print(f"Loading teacher checkpoint: {teacher_ckpt}")
        teacher.load_state_dict(torch.load(teacher_ckpt, map_location=device))
    else:
        print(f"Training teacher ({teacher_name})...")
        t_result = fit(
            teacher, train_ds, val_ds,
            device=device, batch_size=batch_size, epochs=epochs,
            patience=patience, class_weights=class_weights, verbose=verbose,
        )
        teacher.load_state_dict(t_result["best_state"])
        torch.save(t_result["best_state"], teacher_ckpt)
        print(f"  Teacher checkpoint saved: {teacher_ckpt}")

    # Evaluate teacher on test set
    teacher.eval()
    test_loader = make_loader(test_ds, batch_size, shuffle=False)
    loss_fn = torch.nn.CrossEntropyLoss(
        weight=class_weights.to(device) if class_weights is not None else None
    )
    teacher_metrics, _, _ = eval_epoch(teacher, test_loader, loss_fn, device)
    print(f"  Teacher test AUROC: {teacher_metrics['auroc']:.4f}")

    # --- Student: train with KD ---
    print(f"\nTraining student ({student_name}) via KD from {teacher_name}...")
    student = build_model(student_name, student_cfg, n_channels, 2, seq_len).to(device)
    wrapper = KnowledgeDistillationWrapper(teacher, temperature=temperature, alpha=alpha)

    s_result = fit_student_kd(
        student, teacher, train_ds, val_ds, wrapper,
        device=device, batch_size=batch_size, epochs=epochs,
        patience=patience, class_weights=class_weights, verbose=verbose,
    )
    student.load_state_dict(s_result["best_state"])

    student_metrics, _, _ = eval_epoch(student, test_loader, loss_fn, device)
    print(f"  Student test AUROC: {student_metrics['auroc']:.4f}")

    # --- Latency comparison ---
    print("\nMeasuring latency...")
    from data.synthetic import SyntheticDataset
    from torch.utils.data import DataLoader

    prof_ds = SyntheticDataset(
        n_samples=16, seq_len=seq_len, n_channels=n_channels,
        missing_frac=0.4, seed=0,
    )
    prof_loader = DataLoader(prof_ds, batch_size=16, collate_fn=collate_irregular)
    prof_batch = next(iter(prof_loader)).to(device)

    teacher_prof = profile_model(teacher, prof_batch, device=device, n_warmup=10, n_measure=50)
    student_prof = profile_model(student, prof_batch, device=device, n_warmup=10, n_measure=50)

    speedup = (
        teacher_prof["latency_ms"] / student_prof["latency_ms"]
        if student_prof["latency_ms"] and student_prof["latency_ms"] > 0
        else None
    )
    auroc_drop = teacher_metrics["auroc"] - student_metrics["auroc"]
    param_ratio = teacher.param_count() / max(student.param_count(), 1)

    print(f"  Teacher latency: {teacher_prof['latency_ms']:.2f} ms")
    print(f"  Student latency: {student_prof['latency_ms']:.2f} ms")
    print(f"  Speedup: {speedup:.2f}×" if speedup else "  Speedup: n/a")
    print(f"  AUROC drop: {auroc_drop:.4f}")
    print(f"  Param ratio (teacher/student): {param_ratio:.1f}×")

    # --- Save per-seed JSON ---
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    json_path = TABLES_DIR / f"distillation_{dataset}_seed{seed}.json"
    payload = {
        "teacher": teacher_name,
        "student": student_name,
        "dataset": dataset,
        "seed": seed,
        "temperature": temperature,
        "alpha": alpha,
        "teacher_auroc": teacher_metrics["auroc"],
        "student_auroc": student_metrics["auroc"],
        "auroc_drop": auroc_drop,
        "teacher_latency_ms": teacher_prof["latency_ms"],
        "student_latency_ms": student_prof["latency_ms"],
        "speedup": speedup,
        "teacher_params": teacher.param_count(),
        "student_params": student.param_count(),
        "param_ratio": param_ratio,
        "student_history": s_result["history"],
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n  Saved: {json_path}")

    # --- Append to benchmark_inference.csv for consistency ---
    ts = time.strftime("%Y-%m-%dT%H:%M:%S")
    append_row_csv(INFERENCE_CSV, {
        "model": f"kd_{student_name}",
        "complexity_class": student.complexity_class,
        "optimization": f"kd_from_{teacher_name}",
        "baseline_latency_ms": teacher_prof["latency_ms"],
        "opt_latency_ms": student_prof["latency_ms"],
        "speedup": speedup,
        "baseline_mem_mb": teacher_prof["peak_mem_mb"],
        "opt_mem_mb": student_prof["peak_mem_mb"],
        "memory_reduction": (
            teacher_prof["peak_mem_mb"] / student_prof["peak_mem_mb"]
            if student_prof["peak_mem_mb"] else None
        ),
        "params": student.param_count(),
        "profile_device": str(device),
        "dataset_T": seq_len,
        "dataset_d": n_channels,
        "dataset_missing": 0.4,
        "error": None,
        "timestamp": ts,
    })


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, choices=["synthetic", "p12", "p19"])
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2024])
    p.add_argument("--teacher", default="raindrop")
    p.add_argument("--student", default="dlinear")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--temperature", type=float, default=4.0)
    p.add_argument("--alpha", type=float, default=0.3,
                   help="Weight for hard-label CE loss (1-alpha for KD loss)")
    p.add_argument("--device", default="auto")
    p.add_argument("--config_dir", default="config")
    p.add_argument("--verbose", action="store_true", default=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    config_dir = PROJECT_ROOT / args.config_dir
    with open(config_dir / "datasets.yaml") as f:
        ds_cfg = yaml.safe_load(f)

    for seed in args.seeds:
        run_seed(
            seed=seed,
            teacher_name=args.teacher,
            student_name=args.student,
            dataset=args.dataset,
            ds_cfg=ds_cfg,
            config_dir=config_dir,
            device=device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            patience=args.patience,
            temperature=args.temperature,
            alpha=args.alpha,
            verbose=args.verbose,
        )

    print(f"\nDone. Results in {TABLES_DIR} and {INFERENCE_CSV}")


if __name__ == "__main__":
    main()
