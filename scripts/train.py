"""CLI entry point for training a single model on a single dataset.

Usage:
    python scripts/train.py --model dlinear --dataset synthetic --epochs 5
    python scripts/train.py --model gru_d --dataset p12 --seed 42
    python scripts/train.py --model mtan --dataset synthetic --epochs 2 --batch_size 32
"""

import argparse
import json
import os
import sys
import random
from pathlib import Path

# Allow imports from project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import yaml

from data.base import collate_irregular
from data.synthetic import SyntheticDataset
from evaluation.runner import fit, eval_epoch, make_loader
from evaluation.metrics import find_best_threshold


def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True, warn_only=True)


def load_config(config_dir: Path, name: str) -> dict:
    path = config_dir / "models" / f"{name}.yaml"
    if not path.exists():
        return {}
    with open(path) as f:
        return yaml.safe_load(f)


def build_model(name: str, cfg: dict, n_channels: int, n_classes: int, seq_len: int = 500):
    if name == "dlinear":
        from models.dlinear import DLinear
        return DLinear(n_channels, n_classes, seq_len=seq_len, **_filter(cfg, ["hidden_dim", "kernel_size"]))
    elif name == "gru_d":
        from models.gru_d import GRUD
        return GRUD(n_channels, n_classes, **_filter(cfg, ["hidden_dim", "n_layers", "dropout"]))
    elif name == "linear_gru":
        from models.linear_gru import LinearGRU
        return LinearGRU(n_channels, n_classes, **_filter(cfg, ["hidden_dim", "n_layers", "dropout", "interp_steps"]))
    elif name == "mtan":
        from models.mtan import MTAN
        return MTAN(n_channels, n_classes, **_filter(cfg, ["hidden_dim", "n_heads", "n_ref_points", "n_layers", "dropout", "time_embed_dim"]))
    elif name == "latent_ode":
        from models.latent_ode import LatentODE
        return LatentODE(n_channels, n_classes, **_filter(cfg, ["latent_dim", "hidden_dim", "rec_hidden_dim", "solver", "rtol", "atol", "adjoint"]))
    elif name == "neural_cde":
        from models.neural_cde import NeuralCDE
        return NeuralCDE(n_channels, n_classes, **_filter(cfg, ["hidden_dim", "hidden_hidden_dim", "solver", "adjoint", "interpolation"]))
    elif name == "raindrop":
        from models.raindrop import Raindrop
        return Raindrop(n_channels, n_classes, **_filter(cfg, ["d_model", "n_heads", "n_layers", "dropout"]))
    elif name == "s4":
        from models.s4_ts import S4
        return S4(n_channels, n_classes, **_filter(cfg, ["d_model", "d_state", "n_layers", "dropout", "prenorm"]))
    elif name == "mamba":
        from models.mamba_ts import MambaTS
        return MambaTS(n_channels, n_classes, **_filter(cfg, ["d_model", "d_state", "d_conv", "expand", "n_layers", "dropout", "use_mamba_ssm"]))
    elif name == "patchtst":
        from models.patchtst import PatchTST
        return PatchTST(n_channels, n_classes, **_filter(cfg, ["patch_len", "stride", "d_model", "n_heads", "n_layers", "d_ff", "dropout", "use_flash_attn"]))
    else:
        available = "dlinear, gru_d, linear_gru, mtan, latent_ode, neural_cde, raindrop, s4, mamba, patchtst"
        raise ValueError(f"Unknown model: {name}. Available: {available}")


def _filter(cfg: dict, keys: list) -> dict:
    return {k: cfg[k] for k in keys if k in cfg}


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
        # simple manual split
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
        raise ValueError(f"Unknown dataset: {name}. Use: synthetic, p12, p19")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--device", default="auto")
    p.add_argument("--config_dir", default="config")
    p.add_argument("--output_dir", default="results/tables")
    p.add_argument("--verbose", action="store_true", default=True)
    p.add_argument("--save_checkpoint", action="store_true",
                   help="Save best model state_dict to results/checkpoints/")
    return p.parse_args()


def main():
    args = parse_args()
    set_seeds(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    config_dir = PROJECT_ROOT / args.config_dir
    with open(config_dir / "datasets.yaml") as f:
        ds_cfg = yaml.safe_load(f)
    model_cfg = load_config(config_dir, args.model)

    print(f"Loading dataset: {args.dataset} (train)...")
    train_ds, norm_stats, n_channels, seq_len = load_dataset(args.dataset, ds_cfg, "train")
    print(f"Loading dataset: {args.dataset} (val)...")
    val_ds, _, _, _ = load_dataset(args.dataset, ds_cfg, "val", norm_stats)
    print(f"Loading dataset: {args.dataset} (test)...")
    test_ds, _, _, _ = load_dataset(args.dataset, ds_cfg, "test", norm_stats)

    print(f"Building model: {args.model}  (n_channels={n_channels}, seq_len={seq_len})")
    model = build_model(args.model, model_cfg, n_channels=n_channels, n_classes=2, seq_len=seq_len)
    model = model.to(device)
    print(f"Parameters: {model.param_count():,}  |  Complexity: {model.complexity_class}")

    # Class weights from training set
    try:
        class_weights = train_ds.get_class_weights()
    except AttributeError:
        # Subset wrapper — go through dataset attribute
        try:
            class_weights = train_ds.dataset.get_class_weights()
        except Exception:
            class_weights = None

    result = fit(
        model, train_ds, val_ds,
        device=device,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        lr=args.lr,
        class_weights=class_weights,
        verbose=args.verbose,
    )

    # Evaluate on test set with best checkpoint
    model.load_state_dict(result["best_state"])
    test_loader = make_loader(test_ds, args.batch_size, shuffle=False)
    from torch.nn import CrossEntropyLoss
    test_metrics, _, _ = eval_epoch(model, test_loader, CrossEntropyLoss(), device)

    print("\n=== Test Results ===")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")

    # Save results
    out_dir = PROJECT_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{args.model}_{args.dataset}_seed{args.seed}.json"
    payload = {
        "model": args.model,
        "dataset": args.dataset,
        "seed": args.seed,
        "params": model.param_count(),
        "complexity": model.complexity_class,
        "best_val_auroc": result["best_auroc"],
        "test": test_metrics,
        "history": result["history"],
    }
    with open(out_file, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nResults saved to {out_file}")

    if args.save_checkpoint:
        ckpt_dir = PROJECT_ROOT / "results" / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / f"{args.model}_{args.dataset}_seed{args.seed}.pt"
        torch.save(result["best_state"], ckpt_path)
        print(f"Checkpoint saved to {ckpt_path}")


if __name__ == "__main__":
    main()
