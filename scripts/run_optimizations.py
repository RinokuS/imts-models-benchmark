"""Orchestrator for Direction 2 + 3 optimization benchmarks.

Usage:
  python scripts/run_optimizations.py --opt compilation --model all
  python scripts/run_optimizations.py --opt quantization --model raindrop,mamba
  python scripts/run_optimizations.py --opt batching --model all --dataset p12
  python scripts/run_optimizations.py --opt pruning --model raindrop --checkpoint results/checkpoints/raindrop_p12_seed42.pt
  python scripts/run_optimizations.py --opt distillation --teacher-ckpt results/checkpoints/raindrop_p12_seed42.pt

Each run appends a JSON row to results/optimization_results.jsonl, which can
be loaded with pandas for Pareto analysis.
"""

from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.profiler import profile_model       # noqa: E402
from evaluation.metrics import compute_metrics     # noqa: E402

RESULTS_FILE = PROJECT_ROOT / "results" / "optimization_results.jsonl"
OPT_CONFIG_DIR = PROJECT_ROOT / "config" / "optimizations"

ALL_MODELS = [
    "dlinear", "gru_d", "linear_gru", "latent_ode",
    "neural_cde", "mtan", "raindrop", "s4", "mamba", "patchtst",
]


def load_opt_config(opt_name: str) -> dict:
    path = OPT_CONFIG_DIR / f"{opt_name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"No config found at {path}")
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_models(model_arg: str, cfg: dict) -> list[str]:
    if model_arg == "all":
        applicable = cfg.get("apply_to", "all_models")
        if applicable == "all_models":
            return ALL_MODELS
        return applicable
    return [m.strip() for m in model_arg.split(",")]


def append_result(row: dict) -> None:
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "a") as f:
        f.write(json.dumps(row, default=str) + "\n")


def run_compilation(args: argparse.Namespace, cfg: dict) -> None:
    from optimizations.low_level.compilation import CompileWrapper

    models_to_run = resolve_models(args.model, cfg)
    for model_name in models_to_run:
        for variant in cfg["variants"]:
            mode = variant.get("mode")
            print(f"[compile] model={model_name} mode={mode}")
            row = {
                "experiment": "compilation",
                "model": model_name,
                "variant": variant["name"],
                "compile_mode": mode,
                "dataset": args.dataset,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            if mode is not None:
                wrapper = CompileWrapper(mode=mode)
                row["optimization"] = wrapper.name()
                row["metadata"] = wrapper.metadata()
            else:
                row["optimization"] = "no_compile"
            append_result(row)
            print(f"  → logged to {RESULTS_FILE.name}")


def run_batching(args: argparse.Namespace, cfg: dict) -> None:
    print("[batching] Strategy comparison — see collate_fn in batching.py")
    print("  Configure DataLoader with the chosen collate_fn to measure waste.")
    row = {
        "experiment": "batching",
        "dataset": args.dataset,
        "variants": [v["name"] for v in cfg["variants"]],
        "note": "Run data loader with each collate_fn and compare _batching_waste field.",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    append_result(row)


def run_quantization(args: argparse.Namespace, cfg: dict) -> None:
    from optimizations.hardware.quantization import PTQInt8Wrapper, BNBInt4Wrapper

    models_to_run = resolve_models(args.model, cfg)
    for model_name in models_to_run:
        for variant in cfg["variants"]:
            print(f"[quant] model={model_name} variant={variant['name']}")
            row = {
                "experiment": "quantization",
                "model": model_name,
                "variant": variant["name"],
                "dataset": args.dataset,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "requires_checkpoint": cfg.get("requires_trained_checkpoint", True),
            }
            if variant["stack"] == "torch_ao":
                wrapper = PTQInt8Wrapper()
            else:
                wrapper = BNBInt4Wrapper(
                    use_nf4=variant.get("use_nf4", True),
                    double_quant=variant.get("double_quant", True),
                )
            row["optimization"] = wrapper.name()
            row["metadata"] = wrapper.metadata()
            append_result(row)


def run_pruning(args: argparse.Namespace, cfg: dict) -> None:
    from optimizations.hardware.pruning import StructuredPruningWrapper

    models_to_run = resolve_models(args.model, cfg)
    for model_name in models_to_run:
        for variant in cfg["variants"]:
            print(f"[prune] model={model_name} variant={variant['name']}")
            wrapper = StructuredPruningWrapper(amount=variant["amount"])
            row = {
                "experiment": "pruning",
                "model": model_name,
                "variant": variant["name"],
                "dataset": args.dataset,
                "optimization": wrapper.name(),
                "metadata": wrapper.metadata(),
                "requires_checkpoint": True,
                "finetune_epochs": cfg.get("finetune_epochs", 15),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            append_result(row)


def run_distillation(args: argparse.Namespace, cfg: dict) -> None:
    print(f"[distillation] teacher={cfg['teacher']} student={cfg['student']}")
    row = {
        "experiment": "distillation",
        "teacher": cfg["teacher"],
        "student": cfg["student"],
        "kd_temperature": cfg["kd_temperature"],
        "kd_alpha": cfg["kd_alpha"],
        "train_epochs": cfg["train_epochs"],
        "datasets": cfg["datasets"],
        "requires_teacher_checkpoint": cfg.get("requires_trained_teacher_checkpoint", True),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    append_result(row)
    print("  → Use KnowledgeDistillationWrapper in your training loop with distillation_loss()")


_RUNNERS = {
    "compilation": run_compilation,
    "batching": run_batching,
    "quantization": run_quantization,
    "pruning": run_pruning,
    "distillation": run_distillation,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Direction 2/3 optimization benchmark")
    parser.add_argument("--opt", required=True, choices=list(_RUNNERS), help="Optimization type")
    parser.add_argument("--model", default="all", help="Model(s) to test, comma-separated or 'all'")
    parser.add_argument("--dataset", default="p12", help="Dataset name")
    parser.add_argument("--checkpoint", default=None, help="Path to trained model checkpoint")
    parser.add_argument("--teacher-ckpt", default=None, help="Path to teacher checkpoint (distillation)")
    args = parser.parse_args()

    cfg = load_opt_config(args.opt)
    runner = _RUNNERS[args.opt]
    runner(args, cfg)
    print(f"\nDone. Results appended to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
