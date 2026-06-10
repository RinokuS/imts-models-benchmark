"""Run the full benchmark: all models × all datasets × N seeds.

Usage:
    python scripts/run_all.py
    python scripts/run_all.py --models dlinear gru_d --datasets synthetic --epochs 5
    python scripts/run_all.py --datasets synthetic p12 --seeds 42 123 --epochs 20
    python scripts/run_all.py --skip_existing --verbose
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

PYTHON = str(PROJECT_ROOT / ".bench_venv" / "bin" / "python")
if not Path(PYTHON).exists():
    PYTHON = sys.executable

ALL_MODELS = [
    "dlinear",
    "gru_d",
    "linear_gru",
    "mtan",
    "latent_ode",
    "neural_cde",
    "raindrop",
    "s4",
    "mamba",
    "patchtst",
]

ALL_DATASETS = ["synthetic", "p12", "p19"]

DEFAULT_SEEDS = [42, 123, 2024]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=ALL_MODELS)
    p.add_argument("--datasets", nargs="+", default=ALL_DATASETS)
    p.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--device", default="auto")
    p.add_argument("--output_dir", default="results/tables")
    p.add_argument("--skip_existing", action="store_true",
                   help="Skip runs where the result JSON already exists")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def result_path(output_dir: Path, model: str, dataset: str, seed: int) -> Path:
    return output_dir / f"{model}_{dataset}_seed{seed}.json"


def load_result(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def run_one(model: str, dataset: str, seed: int, args) -> dict:
    """Invoke scripts/train.py as a subprocess and return the parsed JSON result."""
    cmd = [
        PYTHON, str(PROJECT_ROOT / "scripts" / "train.py"),
        "--model", model,
        "--dataset", dataset,
        "--seed", str(seed),
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--lr", str(args.lr),
        "--patience", str(args.patience),
        "--device", args.device,
        "--output_dir", args.output_dir
    ]
    cmd.append("--save_checkpoint")
    if args.verbose:
        cmd.append("--verbose")

    t0 = time.time()
    proc = subprocess.run(
        cmd,
        capture_output=not args.verbose,
        text=True,
        cwd=str(PROJECT_ROOT),
    )
    wall = time.time() - t0

    if proc.returncode != 0:
        err = proc.stderr[-500:] if proc.stderr else "(no stderr)"
        print(f"    FAILED ({wall:.0f}s): {err.strip()}")
        return {"status": "error", "model": model, "dataset": dataset, "seed": seed,
                "wall_s": wall, "error": err.strip()}

    out_dir = PROJECT_ROOT / args.output_dir
    result = load_result(result_path(out_dir, model, dataset, seed))
    if result is None:
        return {"status": "error", "model": model, "dataset": dataset, "seed": seed,
                "wall_s": wall, "error": "JSON not written"}

    result["status"] = "ok"
    result["wall_s"] = wall
    return result


def aggregate(results: list[dict]) -> list[dict]:
    """Average test metrics across seeds for each (model, dataset) pair."""
    from collections import defaultdict
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in results:
        if r["status"] == "ok":
            groups[(r["model"], r["dataset"])].append(r)

    rows = []
    for (model, dataset), runs in sorted(groups.items()):
        row = {"model": model, "dataset": dataset, "n_seeds": len(runs)}
        # Numeric test metrics
        metric_keys = [k for k in runs[0].get("test", {}).keys()]
        for k in metric_keys:
            vals = [r["test"][k] for r in runs if k in r.get("test", {})]
            if vals:
                row[f"test_{k}_mean"] = sum(vals) / len(vals)
                row[f"test_{k}_std"] = (
                    (sum((v - row[f"test_{k}_mean"]) ** 2 for v in vals) / len(vals)) ** 0.5
                )
        row["params"] = runs[0].get("params")
        row["complexity"] = runs[0].get("complexity")
        rows.append(row)
    return rows


def save_csv(rows: list[dict], path: Path):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def print_summary(rows: list[dict]):
    if not rows:
        print("No successful runs to summarise.")
        return

    # Find AUROC column
    auroc_col = next((k for k in rows[0] if "auroc" in k.lower() and "mean" in k), None)
    auprc_col = next((k for k in rows[0] if "auprc" in k.lower() and "mean" in k), None)
    f1_col    = next((k for k in rows[0] if k.startswith("test_f1") and "mean" in k), None)

    header_cols = [("AUROC", auroc_col), ("AUPRC", auprc_col), ("F1", f1_col)]
    header_cols = [(name, col) for name, col in header_cols if col]

    col_w = 12
    model_w = 14
    ds_w = 10

    sep = "-" * (model_w + ds_w + col_w * len(header_cols) + 6)
    print("\n" + sep)
    hdr = f"{'Model':<{model_w}} {'Dataset':<{ds_w}}"
    for name, _ in header_cols:
        hdr += f" {name:>{col_w}}"
    print(hdr)
    print(sep)

    for row in rows:
        line = f"{row['model']:<{model_w}} {row['dataset']:<{ds_w}}"
        for _, col in header_cols:
            val = row.get(col)
            if val is None:
                line += f" {'—':>{col_w}}"
            else:
                line += f" {val:>{col_w}.4f}"
        print(line)
    print(sep)


def main():
    args = parse_args()

    out_dir = PROJECT_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    total = len(args.models) * len(args.datasets) * len(args.seeds)
    done = 0
    all_results: list[dict] = []

    print(f"Benchmark: {len(args.models)} models × {len(args.datasets)} datasets "
          f"× {len(args.seeds)} seeds = {total} runs")
    print(f"Output: {out_dir}\n")

    for dataset in args.datasets:
        for model in args.models:
            for seed in args.seeds:
                done += 1
                rpath = result_path(out_dir, model, dataset, seed)

                if args.skip_existing and rpath.exists():
                    existing = load_result(rpath)
                    if existing:
                        existing["status"] = "ok"
                        all_results.append(existing)
                        print(f"[{done:3d}/{total}] {model:12s} {dataset:10s} seed={seed}  SKIP (exists)")
                        continue

                print(f"[{done:3d}/{total}] {model:12s} {dataset:10s} seed={seed} ...", end="", flush=True)
                result = run_one(model, dataset, seed, args)
                all_results.append(result)

                if result["status"] == "ok":
                    auroc = result.get("test", {}).get("auroc", result.get("best_val_auroc"))
                    auroc_str = f"  AUROC={auroc:.4f}" if auroc is not None else ""
                    print(f"  OK  ({result['wall_s']:.0f}s){auroc_str}")
                # error message already printed by run_one

    # Save individual run summary CSV
    csv_all = out_dir / "all_runs.csv"
    flat = []
    for r in all_results:
        row = {
            "model": r.get("model"), "dataset": r.get("dataset"), "seed": r.get("seed"),
            "status": r.get("status"), "wall_s": r.get("wall_s"),
            "params": r.get("params"), "complexity": r.get("complexity"),
            "best_val_auroc": r.get("best_val_auroc"),
        }
        for k, v in r.get("test", {}).items():
            row[f"test_{k}"] = v
        flat.append(row)
    save_csv(flat, csv_all)

    # Aggregated (mean ± std across seeds)
    agg = aggregate(all_results)
    csv_agg = out_dir / "all_results.csv"
    save_csv(agg, csv_agg)

    print(f"\nRaw runs  → {csv_all}")
    print(f"Aggregate → {csv_agg}")
    print_summary(agg)


if __name__ == "__main__":
    main()
