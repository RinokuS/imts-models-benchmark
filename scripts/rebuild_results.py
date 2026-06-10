"""Rebuild all_runs.csv and all_results.csv from individual JSON training files.

Run from benchmark/ directory:
    python scripts/rebuild_results.py

Reads all results/tables/*.json files (one per model×dataset×seed run) and
writes two aggregated CSVs:
  - results/tables/all_runs.csv    — one row per run
  - results/tables/all_results.csv — mean±std per (model, dataset)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "tables"

METRICS = ["test_auroc", "test_auprc", "test_f1", "test_brier", "test_loss"]


def load_runs() -> list[dict]:
    rows = []
    for path in sorted(RESULTS_DIR.glob("*.json")):
        try:
            with open(path) as f:
                d = json.load(f)
        except Exception as e:
            print(f"  SKIP {path.name}: {e}")
            continue

        row = {
            "model": d.get("model", path.stem.split("_")[0]),
            "dataset": d.get("dataset", "unknown"),
            "seed": d.get("seed", -1),
            "status": "ok",
            "wall_s": None,
            "params": d.get("params"),
            "complexity": d.get("complexity"),
            "best_val_auroc": d.get("best_val_auroc"),
            "test_auroc": d.get("test", {}).get("auroc"),
            "test_auprc": d.get("test", {}).get("auprc"),
            "test_f1": d.get("test", {}).get("f1"),
            "test_brier": d.get("test", {}).get("brier"),
            "test_loss": d.get("test", {}).get("loss"),
        }
        rows.append(row)

    return rows


def aggregate(runs: list[dict]) -> list[dict]:
    df = pd.DataFrame(runs)
    results = []
    for (model, dataset), g in df.groupby(["model", "dataset"]):
        row: dict = {
            "model": model,
            "dataset": dataset,
            "n_seeds": len(g),
        }
        for m in METRICS:
            vals = g[m].dropna().values.astype(float)
            row[f"{m}_mean"] = float(np.mean(vals)) if len(vals) else None
            row[f"{m}_std"] = float(np.std(vals, ddof=0)) if len(vals) > 1 else 0.0
        row["params"] = g["params"].dropna().iloc[0] if g["params"].notna().any() else None
        row["complexity"] = g["complexity"].dropna().iloc[0] if g["complexity"].notna().any() else None
        results.append(row)
    return results


def main() -> None:
    print(f"Scanning {RESULTS_DIR} for JSON files …")
    runs = load_runs()
    print(f"  Loaded {len(runs)} runs")

    runs_path = RESULTS_DIR / "all_runs.csv"
    pd.DataFrame(runs).to_csv(runs_path, index=False)
    print(f"  → {runs_path} ({len(runs)} rows)")

    agg = aggregate(runs)
    agg_path = RESULTS_DIR / "all_results.csv"
    pd.DataFrame(agg).to_csv(agg_path, index=False)
    print(f"  → {agg_path} ({len(agg)} rows)")

    print("\nCoverage (model × dataset × n_seeds):")
    for r in sorted(agg, key=lambda x: (x["model"], x["dataset"])):
        auroc = r.get("test_auroc_mean")
        auroc_str = f"{auroc:.4f}" if auroc is not None else "n/a"
        print(f"  {r['model']:12s} {r['dataset']:10s}  n_seeds={r['n_seeds']}  AUROC={auroc_str}")


if __name__ == "__main__":
    main()
