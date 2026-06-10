"""Deduplicate benchmark_complexity.csv.

Removes stale rows that have peak_mem_mb=0 when a newer row with real
memory data exists for the same (model, sweep_type, sweep_value) key.

Run from benchmark/ directory:
    python scripts/clean_complexity.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

CSV_PATH = Path(__file__).resolve().parent.parent / "results" / "benchmark_complexity.csv"


def main() -> None:
    df = pd.read_csv(CSV_PATH)
    before = len(df)
    print(f"Rows before: {before}")

    # Sort so that rows with real memory (peak_mem_mb > 0) come last within each key
    df["_has_mem"] = df["peak_mem_mb"].fillna(0) > 0
    df = df.sort_values(["model", "sweep_type", "sweep_value", "_has_mem"])

    # Keep last (= prefer real-memory row when duplicates exist)
    df = df.drop_duplicates(subset=["model", "sweep_type", "sweep_value"], keep="last")
    df = df.drop(columns=["_has_mem"])

    after = len(df)
    print(f"Rows after:  {after}  (removed {before - after} duplicates)")

    df.to_csv(CSV_PATH, index=False)
    print(f"Saved → {CSV_PATH}")

    print("\nRows per model:")
    print(df.groupby(["model", "sweep_type"]).size().to_string())


if __name__ == "__main__":
    main()
