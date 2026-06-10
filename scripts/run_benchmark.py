"""Unified benchmark: time & memory complexity for all models × optimizations.

Measures time and memory complexity for Direction 1 (architectures) and
Direction 2+3 (optimizations), keeping all measurements comparable via
identical synthetic batches and fixed profiling protocol.

Usage:
  # Complexity stress tests for all models (sweep T, d, missing_frac)
  python scripts/run_benchmark.py --mode complexity
  python scripts/run_benchmark.py --mode complexity --model dlinear,mamba --device cpu

  # Optimization delta at canonical size (T=500, d=10, missing=0.4)
  python scripts/run_benchmark.py --mode optimization --model all --opt quant_int8,compile_default
  python scripts/run_benchmark.py --mode optimization --model raindrop --opt all

  # Both in sequence
  python scripts/run_benchmark.py --mode all

Outputs:
  results/benchmark_complexity.csv   -- stress test sweeps (one row per model × point)
  results/benchmark_inference.csv    -- optimization delta at canonical size
"""

from __future__ import annotations
import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Callable

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.base import collate_irregular                          # noqa: E402
from data.synthetic import SyntheticDataset                     # noqa: E402
from evaluation.profiler import profile_model                   # noqa: E402
from evaluation.stress_tests import (                           # noqa: E402
    run_length_stress,
    run_missing_stress,
    run_dimension_stress,
)
from models.dlinear import DLinear                              # noqa: E402
from models.gru_d import GRUD                                   # noqa: E402
from models.linear_gru import LinearGRU                        # noqa: E402
from models.latent_ode import LatentODE                        # noqa: E402
from models.neural_cde import NeuralCDE                        # noqa: E402
from models.mtan import MTAN                                    # noqa: E402
from models.raindrop import Raindrop                            # noqa: E402
from models.s4_ts import S4                                     # noqa: E402
from models.mamba_ts import MambaTS                            # noqa: E402
from models.patchtst import PatchTST                            # noqa: E402
from optimizations.low_level.compilation import CompileWrapper  # noqa: E402
from optimizations.low_level.scan import MambaScanWrapper       # noqa: E402
from optimizations.hardware.precision import PrecisionWrapper   # noqa: E402
from optimizations.hardware.quantization import PTQInt8Wrapper, BNBInt4Wrapper  # noqa: E402
from optimizations.hardware.pruning import StructuredPruningWrapper  # noqa: E402

RESULTS_DIR = PROJECT_ROOT / "results"
COMPLEXITY_CSV = RESULTS_DIR / "benchmark_complexity.csv"
INFERENCE_CSV = RESULTS_DIR / "benchmark_inference.csv"

# ---------------------------------------------------------------------------
# Model registry
# Each factory accepts n_channels: int and returns a fresh model instance.
# Default hyperparameters are kept at library defaults — no tuning needed for
# pure computational benchmarking.
# ---------------------------------------------------------------------------

MODEL_REGISTRY: dict[str, Callable[[int], torch.nn.Module]] = {
    "dlinear":    lambda d: DLinear(d, n_classes=2, seq_len=500),
    "gru_d":      lambda d: GRUD(d, n_classes=2),
    "linear_gru": lambda d: LinearGRU(d, n_classes=2),
    "latent_ode": lambda d: LatentODE(d, n_classes=2),
    "neural_cde": lambda d: NeuralCDE(d, n_classes=2),
    "mtan":       lambda d: MTAN(d, n_classes=2),
    "raindrop":   lambda d: Raindrop(d, n_classes=2),
    "s4":         lambda d: S4(d, n_classes=2),
    "mamba":      lambda d: MambaTS(d, n_classes=2),
    "patchtst":   lambda d: PatchTST(d, n_classes=2),
}

# Models that support static INT8/INT4 quantization (need at least one Linear layer).
# Excluded: LatentODE/NeuralCDE (ODE-based, no static graph).
# Excluded: raindrop — nn.MultiheadAttention packed weights become callables after
#            quantize_dynamic, breaking TransformerEncoderLayer device checks.
# Excluded: mamba — MambaOfficial CUDA kernel requires GPU, can't run after model.cpu().
_QUANT_MODELS = ["gru_d", "linear_gru", "patchtst", "dlinear"]

# Models with prunable Linear/Conv1d layers.
# Excluded: DLinear (its Linear layers are projection heads, pruning degrades output badly).
_PRUNE_MODELS = ["gru_d", "linear_gru", "mtan", "raindrop", "s4", "patchtst"]

_ALL_MODELS = list(MODEL_REGISTRY)

# ---------------------------------------------------------------------------
# Optimization registry
# Each entry: (opt_name, wrapper_factory, applicable_model_names)
# wrapper_factory is called fresh for each (model, opt) pair to avoid shared state.
# ---------------------------------------------------------------------------

OPT_REGISTRY: list[tuple[str, Callable, list[str]]] = [
    ("compile_default",   lambda: CompileWrapper(mode="default"),             _ALL_MODELS),
    ("compile_reduce_oh", lambda: CompileWrapper(mode="reduce-overhead"),     _ALL_MODELS),
    ("precision_fp16",    lambda: PrecisionWrapper(dtype="fp16"),              _ALL_MODELS),
    ("precision_bf16",    lambda: PrecisionWrapper(dtype="bf16"),              _ALL_MODELS),
    ("quant_int8",        lambda: PTQInt8Wrapper(),                           _QUANT_MODELS),
    ("quant_nf4",         lambda: BNBInt4Wrapper(use_nf4=True),               _QUANT_MODELS),
    ("prune_50",          lambda: StructuredPruningWrapper(amount=0.50),      _PRUNE_MODELS),
    ("prune_75",          lambda: StructuredPruningWrapper(amount=0.75),      _PRUNE_MODELS),
    ("mamba_scan_python", lambda: MambaScanWrapper(backend="python"),         ["mamba"]),
]

_OPT_NAMES = [name for name, _, _ in OPT_REGISTRY]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_canonical_batch(
    T: int,
    d: int,
    missing: float,
    B: int = 16,
    seed: int = 0,
    device: torch.device | None = None,
):
    """Build a single profiling batch from SyntheticDataset with fixed seed.

    Using seed=0 everywhere ensures all models/optimizations see identical data
    at each (T, d, missing) point — the key comparability guarantee.
    """
    ds = SyntheticDataset(n_samples=B, seq_len=T, n_channels=d,
                          missing_frac=missing, seed=seed)
    loader = DataLoader(ds, batch_size=B, collate_fn=collate_irregular)
    batch = next(iter(loader))
    if device is not None:
        batch = batch.to(device)
    return batch


def _timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def append_rows_csv(path: Path, rows: list[dict]) -> None:
    """Append rows to a CSV, writing header only when the file is new."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def _resolve_models(model_arg: str) -> list[str]:
    if model_arg == "all":
        return _ALL_MODELS
    names = [m.strip() for m in model_arg.split(",")]
    unknown = [n for n in names if n not in MODEL_REGISTRY]
    if unknown:
        raise ValueError(f"Unknown models: {unknown}. Valid: {_ALL_MODELS}")
    return names


def _resolve_opts(opt_arg: str) -> list[str]:
    if opt_arg == "all":
        return _OPT_NAMES
    names = [o.strip() for o in opt_arg.split(",")]
    unknown = [n for n in names if n not in _OPT_NAMES]
    if unknown:
        raise ValueError(f"Unknown optimizations: {unknown}. Valid: {_OPT_NAMES}")
    return names


# ---------------------------------------------------------------------------
# Mode: complexity — stress tests sweeping T, d, missing_frac
# ---------------------------------------------------------------------------

def run_complexity_mode(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    models_to_run = _resolve_models(args.model)

    for model_name in models_to_run:
        factory = MODEL_REGISTRY[model_name]
        print(f"\n[complexity] model={model_name}")

        # We need the complexity_class annotation — instantiate with d=10 to read it
        _sample = factory(10)
        complexity_class = _sample.complexity_class
        del _sample

        rows: list[dict] = []

        # 1. Length stress (fixed d=10, missing=0.4)
        print(f"  sweep T ∈ {{100,250,500,1000,2000}} …")
        results_T = run_length_stress(
            model_factory=factory,
            device=device,
            T_values=[100, 250, 500, 1000, 2000],
            n_channels=10,
            missing_frac=0.40,
            batch_size=args.batch_size,
            n_warmup=args.n_warmup,
            n_measure=args.n_measure,
        )
        for r in results_T:
            rows.append({
                "model": model_name,
                "complexity_class": complexity_class,
                "sweep_type": "T",
                "sweep_value": r["T"],
                "latency_ms": r.get("latency_ms"),
                "peak_mem_mb": r.get("peak_mem_mb"),
                "flops": r.get("flops"),
                "params": r.get("params"),
                "timestamp": _timestamp(),
            })

        # 2. Missing stress (fixed T=500, d=10)
        print(f"  sweep missing ∈ {{0%,20%,40%,60%,80%}} …")
        results_M = run_missing_stress(
            model_factory=factory,
            device=device,
            missing_rates=[0.0, 0.2, 0.4, 0.6, 0.8],
            seq_len=500,
            n_channels=10,
            batch_size=args.batch_size,
            n_warmup=args.n_warmup,
            n_measure=args.n_measure,
        )
        for r in results_M:
            rows.append({
                "model": model_name,
                "complexity_class": complexity_class,
                "sweep_type": "missing",
                "sweep_value": r["missing_frac"],
                "latency_ms": r.get("latency_ms"),
                "peak_mem_mb": r.get("peak_mem_mb"),
                "flops": r.get("flops"),
                "params": r.get("params"),
                "timestamp": _timestamp(),
            })

        # 3. Dimension stress (fixed T=500, missing=0.4)
        print(f"  sweep d ∈ {{5,10,20,50,100}} …")
        results_D = run_dimension_stress(
            model_factory=factory,
            device=device,
            d_values=[5, 10, 20, 50, 100],
            seq_len=500,
            missing_frac=0.40,
            batch_size=args.batch_size,
            n_warmup=args.n_warmup,
            n_measure=args.n_measure,
        )
        for r in results_D:
            rows.append({
                "model": model_name,
                "complexity_class": complexity_class,
                "sweep_type": "d",
                "sweep_value": r["d"],
                "latency_ms": r.get("latency_ms"),
                "peak_mem_mb": r.get("peak_mem_mb"),
                "flops": r.get("flops"),
                "params": r.get("params"),
                "timestamp": _timestamp(),
            })

        append_rows_csv(COMPLEXITY_CSV, rows)
        print(f"  → {len(rows)} rows → {COMPLEXITY_CSV.name}")


# ---------------------------------------------------------------------------
# Mode: optimization — baseline vs wrapped model at canonical size
# ---------------------------------------------------------------------------

_CANONICAL_T = 500
_CANONICAL_D = 10
_CANONICAL_MISSING = 0.4
_CANONICAL_B = 16


def run_optimization_mode(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    models_to_run = _resolve_models(args.model)
    opts_to_run = set(_resolve_opts(args.opt))

    batch = make_canonical_batch(
        T=_CANONICAL_T, d=_CANONICAL_D, missing=_CANONICAL_MISSING,
        B=_CANONICAL_B, seed=0, device=device,
    )

    rows: list[dict] = []

    for opt_name, wrapper_factory, applicable_models in OPT_REGISTRY:
        if opt_name not in opts_to_run:
            continue
        for model_name in models_to_run:
            if model_name not in applicable_models:
                continue

            print(f"[optimization] model={model_name}  opt={opt_name}")
            factory = MODEL_REGISTRY[model_name]
            complexity_class = factory(10).complexity_class

            # Baseline: fresh model, no optimization
            baseline_model = factory(_CANONICAL_D).to(device)
            try:
                baseline = profile_model(
                    baseline_model, batch, device=device,
                    n_warmup=args.n_warmup, n_measure=args.n_measure,
                )
            except Exception as e:
                print(f"  baseline failed: {e}")
                continue

            # Optimized: fresh model + wrapper
            opt_model = factory(_CANONICAL_D).to(device)
            actual_device = device  # may change if optimization moves model to CPU
            try:
                wrapper = wrapper_factory()
                opt_model = wrapper.apply(opt_model)

                # Detect actual device post-optimization (quant_int8 moves to CPU).
                # Check explicit flag first — quantize_dynamic replaces Linear with
                # DynamicQuantizedLinear (no standard parameters), so parameters()
                # may return non-Linear params still on CUDA, giving a wrong device.
                if getattr(opt_model, "_ptq_on_cpu", False):
                    actual_device = torch.device("cpu")
                else:
                    try:
                        actual_device = next(opt_model.parameters()).device
                    except StopIteration:
                        actual_device = torch.device("cpu")
                profile_batch = batch if actual_device.type == device.type else batch.to(actual_device)

                # Re-run baseline on actual_device for a fair comparison
                compare_baseline = baseline
                if actual_device.type != device.type:
                    compare_baseline = profile_model(
                        factory(_CANONICAL_D).to(actual_device), profile_batch,
                        device=actual_device, n_warmup=args.n_warmup, n_measure=args.n_measure,
                    )

                opt_stats = profile_model(
                    opt_model, profile_batch, device=actual_device,
                    n_warmup=args.n_warmup, n_measure=args.n_measure,
                )
                speedup = (
                    compare_baseline["latency_ms"] / opt_stats["latency_ms"]
                    if opt_stats["latency_ms"] and opt_stats["latency_ms"] > 0
                    else None
                )
                mem_reduction = (
                    compare_baseline["peak_mem_mb"] / opt_stats["peak_mem_mb"]
                    if opt_stats["peak_mem_mb"] and opt_stats["peak_mem_mb"] > 0
                    else None
                )
                opt_latency = opt_stats["latency_ms"]
                opt_mem = opt_stats["peak_mem_mb"]
                err = None
            except Exception as e:
                speedup = mem_reduction = opt_latency = opt_mem = None
                err = str(e)
                print(f"  optimization failed: {e}")

            print(
                f"  baseline={baseline['latency_ms']:.2f}ms "
                f"opt={opt_latency}ms speedup={speedup}"
            )

            rows.append({
                "model": model_name,
                "complexity_class": complexity_class,
                "optimization": opt_name,
                "baseline_latency_ms": baseline["latency_ms"],
                "opt_latency_ms": opt_latency,
                "speedup": speedup,
                "baseline_mem_mb": baseline["peak_mem_mb"],
                "opt_mem_mb": opt_mem,
                "memory_reduction": mem_reduction,
                "params": baseline["params"],
                "profile_device": str(actual_device),
                "dataset_T": _CANONICAL_T,
                "dataset_d": _CANONICAL_D,
                "dataset_missing": _CANONICAL_MISSING,
                "error": err,
                "timestamp": _timestamp(),
            })

    append_rows_csv(INFERENCE_CSV, rows)
    print(f"\n→ {len(rows)} rows → {INFERENCE_CSV.name}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified time & memory benchmark for models × optimizations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mode", required=True,
        choices=["complexity", "optimization", "all"],
        help="complexity: stress sweeps per model; optimization: delta vs baseline; all: both",
    )
    parser.add_argument(
        "--model", default="all",
        help="Model(s) to include: 'all' or comma-separated from "
             + ", ".join(MODEL_REGISTRY),
    )
    parser.add_argument(
        "--opt", default="all",
        help="Optimization(s) to include (only for --mode optimization/all): 'all' or "
             "comma-separated from " + ", ".join(_OPT_NAMES),
    )
    parser.add_argument(
        "--device", default="cpu",
        help="Compute device: 'cpu' or 'cuda' (default: cpu)",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--n-warmup", type=int, default=5,
        help="Warmup iterations before measurement (stress tests; inference uses 10)",
    )
    parser.add_argument(
        "--n-measure", type=int, default=20,
        help="Timed iterations to average (stress tests; inference uses 50)",
    )
    args = parser.parse_args()

    if args.mode in ("complexity", "all"):
        run_complexity_mode(args)

    if args.mode in ("optimization", "all"):
        # Inference profiling uses a tighter fixed protocol for comparability
        args_inf = argparse.Namespace(**vars(args))
        args_inf.n_warmup = 10
        args_inf.n_measure = 50
        run_optimization_mode(args_inf)

    print("\nDone.")
    if args.mode in ("complexity", "all"):
        print(f"  Complexity results: {COMPLEXITY_CSV}")
    if args.mode in ("optimization", "all"):
        print(f"  Inference results:  {INFERENCE_CSV}")


if __name__ == "__main__":
    main()
