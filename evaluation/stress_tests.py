"""Scalability stress tests for benchmark models.

Three test series:
  1. Length stress:   T ∈ {100, 250, 500, 1000, 2000, 5000} — fixed d=10, miss=40%
  2. Missing stress:  r ∈ {0%, 20%, 40%, 60%, 80%} — fixed T=500, d=10
  3. Dimension stress: d ∈ {5, 10, 20, 50, 100} — fixed T=500, miss=40%
"""

import time
from typing import Callable

import torch
from torch.utils.data import DataLoader

from data.base import collate_irregular
from data.synthetic import SyntheticDataset
from evaluation.profiler import profile_model


def _make_batch(seq_len: int, n_channels: int, missing_frac: float,
                batch_size: int, device: torch.device):
    """Build a single profiling batch from the synthetic dataset."""
    ds = SyntheticDataset(
        n_samples=batch_size,
        seq_len=seq_len,
        n_channels=n_channels,
        missing_frac=missing_frac,
        seed=0,
    )
    loader = DataLoader(ds, batch_size=batch_size, collate_fn=collate_irregular)
    batch = next(iter(loader))
    return batch.to(device)


def run_length_stress(
    model_factory: Callable,
    device: torch.device,
    T_values: list[int] = (100, 250, 500, 1000, 2000, 5000),
    n_channels: int = 10,
    missing_frac: float = 0.40,
    batch_size: int = 16,
    n_warmup: int = 5,
    n_measure: int = 20,
) -> list[dict]:
    """Sweep over sequence lengths — measure time and peak memory.

    Args:
        model_factory: callable(n_channels) → IrregularTSModel (freshly initialised)

    Returns: list of dicts with keys: T, latency_ms, peak_mem_mb, params
    """
    results = []
    for T in T_values:
        batch = _make_batch(T, n_channels, missing_frac, batch_size, device)
        model = model_factory(n_channels).to(device)
        model.eval()

        try:
            stats = profile_model(model, batch, device=device,
                                  n_warmup=n_warmup, n_measure=n_measure)
            results.append({"T": T, **stats})
            print(f"  T={T:5d}  latency={stats['latency_ms']:.1f}ms  "
                  f"mem={stats['peak_mem_mb']:.1f}MB")
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            print(f"  T={T:5d}  OOM or error: {e}")
            results.append({"T": T, "latency_ms": None, "peak_mem_mb": None,
                            "flops": None, "params": None})
    return results


def run_missing_stress(
    model_factory: Callable,
    device: torch.device,
    missing_rates: list[float] = (0.0, 0.2, 0.4, 0.6, 0.8),
    seq_len: int = 500,
    n_channels: int = 10,
    batch_size: int = 16,
    n_warmup: int = 5,
    n_measure: int = 20,
) -> list[dict]:
    """Sweep over MCAR missing rates.

    Returns: list of dicts with keys: missing_frac, latency_ms, peak_mem_mb
    """
    results = []
    for r in missing_rates:
        batch = _make_batch(seq_len, n_channels, r, batch_size, device)
        model = model_factory(n_channels).to(device)
        model.eval()

        try:
            stats = profile_model(model, batch, device=device,
                                  n_warmup=n_warmup, n_measure=n_measure)
            results.append({"missing_frac": r, **stats})
            print(f"  missing={r:.0%}  latency={stats['latency_ms']:.1f}ms  "
                  f"mem={stats['peak_mem_mb']:.1f}MB")
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            print(f"  missing={r:.0%}  OOM or error: {e}")
            results.append({"missing_frac": r, "latency_ms": None,
                            "peak_mem_mb": None, "flops": None, "params": None})
    return results


def run_dimension_stress(
    model_factory: Callable,
    device: torch.device,
    d_values: list[int] = (5, 10, 20, 50, 100),
    seq_len: int = 500,
    missing_frac: float = 0.40,
    batch_size: int = 16,
    n_warmup: int = 5,
    n_measure: int = 20,
) -> list[dict]:
    """Sweep over channel dimensionality d.

    Returns: list of dicts with keys: d, latency_ms, peak_mem_mb
    """
    results = []
    for d in d_values:
        batch = _make_batch(seq_len, d, missing_frac, batch_size, device)
        model = model_factory(d).to(device)
        model.eval()

        try:
            stats = profile_model(model, batch, device=device,
                                  n_warmup=n_warmup, n_measure=n_measure)
            results.append({"d": d, **stats})
            print(f"  d={d:3d}  latency={stats['latency_ms']:.1f}ms  "
                  f"mem={stats['peak_mem_mb']:.1f}MB")
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            print(f"  d={d:3d}  OOM or error: {e}")
            results.append({"d": d, "latency_ms": None, "peak_mem_mb": None,
                            "flops": None, "params": None})
    return results
