"""GPU/CPU profiling utilities: wall-clock time, peak memory, FLOPs, param count."""

import time
from contextlib import contextmanager
from typing import Any

import torch
from torch import Tensor

try:
    from fvcore.nn import FlopCountAnalysis
    _FVCORE = True
except ImportError:
    _FVCORE = False


def profile_model(
    model: torch.nn.Module,
    batch,
    device: torch.device | None = None,
    n_warmup: int = 10,
    n_measure: int = 50,
) -> dict[str, Any]:
    """Profile inference latency, peak GPU memory and FLOPs.

    Args:
        model: must accept an IrregularBatch and return a Tensor
        batch: IrregularBatch to profile on (moved to device internally)
        device: target device; defaults to next(model.parameters()).device
        n_warmup: ignored iterations before measurement begins
        n_measure: number of timed iterations to average

    Returns:
        {
            "latency_ms": float,
            "peak_mem_mb": float,      # 0 on CPU
            "flops": int | None,       # None if fvcore unavailable
            "params": int,
        }
    """
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

    model.eval()
    batch = batch.to(device)

    use_cuda = device.type == "cuda" and torch.cuda.is_available()

    # warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            model(batch)

    if use_cuda:
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        with torch.no_grad():
            for _ in range(n_measure):
                model(batch)
        end_event.record()
        torch.cuda.synchronize(device)
        latency_ms = start_event.elapsed_time(end_event) / n_measure
        peak_mem_mb = torch.cuda.max_memory_allocated(device) / 1024 ** 2
    else:
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(n_measure):
                model(batch)
        latency_ms = (time.perf_counter() - t0) / n_measure * 1000
        peak_mem_mb = 0.0

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    flops = None
    if _FVCORE:
        try:
            with torch.no_grad():
                analysis = FlopCountAnalysis(model, batch)
                analysis.unsupported_ops_warnings(False)
                analysis.uncalled_modules_warnings(False)
                flops = int(analysis.total())
        except Exception:
            flops = None

    return {
        "latency_ms": latency_ms,
        "peak_mem_mb": peak_mem_mb,
        "flops": flops,
        "params": params,
    }


def profile_training_epoch(
    model: torch.nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    loss_fn,
    device: torch.device,
) -> dict[str, float]:
    """Measure wall-clock time and peak GPU memory for a full training epoch."""
    model.train()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter()

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch)
        loss = loss_fn(logits, batch.labels)
        loss.backward()
        optimizer.step()

    elapsed = time.perf_counter() - t0
    peak_mem = torch.cuda.max_memory_allocated(device) / 1024 ** 2 if device.type == "cuda" else 0.0

    return {"epoch_time_s": elapsed, "peak_mem_mb": peak_mem}
