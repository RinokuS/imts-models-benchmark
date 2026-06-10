"""Smoke tests for benchmark/optimizations/.

Run:  .bench_venv/bin/python -m pytest tests/test_optimizations.py -v

Covers Direction 2 (Low-Level) and Direction 3 (Hardware) wrappers:
  - all wrappers import cleanly
  - apply() returns an nn.Module
  - name() / metadata() return sensible values
  - functional correctness on a tiny dummy model (Linear + GRU stack)
"""

from __future__ import annotations
import sys, types
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch import Tensor

# ── project root on path ─────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ── shared dummy model ────────────────────────────────────────────────────────

class TinyClassifier(nn.Module):
    """Minimal irregular-TS-shaped model: GRU + linear head."""

    def __init__(self, d_in: int = 8, d_hidden: int = 16, n_classes: int = 2):
        super().__init__()
        self.gru = nn.GRU(d_in, d_hidden, batch_first=True)
        self.head = nn.Linear(d_hidden, n_classes)
        self.solver = "dopri5"       # ODE solver attribute (for ODESolverWrapper)
        self._mamba_scan_backend = "cuda"  # for MambaScanWrapper

    def forward(self, values: Tensor, **_) -> Tensor:
        out, _ = self.gru(values)
        return self.head(out[:, -1])


def make_model() -> TinyClassifier:
    return TinyClassifier()


def make_batch(B: int = 4, T: int = 20, D: int = 8):
    return {
        "values": torch.randn(B, T, D),
        "mask":   torch.ones(B, T, D),
        "times":  torch.linspace(0, 1, T).unsqueeze(0).expand(B, -1),
        "labels": torch.randint(0, 2, (B,)),
    }


# ═════════════════════════════════════════════════════════════════════════════
# Direction 2 — Low-Level
# ═════════════════════════════════════════════════════════════════════════════

class TestCompileWrapper:
    # torch.compile in PyTorch 2.11 triggers DeprecationWarning from its own
    # torch.utils.mkldnn internals (@torch.jit.script_method). Filter it so
    # -W error::DeprecationWarning doesn't falsely fail these tests.
    _compile_filter = pytest.mark.filterwarnings(
        "ignore:`torch.jit.script_method` is deprecated:DeprecationWarning"
    )

    def test_import(self):
        from optimizations.low_level.compilation import CompileWrapper
        assert CompileWrapper

    def test_invalid_mode_raises(self):
        from optimizations.low_level.compilation import CompileWrapper
        with pytest.raises(ValueError, match="mode must be one of"):
            CompileWrapper(mode="turbo")

    @_compile_filter
    @pytest.mark.parametrize("mode", ["default", "reduce-overhead"])
    def test_apply_returns_module(self, mode):
        from optimizations.low_level.compilation import CompileWrapper
        model = make_model()
        wrapper = CompileWrapper(mode=mode)
        compiled = wrapper.apply(model)
        assert isinstance(compiled, nn.Module)

    def test_name_format(self):
        from optimizations.low_level.compilation import CompileWrapper
        assert CompileWrapper("reduce-overhead").name() == "compile_reduce_overhead"
        assert CompileWrapper("max-autotune").name() == "compile_max_autotune"

    @_compile_filter
    def test_compiled_model_runs(self):
        from optimizations.low_level.compilation import CompileWrapper
        model = make_model()
        wrapper = CompileWrapper(mode="default")
        compiled = wrapper.apply(model)
        batch = make_batch()
        out = compiled(batch["values"])
        assert out.shape == (4, 2)

    @_compile_filter
    def test_metadata_contains_compile_mode(self):
        from optimizations.low_level.compilation import CompileWrapper
        wrapper = CompileWrapper(mode="default")
        model = make_model()
        wrapper.apply(model)
        meta = wrapper.metadata()
        assert meta["compile_mode"] == "default"
        assert "compile_time_s" in meta


class TestBatchingCollate:
    def _make_samples(self, lengths=(10, 25, 50)):
        D = 4
        return [
            {
                "values": torch.randn(T, D),
                "mask":   torch.ones(T, D),
                "times":  torch.linspace(0, 1, T),
                "label":  torch.tensor(0),
            }
            for T in lengths
        ]

    def test_padding_batch_shape(self):
        from optimizations.low_level.batching import PaddingBatching
        wb = PaddingBatching()
        samples = self._make_samples((10, 25, 50))
        batch = wb.collate_fn(samples)
        assert batch["values"].shape == (3, 50, 4)
        assert batch["mask"].shape == (3, 50, 4)

    def test_padding_waste_nonzero(self):
        from optimizations.low_level.batching import PaddingBatching
        wb = PaddingBatching()
        samples = self._make_samples((5, 50))   # one very short, one full
        batch = wb.collate_fn(samples)
        assert batch["_batching_waste"] > 0.0

    def test_nested_batch_created(self):
        from optimizations.low_level.batching import NestedTensorBatching
        wb = NestedTensorBatching()
        samples = self._make_samples((10, 25, 50))
        batch = wb.collate_fn(samples)
        assert batch["_batching_waste"] == 0.0
        assert batch["_is_nested"] is True
        assert batch["values"].is_nested

    def test_bucket_name(self):
        from optimizations.low_level.batching import BucketBatching
        assert BucketBatching(n_buckets=4).name() == "batching_bucket_4"

    def test_padding_apply_noop(self):
        from optimizations.low_level.batching import PaddingBatching
        model = make_model()
        assert PaddingBatching().apply(model) is model


class TestODESolverWrapper:
    def test_dopri5_sets_solver(self):
        from optimizations.low_level.ode_solvers import ODESolverWrapper
        model = make_model()
        ODESolverWrapper(backend="dopri5_adaptive").apply(model)
        assert model.solver == "dopri5"

    def test_rk4_sets_solver_and_options(self):
        from optimizations.low_level.ode_solvers import ODESolverWrapper
        model = make_model()
        ODESolverWrapper(backend="rk4_fixed", n_fixed_steps=10).apply(model)
        assert model.solver == "rk4"
        assert model.options["step_size"] == pytest.approx(0.1)

    def test_invalid_backend_raises(self):
        from optimizations.low_level.ode_solvers import ODESolverWrapper
        with pytest.raises(ValueError):
            ODESolverWrapper(backend="euler_explicit")

    def test_name_rk4(self):
        from optimizations.low_level.ode_solvers import ODESolverWrapper
        assert "rk4_fixed_10steps" in ODESolverWrapper("rk4_fixed", 10).name()

    def test_missing_solver_attr_raises(self):
        from optimizations.low_level.ode_solvers import ODESolverWrapper

        class NoSolverModel(nn.Module):
            def forward(self, x):
                return x

        with pytest.raises(AttributeError, match="solver"):
            ODESolverWrapper().apply(NoSolverModel())


class TestFlashAttnVariant:
    def test_sdpa_auto_noop(self):
        from optimizations.low_level.flashattn import FlashAttnVariant
        model = make_model()
        result = FlashAttnVariant("sdpa_auto").apply(model)
        assert result is model

    def test_invalid_variant_raises(self):
        from optimizations.low_level.flashattn import FlashAttnVariant
        with pytest.raises(ValueError):
            FlashAttnVariant("flash_v99")

    def test_name(self):
        from optimizations.low_level.flashattn import FlashAttnVariant
        assert FlashAttnVariant("sdpa_auto").name() == "flashattn_sdpa_auto"


class TestMambaScanWrapper:
    def test_sets_backend(self):
        from optimizations.low_level.scan import MambaScanWrapper
        model = make_model()
        MambaScanWrapper("python").apply(model)
        assert model._mamba_scan_backend == "python"

    def test_missing_attr_raises(self):
        from optimizations.low_level.scan import MambaScanWrapper

        class Plain(nn.Module):
            def forward(self, x):
                return x

        with pytest.raises(AttributeError):
            MambaScanWrapper("cuda").apply(Plain())

    def test_invalid_backend_raises(self):
        from optimizations.low_level.scan import MambaScanWrapper
        with pytest.raises(ValueError):
            MambaScanWrapper("tpu")


# ═════════════════════════════════════════════════════════════════════════════
# Direction 3 — Hardware
# ═════════════════════════════════════════════════════════════════════════════

class TestPrecisionWrapper:
    @pytest.mark.parametrize("dtype", ["fp32", "fp16", "bf16"])
    def test_apply_changes_dtype(self, dtype):
        from optimizations.hardware.precision import PrecisionWrapper
        model = make_model()
        wrapper = PrecisionWrapper(dtype=dtype)
        result = wrapper.apply(model)
        expected = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[dtype]
        for p in result.parameters():
            assert p.dtype == expected
            break  # check first param only

    def test_invalid_dtype_raises(self):
        from optimizations.hardware.precision import PrecisionWrapper
        with pytest.raises(ValueError):
            PrecisionWrapper(dtype="int8")

    def test_name_format(self):
        from optimizations.hardware.precision import PrecisionWrapper
        assert PrecisionWrapper("bf16").name() == "precision_bf16"

    def test_fp16_model_runs(self):
        from optimizations.hardware.precision import PrecisionWrapper
        model = make_model()
        model = PrecisionWrapper("fp16").apply(model)
        x = torch.randn(2, 10, 8, dtype=torch.float16)
        out = model(x)
        assert out.dtype == torch.float16


class TestPTQInt8Wrapper:
    def test_apply_returns_module(self):
        from optimizations.hardware.quantization import PTQInt8Wrapper
        model = make_model()
        result = PTQInt8Wrapper().apply(model)
        assert isinstance(result, nn.Module)

    def test_name(self):
        from optimizations.hardware.quantization import PTQInt8Wrapper
        assert PTQInt8Wrapper().name() == "quant_int8_ptq_torchao"

    def test_quantized_model_runs_cpu(self):
        from optimizations.hardware.quantization import PTQInt8Wrapper
        model = make_model().cpu()
        quantized = PTQInt8Wrapper().apply(model)
        x = torch.randn(2, 10, 8)
        out = quantized(x)
        assert out.shape == (2, 2)

    def test_linear_layers_quantized(self):
        from optimizations.hardware.quantization import PTQInt8Wrapper
        model = make_model().cpu()
        quantized = PTQInt8Wrapper().apply(model)
        # torch.quantization.quantize_dynamic wraps linear layers in DynamicQuantizedLinear
        head = quantized.head
        assert "DynamicQuantized" in type(head).__name__ or "quantized" in str(type(head)).lower()


class TestBNBInt4Wrapper:
    def test_missing_bnb_raises_import_error(self):
        """bitsandbytes not installed in test env → ImportError on apply()."""
        try:
            import bitsandbytes  # noqa: F401
            pytest.skip("bitsandbytes is installed; skip ImportError test")
        except ImportError:
            pass

        from optimizations.hardware.quantization import BNBInt4Wrapper
        model = make_model()
        with pytest.raises(ImportError, match="bitsandbytes"):
            BNBInt4Wrapper().apply(model)

    def test_name_nf4(self):
        from optimizations.hardware.quantization import BNBInt4Wrapper
        assert BNBInt4Wrapper(use_nf4=True, double_quant=True).name() == "quant_nf4_bnb_dq"

    def test_name_int4(self):
        from optimizations.hardware.quantization import BNBInt4Wrapper
        assert BNBInt4Wrapper(use_nf4=False, double_quant=False).name() == "quant_int4_bnb"


class TestStructuredPruningWrapper:
    def test_amount_validation(self):
        from optimizations.hardware.pruning import StructuredPruningWrapper
        with pytest.raises(ValueError):
            StructuredPruningWrapper(amount=0.0)
        with pytest.raises(ValueError):
            StructuredPruningWrapper(amount=1.0)

    def test_apply_reduces_nonzero_params(self):
        from optimizations.hardware.pruning import StructuredPruningWrapper
        model = make_model()
        original_nnz = sum(p.numel() for p in model.parameters())
        pruned = StructuredPruningWrapper(amount=0.5).apply(model)
        pruned_nnz = sum((p != 0).sum().item() for p in pruned.parameters())
        assert pruned_nnz < original_nnz

    def test_pruned_model_still_runs(self):
        from optimizations.hardware.pruning import StructuredPruningWrapper
        model = make_model()
        pruned = StructuredPruningWrapper(amount=0.5).apply(model)
        x = torch.randn(2, 10, 8)
        out = pruned(x)
        assert out.shape == (2, 2)

    def test_name_format(self):
        from optimizations.hardware.pruning import StructuredPruningWrapper
        assert StructuredPruningWrapper(0.75).name() == "structured_pruning_75pct"

    def test_compression_ratio_gt1(self):
        from optimizations.hardware.pruning import StructuredPruningWrapper
        import copy
        model = make_model()
        pruned = StructuredPruningWrapper(0.5).apply(copy.deepcopy(model))
        ratio = StructuredPruningWrapper.compression_ratio(model, pruned)
        assert ratio > 1.0


class TestKnowledgeDistillationWrapper:
    def _make_kd(self):
        from optimizations.hardware.distillation import KnowledgeDistillationWrapper
        teacher = make_model()
        return KnowledgeDistillationWrapper(teacher=teacher, temperature=4.0, alpha=0.3)

    def test_teacher_frozen(self):
        kd = self._make_kd()
        for p in kd.teacher.parameters():
            assert not p.requires_grad

    def test_apply_attaches_refs(self):
        kd = self._make_kd()
        student = make_model()
        kd.apply(student)
        assert hasattr(student, "_kd_teacher")
        assert student._kd_temperature == 4.0
        assert student._kd_alpha == 0.3

    def test_distillation_loss_scalar(self):
        kd = self._make_kd()
        B, C = 4, 2
        student_logits = torch.randn(B, C)
        teacher_logits = torch.randn(B, C)
        labels = torch.randint(0, C, (B,))
        loss = kd.distillation_loss(student_logits, teacher_logits, labels)
        assert loss.ndim == 0   # scalar
        assert loss.item() > 0

    def test_distillation_loss_differentiable(self):
        kd = self._make_kd()
        student_logits = torch.randn(4, 2, requires_grad=True)
        teacher_logits = torch.randn(4, 2).detach()
        labels = torch.randint(0, 2, (4,))
        loss = kd.distillation_loss(student_logits, teacher_logits, labels)
        loss.backward()
        assert student_logits.grad is not None

    def test_metadata_keys(self):
        kd = self._make_kd()
        meta = kd.metadata()
        assert meta["kd_temperature"] == 4.0
        assert meta["kd_alpha"] == 0.3
        assert "teacher" in meta

    def test_name_contains_temp(self):
        kd = self._make_kd()
        assert "T4.0" in kd.name()


# ═════════════════════════════════════════════════════════════════════════════
# Base class contract
# ═════════════════════════════════════════════════════════════════════════════

class TestBaseContract:
    """Every concrete wrapper must satisfy the OptimizationWrapper contract."""

    def _all_wrappers(self):
        from optimizations.low_level.compilation import CompileWrapper
        from optimizations.low_level.batching import PaddingBatching, NestedTensorBatching
        from optimizations.low_level.ode_solvers import ODESolverWrapper
        from optimizations.low_level.flashattn import FlashAttnVariant
        from optimizations.hardware.precision import PrecisionWrapper
        from optimizations.hardware.quantization import PTQInt8Wrapper, BNBInt4Wrapper
        from optimizations.hardware.pruning import StructuredPruningWrapper
        return [
            CompileWrapper("default"),
            PaddingBatching(),
            NestedTensorBatching(),
            ODESolverWrapper(),
            FlashAttnVariant("sdpa_auto"),
            PrecisionWrapper("bf16"),
            PTQInt8Wrapper(),
            BNBInt4Wrapper(),
            StructuredPruningWrapper(0.5),
        ]

    def test_all_have_name(self):
        for w in self._all_wrappers():
            assert isinstance(w.name(), str) and len(w.name()) > 0, f"Empty name: {type(w)}"

    def test_all_have_metadata(self):
        for w in self._all_wrappers():
            meta = w.metadata()
            assert isinstance(meta, dict)
            assert "optimization" in meta, f"Missing 'optimization' key in {type(w)}"
