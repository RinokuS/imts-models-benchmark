"""PhysioNet Challenge 2012 loader.

Downloads and parses the raw set-a/set-b/set-c CSV files from PhysioNet.
Each patient record is a CSV with columns: Time,Parameter,Value.

Usage:
    ds = PhysioNetP12Dataset(data_dir="data/raw/physionet2012", split="train")
    t, v, m, y = ds[0]

Download the data manually from https://physionet.org/content/challenge-2012/1.0.0/
and place the extracted folders (set-a/, set-b/, set-c/, Outcomes-a.txt,
Outcomes-b.txt) under data_dir.
"""

import os
import csv
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

VARIABLES = [
    "ALP", "ALT", "AST", "Albumin", "BUN", "Bilirubin", "Cholesterol",
    "Creatinine", "DiasABP", "FiO2", "GCS", "Glucose", "HCO3", "HCT",
    "HR", "K", "Lactate", "MAP", "MechVent", "Mg", "NIDiasABP", "NIMAP",
    "NISysABP", "Na", "PaCO2", "PaO2", "Platelets", "RespRate", "SaO2",
    "SysABP", "Temp", "TroponinI", "TroponinT", "Urine", "WBC", "Weight",
]
VAR2IDX = {v: i for i, v in enumerate(VARIABLES)}
N_CHANNELS = len(VARIABLES)  # 36


def _parse_record(filepath: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse a single patient CSV into (times, values, mask) arrays."""
    obs: list[tuple[float, int, float]] = []

    with open(filepath) as f:
        reader = csv.DictReader(f)
        for row in reader:
            param = row["Parameter"].strip()
            if param not in VAR2IDX:
                continue
            try:
                h, m = row["Time"].strip().split(":")
                t = int(h) * 60 + int(m)  # minutes from admission
                v = float(row["Value"])
                obs.append((t / 2880.0, VAR2IDX[param], v))  # normalise to [0,1]
            except (ValueError, KeyError):
                continue

    if not obs:
        times = np.zeros((1,), dtype=np.float32)
        values = np.zeros((1, N_CHANNELS), dtype=np.float32)
        mask = np.zeros((1, N_CHANNELS), dtype=bool)
        return times, values, mask

    obs.sort(key=lambda x: x[0])
    unique_times = sorted(set(o[0] for o in obs))
    T = len(unique_times)
    t2idx = {t: i for i, t in enumerate(unique_times)}

    times = np.array(unique_times, dtype=np.float32)
    values = np.zeros((T, N_CHANNELS), dtype=np.float32)
    mask = np.zeros((T, N_CHANNELS), dtype=bool)

    for t, var_idx, val in obs:
        ti = t2idx[t]
        values[ti, var_idx] = val
        mask[ti, var_idx] = True

    return times, values, mask


def _load_outcomes(path: Path) -> dict[str, int]:
    outcomes: dict[str, int] = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row["RecordID"].strip()
            label = int(float(row["In-hospital_death"]))
            outcomes[pid] = label
    return outcomes


class PhysioNetP12Dataset(Dataset):
    """PhysioNet Challenge 2012 irregular MVTS dataset.

    Args:
        data_dir: directory containing set-a/, set-b/, Outcomes-a.txt, etc.
        split: "train" | "val" | "test"
        split_ratio: (train, val, test) fractions
        seed: random seed for stratified split
        normalize: fit and apply zscore on train; reuse stats for val/test
        _stats: optional (mean, std) tuple for val/test normalisation
    """

    def __init__(
        self,
        data_dir: str | Path = "data/raw/physionet2012",
        split: Literal["train", "val", "test"] = "train",
        split_ratio: tuple[float, float, float] = (0.8, 0.1, 0.1),
        seed: int = 42,
        normalize: bool = True,
        _stats: tuple[np.ndarray, np.ndarray] | None = None,
    ):
        super().__init__()
        data_dir = Path(data_dir)
        if not data_dir.exists():
            raise FileNotFoundError(
                f"PhysioNet P12 data not found at {data_dir}. "
                "Download from https://physionet.org/content/challenge-2012/1.0.0/ "
                "and place set-a/, set-b/, Outcomes-a.txt under data_dir."
            )

        all_times, all_values, all_masks, all_labels, all_pids = self._load_all(data_dir)
        indices = self._stratified_split(all_labels, split_ratio, seed, split)

        self.times = [all_times[i] for i in indices]
        self.values = [all_values[i] for i in indices]
        self.masks = [all_masks[i] for i in indices]
        self.labels = torch.tensor([all_labels[i] for i in indices], dtype=torch.long)

        if normalize:
            if _stats is None:
                self._stats = self._compute_stats()
            else:
                self._stats = _stats
            self._apply_normalization()

    def _load_all(self, data_dir: Path):
        subsets = ["set-a", "set-b"]
        outcome_files = ["Outcomes-a.txt", "Outcomes-b.txt"]
        outcomes: dict[str, int] = {}
        for of in outcome_files:
            p = data_dir / of
            if p.exists():
                outcomes.update(_load_outcomes(p))

        all_times, all_values, all_masks, all_labels, all_pids = [], [], [], [], []
        for subset in subsets:
            folder = data_dir / subset
            if not folder.exists():
                continue
            for filepath in sorted(folder.glob("*.txt")):
                pid = filepath.stem
                if pid not in outcomes:
                    continue
                t, v, m = _parse_record(filepath)
                all_times.append(torch.from_numpy(t))
                all_values.append(torch.from_numpy(v))
                all_masks.append(torch.from_numpy(m))
                all_labels.append(outcomes[pid])
                all_pids.append(pid)

        return all_times, all_values, all_masks, all_labels, all_pids

    @staticmethod
    def _stratified_split(labels, ratio, seed, split):
        from sklearn.model_selection import train_test_split

        n = len(labels)
        idx = np.arange(n)
        y = np.array(labels)
        train_r, val_r, test_r = ratio
        idx_train, idx_tmp, y_train, y_tmp = train_test_split(
            idx, y, test_size=(1 - train_r), random_state=seed, stratify=y
        )
        rel_val = val_r / (val_r + test_r)
        idx_val, idx_test = train_test_split(
            idx_tmp, test_size=(1 - rel_val), random_state=seed, stratify=y_tmp
        )
        return {"train": idx_train, "val": idx_val, "test": idx_test}[split]

    def _compute_stats(self):
        all_obs = [[] for _ in range(N_CHANNELS)]
        for v, m in zip(self.values, self.masks):
            for c in range(N_CHANNELS):
                obs = v[:, c][m[:, c]].numpy()
                all_obs[c].extend(obs.tolist())
        mu = np.array([np.mean(o) if o else 0.0 for o in all_obs], dtype=np.float32)
        sigma = np.array([np.std(o) if o else 1.0 for o in all_obs], dtype=np.float32)
        sigma[sigma < 1e-6] = 1.0
        return mu, sigma

    def _apply_normalization(self):
        mu, sigma = self._stats
        mu_t = torch.from_numpy(mu)
        sigma_t = torch.from_numpy(sigma)
        for i in range(len(self.values)):
            m = self.masks[i]
            self.values[i] = torch.where(m, (self.values[i] - mu_t) / sigma_t, self.values[i])

    def get_norm_stats(self):
        return self._stats

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.times[idx], self.values[idx], self.masks[idx], self.labels[idx]

    def get_class_weights(self) -> Tensor:
        counts = torch.bincount(self.labels, minlength=2).float()
        weights = 1.0 / (counts + 1e-6)
        return weights / weights.sum() * 2
