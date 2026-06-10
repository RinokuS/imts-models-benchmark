"""PhysioNet Challenge 2019 loader.

Parses the official PSV (pipe-separated values) patient records.
Each file: one row per hour; columns include 34 clinical variables + SepsisLabel.

Download training_setA.zip and training_setB.zip from
https://physionet.org/content/challenge-2019/1.0.0/
Extract under data_dir/training/ so that you have:
    data_dir/training/p000001.psv  ...

Sepsis label logic: a patient is positive if SepsisLabel==1 at t-6h or earlier.
We use the last observed SepsisLabel to determine the sequence-level label.
"""

from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

VARIABLES = [
    "HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2",
    "BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST",
    "BUN", "Alkalinephos", "Calcium", "Chloride", "Creatinine",
    "Bilirubin_direct", "Glucose", "Lactate", "Magnesium", "Phosphate",
    "Potassium", "Bilirubin_total", "TroponinI", "Hct", "Hgb",
    "PTT", "WBC", "Fibrinogen", "Platelets",
]
N_CHANNELS = len(VARIABLES)  # 34
VAR2IDX = {v: i for i, v in enumerate(VARIABLES)}


def _parse_psv(filepath: Path):
    """Parse a PSV file into (times, values, mask, label)."""
    with open(filepath) as f:
        header = f.readline().strip().split("|")
        rows = [line.strip().split("|") for line in f if line.strip()]

    if not rows:
        times = np.zeros((1,), dtype=np.float32)
        values = np.zeros((1, N_CHANNELS), dtype=np.float32)
        mask = np.zeros((1, N_CHANNELS), dtype=bool)
        return times, values, mask, 0

    col2idx = {c: i for i, c in enumerate(header)}
    T = len(rows)
    times = np.arange(T, dtype=np.float32) / 48.0  # hours / 48 → [0, ~]

    values = np.zeros((T, N_CHANNELS), dtype=np.float32)
    mask = np.zeros((T, N_CHANNELS), dtype=bool)

    for t, row in enumerate(rows):
        for var, cidx in VAR2IDX.items():
            if var not in col2idx:
                continue
            raw = row[col2idx[var]].strip()
            if raw and raw != "NaN":
                try:
                    values[t, cidx] = float(raw)
                    mask[t, cidx] = True
                except ValueError:
                    pass

    # Sequence-level label: positive if any SepsisLabel==1
    label = 0
    if "SepsisLabel" in col2idx:
        sl_col = col2idx["SepsisLabel"]
        for row in rows:
            try:
                if int(float(row[sl_col])) == 1:
                    label = 1
                    break
            except (ValueError, IndexError):
                pass

    return times, values, mask, label


class PhysioNetP19Dataset(Dataset):
    """PhysioNet Challenge 2019 irregular MVTS dataset.

    Args:
        data_dir: path containing training/ subdirectory with *.psv files
        split: "train" | "val" | "test"
        split_ratio: (train, val, test) fractions
        seed: random seed for stratified split
        normalize: apply zscore using train statistics
        _stats: optional (mu, sigma) for val/test
    """

    def __init__(
        self,
        data_dir: str | Path = "data/raw/physionet2019",
        split: Literal["train", "val", "test"] = "train",
        split_ratio: tuple[float, float, float] = (0.7, 0.15, 0.15),
        seed: int = 42,
        normalize: bool = True,
        _stats=None,
    ):
        super().__init__()
        data_dir = Path(data_dir)
        training_dir = data_dir / "training"
        if not training_dir.exists():
            raise FileNotFoundError(
                f"PhysioNet P19 data not found at {training_dir}. "
                "Download from https://physionet.org/content/challenge-2019/1.0.0/"
            )

        # Support both flat layout (training/*.psv) and nested (training/training_setA/*.psv)
        all_files = sorted(training_dir.glob("*.psv"))
        if not all_files:
            all_files = sorted(training_dir.glob("**/*.psv"))
        all_times, all_values, all_masks, all_labels = [], [], [], []
        for fp in all_files:
            t, v, m, y = _parse_psv(fp)
            all_times.append(torch.from_numpy(t))
            all_values.append(torch.from_numpy(v))
            all_masks.append(torch.from_numpy(m))
            all_labels.append(y)

        indices = self._stratified_split(all_labels, split_ratio, seed, split)

        self.times = [all_times[i] for i in indices]
        self.values = [all_values[i] for i in indices]
        self.masks = [all_masks[i] for i in indices]
        self.labels = torch.tensor([all_labels[i] for i in indices], dtype=torch.long)

        if normalize:
            self._stats = _stats if _stats else self._compute_stats()
            self._apply_normalization()

    @staticmethod
    def _stratified_split(labels, ratio, seed, split):
        import numpy as np
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
        mu_t = torch.from_numpy(self._stats[0])
        sigma_t = torch.from_numpy(self._stats[1])
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
