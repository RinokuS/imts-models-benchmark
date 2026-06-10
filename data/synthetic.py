"""Synthetic irregular MVTS dataset for scalability stress tests.

Generates multi-channel sinusoidal signals with controlled MCAR missingness
and asynchronous channel patterns. No files needed — everything is in memory.
"""

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


class SyntheticDataset(Dataset):
    """Synthetic irregular MVTS with controllable length and missingness.

    Args:
        n_samples: number of time series
        seq_len: number of time steps in the dense grid before missingness
        n_channels: number of channels (d)
        missing_frac: MCAR fraction [0, 1) — applied independently per channel
        n_classes: number of output classes
        seed: random seed for reproducibility
        normalize: whether to zscore the values
    """

    def __init__(
        self,
        n_samples: int = 5000,
        seq_len: int = 500,
        n_channels: int = 10,
        missing_frac: float = 0.40,
        n_classes: int = 2,
        seed: int = 0,
        normalize: bool = True,
    ):
        super().__init__()
        self.n_samples = n_samples
        self.seq_len = seq_len
        self.n_channels = n_channels
        self.missing_frac = missing_frac
        self.n_classes = n_classes

        rng = np.random.default_rng(seed)
        self.times, self.values, self.mask, self.labels = self._generate(rng, normalize)

    def _generate(self, rng, normalize):
        T = self.seq_len
        d = self.n_channels
        N = self.n_samples

        t_grid = np.linspace(0.0, 1.0, T, dtype=np.float32)  # (T,)

        # Build signals: superposition of 2-4 sinusoids per channel
        n_freqs = rng.integers(2, 5, size=(N, d))
        freqs = rng.uniform(1.0, 20.0, size=(N, d, 4))
        amps = rng.uniform(0.5, 2.0, size=(N, d, 4))
        phases = rng.uniform(0, 2 * np.pi, size=(N, d, 4))

        # (N, T, d)
        signals = np.zeros((N, T, d), dtype=np.float32)
        for i in range(N):
            for j in range(d):
                for f in range(n_freqs[i, j]):
                    signals[i, :, j] += amps[i, j, f] * np.sin(
                        2 * np.pi * freqs[i, j, f] * t_grid + phases[i, j, f]
                    )
        signals += rng.normal(0, 0.1, size=signals.shape).astype(np.float32)

        # Binary labels from a deterministic function of the clean signal
        label_scores = signals[:, T // 4 : 3 * T // 4, 0].mean(axis=1)
        labels = (label_scores > np.median(label_scores)).astype(np.int64)

        # Apply MCAR per channel with different patterns (asynchronous)
        mask = np.ones((N, T, d), dtype=bool)
        for j in range(d):
            drop = rng.random(size=(N, T)) < self.missing_frac
            # shift pattern slightly per channel to create asynchrony
            drop = np.roll(drop, shift=j * 3, axis=1)
            mask[:, :, j] = ~drop

        values = signals * mask  # zero-fill missing

        if normalize:
            # zscore per channel using observed values
            for j in range(d):
                obs = signals[:, :, j][mask[:, :, j]]
                mu, sigma = obs.mean(), obs.std() + 1e-6
                values[:, :, j] = (values[:, :, j] - mu) / sigma

        times = np.tile(t_grid, (N, 1))  # (N, T) — same grid for all
        return (
            torch.from_numpy(times),
            torch.from_numpy(values),
            torch.from_numpy(mask),
            torch.from_numpy(labels),
        )

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int):
        return (
            self.times[idx],    # (T,)
            self.values[idx],   # (T, d)
            self.mask[idx],     # (T, d)
            self.labels[idx],   # scalar
        )

    def get_class_weights(self) -> Tensor:
        counts = torch.bincount(self.labels, minlength=self.n_classes).float()
        weights = 1.0 / (counts + 1e-6)
        return weights / weights.sum() * self.n_classes


if __name__ == "__main__":
    ds = SyntheticDataset(n_samples=200, seq_len=100, n_channels=5, missing_frac=0.4)
    t, v, m, y = ds[0]
    print(f"times: {t.shape}, values: {v.shape}, mask: {m.shape}, label: {y.item()}")
    print(f"observed fraction: {m.float().mean():.2%}")
    print(f"class balance: {ds.labels.float().mean():.2%} positive")
