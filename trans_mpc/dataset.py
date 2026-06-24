from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset


LABEL_KEYS = ("r_low", "r_ey", "r_stab", "k_v")


@dataclass
class FeatureStandardizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, features: np.ndarray) -> "FeatureStandardizer":
        mean = np.mean(features, axis=0)
        std = np.std(features, axis=0)
        std = np.where(std < 1.0e-6, 1.0, std)
        return cls(mean=mean, std=std)

    def transform(self, features: np.ndarray) -> np.ndarray:
        return (features - self.mean) / self.std

    def inverse_transform(self, features: np.ndarray) -> np.ndarray:
        return features * self.std + self.mean


def build_windowed_arrays(
    features: np.ndarray,
    labels: dict[str, np.ndarray],
    history_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2:
        raise ValueError("features must be a 2D array")
    if len(features) < history_len:
        return np.empty((0, history_len, features.shape[1]), dtype=np.float32), np.empty((0, 4), dtype=np.float32)

    label_matrix = np.column_stack([np.asarray(labels[key], dtype=np.float32) for key in LABEL_KEYS])
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for end in range(history_len - 1, len(features)):
        start = end - history_len + 1
        xs.append(features[start : end + 1])
        ys.append(label_matrix[end])
    return np.stack(xs).astype(np.float32), np.stack(ys).astype(np.float32)


class WindowedRiskDataset(Dataset):
    def __init__(self, windows: np.ndarray, targets: np.ndarray):
        self.windows = torch.as_tensor(windows, dtype=torch.float32)
        self.targets = torch.as_tensor(targets, dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.windows.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.windows[index], self.targets[index]


def train_val_split(
    windows: np.ndarray,
    targets: np.ndarray,
    val_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(windows)
    rng = np.random.default_rng(seed)
    indices = np.arange(n)
    rng.shuffle(indices)
    val_count = max(1, int(round(n * val_fraction))) if n > 1 else 0
    val_idx = indices[:val_count]
    train_idx = indices[val_count:]
    return windows[train_idx], targets[train_idx], windows[val_idx], targets[val_idx]
