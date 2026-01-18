from __future__ import annotations

from dataclasses import asdict
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .config import LiQSSConfig
from .utils import StandardScaler


def load_windows(cfg: LiQSSConfig):
    x = np.load(cfg.data_path_x, allow_pickle=True)
    y = np.load(cfg.data_path_y, allow_pickle=True)

    if x.ndim != 3 or y.ndim != 2:
        raise ValueError(f"Expected x (N,L,K) and y (N,K). Got {x.shape}, {y.shape}.")

    N, L, K = x.shape
    if y.shape[0] != N or y.shape[1] != K:
        raise ValueError(f"y must be (N,K) with same N,K as x. Got x={x.shape}, y={y.shape}.")

    if K != len(cfg.feature_names):
        raise ValueError(
            f"Config feature_names length ({len(cfg.feature_names)}) does not match K ({K})."
        )

    n_train = int(N * cfg.train_ratio)
    n_val = int(N * cfg.val_ratio)

    idx_train = slice(0, n_train)
    idx_val = slice(n_train, n_train + n_val)
    idx_test = slice(n_train + n_val, N)

    x_train, x_val, x_test = x[idx_train], x[idx_val], x[idx_test]
    y_train, y_val, y_test = y[idx_train], y[idx_val], y[idx_test]

    # Train-only input scaler (flatten windows)
    x_train_flat = x_train.reshape(-1, K)
    x_mean = x_train_flat.mean(0)
    x_std = x_train_flat.std(0)
    x_scaler = StandardScaler(x_mean, x_std)

    x_train = x_scaler.transform(x_train)
    x_val = x_scaler.transform(x_val)
    x_test = x_scaler.transform(x_test)

    # Target scaler (single KPI by default)
    t = cfg.target_index
    y_train_t = y_train[:, t : t + 1]
    y_val_t = y_val[:, t : t + 1]
    y_test_t = y_test[:, t : t + 1]

    y_mean = y_train_t.mean(0)
    y_std = y_train_t.std(0)
    y_scaler = StandardScaler(y_mean, y_std)

    y_train_t = y_scaler.transform(y_train_t)
    y_val_t = y_scaler.transform(y_val_t)
    y_test_t = y_scaler.transform(y_test_t)

    # Torch tensors
    x_train_t = torch.tensor(x_train, dtype=torch.float32)
    x_val_t = torch.tensor(x_val, dtype=torch.float32)
    x_test_t = torch.tensor(x_test, dtype=torch.float32)

    y_train_t = torch.tensor(y_train_t, dtype=torch.float32)
    y_val_t = torch.tensor(y_val_t, dtype=torch.float32)
    y_test_t = torch.tensor(y_test_t, dtype=torch.float32)

    train_ds = TensorDataset(x_train_t, y_train_t)
    val_ds = TensorDataset(x_val_t, y_val_t)
    test_ds = TensorDataset(x_test_t, y_test_t)

    meta: Dict = {
        "N": int(N),
        "L": int(L),
        "K": int(K),
        "splits": {"train": int(n_train), "val": int(n_val), "test": int(N - n_train - n_val)},
        "target": cfg.target,
        "target_index": int(cfg.target_index),
        "feature_names": list(cfg.feature_names),
        "x_mean": x_mean.tolist(),
        "x_std": x_std.tolist(),
        "y_mean": y_mean.tolist(),
        "y_std": y_std.tolist(),
    }

    return train_ds, val_ds, test_ds, meta, x_scaler, y_scaler


def make_loaders(cfg: LiQSSConfig, train_ds, val_ds, test_ds):
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, pin_memory=True)
    return train_loader, val_loader, test_loader
