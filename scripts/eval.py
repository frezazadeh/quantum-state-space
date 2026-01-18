#!/usr/bin/env python
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch
import yaml

from liqss.config import LiQSSConfig
from liqss.data import load_windows, make_loaders
from liqss.model import LiQSSForecaster
from liqss.utils import StandardScaler, save_json


def load_cfg(path: str) -> LiQSSConfig:
    d = yaml.safe_load(Path(path).read_text())
    return LiQSSConfig(**d)


@torch.no_grad()
def eval_ckpt(cfg: LiQSSConfig, ckpt_path: str):
    train_ds, val_ds, test_ds, meta, x_scaler, y_scaler = load_windows(cfg)
    _, _, test_loader = make_loaders(cfg, train_ds, val_ds, test_ds)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = LiQSSForecaster(
        in_dim=meta["K"],
        d_model=cfg.d_model,
        out_dim=1,
        L=meta["L"],
        n_layers=cfg.n_layers,
        state_dim=cfg.ssm_state_dim,
        dropout=cfg.dropout,
        dt_init=cfg.hippo_dt_init,
        n_components=cfg.n_components,
        use_tt_inproj=cfg.use_tt_inproj,
        tt_in_in_modes=cfg.tt_in_in_modes,
        tt_in_out_modes=cfg.tt_in_out_modes,
        tt_in_rank=cfg.tt_in_rank,
        use_tt_head=cfg.use_tt_head,
        tt_head_in_modes=cfg.tt_head_in_modes,
        tt_head_out_modes=cfg.tt_head_out_modes,
        tt_rank=cfg.tt_rank,
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    preds, trues = [], []
    sse = 0.0
    sae = 0.0
    n = 0

    for xb, yb in test_loader:
        xb = xb.to(device)
        p = model(xb).cpu().numpy()
        y = yb.cpu().numpy()

        p_inv = y_scaler.inverse_transform(p)
        y_inv = y_scaler.inverse_transform(y)

        preds.append(p_inv)
        trues.append(y_inv)

        sse += float(((p_inv - y_inv) ** 2).sum())
        sae += float(np.abs(p_inv - y_inv).sum())
        n += p_inv.size

    mse = sse / max(1, n)
    mae = sae / max(1, n)
    rmse = math.sqrt(mse)

    out = {"test": {"mse": mse, "rmse": rmse, "mae": mae}}
    save_json(out, Path(cfg.run_dir) / "eval_metrics.json")

    print(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    eval_ckpt(cfg, args.ckpt)


if __name__ == "__main__":
    main()
