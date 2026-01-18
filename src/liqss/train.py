from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from .config import LiQSSConfig
from .data import load_windows, make_loaders
from .model import LiQSSForecaster
from .utils import ensure_dir, save_json, set_seed


def build_model(cfg: LiQSSConfig, L: int, K: int) -> LiQSSForecaster:
    return LiQSSForecaster(
        in_dim=K,
        d_model=cfg.d_model,
        out_dim=1,
        L=L,
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
    )


@torch.no_grad()
def eval_in_original_units(model, loader, y_scaler, device: str) -> Tuple[float, float]:
    model.eval()
    sse = 0.0
    sae = 0.0
    n = 0
    for xb, yb in loader:
        xb = xb.to(device)
        pred = model(xb).cpu().numpy()
        y = yb.cpu().numpy()
        pred_inv = y_scaler.inverse_transform(pred)
        y_inv = y_scaler.inverse_transform(y)
        sse += float(((pred_inv - y_inv) ** 2).sum())
        sae += float(np.abs(pred_inv - y_inv).sum())
        n += pred_inv.size
    mse = sse / max(1, n)
    mae = sae / max(1, n)
    return mse, mae


def train(cfg: LiQSSConfig) -> Dict:
    torch.set_grad_enabled(True)
    set_seed(cfg.seed)

    run_dir = ensure_dir(cfg.run_dir)

    train_ds, val_ds, test_ds, meta, x_scaler, y_scaler = load_windows(cfg)
    train_loader, val_loader, test_loader = make_loaders(cfg, train_ds, val_ds, test_ds)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = build_model(cfg, L=meta["L"], K=meta["K"]).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=2)
    loss_fn = nn.MSELoss()

    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    best_val = float("inf")
    patience = cfg.patience
    best_path = run_dir / "best.pt"

    save_json(cfg.as_dict(), run_dir / "config.json")
    save_json(meta, run_dir / "meta.json")

    for epoch in range(1, cfg.num_epochs + 1):
        model.train()
        tr_sum = 0.0
        ntr = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:03d}", leave=False)
        with torch.enable_grad():
            for xb, yb in pbar:
                xb = xb.to(device)
                yb = yb.to(device)
                opt.zero_grad(set_to_none=True)

                with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                    pred = model(xb)
                    loss = loss_fn(pred, yb)

                scaler.scale(loss).backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(opt)
                scaler.update()

                tr_sum += float(loss.item()) * xb.size(0)
                ntr += xb.size(0)
                pbar.set_postfix({"train_mse": tr_sum / max(1, ntr)})

        tr_loss = tr_sum / max(1, ntr)

        model.eval()
        with torch.no_grad():
            va_sum = 0.0
            nva = 0
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                va_sum += float(loss_fn(pred, yb).item()) * xb.size(0)
                nva += xb.size(0)
            va_loss = va_sum / max(1, nva)

        sched.step(va_loss)

        # Early stopping
        if va_loss < best_val - 1e-6:
            best_val = va_loss
            torch.save({"model": model.state_dict(), "meta": meta, "config": cfg.as_dict()}, best_path)
            patience = cfg.patience
        else:
            patience -= 1
            if patience == 0:
                break

    # Reload best and compute original-unit metrics
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])

    tr_mse, tr_mae = eval_in_original_units(model, train_loader, y_scaler, device)
    va_mse, va_mae = eval_in_original_units(model, val_loader, y_scaler, device)
    te_mse, te_mae = eval_in_original_units(model, test_loader, y_scaler, device)

    metrics = {
        "train": {"mse": tr_mse, "rmse": float(math.sqrt(tr_mse)), "mae": tr_mae},
        "val": {"mse": va_mse, "rmse": float(math.sqrt(va_mse)), "mae": va_mae},
        "test": {"mse": te_mse, "rmse": float(math.sqrt(te_mse)), "mae": te_mae},
        "best_val_mse": float(best_val),
        "n_params": int(n_params),
    }

    save_json(metrics, run_dir / "metrics.json")

    return metrics
