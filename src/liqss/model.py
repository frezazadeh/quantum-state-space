from __future__ import annotations

import torch
import torch.nn as nn

from .blocks import SSMBlock
from .tt import TTLinear, _prod


class QTNHead(nn.Module):
    def __init__(
        self,
        d_model: int,
        out_dim: int,
        in_modes=(4, 4, 4),
        out_modes=(1,),
        tt_rank: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        if _prod(in_modes) != int(d_model):
            raise ValueError("Product of tt_head_in_modes must equal d_model")

        d = len(in_modes)
        out_dim = int(out_dim)

        out_modes = tuple(int(x) for x in out_modes)
        if len(out_modes) == 1 and d > 1 and out_dim == 1 and out_modes[0] == 1:
            out_modes = (1,) * d

        if len(out_modes) != d or _prod(out_modes) != out_dim:
            raise ValueError("Product of tt_head_out_modes must equal out_dim")

        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.tt = TTLinear(in_modes, out_modes, tt_rank=tt_rank, bias=True)

    def forward(self, h_last: torch.Tensor) -> torch.Tensor:
        return self.tt(self.drop(self.norm(h_last)))


class LiQSSForecaster(nn.Module):
    """LiQSS model: TT input projection + stacked SSM blocks + TT head."""

    def __init__(
        self,
        in_dim: int,
        d_model: int,
        out_dim: int,
        L: int,
        n_layers: int,
        state_dim: int,
        dropout: float,
        dt_init: float,
        n_components: int,
        use_tt_inproj: bool,
        tt_in_in_modes,
        tt_in_out_modes,
        tt_in_rank: int,
        use_tt_head: bool,
        tt_head_in_modes,
        tt_head_out_modes,
        tt_rank: int,
    ):
        super().__init__()

        if use_tt_inproj:
            if _prod(tt_in_in_modes) != in_dim:
                raise ValueError("Product of tt_in_in_modes must equal in_dim")
            if _prod(tt_in_out_modes) != d_model:
                raise ValueError("Product of tt_in_out_modes must equal d_model")
            self.in_proj = TTLinear(tt_in_in_modes, tt_in_out_modes, tt_rank=tt_in_rank, bias=True)
        else:
            self.in_proj = nn.Linear(in_dim, d_model)

        self.blocks = nn.ModuleList(
            [
                SSMBlock(d_model, state_dim, L, dropout=dropout, dt_init=dt_init, n_components=n_components)
                for _ in range(n_layers)
            ]
        )

        if use_tt_head:
            self.head = QTNHead(
                d_model,
                out_dim,
                in_modes=tt_head_in_modes,
                out_modes=tt_head_out_modes,
                tt_rank=tt_rank,
                dropout=dropout,
            )
        else:
            self.head = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, out_dim),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,L,K)
        h = self.in_proj(x)  # (B,L,D)
        for blk in self.blocks:
            h = blk(h)
        return self.head(h[:, -1, :])
