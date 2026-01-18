from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .kernels import SSMMixtureKernel


class ChannelMix(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, mult: int = 1):
        super().__init__()
        hidden = mult * d_model
        self.up = nn.Linear(d_model, 2 * hidden)
        self.act = nn.GELU()
        self.down = nn.Linear(hidden, d_model)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = self.up(x)
        a, g = torch.chunk(u, 2, dim=-1)
        y = self.down(self.act(a) * torch.sigmoid(g))
        y = self.drop(y)
        return self.norm(x + y)


class SEGate(nn.Module):
    def __init__(self, d_model: int, r: int = 16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(d_model, max(1, d_model // r)),
            nn.ReLU(inplace=True),
            nn.Linear(max(1, d_model // r), d_model),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,T,D)
        s = self.pool(x.transpose(1, 2)).squeeze(-1)  # (B,D)
        g = self.fc(s).unsqueeze(1)  # (B,1,D)
        return x * g


class SSMBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        state_dim: int,
        L: int,
        dropout: float = 0.1,
        dt_init: float = 0.1,
        n_components: int = 2,
    ):
        super().__init__()
        self.kernel = SSMMixtureKernel(d_model, state_dim, L, n_components=n_components, dt_init=dt_init)
        self.drop = nn.Dropout(dropout)
        self.se = SEGate(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.mix = ChannelMix(d_model, dropout=dropout, mult=1)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        k = self.kernel()  # (C,1,L)
        Lk = k.shape[-1]

        x_dw = x.transpose(1, 2)  # (B,C,T)
        y = F.conv1d(F.pad(x_dw, (Lk - 1, 0)), k, groups=C).transpose(1, 2)  # (B,T,C)

        y = self.se(y)
        y = self.drop(y)
        y = self.norm1(x + y)
        z = self.mix(y)
        return self.norm2(y + z)
