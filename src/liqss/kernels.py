from __future__ import annotations

import torch
import torch.nn as nn

from .hippo import hippo_legs_matrix, bilinear_discretize


class HiPPOLegSKernel(nn.Module):
    """HiPPO-LegS kernel generator producing depthwise convolution taps.

    Produces K with shape (D, 1, L).
    """

    def __init__(self, d_model: int, state_dim: int, L: int, dt_init: float = 0.1):
        super().__init__()
        self.d_model, self.N, self.L = d_model, state_dim, L

        A, B = hippo_legs_matrix(state_dim)
        self.register_buffer("A_ct", A)
        self.register_buffer("B_ref", B)

        B0 = B[None, :].repeat(d_model, 1)
        self.B = nn.Parameter(B0 + 0.01 * torch.randn_like(B0))
        self.C = nn.Parameter(torch.randn(d_model, state_dim) * 0.1)
        self.D = nn.Parameter(torch.zeros(d_model))
        self.log_dt = nn.Parameter(torch.log(torch.tensor(dt_init, dtype=torch.float32)))

    def forward(self) -> torch.Tensor:
        A_bar, B_bar = bilinear_discretize(self.A_ct, self.B, self.log_dt)
        C = self.C
        A_bar_T = A_bar.transpose(0, 1)

        x = B_bar  # (D, N)
        out = []
        for _ in range(self.L):
            y = (C * x).sum(dim=-1)  # (D,)
            out.append(y)
            x = x @ A_bar_T

        K = torch.stack(out, dim=1)  # (D, L)
        K[:, 0] += self.D
        return K.unsqueeze(1)  # (D,1,L)


class SSMMixtureKernel(nn.Module):
    def __init__(self, d_model: int, state_dim: int, L: int, n_components: int = 2, dt_init: float = 0.1):
        super().__init__()
        self.components = nn.ModuleList(
            [
                HiPPOLegSKernel(d_model, state_dim, L, dt_init=dt_init * (1.5**i))
                for i in range(n_components)
            ]
        )

    def forward(self) -> torch.Tensor:
        return torch.stack([c() for c in self.components], dim=0).sum(0)
