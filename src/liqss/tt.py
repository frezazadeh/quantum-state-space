from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import torch
import torch.nn as nn


def _prod(xs: Sequence[int]) -> int:
    p = 1
    for v in xs:
        p *= int(v)
    return int(p)


class TTLinear(nn.Module):
    """Tensor-Train (TT/MPS) factorized linear layer.

    This layer parameterizes a matrix W (out_dim x in_dim) via TT cores.
    Input is reshaped into in_modes; output is reshaped into out_modes.

    Note: This is designed for clarity and reproducibility; if you need maximum speed,
    consider specialized TT implementations.
    """

    def __init__(
        self,
        in_modes: Sequence[int],
        out_modes: Sequence[int],
        tt_rank: int = 4,
        bias: bool = True,
        init_std: float = 0.02,
    ):
        super().__init__()
        self.in_modes = tuple(int(x) for x in in_modes)
        self.out_modes = tuple(int(x) for x in out_modes)
        if len(self.in_modes) != len(self.out_modes):
            raise ValueError("in_modes and out_modes must have the same length")

        self.d = len(self.in_modes)
        self.in_dim = _prod(self.in_modes)
        self.out_dim = _prod(self.out_modes)

        if isinstance(tt_rank, int):
            self.ranks = [1] + [int(tt_rank)] * (self.d - 1) + [1]
        else:
            self.ranks = list(tt_rank)
            if len(self.ranks) != self.d + 1 or self.ranks[0] != 1 or self.ranks[-1] != 1:
                raise ValueError("tt_rank list must have length d+1 and start/end with 1")

        self.cores = nn.ParameterList()
        for k in range(self.d):
            r0, r1 = self.ranks[k], self.ranks[k + 1]
            ik, ok = self.in_modes[k], self.out_modes[k]
            self.cores.append(nn.Parameter(torch.randn(r0, ik, ok, r1) * init_std))

        self.bias = nn.Parameter(torch.zeros(self.out_dim)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig = x.shape[:-1]
        x = x.reshape(-1, self.in_dim)
        B = x.size(0)

        X = x.reshape(B, *self.in_modes)
        S = X.reshape(B, 1, 1, self.in_dim)
        in_remaining = self.in_dim
        out_prod = 1

        for k in range(self.d):
            ik = self.in_modes[k]
            ok = self.out_modes[k]
            r0 = self.ranks[k]
            r1 = self.ranks[k + 1]

            in_remaining //= ik
            S = S.reshape(B, r0, out_prod, ik, in_remaining)

            S_flat = S.permute(0, 2, 4, 1, 3).contiguous()
            S_flat = S_flat.view(B * out_prod * in_remaining, r0 * ik)

            G = self.cores[k].view(r0 * ik, ok * r1)
            Y = S_flat @ G
            Y = Y.view(B, out_prod, in_remaining, ok, r1)

            out_prod *= ok
            S = Y.permute(0, 4, 1, 3, 2).contiguous()
            S = S.view(B, r1, out_prod, in_remaining)

        S = S.view(B, self.out_dim)
        if self.bias is not None:
            S = S + self.bias
        return S.reshape(*orig, self.out_dim)
