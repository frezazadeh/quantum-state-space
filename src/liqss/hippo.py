from __future__ import annotations

import torch
import torch.nn.functional as F


def hippo_legs_matrix(N: int, device=None, dtype=torch.float32):
    """HiPPO-LegS continuous-time generator A_ct and reference input B_ref.

    This matches the structure used in many HiPPO/SSM implementations.
    """
    n = torch.arange(N, device=device, dtype=dtype)
    two_n1 = 2 * n + 1
    sq = torch.sqrt(two_n1[:, None] * two_n1[None, :])
    A = torch.tril(sq, diagonal=-1) * (-1.0)
    A = A + torch.diag(-(n + 1.0))
    B = torch.sqrt(two_n1)
    return A, B


def bilinear_discretize(A_ct: torch.Tensor, B: torch.Tensor, log_dt: torch.Tensor):
    """Tustin/bilinear discretization.

    Args:
        A_ct: (N,N)
        B: (D,N)
        log_dt: scalar parameter

    Returns:
        A_bar: (N,N)
        B_bar: (D,N)
    """
    dt = F.softplus(log_dt) + 1e-6
    N = A_ct.size(0)
    I = torch.eye(N, device=A_ct.device, dtype=A_ct.dtype)
    lhs = I - 0.5 * dt * A_ct
    rhs = I + 0.5 * dt * A_ct
    A_bar = torch.linalg.solve(lhs, rhs)

    # Solve lhs * X = dt*B^T, then transpose back
    B_rhs = (dt * B).transpose(0, 1)  # (N, D)
    B_bar_T = torch.linalg.solve(lhs, B_rhs)
    B_bar = B_bar_T.transpose(0, 1)  # (D, N)
    return A_bar, B_bar
