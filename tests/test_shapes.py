import torch

from liqss.model import LiQSSForecaster


def test_forward_shapes():
    B, L, K = 2, 16, 13
    x = torch.randn(B, L, K)

    model = LiQSSForecaster(
        in_dim=K,
        d_model=64,
        out_dim=1,
        L=L,
        n_layers=2,
        state_dim=32,
        dropout=0.1,
        dt_init=0.1,
        n_components=2,
        use_tt_inproj=True,
        tt_in_in_modes=(1, 1, 13),
        tt_in_out_modes=(4, 4, 4),
        tt_in_rank=4,
        use_tt_head=True,
        tt_head_in_modes=(4, 4, 4),
        tt_head_out_modes=(1,),
        tt_rank=4,
    )

    y = model(x)
    assert y.shape == (B, 1)
