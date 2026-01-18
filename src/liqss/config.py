from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Sequence, Tuple


@dataclass
class LiQSSConfig:
    # Data
    data_dir: str = "data"
    x_file: str = "x.npy"
    y_file: str = "y.npy"
    feature_names: Tuple[str, ...] = (
        "MCS","CQI","RI","PMI","Buffer","PRBs","RSRQ","RSRP","RSSI","SINR","SE","BLER","Delay"
    )
    target: str = "RSRP"
    target_index: int = 7

    # Splits
    train_ratio: float = 0.70
    val_ratio: float = 0.15

    # Optimization
    batch_size: int = 256
    num_epochs: int = 120
    lr: float = 3e-3
    weight_decay: float = 1e-4
    patience: int = 30
    grad_clip: float = 1.0
    seed: int = 42

    # Model
    d_model: int = 64
    n_layers: int = 2
    ssm_state_dim: int = 32
    dropout: float = 0.1
    n_components: int = 2
    hippo_dt_init: float = 0.1

    # TT knobs
    use_tt_inproj: bool = True
    use_tt_head: bool = True

    tt_in_in_modes: Tuple[int, ...] = (1, 1, 13)   # product = K
    tt_in_out_modes: Tuple[int, ...] = (4, 4, 4)   # product = D
    tt_in_rank: int = 4

    tt_head_in_modes: Tuple[int, ...] = (4, 4, 4)  # product = D
    tt_head_out_modes: Tuple[int, ...] = (1,)      # expanded to (1,1,1) for scalar out
    tt_rank: int = 4

    # Output
    run_dir: str = "runs/liqss_default"

    def as_dict(self):
        return asdict(self)

    @property
    def data_path_x(self) -> Path:
        return Path(self.data_dir) / self.x_file

    @property
    def data_path_y(self) -> Path:
        return Path(self.data_dir) / self.y_file
