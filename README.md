# LiQSS: Linear Quantum-Inspired State-Space Tensor Networks for Real-Time 6G

This repository contains the **reference implementation** for the paper:

> **LiQSS: Post-Transformer Linear Quantum-Inspired State-Space Tensor Networks for Real-Time 6G**

LiQSS is a **post-Transformer** forecaster designed for **Near-RT O-RAN** telemetry. It replaces attention with **stable HiPPO–LegS structured state-space kernels** (linear-time in sequence length) and compresses global projections using **TT/MPS (Tensor Train / Matrix Product State) tensor networks**.

---

## Highlights

- **Linear-time** sequence modeling via **structured SSM** kernels (depthwise causal convolution)
- **Tensor-network (TT/MPS) compression** of the input embedding and prediction head
- **Leakage-safe** time-series pipeline: chronological split + train-only normalization
- Deployment-friendly: optimized for small footprint and low latency

---

## Repository structure

```
liqss/
  src/liqss/               Core library (model, TT layers, kernels, utils)
  scripts/                 Train/eval CLIs
  configs/                 Reproducible experiment configs (YAML)
  tests/                   Lightweight unit tests
  data/                    Data folder
  assets/                  Figures / misc
```

---

## Quickstart

### 1) Create an environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

### 2) Install dependencies

If you prefer explicit requirements:

```bash
pip install -r requirements.txt
```

### 3) Prepare data

This code expects **NumPy arrays**:

- `x.npy`: shape **(N, L, K)** (windows)
- `y.npy`: shape **(N, K)** (next-step labels for all KPIs)

Where:
- `N` = number of windows
- `L` = lookback window length
- `K` = number of KPIs (default: 13)

Place them under `data/` (or point the config to your directory).

**Example:**

```bash
mkdir -p data
cp /path/to/x.npy data/x.npy
cp /path/to/y.npy data/y.npy
```

---

## Run training

The recommended way is via the CLI.

### Train (default config)

```bash
python scripts/train.py --config configs/liqss_default.yaml
```

Artifacts are saved under `runs/<run_name>/`:

- `best.pt` (best checkpoint)
- `metrics.json`
- `test_pred.npy`, `test_true.npy`
- `meta.json` (data/meta)

### Evaluate an existing checkpoint

```bash
python scripts/eval.py \
  --config configs/liqss_default.yaml \
  --ckpt runs/liqss_default/best.pt
```

---

## Reproducibility

- Chronological splits (`train_ratio`, `val_ratio`) are applied exactly.
- Input and target scalers are computed on **training data only**.
- Fixed seed (default: 42).

> Note: GPU speed/latency will vary by device and CUDA/cuDNN versions.

---

## Configuration

All experiment knobs live in `configs/*.yaml`:

- model width (`d_model`)
- number of SSM blocks (`n_layers`)
- state dimension (`ssm_state_dim`)
- mixture size (`n_components`)
- TT ranks (`tt_in_rank`, `tt_rank`)
- optimizer params (`lr`, `weight_decay`)

---

## Citation

If you use this code, please cite the paper:

```bibtex
@article{rezazadeh2025liqss,
  title   = {LiQSS: Post-Transformer Linear Quantum-Inspired State-Space Tensor Networks for Real-Time 6G},
  author  = {Rezazadeh, Farhad and Chergui, Hatim and Bennis, Mehdi and Song, Houbing and Liu, Lingjia and Niyato, Dusit and Debbah, Merouane},
  journal = {arXiv:2601.12375},
  year    = {2026}
}
```

A machine-readable citation is also provided in [`CITATION.cff`](CITATION.cff).

---

## License

This code is released under the MIT License (see `LICENSE`).

If you use this implementation in academic work, please consider citing the corresponding paper

---

## Acknowledgements

- HiPPO / structured state-space models
- Tensor Train / MPS tensor networks
- O-RAN Near-RT RIC analytics context

---

## Contact

For questions, issues, or collaboration, open a GitHub issue or contact the corresponding author.
