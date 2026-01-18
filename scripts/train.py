#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from liqss.config import LiQSSConfig
from liqss.train import train


def load_cfg(path: str) -> LiQSSConfig:
    d = yaml.safe_load(Path(path).read_text())
    return LiQSSConfig(**d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    metrics = train(cfg)
    print("Training complete. Metrics:")
    print(metrics)


if __name__ == "__main__":
    main()
