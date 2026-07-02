#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from stitch_kimodo_chunks import _qpos_to_kimodo_pose


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert MuJoCo qpos36 CSV to a Kimodo-viewer-compatible NPZ.")
    parser.add_argument("csv", type=Path, help="CSV with columns root_xyz3 + root_quat_wxyz4 + body29.")
    parser.add_argument("--output", type=Path, default=None, help="Defaults to CSV with .kimodo.npz suffix.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    qpos = np.loadtxt(args.csv, delimiter=",")
    if qpos.ndim == 1:
        qpos = qpos[None]
    if qpos.shape[1] != 36:
        raise ValueError(f"Expected qpos36 CSV, got shape {qpos.shape}")
    output = args.output or args.csv.with_suffix(".kimodo.npz")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, **_qpos_to_kimodo_pose(qpos))
    print(output)


if __name__ == "__main__":
    main()
