#!/usr/bin/env python3
"""Validate the EE-augmented MovePick dataset.

Checks:
  - augmented parquet count and required columns
  - per-row dimensions and finite values
  - root xyz/rpy reconstructed from reset pose + vx/vy integration
  - SIMPLE/MuJoCo root_xyzyaw consistency
  - optional FK recomputation for full 28D EE constraints
  - policy action = hand joints + root xyz/rpy + four EE poses
  - global stats.json dimensions and values for the new columns
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_movepick_kimodo_dataset import (  # noqa: E402
    DEFAULT_DATASET,
    DEFAULT_URDF,
    build_ee_constraint,
    feature_stats,
    integrate_root_xyz,
    load_episode_configs,
    make_fk,
    quat_wxyz_to_yaw,
    robot_reset_pose,
    unwrap_yaw,
)


NEW_FEATURES = {
    "observation.full_state": 49,
    "observation.root_state_6": 6,
    "action.ee_constraints": 28,
    "action.ee_root_xyzyaw": 4,
    "action.policy_action": 44,
}


def stack_column(df: pd.DataFrame, key: str) -> np.ndarray:
    return np.stack(df[key].to_numpy()).astype(np.float32)


def max_abs(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 and b.size == 0:
        return 0.0
    return float(np.max(np.abs(a - b)))


def expected_root(src_df: pd.DataFrame, cfg: dict[str, Any], fps: float) -> tuple[np.ndarray, np.ndarray]:
    actions = stack_column(src_df, "action")
    reset_pos, reset_quat = robot_reset_pose(cfg)
    reset_yaw = quat_wxyz_to_yaw(reset_quat)

    root_rpy = actions[:, 28:31].astype(np.float64)
    root_rpy[:, 2] = unwrap_yaw(root_rpy[:, 2], reset_yaw)
    root_xyz = integrate_root_xyz(
        reset_pos,
        actions[:, 31].astype(np.float64),
        root_rpy[:, 2],
        actions[:, 32].astype(np.float64),
        actions[:, 33].astype(np.float64),
        fps,
    )
    return root_xyz.astype(np.float32), root_rpy.astype(np.float32)


def expected_full_state(src_df: pd.DataFrame, root_state_6: np.ndarray) -> np.ndarray:
    hand = stack_column(src_df, "observation.hand_joints")
    leg = stack_column(src_df, "observation.leg_joints")
    arm = stack_column(src_df, "observation.arm_joints")
    return np.concatenate([hand, leg, arm, root_state_6], axis=1).astype(np.float32)


def expected_policy_action(src_df: pd.DataFrame, root_state_6: np.ndarray, ee_constraints: np.ndarray) -> np.ndarray:
    actions = stack_column(src_df, "action")
    return np.concatenate([actions[:, 0:14], root_state_6, ee_constraints[:, 4:28]], axis=1).astype(np.float32)


def expected_root_xyzyaw(root_xyz: np.ndarray, root_rpy: np.ndarray) -> np.ndarray:
    return np.concatenate([root_xyz, root_rpy[:, 2:3]], axis=1).astype(np.float32)


def check_shapes_and_finiteness(df: pd.DataFrame, rel: Path) -> list[str]:
    errors = []
    for key, dim in NEW_FEATURES.items():
        if key not in df.columns:
            errors.append(f"{rel}: missing column {key}")
            continue
        values = stack_column(df, key)
        if values.ndim != 2 or values.shape[1] != dim:
            errors.append(f"{rel}: {key} shape {values.shape}, expected [T, {dim}]")
        if not np.isfinite(values).all():
            errors.append(f"{rel}: {key} contains NaN/Inf")
    return errors


def sample_indices(n_frames: int, per_episode: int) -> list[int]:
    if per_episode <= 0 or n_frames <= 0:
        return []
    if per_episode >= n_frames:
        return list(range(n_frames))
    return sorted(set(np.linspace(0, n_frames - 1, per_episode).round().astype(int).tolist()))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, default=DEFAULT_DATASET, help="Original MovePick dataset dir.")
    parser.add_argument("--aug", type=Path, default=None, help="Augmented dataset dir. Default: <src>/kimodo_augmented_v2.")
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF, help="G1 URDF for optional FK checks.")
    parser.add_argument("--fps", type=float, default=None, help="Override dataset fps.")
    parser.add_argument("--limit-episodes", type=int, default=None, help="Only validate the first N episodes.")
    parser.add_argument(
        "--fk-samples-per-episode",
        type=int,
        default=3,
        help="Recompute full 28D constraints for this many frames per episode. Use 0 to skip FK.",
    )
    parser.add_argument("--tol", type=float, default=1e-4, help="Max allowed absolute error.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    src = args.src.resolve()
    aug = (args.aug or (src / "kimodo_augmented_v2")).resolve()
    info = json.loads((src / "meta" / "info.json").read_text())
    fps = float(args.fps or info["fps"])
    episode_cfgs = load_episode_configs(src / "meta")

    src_files = sorted((src / "data").glob("chunk-*/episode_*.parquet"))
    aug_files = sorted((aug / "data").glob("chunk-*/episode_*.parquet"))
    if args.limit_episodes is not None:
        src_files = src_files[: args.limit_episodes]

    errors: list[str] = []
    metrics = {
        "root_state_6": 0.0,
        "root_xyzyaw": 0.0,
        "full_state": 0.0,
        "ee_constraints_sampled": 0.0,
        "policy_action": 0.0,
        "stats": 0.0,
    }

    if args.limit_episodes is None and len(aug_files) != len(src_files):
        errors.append(f"episode count mismatch: src={len(src_files)}, aug={len(aug_files)}")

    fk = make_fk(args.urdf) if args.fk_samples_per_episode > 0 else None
    all_new_values: dict[str, list[np.ndarray]] = {key: [] for key in NEW_FEATURES}

    for src_path in tqdm(src_files, desc="Validating EE augmented"):
        rel = src_path.relative_to(src)
        aug_path = aug / rel
        if not aug_path.exists():
            errors.append(f"{rel}: missing augmented parquet")
            continue

        episode_index = int(src_path.stem.split("_")[-1])
        src_df = pd.read_parquet(src_path)
        aug_df = pd.read_parquet(aug_path)
        if len(src_df) != len(aug_df):
            errors.append(f"{rel}: row count mismatch src={len(src_df)}, aug={len(aug_df)}")
            continue

        errors.extend(check_shapes_and_finiteness(aug_df, rel))
        root_xyz, root_rpy = expected_root(src_df, episode_cfgs[episode_index], fps)
        root_state_6 = np.concatenate([root_xyz, root_rpy], axis=1).astype(np.float32)
        root_xyzyaw = expected_root_xyzyaw(root_xyz, root_rpy)
        full_state = expected_full_state(src_df, root_state_6)

        metrics["root_state_6"] = max(metrics["root_state_6"], max_abs(stack_column(aug_df, "observation.root_state_6"), root_state_6))
        metrics["root_xyzyaw"] = max(
            metrics["root_xyzyaw"],
            max_abs(stack_column(aug_df, "action.ee_root_xyzyaw"), root_xyzyaw),
            max_abs(stack_column(aug_df, "action.ee_constraints")[:, :4], root_xyzyaw),
        )
        metrics["full_state"] = max(metrics["full_state"], max_abs(stack_column(aug_df, "observation.full_state"), full_state))
        metrics["policy_action"] = max(
            metrics["policy_action"],
            max_abs(
                stack_column(aug_df, "action.policy_action"),
                expected_policy_action(src_df, root_state_6, stack_column(aug_df, "action.ee_constraints")),
            ),
        )

        for idx in sample_indices(len(src_df), args.fk_samples_per_episode):
            expected = build_ee_constraint(fk, src_df.iloc[idx], root_xyz[idx], root_rpy[idx])
            actual = np.asarray(aug_df.iloc[idx]["action.ee_constraints"], dtype=np.float32)
            metrics["ee_constraints_sampled"] = max(metrics["ee_constraints_sampled"], max_abs(actual, expected))

        for key in NEW_FEATURES:
            all_new_values[key].append(stack_column(aug_df, key))

    stats_path = aug / "meta" / "stats.json"
    if stats_path.exists():
        stats = json.loads(stats_path.read_text())
        for key, chunks in all_new_values.items():
            if key not in stats:
                errors.append(f"stats.json missing {key}")
                continue
            expected = feature_stats(np.concatenate(chunks, axis=0))
            for stat_name, expected_values in expected.items():
                actual = np.asarray(stats[key][stat_name], dtype=np.float32)
                exp = np.asarray(expected_values, dtype=np.float32)
                metrics["stats"] = max(metrics["stats"], max_abs(actual, exp))
    else:
        errors.append(f"missing {stats_path}")

    for name, value in metrics.items():
        print(f"{name:28s} max_abs_error = {value:.8g}")
        if value > args.tol:
            errors.append(f"{name} max_abs_error {value:.8g} > tol {args.tol}")

    if errors:
        print("\nFAILED")
        for err in errors[:50]:
            print(f"- {err}")
        if len(errors) > 50:
            print(f"- ... {len(errors) - 50} more")
        raise SystemExit(1)

    print(f"\nOK: validated {len(src_files)} episode(s) from {aug}")


if __name__ == "__main__":
    main()
