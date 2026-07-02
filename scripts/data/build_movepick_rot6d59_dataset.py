#!/usr/bin/env python3
"""Build a MovePick dataset for Psi0 -> hand14 + root/EE rot6d policy actions.

The output action layout is:

    hand14 + root9 + four EE poses * 9 = 59D

where each 9D pose is xyz + rotation_6d.  The policy action root/EE poses are
future targets shifted from the observation stream by ``--target-shift`` frames
(default: 1).  The final frames are padded by repeating the last available pose.
The EE link poses use the same MuJoCo link names as the Kimodo adapter:
left/right wrist_yaw_link and left/right ankle_roll_link.

Two source layouts are supported automatically:

1. Legacy SIMPLE datasets containing ``observation.state``, base pose and EE
   link-pose streams.  The 52D/59D terms are constructed from those streams.
2. Multi-chain datasets that already contain ``observation.full_state_rot6d``
   and ``action.policy_action_rot6d59``.  Those terms are validated and copied
   without applying another temporal shift.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SRC = REPO_ROOT / "data/simple/G1WholebodyXMovePickTeleop-v0-eval-fullstate-visualdr-99-merged"

HAND14_ORDER = [
    "left_hand_thumb_0_joint",
    "left_hand_thumb_1_joint",
    "left_hand_thumb_2_joint",
    "left_hand_middle_0_joint",
    "left_hand_middle_1_joint",
    "left_hand_index_0_joint",
    "left_hand_index_1_joint",
    "right_hand_thumb_0_joint",
    "right_hand_thumb_1_joint",
    "right_hand_thumb_2_joint",
    "right_hand_index_0_joint",
    "right_hand_index_1_joint",
    "right_hand_middle_0_joint",
    "right_hand_middle_1_joint",
]

BODY29_ORDER = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

POSE9_KEYS = [
    "observation.left_wrist_yaw_link_pose",
    "observation.right_wrist_yaw_link_pose",
    "observation.left_ankle_roll_link_pose",
    "observation.right_ankle_roll_link_pose",
]

POSE9_NAMES = ["x", "y", "z", "r00", "r01", "r10", "r11", "r20", "r21"]
ROOT9_NAMES = [f"root.{name}" for name in POSE9_NAMES]
EE9_NAMES = [
    f"{prefix}.{name}"
    for prefix in ("left_hand_pose", "right_hand_pose", "left_foot_pose", "right_foot_pose")
    for name in POSE9_NAMES
]
ACTION59_NAMES = [f"hand.{name}" for name in HAND14_ORDER] + ROOT9_NAMES + EE9_NAMES
STATE52_NAMES = [f"hand.{name}" for name in HAND14_ORDER] + [f"body.{name}" for name in BODY29_ORDER] + ROOT9_NAMES


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=4, ensure_ascii=False) + "\n", encoding="utf-8")


def stack_column(df: pd.DataFrame, key: str) -> np.ndarray:
    return np.stack(df[key].to_numpy()).astype(np.float32)


def feature_stats(values: np.ndarray) -> dict[str, list[float]]:
    values = np.asarray(values, dtype=np.float32)
    return {
        "mean": values.mean(axis=0).astype(float).tolist(),
        "std": values.std(axis=0).astype(float).tolist(),
        "min": values.min(axis=0).astype(float).tolist(),
        "max": values.max(axis=0).astype(float).tolist(),
        "q01": np.quantile(values, 0.01, axis=0).astype(float).tolist(),
        "q99": np.quantile(values, 0.99, axis=0).astype(float).tolist(),
        "count": [int(values.shape[0])],
    }


def index_by_name(info: dict, feature_key: str) -> dict[str, int]:
    return {name: idx for idx, name in enumerate(info["features"][feature_key]["names"])}


def select_by_names(values: np.ndarray, name_to_idx: dict[str, int], names: list[str]) -> np.ndarray:
    return values[:, [name_to_idx[name] for name in names]].astype(np.float32)


def patch_meta(src: Path, out: Path, *, target_shift: int, prebuilt: bool) -> None:
    out_meta = out / "meta"
    out_meta.mkdir(parents=True, exist_ok=True)
    for item in (src / "meta").iterdir():
        if item.is_file():
            shutil.copy2(item, out_meta / item.name)

    info = read_json(out_meta / "info.json")
    features = info.setdefault("features", {})
    features["observation.full_state_rot6d"] = {
        "dtype": "float32",
        "shape": [52],
        "names": STATE52_NAMES,
    }
    features["observation.root_pose_rot6d"] = {
        "dtype": "float32",
        "shape": [9],
        "names": ROOT9_NAMES,
    }
    features["action.policy_action_rot6d59"] = {
        "dtype": "float32",
        "shape": [59],
        "names": ACTION59_NAMES,
    }
    write_json(out_meta / "info.json", info)

    modality_path = out_meta / "modality.json"
    modality = read_json(modality_path) if modality_path.exists() else {}
    modality.setdefault("state", {}).update(
        {
            "rot59_hand": {
                "start": 0,
                "end": 14,
                "original_key": "observation.full_state_rot6d",
            },
            "rot59_body": {
                "start": 14,
                "end": 43,
                "original_key": "observation.full_state_rot6d",
            },
            "rot59_root": {
                "start": 43,
                "end": 52,
                "original_key": "observation.full_state_rot6d",
            },
        }
    )
    modality.setdefault("action", {}).update(
        {
            "rot59_hand": {
                "start": 0,
                "end": 14,
                "original_key": "action.policy_action_rot6d59",
            },
            "rot59_root": {
                "start": 14,
                "end": 23,
                "original_key": "action.policy_action_rot6d59",
            },
            "rot59_left_hand_pose": {
                "start": 23,
                "end": 32,
                "original_key": "action.policy_action_rot6d59",
            },
            "rot59_right_hand_pose": {
                "start": 32,
                "end": 41,
                "original_key": "action.policy_action_rot6d59",
            },
            "rot59_left_foot_pose": {
                "start": 41,
                "end": 50,
                "original_key": "action.policy_action_rot6d59",
            },
            "rot59_right_foot_pose": {
                "start": 50,
                "end": 59,
                "original_key": "action.policy_action_rot6d59",
            },
        }
    )
    write_json(modality_path, modality)

    timing_note = (
        "Copied from prebuilt action.policy_action_rot6d59. Verified that action root9 "
        "at every non-terminal frame t equals observation root9 at frame t+1. The "
        "source terminal target is preserved unchanged."
        if prebuilt
        else (
            "action.policy_action_rot6d59 root/EE pose targets are taken from "
            f"observation pose streams at frame t+{target_shift}; final frames are "
            "padded with the last pose. Use --target-shift=0 to reproduce the old "
            "current-frame behavior."
        )
    )
    schema = {
        "action.policy_action_rot6d59": {
            "dim": 59,
            "layout": "hand14 + root9(xyz,rot6d) + 4 ee poses * 9(xyz,rot6d)",
            "names": ACTION59_NAMES,
        },
        "observation.full_state_rot6d": {
            "dim": 52,
            "layout": "hand14 + body29 + root9(xyz,rot6d)",
            "names": STATE52_NAMES,
        },
        "coordinate_notes": {
            "coordinates": "SIMPLE/MuJoCo z-up world coordinates.",
            "rotation_6d": "First two rotation-matrix columns flattened as [r00,r01,r10,r11,r20,r21], matching the source pose9 fields.",
            "action_target_timing": timing_note,
            "source_mode": "prebuilt_rot6d59_passthrough" if prebuilt else "legacy_pose_conversion",
            "kimodo_adapter": "For Kimodo, convert root/EE rot6d back to rotation matrices/RPY and then apply the existing MuJoCo->Kimodo coordinate conversion.",
            "ee_link_sources": POSE9_KEYS,
        },
    }
    write_json(out_meta / "rot6d59_schema.json", schema)


def ensure_video_link(src: Path, out: Path) -> None:
    link = out / "videos"
    if link.exists() or link.is_symlink():
        return
    target = os.path.relpath(src / "videos", out)
    link.symlink_to(target)


def shift_future(values: np.ndarray, shift: int) -> np.ndarray:
    """Return values[t + shift], padding the tail with the final row."""
    values = np.asarray(values)
    if shift < 0:
        raise ValueError(f"--target-shift must be >= 0, got {shift}")
    if shift == 0 or values.shape[0] == 0:
        return values.copy()
    if shift >= values.shape[0]:
        return np.repeat(values[-1:], values.shape[0], axis=0)
    return np.concatenate([values[shift:], np.repeat(values[-1:], shift, axis=0)], axis=0)


def validate_rot6d(values: np.ndarray, label: str, *, atol: float = 1e-5) -> None:
    """Validate [r00,r01,r10,r11,r20,r21] as two orthonormal columns."""
    values = np.asarray(values)
    c0 = values[:, [0, 2, 4]]
    c1 = values[:, [1, 3, 5]]
    norm0_error = np.max(np.abs(np.linalg.norm(c0, axis=1) - 1.0))
    norm1_error = np.max(np.abs(np.linalg.norm(c1, axis=1) - 1.0))
    dot_error = np.max(np.abs(np.sum(c0 * c1, axis=1)))
    error = max(float(norm0_error), float(norm1_error), float(dot_error))
    if not np.isfinite(error) or error > atol:
        raise ValueError(f"{label} is not a valid rot6d field: max error={error:.6g}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--use-policy-action-hand14",
        action="store_true",
        help="Use action.policy_action[:, :14] for hand supervision when available.",
    )
    parser.add_argument(
        "--target-shift",
        type=int,
        default=1,
        help=(
            "Use future observation pose streams at t+SHIFT for action root/EE targets. "
            "Default 1 learns next-frame targets; 0 reproduces the old current-frame labels."
        ),
    )
    args = parser.parse_args()

    src = args.src.resolve()
    out = (args.out or src.with_name(src.name + "-rot6d59")).resolve()
    if out == src:
        raise ValueError("--out must not be the same as --src")
    if out.exists():
        if not args.overwrite:
            raise FileExistsError(f"{out} exists; pass --overwrite to replace it")
        shutil.rmtree(out)

    info = read_json(src / "meta/info.json")
    features = info.get("features", {})
    prebuilt = (
        "observation.full_state_rot6d" in features
        and "action.policy_action_rot6d59" in features
    )
    if prebuilt:
        print("Source mode: prebuilt rot6d59 passthrough (no additional target shift)")
        obs_idx = action_idx = None
    else:
        print(f"Source mode: legacy pose conversion (target shift={args.target_shift})")
        obs_idx = index_by_name(info, "observation.state")
        action_idx = index_by_name(info, "action")

    patch_meta(src, out, target_shift=args.target_shift, prebuilt=prebuilt)
    ensure_video_link(src, out)

    all_policy59: list[np.ndarray] = []
    all_state52: list[np.ndarray] = []
    all_root9: list[np.ndarray] = []
    episodes_stats = []

    data_files = sorted((src / "data").glob("chunk-*/episode_*.parquet"))
    for src_path in tqdm(data_files, desc="Building rot6d59"):
        rel = src_path.relative_to(src)
        dst_path = out / rel
        df = pd.read_parquet(src_path)

        if prebuilt:
            state52 = stack_column(df, "observation.full_state_rot6d")
            policy59 = stack_column(df, "action.policy_action_rot6d59")
            if state52.shape[1] != 52 or policy59.shape[1] != 59:
                raise ValueError(
                    f"Expected prebuilt state/action dimensions 52/59, got "
                    f"{state52.shape}/{policy59.shape} in {src_path}"
                )
            root9_obs = state52[:, 43:52].copy()
            validate_rot6d(root9_obs[:, 3:9], f"{src_path}: observation root")
            for label, start in (
                ("action root", 17),
                ("left hand pose", 26),
                ("right hand pose", 35),
                ("left foot pose", 44),
                ("right foot pose", 53),
            ):
                validate_rot6d(policy59[:, start : start + 6], f"{src_path}: {label}")
            if len(state52) > 1 and not np.allclose(
                policy59[:-1, 14:23], state52[1:, 43:52], rtol=0.0, atol=1e-6
            ):
                error = np.max(np.abs(policy59[:-1, 14:23] - state52[1:, 43:52]))
                raise ValueError(
                    f"{src_path}: action root is not aligned to next-frame observation "
                    f"root (max error={error:.6g})"
                )
        else:
            obs_state = stack_column(df, "observation.state")
            action = stack_column(df, "action")
            root9_obs = stack_column(df, "observation.base_pose")
            ee36_obs = np.concatenate([stack_column(df, key) for key in POSE9_KEYS], axis=1).astype(np.float32)
            root9_action = shift_future(root9_obs, args.target_shift).astype(np.float32)
            ee36_action = shift_future(ee36_obs, args.target_shift).astype(np.float32)

            if args.use_policy_action_hand14:
                if "action.policy_action" not in df.columns:
                    raise KeyError(
                        "--use-policy-action-hand14 was set, but action.policy_action is missing "
                        f"from {src_path}"
                    )
                policy_action = stack_column(df, "action.policy_action")
                if policy_action.shape[1] < 14:
                    raise ValueError(
                        f"Expected action.policy_action to have at least 14 dims, got {policy_action.shape}"
                    )
                hand14_action = policy_action[:, :14].astype(np.float32)
            else:
                hand14_action = select_by_names(action, action_idx, HAND14_ORDER)
            hand14_state = select_by_names(obs_state, obs_idx, HAND14_ORDER)
            body29_state = select_by_names(obs_state, obs_idx, BODY29_ORDER)

            policy59 = np.concatenate([hand14_action, root9_action, ee36_action], axis=1).astype(np.float32)
            state52 = np.concatenate([hand14_state, body29_state, root9_obs], axis=1).astype(np.float32)

        df["observation.root_pose_rot6d"] = list(root9_obs)
        df["observation.full_state_rot6d"] = list(state52)
        df["action.policy_action_rot6d59"] = list(policy59)

        dst_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pandas(df, preserve_index=False), dst_path)

        ep_idx = int(src_path.stem.split("_")[-1])
        episodes_stats.append(
            {
                "episode_index": ep_idx,
                "stats": {
                    "observation.root_pose_rot6d": feature_stats(root9_obs),
                    "observation.full_state_rot6d": feature_stats(state52),
                    "action.policy_action_rot6d59": feature_stats(policy59),
                },
            }
        )
        all_root9.append(root9_obs)
        all_state52.append(state52)
        all_policy59.append(policy59)

    stats = {}
    src_stats = src / "meta/stats.json"
    if src_stats.exists():
        stats = read_json(src_stats)
    stats["observation.root_pose_rot6d"] = feature_stats(np.concatenate(all_root9, axis=0))
    stats["observation.full_state_rot6d"] = feature_stats(np.concatenate(all_state52, axis=0))
    stats["action.policy_action_rot6d59"] = feature_stats(np.concatenate(all_policy59, axis=0))
    write_json(out / "meta/stats.json", stats)

    existing_episode_stats = {}
    src_episode_stats = src / "meta/episodes_stats.jsonl"
    if src_episode_stats.exists():
        with src_episode_stats.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    existing_episode_stats[int(row["episode_index"])] = row
    for row in episodes_stats:
        existing = existing_episode_stats.setdefault(row["episode_index"], {"episode_index": row["episode_index"], "stats": {}})
        existing.setdefault("stats", {}).update(row["stats"])
    with (out / "meta/episodes_stats.jsonl").open("w", encoding="utf-8") as f:
        for ep_idx in sorted(existing_episode_stats):
            f.write(json.dumps(existing_episode_stats[ep_idx], separators=(",", ":")) + "\n")

    print(f"Saved rot6d59 dataset to: {out}")
    print("Added columns: observation.root_pose_rot6d, observation.full_state_rot6d, action.policy_action_rot6d59")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
