#!/usr/bin/env python3
"""Extract a GR00T N1.7 -> Sonic LeRobot-v2 dataset from a multi-chain dataset.

The exported policy interface is:

    observation.sonic_state: body29 + hand14 + projected_gravity3 = 46D
    action.sonic:            motion_token64 + target_hand14       = 78D

The source values are copied without temporal shifting or recomputation.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SRC = REPO_ROOT / "data/output/fullstate_20260615_task1"
DEFAULT_OUT = REPO_ROOT / "data/output/fullstate_20260615_task1_gr00t"

STATE_KEY = "observation.sonic_state"
ACTION_KEY = "action.sonic"
VIDEO_KEY = "observation.images.ego_view"
LOWDIM_KEYS = [
    STATE_KEY,
    ACTION_KEY,
    "timestamp",
    "frame_index",
    "episode_index",
    "index",
    "task_index",
]

# Source observation.sonic_state already follows the official RobotModel layout:
#   left_leg6, right_leg6, waist3, left_arm7, left_hand7,
#   right_arm7, right_hand7, gravity3
# Its state hand blocks follow RobotModel joint order:
#   index0, index1, middle0, middle1, thumb0, thumb1, thumb2
STATE_REORDER = list(range(46))

# Official GEAR-SONIC teleop hand-command order (IK solver output):
#   thumb0, thumb1, thumb2, index0, index1, middle0, middle1
# Source action.sonic hand order follows Psi hand14:
#   left:  thumb0, thumb1, thumb2, middle0, middle1, index0, index1
#   right: thumb0, thumb1, thumb2, index0, index1, middle0, middle1
LEFT_ACTION_HAND_REORDER = [0, 1, 2, 5, 6, 3, 4]
RIGHT_ACTION_HAND_REORDER = [0, 1, 2, 3, 4, 5, 6]

OFFICIAL_STATE_NAMES = [
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
    "left_hand_index_0_joint",
    "left_hand_index_1_joint",
    "left_hand_middle_0_joint",
    "left_hand_middle_1_joint",
    "left_hand_thumb_0_joint",
    "left_hand_thumb_1_joint",
    "left_hand_thumb_2_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
    "right_hand_index_0_joint",
    "right_hand_index_1_joint",
    "right_hand_middle_0_joint",
    "right_hand_middle_1_joint",
    "right_hand_thumb_0_joint",
    "right_hand_thumb_1_joint",
    "right_hand_thumb_2_joint",
    "gravity_x",
    "gravity_y",
    "gravity_z",
]

OFFICIAL_HAND_NAMES = [
    "thumb_0_joint",
    "thumb_1_joint",
    "thumb_2_joint",
    "index_0_joint",
    "index_1_joint",
    "middle_0_joint",
    "middle_1_joint",
]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=4, ensure_ascii=False) + "\n", encoding="utf-8")


def link_videos(src: Path, out: Path) -> None:
    target = os.path.relpath(src / "videos", out)
    (out / "videos").symlink_to(target)


def validate_episode(table: pa.Table, path: Path) -> None:
    state = np.asarray(table[STATE_KEY].to_pylist(), dtype=np.float32)
    action = np.asarray(table[ACTION_KEY].to_pylist(), dtype=np.float32)
    if state.ndim != 2 or state.shape[1] != 46:
        raise ValueError(f"{path}: expected {STATE_KEY} shape (T,46), got {state.shape}")
    if action.ndim != 2 or action.shape[1] != 78:
        raise ValueError(f"{path}: expected {ACTION_KEY} shape (T,78), got {action.shape}")
    if not np.isfinite(state).all() or not np.isfinite(action).all():
        raise ValueError(f"{path}: state/action contains NaN or Inf")

    gravity_norm = np.linalg.norm(state[:, 43:46], axis=1)
    max_gravity_error = float(np.max(np.abs(gravity_norm - 1.0)))
    if max_gravity_error > 1e-4:
        raise ValueError(
            f"{path}: projected gravity is not unit length "
            f"(max norm error={max_gravity_error:.6g})"
        )


def build_info(src_info: dict) -> dict:
    required = [VIDEO_KEY, *LOWDIM_KEYS]
    missing = [key for key in required if key not in src_info.get("features", {})]
    if missing:
        raise KeyError(f"Source info.json is missing required features: {missing}")

    info = dict(src_info)
    info["features"] = {key: src_info["features"][key] for key in required}
    info["features"][STATE_KEY] = dict(info["features"][STATE_KEY])
    info["features"][STATE_KEY]["dtype"] = "float32"
    info["features"][STATE_KEY]["names"] = OFFICIAL_STATE_NAMES
    info["features"][ACTION_KEY] = dict(info["features"][ACTION_KEY])
    info["features"][ACTION_KEY]["dtype"] = "float32"
    info["features"][ACTION_KEY]["names"] = (
        [f"motion_token.{index}" for index in range(64)]
        + [f"left_hand_joints.{name}" for name in OFFICIAL_HAND_NAMES]
        + [f"right_hand_joints.{name}" for name in OFFICIAL_HAND_NAMES]
    )
    info["script_config"] = {
        "source": "multichain_gr00t_sonic_passthrough",
        "source_dataset": str(DEFAULT_SRC),
        "state": f"{STATE_KEY}(46=body29+hand14+projected_gravity3)",
        "action": f"{ACTION_KEY}(78=motion_token64+target_hand14)",
        "temporal_transform": "none",
    }
    return info


def build_modality() -> dict:
    return {
        "state": {
            "left_leg": {"start": 0, "end": 6, "original_key": STATE_KEY},
            "right_leg": {"start": 6, "end": 12, "original_key": STATE_KEY},
            "waist": {"start": 12, "end": 15, "original_key": STATE_KEY},
            "left_arm": {"start": 15, "end": 22, "original_key": STATE_KEY},
            "left_hand": {"start": 22, "end": 29, "original_key": STATE_KEY},
            "right_arm": {"start": 29, "end": 36, "original_key": STATE_KEY},
            "right_hand": {"start": 36, "end": 43, "original_key": STATE_KEY},
            "projected_gravity": {"start": 43, "end": 46, "original_key": STATE_KEY},
        },
        "action": {
            "motion_token": {"start": 0, "end": 64, "original_key": ACTION_KEY},
            "left_hand_joints": {"start": 64, "end": 71, "original_key": ACTION_KEY},
            "right_hand_joints": {"start": 71, "end": 78, "original_key": ACTION_KEY},
        },
        "video": {
            "ego_view": {"original_key": VIDEO_KEY},
        },
        "annotation": {
            "human.task_description": {"original_key": "task_index"},
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    src = args.src.resolve()
    out = args.out.resolve()
    if src == out:
        raise ValueError("--src and --out must be different")
    if out.exists():
        if not args.overwrite:
            raise FileExistsError(f"{out} already exists; pass --overwrite to replace it")
        shutil.rmtree(out)

    out_meta = out / "meta"
    out_meta.mkdir(parents=True)
    for name in ("episodes.jsonl", "tasks.jsonl"):
        shutil.copy2(src / "meta" / name, out_meta / name)
    src_episode_stats = src / "meta" / "episodes_stats.jsonl"
    if src_episode_stats.exists():
        shutil.copy2(src_episode_stats, out_meta / "episodes_stats.jsonl")

    info = build_info(read_json(src / "meta" / "info.json"))
    info["script_config"]["source_dataset"] = str(src)
    write_json(out_meta / "info.json", info)
    write_json(out_meta / "modality.json", build_modality())
    link_videos(src, out)

    files = sorted((src / "data").glob("chunk-*/episode_*.parquet"))
    if len(files) != int(info["total_episodes"]):
        raise ValueError(
            f"Found {len(files)} parquet files, but info.json declares "
            f"{info['total_episodes']} episodes"
        )

    total_rows = 0
    selected = LOWDIM_KEYS
    for src_path in tqdm(files, desc="Extracting GR00T -> Sonic"):
        table = pq.read_table(src_path, columns=selected)
        validate_episode(table, src_path)
        source_state = np.asarray(table[STATE_KEY].to_pylist(), dtype=np.float32)
        source_action = np.asarray(table[ACTION_KEY].to_pylist(), dtype=np.float32)
        official_state = source_state[:, STATE_REORDER]
        source_left_hand = source_action[:, 64:71]
        source_right_hand = source_action[:, 71:78]
        official_action = np.concatenate(
            [
                source_action[:, :64],
                source_left_hand[:, LEFT_ACTION_HAND_REORDER],
                source_right_hand[:, RIGHT_ACTION_HAND_REORDER],
            ],
            axis=1,
        ).astype(np.float32)

        # Store policy arrays as float32, as required by GR00T LeRobot.
        fields = []
        arrays = []
        for name in selected:
            column = table[name]
            if name in (STATE_KEY, ACTION_KEY):
                width = 46 if name == STATE_KEY else 78
                values = official_state if name == STATE_KEY else official_action
                column = pa.array(values.tolist(), type=pa.list_(pa.float32(), width))
            fields.append(pa.field(name, column.type))
            arrays.append(column)
        output_table = pa.Table.from_arrays(arrays, schema=pa.schema(fields))

        dst = out / src_path.relative_to(src)
        dst.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(output_table, dst)
        total_rows += output_table.num_rows

    if total_rows != int(info["total_frames"]):
        raise ValueError(
            f"Exported {total_rows} rows, but info.json declares {info['total_frames']} frames"
        )

    print(f"Saved GR00T -> Sonic dataset to: {out}")
    print(f"Episodes: {len(files)}, frames: {total_rows}, state/action: 46D/78D")
    print("Next: run gr00t/data/stats.py to generate meta/stats.json and relative_stats.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
