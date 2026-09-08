#!/usr/bin/env python3
"""Replay one LeRobot episode's measured state or WBC target on the G1 MuJoCo model."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

from replay_realtask_action_pair_mujoco import (
    DEFAULT_MODEL,
    G1ReplayRenderer,
    VideoWriter,
    labeled_frame,
    load_episode,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--label")
    parser.add_argument(
        "--joint-source",
        choices=("observation.state", "action.wbc"),
        default="observation.state",
        help="Measured collection pose by default; action.wbc is the WBC command target.",
    )
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--save-snapshots",
        action="store_true",
        help="Also save first, middle, and last frames as PNG files.",
    )
    args = parser.parse_args()

    dataset = args.dataset.resolve()
    label = args.label or dataset.name
    motion = load_episode(dataset, args.episode, joint_key=args.joint_source)
    action = np.asarray(motion["action"])
    root_pos = np.asarray(motion["root_pos"])
    root_quat = np.asarray(motion["root_quat_xyzw"])
    timestamp = np.asarray(motion["timestamp"])
    joint_names = list(motion["joint_names"])
    actual_joint_source = str(motion["joint_source"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", label)
    safe_source = actual_joint_source.replace(".", "_")
    stem = f"episode_{args.episode:06d}_{safe_label}_{safe_source}_mujoco"
    video_path = args.output_dir / f"{stem}.mp4"
    writer = VideoWriter(video_path, 640, 556, args.fps)
    replay = G1ReplayRenderer(args.model.resolve())
    mapping = []
    seen_qpos_addresses = set()
    for dataset_index, name in enumerate(joint_names):
        joint_id = mujoco.mj_name2id(
            replay.model, mujoco.mjtObj.mjOBJ_JOINT, name
        )
        if joint_id < 0:
            raise KeyError(f"MuJoCo model is missing dataset joint: {name}")
        qpos_address = int(replay.model.jnt_qposadr[joint_id])
        if qpos_address in seen_qpos_addresses:
            raise ValueError(f"Duplicate MuJoCo qpos address {qpos_address} for {name}")
        seen_qpos_addresses.add(qpos_address)
        mapping.append(
            {
                "dataset_index": dataset_index,
                "dataset_joint_name": name,
                "mujoco_joint_id": int(joint_id),
                "mujoco_qpos_address": qpos_address,
            }
        )
    if len(mapping) != 43:
        raise ValueError(f"Expected 43 joints, got {len(mapping)}")
    mapping_path = args.output_dir / f"{stem}_joint_mapping.json"
    mapping_path.write_text(
        json.dumps(
            {
                "dataset": str(dataset),
                "episode": args.episode,
                "joint_source": actual_joint_source,
                "model": str(args.model.resolve()),
                "mapping": mapping,
            },
            indent=2,
        )
        + "\n"
    )
    snapshots = {0, len(action) // 2, len(action) - 1} if args.save_snapshots else set()
    try:
        for frame in range(len(action)):
            replay.set_pose(
                joint_names,
                action[frame],
                root_pos[frame],
                root_quat[frame],
            )
            lookat = root_pos[frame].copy()
            lookat[2] += 0.15
            rendered = replay.render(lookat)
            panel = labeled_frame(
                rendered,
                f"{label} [{actual_joint_source}]",
                frame,
                len(action),
                float(timestamp[frame]),
                root_pos[frame],
            )
            writer.write(panel)
            if frame in snapshots:
                Image.fromarray(panel).save(
                    args.output_dir / f"{stem}_frame_{frame:03d}.png"
                )
    finally:
        replay.close()
        writer.close()

    print(f"Saved: {video_path}")
    print(f"Saved mapping: {mapping_path}")
    print(f"Frames={len(action)}, fps={args.fps}, duration={len(action) / args.fps:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
