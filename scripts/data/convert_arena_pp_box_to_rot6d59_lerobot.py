#!/usr/bin/env python3
"""Convert external-video HumanoidArena raw NPZ recordings to rot6d59 data.

This produces the same state/action schema as ``arena_football_rot6d59``:

    observation.images.ego_view
    observation.full_state_rot6d: hand14 + body29 + root9 = 52D
    action.policy_action_rot6d59:
        hand14 + root9 + left/right hand pose9 + left/right foot pose9 = 59D

P&P Box and OpenDoor raw episodes store RGB as external MP4 files referenced
from the NPZ. Videos may be re-encoded, copied, or linked into the LeRobot
video tree.
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

import convert_arena_football_to_rot6d52_lerobot as arena


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SRC = Path("/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/dataset/HumanoidArena_pp_box")
DEFAULT_OUT = REPO_ROOT / "data/output/arena_pp_box_rot6d59"
DEFAULT_MJCF = arena.DEFAULT_MJCF
DEFAULT_BRANCHES = ("sonic/yb", "twist2/yb")
DEFAULT_TASK = "Pick up the box and place it on the shelf."
DEFAULT_OPEN_DOOR_SRC = Path(
    "/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/dataset/"
    "HumanoidArena_open_door/HSI_open_door"
)
DEFAULT_OPEN_DOOR_OUT = REPO_ROOT / "data/output/arena_open_door_sonic_rot6d59_v1"
DEFAULT_OPEN_DOOR_BRANCHES = ("sonic/zz",)
DEFAULT_OPEN_DOOR_TASK = "Open the door."

# SONIC records ``robot_qpos_before_decimation`` in the interleaved Isaac Lab
# order used by its controller.  TWIST2 records the same field in the grouped
# MuJoCo order used by rot6d59, Kimodo, and TextOp.  A 29D shape check cannot
# detect this semantic difference, so make the source order explicit here.
SONIC_ISAACLAB_BODY29_ORDER = (
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
)
SONIC_TO_BODY29 = np.asarray(
    [SONIC_ISAACLAB_BODY29_ORDER.index(name) for name in arena.BODY29_ORDER],
    dtype=np.int64,
)

# SONIC records hand targets in its provider order (thumb, middle, index).
# rot6d59 uses that order for the left hand and (thumb, index, middle) for the
# right hand. TWIST2 recordings use (thumb, index, middle) on both sides.
SONIC_LEFT_TO_ROT59 = np.arange(7, dtype=np.int64)
SONIC_RIGHT_TO_ROT59 = np.asarray([0, 1, 2, 5, 6, 3, 4], dtype=np.int64)


def scalar_text(data: np.lib.npyio.NpzFile, key: str, default: str = "") -> str:
    if key not in data.files:
        return default
    value = data[key]
    if value.shape == ():
        return str(value.item())
    return str(value)


def scalar_int(data: np.lib.npyio.NpzFile, key: str, default: int = 0) -> int:
    if key not in data.files:
        return default
    return int(data[key].item() if data[key].shape == () else data[key])


def hand_key(data: np.lib.npyio.NpzFile, side: str) -> str:
    candidates = (
        f"human_hand_{side}",
        f"human_{side}_hand",
        f"hand_action_{side}",
    )
    for key in candidates:
        if key in data.files:
            return key
    raise KeyError(f"Could not find {side} hand stream; tried {candidates}")


def reorder_source_qpos_to_body29(
    qpos: np.ndarray,
    *,
    source_family: str,
) -> np.ndarray:
    """Return body joints in the grouped rot6d59/TWIST2 MuJoCo order."""
    qpos = np.asarray(qpos, dtype=np.float32)
    if qpos.ndim != 2 or qpos.shape[1] != len(arena.BODY29_ORDER):
        raise ValueError(f"Expected body qpos [T, 29], got {qpos.shape}")

    normalized_family = source_family.strip().lower()
    if normalized_family == "sonic":
        return qpos[:, SONIC_TO_BODY29].copy()
    if normalized_family == "twist2":
        return qpos.copy()
    raise ValueError(
        f"Unsupported source family {source_family!r}; expected 'sonic' or 'twist2'"
    )


def hand14_from_source(
    left_hand: np.ndarray,
    right_hand: np.ndarray,
    *,
    source_family: str,
) -> np.ndarray:
    """Return hand joints in the asymmetric rot6d59 hand order."""
    normalized_family = source_family.strip().lower()
    if normalized_family == "sonic":
        return np.concatenate(
            [
                left_hand[:, SONIC_LEFT_TO_ROT59],
                right_hand[:, SONIC_RIGHT_TO_ROT59],
            ],
            axis=1,
        ).astype(np.float32)
    if normalized_family == "twist2":
        return arena.hand14_from_arena(left_hand, right_hand)
    raise ValueError(
        f"Unsupported source family {source_family!r}; expected 'sonic' or 'twist2'"
    )


def load_episode_arrays(
    data: np.lib.npyio.NpzFile,
    length: int,
    fk: arena.G1MjcfFK,
    *,
    source_family: str,
    target_shift: int,
    hand_target_shift: int,
) -> dict[str, np.ndarray]:
    source_qpos = np.asarray(
        data["robot_qpos_before_decimation"][:length], dtype=np.float32
    )
    qpos = reorder_source_qpos_to_body29(
        source_qpos,
        source_family=source_family,
    )
    left_hand = np.asarray(data[hand_key(data, "left")][:length], dtype=np.float32)
    right_hand = np.asarray(data[hand_key(data, "right")][:length], dtype=np.float32)
    root_pos = np.asarray(data["robot_root_position"][:length], dtype=np.float32)
    root_quat = np.asarray(data["robot_root_orientation"][:length], dtype=np.float32)

    if qpos.shape != (length, 29):
        raise ValueError(f"Expected robot_qpos_before_decimation {(length, 29)}, got {qpos.shape}")
    if left_hand.shape != (length, 7) or right_hand.shape != (length, 7):
        raise ValueError(f"Expected hand shapes {(length, 7)}, got {left_hand.shape}, {right_hand.shape}")
    if root_pos.shape != (length, 3) or root_quat.shape != (length, 4):
        raise ValueError(f"Expected root shapes {(length, 3)}/{(length, 4)}, got {root_pos.shape}/{root_quat.shape}")

    hand14 = hand14_from_source(
        left_hand,
        right_hand,
        source_family=source_family,
    )
    root9 = np.concatenate([root_pos, arena.quat_wxyz_to_rot6d(root_quat)], axis=1).astype(np.float32)
    state52 = np.concatenate([hand14, qpos, root9], axis=1).astype(np.float32)
    if state52.shape != (length, 52):
        raise ValueError(f"Built invalid {arena.STATE_KEY} shape {state52.shape}")

    ee36 = np.stack(
        [fk.poses(qpos[i], root_pos[i], root_quat[i]) for i in range(length)],
        axis=0,
    ).astype(np.float32)
    hand_action = arena.shift_future(hand14, hand_target_shift).astype(np.float32)
    root_action = arena.shift_future(root9, target_shift).astype(np.float32)
    ee_action = arena.shift_future(ee36, target_shift).astype(np.float32)
    ref_body29 = arena.shift_future(qpos, target_shift).astype(np.float32)
    action59 = np.concatenate([hand_action, root_action, ee_action], axis=1).astype(np.float32)

    arena.validate_rot6d(root9[:, 3:9], arena.STATE_KEY)
    arena.validate_rot6d(action59[:, 17:23], f"{arena.ACTION_KEY}.root")
    arena.validate_rot6d(action59[:, 26:32], f"{arena.ACTION_KEY}.left_hand_pose")
    arena.validate_rot6d(action59[:, 35:41], f"{arena.ACTION_KEY}.right_hand_pose")
    arena.validate_rot6d(action59[:, 44:50], f"{arena.ACTION_KEY}.left_foot_pose")
    arena.validate_rot6d(action59[:, 53:59], f"{arena.ACTION_KEY}.right_foot_pose")

    return {
        arena.STATE_KEY: state52,
        arena.ROOT_KEY: root9,
        arena.ACTION_KEY: action59,
        arena.REF_BODY_KEY: ref_body29,
    }


def selected_npz_files(src: Path, branches: tuple[str, ...]) -> list[Path]:
    files: list[Path] = []
    for branch in branches:
        branch_dir = src / branch
        if not branch_dir.exists():
            raise FileNotFoundError(f"Branch directory does not exist: {branch_dir}")
        files.extend(sorted(branch_dir.glob("*.npz")))
    if not files:
        raise FileNotFoundError(f"No .npz files found under selected branches: {branches}")
    return files


def video_path_for_episode(npz_path: Path, data: np.lib.npyio.NpzFile) -> Path:
    rel = scalar_text(data, "vision_rgb_video_path", f"videos/{npz_path.stem}_front_rgb.mp4")
    path = (npz_path.parent / rel).resolve()
    return path


def copy_video(src_video: Path, out_video: Path) -> None:
    out_video.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_video, out_video)


def link_video(src_video: Path, out_video: Path) -> None:
    out_video.parent.mkdir(parents=True, exist_ok=True)
    out_video.symlink_to(os.path.relpath(src_video.resolve(), out_video.parent))


def encode_h264_video(src_video: Path, out_video: Path) -> None:
    out_video.parent.mkdir(parents=True, exist_ok=True)
    tmp_video = out_video.with_name(f"{out_video.stem}.tmp{out_video.suffix}")
    if tmp_video.exists():
        tmp_video.unlink()
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(src_video),
        "-an",
        "-c:v",
        "libx264",
        "-profile:v",
        "baseline",
        "-level",
        "3.1",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(tmp_video),
    ]
    subprocess.run(cmd, check=True)
    tmp_video.replace(out_video)


def discover_episodes(
    src: Path,
    branches: tuple[str, ...],
    *,
    max_episodes: int | None,
    require_videos: bool,
) -> tuple[list[Path], list[dict[str, Any]]]:
    selected: list[Path] = []
    skipped: list[dict[str, Any]] = []
    for npz_path in selected_npz_files(src, branches):
        with np.load(npz_path, allow_pickle=True) as data:
            video_path = video_path_for_episode(npz_path, data)
            if require_videos and not video_path.exists():
                skipped.append(
                    {
                        "source_npz": str(npz_path),
                        "reason": "missing_front_video",
                        "expected_video": str(video_path),
                    }
                )
                continue
        selected.append(npz_path)
        if max_episodes is not None and len(selected) >= max_episodes:
            break
    return selected, skipped


def build_info(
    *,
    total_episodes: int,
    total_frames: int,
    total_videos: int,
    task: str,
    source_root: Path,
    branches: tuple[str, ...],
    mjcf_path: Path,
    target_shift: int,
    hand_target_shift: int,
    skipped: list[dict[str, Any]],
    video_codec: str,
    kind: str,
) -> dict[str, Any]:
    source_families = tuple(
        sorted({branch.split("/", maxsplit=1)[0].lower() for branch in branches})
    )
    info = arena.build_info(
        total_episodes=total_episodes,
        total_frames=total_frames,
        task=task,
        source_root=source_root,
        mjcf_path=mjcf_path,
        target_shift=target_shift,
        hand_target_shift=hand_target_shift,
        source_families=source_families,
    )
    open_door_source = "humanoid_arena_open_door_to_rot6d59"
    open_door_schema = "arena_open_door_rot6d59_v1"
    if source_families == ("sonic",):
        open_door_source = "humanoid_arena_sonic_open_door_to_rot6d59"
        open_door_schema = "arena_open_door_sonic_rot6d59_v1"
    elif source_families == ("twist2",):
        open_door_source = "humanoid_arena_twist2_open_door_to_rot6d59"
        open_door_schema = "arena_open_door_twist2_rot6d59_v1"

    kind_settings = {
        "pp_box": {
            "robot_type": "g1_dex3_arena_pp_box",
            "source": "humanoid_arena_npz_pp_box_to_rot6d59",
            "schema_version": "arena_pp_box_rot6d59_v3",
        },
        "open_door": {
            "robot_type": "g1_dex3_arena_open_door",
            "source": open_door_source,
            "schema_version": open_door_schema,
        },
    }[kind]
    info["robot_type"] = kind_settings["robot_type"]
    info["total_videos"] = total_videos
    info["features"][arena.VIDEO_KEY]["info"]["video.codec"] = video_codec
    info["script_config"].update(
        {
            "source": kind_settings["source"],
            "schema_version": kind_settings["schema_version"],
            "kind": kind,
            "source_branches": list(branches),
            "source_body29_orders": {
                "sonic": "SONIC IsaacLab interleaved -> reordered to rot6d59 BODY29_ORDER",
                "twist2": "already rot6d59 BODY29_ORDER",
            },
            "source_hand_orders": {
                "sonic": (
                    "provider thumb,middle,index -> rot6d59 left thumb,middle,index; "
                    "right thumb,index,middle"
                ),
                "twist2": (
                    "thumb,index,middle -> rot6d59 left thumb,middle,index; "
                    "right thumb,index,middle"
                ),
            },
            "video_source": "external front_rgb mp4 from vision_rgb_video_path",
            "skipped_episodes": len(skipped),
            "body29_source": (
                "robot_qpos_before_decimation; sonic reordered from SONIC IsaacLab "
                "interleaved order to TWIST2/MuJoCo BODY29_ORDER"
            ),
            "hand14_source": (
                "human hand targets; source-family-aware reorder into asymmetric "
                "rot6d59 HAND14_ORDER"
            ),
            "task": task,
        }
    )
    return info


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("pp_box", "open_door"), default="pp_box")
    parser.add_argument("--src", type=Path, help="HumanoidArena dataset root")
    parser.add_argument("--out", type=Path, help="Output LeRobot dataset directory")
    parser.add_argument("--mjcf", type=Path, default=DEFAULT_MJCF, help="G1 MJCF used for FK")
    parser.add_argument(
        "--branches",
        default=None,
        help="Comma-separated source branches relative to --src",
    )
    parser.add_argument("--max-episodes", type=int, default=None, help="Convert only the first N selected episodes")
    parser.add_argument("--max-frames", type=int, default=None, help="Cap frames per episode, useful for smoke tests")
    parser.add_argument("--target-shift", type=int, default=1, help="Future shift for root/EE/body targets")
    parser.add_argument("--hand-target-shift", type=int, default=0, help="Future shift for hand target commands")
    parser.add_argument("--task")
    parser.add_argument("--skip-videos", action="store_true", help="Write parquet/meta only")
    parser.add_argument("--copy-videos", action="store_true", help="Copy raw MP4s instead of re-encoding to h264")
    parser.add_argument("--link-videos", action="store_true", help="Use relative symlinks to raw MP4s")
    parser.add_argument("--include-missing-video", action="store_true", help="Do not skip episodes missing front video")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    defaults = {
        "pp_box": {
            "src": DEFAULT_SRC,
            "out": DEFAULT_OUT,
            "branches": DEFAULT_BRANCHES,
            "task": DEFAULT_TASK,
        },
        "open_door": {
            "src": DEFAULT_OPEN_DOOR_SRC,
            "out": DEFAULT_OPEN_DOOR_OUT,
            "branches": DEFAULT_OPEN_DOOR_BRANCHES,
            "task": DEFAULT_OPEN_DOOR_TASK,
        },
    }[args.kind]
    src = (args.src or defaults["src"]).resolve()
    out = (args.out or defaults["out"]).resolve()
    mjcf = args.mjcf.resolve()
    branch_value = args.branches or ",".join(defaults["branches"])
    branches = tuple(branch.strip() for branch in branch_value.split(",") if branch.strip())
    task = args.task or defaults["task"]
    if not branches:
        raise ValueError("--branches must contain at least one branch")
    if args.copy_videos and args.link_videos:
        raise ValueError("--copy-videos and --link-videos are mutually exclusive")
    if out.exists():
        if not args.overwrite:
            raise FileExistsError(f"{out} exists; pass --overwrite to replace it")
        shutil.rmtree(out)
    if not mjcf.exists():
        raise FileNotFoundError(f"G1 MJCF does not exist: {mjcf}")

    npz_files, skipped = discover_episodes(
        src,
        branches,
        max_episodes=args.max_episodes,
        require_videos=not args.skip_videos and not args.include_missing_video,
    )
    if not npz_files:
        raise FileNotFoundError("No episodes selected after filtering")

    fk = arena.G1MjcfFK(mjcf)
    all_stats_arrays: dict[str, list[np.ndarray]] = {
        arena.STATE_KEY: [],
        arena.ROOT_KEY: [],
        arena.ACTION_KEY: [],
        arena.REF_BODY_KEY: [],
        "timestamp": [],
        "frame_index": [],
        "episode_index": [],
        "index": [],
        "task_index": [],
    }
    episodes_rows: list[dict[str, Any]] = []
    episodes_stats_rows: list[dict[str, Any]] = []
    total_frames = 0

    for episode_index, npz_path in enumerate(npz_files):
        relative_parts = npz_path.relative_to(src).parts
        if not relative_parts:
            raise ValueError(f"Could not resolve source family for {npz_path}")
        source_family = relative_parts[0].lower()
        with np.load(npz_path, allow_pickle=True) as data:
            length = scalar_int(data, "num_frames")
            if args.max_frames is not None:
                length = min(length, int(args.max_frames))
            if length <= 0:
                raise ValueError(f"{npz_path} has invalid length {length}")

            arrays = load_episode_arrays(
                data,
                length,
                fk,
                source_family=source_family,
                target_shift=args.target_shift,
                hand_target_shift=args.hand_target_shift,
            )

            chunk = episode_index // arena.CHUNKS_SIZE
            parquet_path = out / "data" / f"chunk-{chunk:03d}" / f"episode_{episode_index:06d}.parquet"
            parquet_path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(arena.build_table(arrays, episode_index, total_frames), parquet_path)

            video_path = video_path_for_episode(npz_path, data)
            if not args.skip_videos:
                if not video_path.exists():
                    raise FileNotFoundError(f"Missing front video for {npz_path}: {video_path}")
                out_video = out / "videos" / f"chunk-{chunk:03d}" / arena.VIDEO_KEY / f"episode_{episode_index:06d}.mp4"
                if args.link_videos:
                    link_video(video_path, out_video)
                elif args.copy_videos:
                    copy_video(video_path, out_video)
                else:
                    encode_h264_video(video_path, out_video)

            frame_indices = np.arange(length, dtype=np.int64)
            all_stats_arrays[arena.STATE_KEY].append(arrays[arena.STATE_KEY])
            all_stats_arrays[arena.ROOT_KEY].append(arrays[arena.ROOT_KEY])
            all_stats_arrays[arena.ACTION_KEY].append(arrays[arena.ACTION_KEY])
            all_stats_arrays[arena.REF_BODY_KEY].append(arrays[arena.REF_BODY_KEY])
            all_stats_arrays["timestamp"].append(frame_indices.astype(np.float32) / float(arena.FPS))
            all_stats_arrays["frame_index"].append(frame_indices)
            all_stats_arrays["episode_index"].append(np.full(length, episode_index, dtype=np.int64))
            all_stats_arrays["index"].append(total_frames + frame_indices)
            all_stats_arrays["task_index"].append(np.zeros(length, dtype=np.int64))
            episodes_rows.append(
                {
                    "episode_index": episode_index,
                    "tasks": [task],
                    "length": length,
                    "source_npz": str(npz_path),
                    "source_family": source_family,
                    "source_schema_version": scalar_text(data, "schema_version"),
                    "source_video": str(video_path),
                }
            )
            episodes_stats_rows.append(arena.episode_stats(arrays, episode_index, total_frames))
            total_frames += length
            print(f"[{episode_index + 1}/{len(npz_files)}] {npz_path.name}: frames={length}")

    meta = out / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    arena.write_jsonl(meta / "tasks.jsonl", [{"task_index": 0, "task": task}])
    arena.write_jsonl(meta / "episodes.jsonl", episodes_rows)
    arena.write_jsonl(meta / "episodes_stats.jsonl", episodes_stats_rows)

    stats = {
        key: arena.feature_stats(np.concatenate(values, axis=0))
        for key, values in all_stats_arrays.items()
    }
    arena.write_json(meta / "stats.json", stats)
    arena.write_json(meta / "relative_stats.json", {"__fingerprints__": {}})
    arena.write_json(meta / "modality.json", arena.build_modality())
    arena.write_json(meta / "rot6d59_schema.json", arena.build_schema(args.target_shift, args.hand_target_shift))
    arena.write_json(
        meta / "info.json",
        build_info(
            total_episodes=len(npz_files),
            total_frames=total_frames,
            total_videos=0 if args.skip_videos else len(npz_files),
            task=task,
            source_root=src,
            branches=branches,
            mjcf_path=mjcf,
            target_shift=args.target_shift,
            hand_target_shift=args.hand_target_shift,
            skipped=skipped,
            video_codec="mpeg4" if (args.copy_videos or args.link_videos) else "h264",
            kind=args.kind,
        ),
    )
    if skipped:
        arena.write_json(meta / "conversion_report.json", {"skipped": skipped})

    print(f"source={src}")
    print(f"branches={','.join(branches)}")
    print(f"out={out}")
    print(f"episodes={len(npz_files)} frames={total_frames} fps={arena.FPS}")
    print(f"skipped_missing_video={len(skipped)}")
    print(f"{arena.STATE_KEY}=52D {arena.ACTION_KEY}=59D")
    if args.skip_videos:
        print("videos=skipped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
