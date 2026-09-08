#!/usr/bin/env python3
"""Convert HumanoidArena demonstrations to native GR00T/SONIC data.

The exported LeRobot-v2 policy interface is the built-in
``UNITREE_G1_SONIC`` interface from GR00T N1.7:

    observation.sonic_state: body29 + hand14 + projected_gravity3 = 46D
    action.sonic:            motion_token64 + target_hand14       = 78D

For TWIST2 recordings, the 64D motion token is produced by the released SONIC
encoder from the realized future body trajectory at t+1, t+6, ..., t+46
(50 Hz). Native SONIC recordings may instead reuse their recorded
``encoder_latent`` values without re-encoding or dropping the final frame.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any

import numpy as np
import onnxruntime as ort
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

import convert_arena_football_to_rot6d52_lerobot as arena
import convert_arena_pp_box_to_rot6d59_lerobot as pp_box


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENCODER = Path(
    "/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/SONIC_my/"
    "gear_sonic_deploy/policy/release/model_encoder.onnx"
)
DEFAULT_FOOTBALL_SRC = Path(
    "/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/dataset/"
    "HumanoidArena_football/HOI_football_v2/TWIST2"
)
DEFAULT_PP_BOX_SRC = Path(
    "/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/dataset/"
    "HumanoidArena_pp_box/twist2/yb"
)
DEFAULT_FOOTBALL_OUT = REPO_ROOT / "data/output/arena_football_twist2_sonic"
DEFAULT_PP_BOX_OUT = REPO_ROOT / "data/output/arena_pp_box_twist2_sonic"
DEFAULT_FOOTBALL_VIDEO_DATASET = REPO_ROOT / "data/output/arena_football_rot6d59"
DEFAULT_PP_BOX_VIDEO_DATASET = REPO_ROOT / "data/output/arena_pp_box_twist2_rot6d59"
DEFAULT_OPEN_DOOR_SRC = Path(
    "/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/dataset/"
    "HumanoidArena_open_door/HSI_open_door/sonic/zz"
)
DEFAULT_OPEN_DOOR_OUT = (
    REPO_ROOT / "data/output/arena_open_door_sonic_native_realized_v1"
)
DEFAULT_OPEN_DOOR_VIDEO_DATASET = (
    REPO_ROOT / "data/output/arena_open_door_sonic_rot6d59_v1"
)

STATE_KEY = "observation.sonic_state"
ACTION_KEY = "action.sonic"
VIDEO_KEY = "observation.images.ego_view"
FPS = 50
CHUNKS_SIZE = 1000
TOKEN_DIM = 64
BODY_DIM = 29
FUTURE_FRAMES = 10
FUTURE_STEP = 5
ENCODER_INPUT_DIM = 1762
MOTION_TOKEN_SOURCE_FUTURE = "future_reencode"
MOTION_TOKEN_SOURCE_RECORDED = "recorded_encoder_latent"
MOTION_TOKEN_SOURCES = (
    MOTION_TOKEN_SOURCE_FUTURE,
    MOTION_TOKEN_SOURCE_RECORDED,
)
SOURCE_FAMILY_AUTO = "auto"
SOURCE_FAMILY_TWIST2 = "twist2"
SOURCE_FAMILY_SONIC = "sonic"
SOURCE_FAMILIES = (
    SOURCE_FAMILY_AUTO,
    SOURCE_FAMILY_TWIST2,
    SOURCE_FAMILY_SONIC,
)

ENCODER_BODY_POS_SLICE = slice(4, 294)
ENCODER_BODY_VEL_SLICE = slice(294, 584)
ENCODER_ANCHOR_ROT6D_SLICE = slice(601, 661)

# Index a MuJoCo-order body vector to produce the IsaacLab order consumed by
# the released SONIC encoder/decoder.
MUJOCO_TO_ISAACLAB = np.asarray(
    [
        0,
        6,
        12,
        1,
        7,
        13,
        2,
        8,
        14,
        3,
        9,
        15,
        22,
        4,
        10,
        16,
        23,
        5,
        11,
        17,
        24,
        18,
        25,
        19,
        26,
        20,
        27,
        21,
        28,
    ],
    dtype=np.int64,
)

# TWIST2 hand streams are thumb, index, middle. Native SONIC recordings expose
# the provider target order thumb, middle, index. SONIC proprioception uses the
# RobotModel order index, middle, thumb, while GR00T SONIC actions use
# thumb, index, middle.
TWIST2_TO_STATE_HAND = np.asarray([3, 4, 5, 6, 0, 1, 2], dtype=np.int64)
SONIC_PROVIDER_TO_STATE_HAND = np.asarray(
    [5, 6, 3, 4, 0, 1, 2], dtype=np.int64
)
SONIC_PROVIDER_TO_ACTION_HAND = np.asarray(
    [0, 1, 2, 5, 6, 3, 4], dtype=np.int64
)

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
ACTION_NAMES = (
    [f"motion_token.{index}" for index in range(TOKEN_DIM)]
    + [f"left_hand_joints.{name}" for name in OFFICIAL_HAND_NAMES]
    + [f"right_hand_joints.{name}" for name in OFFICIAL_HAND_NAMES]
)


class SonicEncoder:
    def __init__(self, path: Path, provider: str, num_threads: int) -> None:
        available = ort.get_available_providers()
        if provider not in available:
            raise RuntimeError(
                f"ONNX Runtime provider {provider!r} is unavailable; available={available}"
            )
        options = ort.SessionOptions()
        if num_threads > 0:
            options.intra_op_num_threads = num_threads
        self.session = ort.InferenceSession(
            str(path), sess_options=options, providers=[provider]
        )
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        input_spec = [(item.name, item.shape) for item in inputs]
        output_spec = [(item.name, item.shape) for item in outputs]
        if input_spec != [("obs_dict", [1, ENCODER_INPUT_DIM])]:
            raise ValueError(f"Unexpected SONIC encoder inputs: {input_spec}")
        if output_spec != [("encoded_tokens", [1, TOKEN_DIM])]:
            raise ValueError(f"Unexpected SONIC encoder outputs: {output_spec}")
        self.input = np.zeros((1, ENCODER_INPUT_DIM), dtype=np.float32)

    def encode_episode(
        self,
        body_positions: np.ndarray,
        body_velocities: np.ndarray,
        root_rotations: np.ndarray,
        output_length: int,
        *,
        label: str,
    ) -> np.ndarray:
        tokens = np.empty((output_length, TOKEN_DIM), dtype=np.float32)
        for frame_idx in tqdm(range(output_length), desc=label, leave=False):
            indices = np.minimum(
                frame_idx + 1 + np.arange(FUTURE_FRAMES) * FUTURE_STEP,
                len(body_positions) - 1,
            )
            relative_rotations = np.einsum(
                "ij,njk->nik",
                root_rotations[frame_idx].T,
                root_rotations[indices],
            )
            self.input.fill(0.0)
            self.input[0, ENCODER_BODY_POS_SLICE] = body_positions[indices].reshape(-1)
            self.input[0, ENCODER_BODY_VEL_SLICE] = body_velocities[indices].reshape(-1)
            self.input[0, ENCODER_ANCHOR_ROT6D_SLICE] = relative_rotations[
                :, :, :2
            ].reshape(-1)
            token = self.session.run(None, {"obs_dict": self.input})[0]
            tokens[frame_idx] = np.asarray(token, dtype=np.float32).reshape(TOKEN_DIM)
        if not np.isfinite(tokens).all():
            raise ValueError(f"{label}: SONIC encoder returned NaN or Inf")
        return tokens


def projected_gravity(root_rotations: np.ndarray) -> np.ndarray:
    gravity_world = np.asarray([0.0, 0.0, -1.0], dtype=np.float64)
    gravity = np.einsum(
        "nij,j->ni", root_rotations.transpose(0, 2, 1), gravity_world
    ).astype(np.float32)
    norm_error = float(np.max(np.abs(np.linalg.norm(gravity, axis=1) - 1.0)))
    if norm_error > 1e-5:
        raise ValueError(f"Projected gravity norm error is too large: {norm_error}")
    return gravity


def resolve_source_family(source_family: str, motion_token_source: str) -> str:
    """Resolve recording layout independently from the motion-token source."""
    if source_family == SOURCE_FAMILY_AUTO:
        return (
            SOURCE_FAMILY_SONIC
            if motion_token_source == MOTION_TOKEN_SOURCE_RECORDED
            else SOURCE_FAMILY_TWIST2
        )
    if source_family not in (SOURCE_FAMILY_TWIST2, SOURCE_FAMILY_SONIC):
        raise ValueError(f"Unsupported source family: {source_family!r}")
    if (
        source_family == SOURCE_FAMILY_TWIST2
        and motion_token_source == MOTION_TOKEN_SOURCE_RECORDED
    ):
        raise ValueError(
            "recorded_encoder_latent is only supported for native SONIC recordings"
        )
    return source_family


def output_length_for_episode(
    raw_length: int, motion_token_source: str, keep_final_frame: bool
) -> int:
    if raw_length < 2:
        raise ValueError("fewer than two usable frames")
    if (
        motion_token_source == MOTION_TOKEN_SOURCE_FUTURE
        and not keep_final_frame
    ):
        return raw_length - 1
    return raw_length


def load_episode_arrays(
    data: np.lib.npyio.NpzFile,
    source_length: int,
    output_length: int,
    encoder: SonicEncoder | None,
    *,
    label: str,
    motion_token_source: str = MOTION_TOKEN_SOURCE_FUTURE,
    source_family: str = SOURCE_FAMILY_AUTO,
) -> dict[str, np.ndarray]:
    source_qpos = np.asarray(
        data["robot_qpos_before_decimation"][:source_length], dtype=np.float32
    )
    source_qvel = np.asarray(
        data["robot_qvel_before_decimation"][:source_length], dtype=np.float32
    )
    left_hand = np.asarray(data[pp_box.hand_key(data, "left")][:source_length], dtype=np.float32)
    right_hand = np.asarray(data[pp_box.hand_key(data, "right")][:source_length], dtype=np.float32)
    root_quat = np.asarray(data["robot_root_orientation"][:source_length], dtype=np.float64)

    expected = {
        "qpos": ((source_length, BODY_DIM), source_qpos.shape),
        "qvel": ((source_length, BODY_DIM), source_qvel.shape),
        "left_hand": ((source_length, 7), left_hand.shape),
        "right_hand": ((source_length, 7), right_hand.shape),
        "root_quat": ((source_length, 4), root_quat.shape),
    }
    bad = {name: value for name, value in expected.items() if value[0] != value[1]}
    if bad:
        raise ValueError(f"{label}: unexpected source shapes: {bad}")

    # Native SONIC recordings use the interleaved Isaac Lab controller order,
    # while UNITREE_G1_SONIC state features use the grouped order documented by
    # OFFICIAL_STATE_NAMES. TWIST2 recordings are already grouped. Source layout
    # is independent from whether tokens are recorded or re-encoded.
    source_family = resolve_source_family(source_family, motion_token_source)
    qpos = pp_box.reorder_source_qpos_to_body29(
        source_qpos,
        source_family=source_family,
    )
    qvel = pp_box.reorder_source_qpos_to_body29(
        source_qvel,
        source_family=source_family,
    )
    if source_family == "sonic":
        state_hand_reorder = SONIC_PROVIDER_TO_STATE_HAND
        left_action = left_hand[:, SONIC_PROVIDER_TO_ACTION_HAND]
        right_action = right_hand[:, SONIC_PROVIDER_TO_ACTION_HAND]
    else:
        state_hand_reorder = TWIST2_TO_STATE_HAND
        left_action = left_hand
        right_action = right_hand

    root_rotations = arena.quat_wxyz_to_matrix(root_quat)
    gravity = projected_gravity(root_rotations)
    if motion_token_source == MOTION_TOKEN_SOURCE_FUTURE:
        if encoder is None:
            raise ValueError(f"{label}: future_reencode requires a SONIC encoder")
        body_positions_encoder = qpos[:, MUJOCO_TO_ISAACLAB]
        body_velocities_encoder = qvel[:, MUJOCO_TO_ISAACLAB]
        motion_tokens = encoder.encode_episode(
            body_positions_encoder,
            body_velocities_encoder,
            root_rotations,
            output_length,
            label=f"SONIC {label}",
        )
    elif motion_token_source == MOTION_TOKEN_SOURCE_RECORDED:
        if "encoder_latent" not in data.files:
            raise KeyError(f"{label}: source NPZ has no encoder_latent array")
        motion_tokens = np.asarray(
            data["encoder_latent"][:output_length], dtype=np.float32
        )
        if motion_tokens.shape != (output_length, TOKEN_DIM):
            raise ValueError(
                f"{label}: expected encoder_latent shape "
                f"({output_length}, {TOKEN_DIM}), got {motion_tokens.shape}"
            )
        if not np.isfinite(motion_tokens).all():
            raise ValueError(f"{label}: encoder_latent contains NaN or Inf")
    else:
        raise ValueError(
            f"{label}: unsupported motion token source {motion_token_source!r}"
        )

    state = np.concatenate(
        [
            qpos[:output_length, :22],
            left_hand[:output_length, state_hand_reorder],
            qpos[:output_length, 22:],
            right_hand[:output_length, state_hand_reorder],
            gravity[:output_length],
        ],
        axis=1,
    ).astype(np.float32)
    action = np.concatenate(
        [
            motion_tokens,
            left_action[:output_length],
            right_action[:output_length],
        ],
        axis=1,
    ).astype(np.float32)
    if state.shape != (output_length, 46) or action.shape != (output_length, 78):
        raise ValueError(
            f"{label}: invalid state/action shapes {state.shape}/{action.shape}"
        )
    if not np.isfinite(state).all() or not np.isfinite(action).all():
        raise ValueError(f"{label}: state/action contains NaN or Inf")
    return {STATE_KEY: state, ACTION_KEY: action}


def build_table(
    arrays: dict[str, np.ndarray], episode_index: int, global_offset: int
) -> pa.Table:
    length = arrays[STATE_KEY].shape[0]
    frame_indices = np.arange(length, dtype=np.int64)
    fields = [
        pa.field(STATE_KEY, pa.list_(pa.float32(), 46)),
        pa.field(ACTION_KEY, pa.list_(pa.float32(), 78)),
        pa.field("timestamp", pa.float32()),
        pa.field("frame_index", pa.int64()),
        pa.field("episode_index", pa.int64()),
        pa.field("index", pa.int64()),
        pa.field("task_index", pa.int64()),
    ]
    parquet_arrays = [
        pa.array(arrays[STATE_KEY].tolist(), type=pa.list_(pa.float32(), 46)),
        pa.array(arrays[ACTION_KEY].tolist(), type=pa.list_(pa.float32(), 78)),
        pa.array(frame_indices.astype(np.float32) / FPS, type=pa.float32()),
        pa.array(frame_indices, type=pa.int64()),
        pa.array(np.full(length, episode_index, dtype=np.int64), type=pa.int64()),
        pa.array(global_offset + frame_indices, type=pa.int64()),
        pa.array(np.zeros(length, dtype=np.int64), type=pa.int64()),
    ]
    return pa.Table.from_arrays(parquet_arrays, schema=pa.schema(fields))


def modality() -> dict[str, Any]:
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
        "video": {"ego_view": {"original_key": VIDEO_KEY}},
        "annotation": {
            "human.task_description": {"original_key": "task_index"}
        },
    }


def feature_stats(values: np.ndarray) -> dict[str, list[Any]]:
    return arena.feature_stats(values)


def episode_stats(
    arrays: dict[str, np.ndarray], episode_index: int, global_offset: int
) -> dict[str, Any]:
    length = arrays[STATE_KEY].shape[0]
    frame_indices = np.arange(length, dtype=np.int64)
    return {
        "episode_index": episode_index,
        "stats": {
            STATE_KEY: feature_stats(arrays[STATE_KEY]),
            ACTION_KEY: feature_stats(arrays[ACTION_KEY]),
            "timestamp": feature_stats(frame_indices.astype(np.float32) / FPS),
            "frame_index": feature_stats(frame_indices),
            "episode_index": feature_stats(
                np.full(length, episode_index, dtype=np.int64)
            ),
            "index": feature_stats(global_offset + frame_indices),
            "task_index": feature_stats(np.zeros(length, dtype=np.int64)),
        },
    }


def build_info(
    *,
    kind: str,
    total_episodes: int,
    total_frames: int,
    total_videos: int,
    task: str,
    source_root: Path,
    encoder_path: Path,
    skipped: list[dict[str, Any]],
    reused_video_dataset: Path | None,
    motion_token_source: str,
    video_codec: str,
    source_family: str = SOURCE_FAMILY_AUTO,
    keep_final_frame: bool = False,
) -> dict[str, Any]:
    source_family = resolve_source_family(source_family, motion_token_source)
    recorded_tokens = motion_token_source == MOTION_TOKEN_SOURCE_RECORDED
    realized_sonic = (
        source_family == SOURCE_FAMILY_SONIC
        and motion_token_source == MOTION_TOKEN_SOURCE_FUTURE
    )
    if realized_sonic:
        source_name = "humanoid_arena_sonic_realized_to_native_sonic"
        schema_version = "arena_sonic_native_realized_v1"
    elif recorded_tokens:
        source_name = "humanoid_arena_sonic_to_native_sonic"
        schema_version = "arena_sonic_native_v2"
    else:
        source_name = "humanoid_arena_twist2_to_native_sonic"
        schema_version = "arena_twist2_sonic_v1"
    return {
        "codebase_version": "v2.1",
        "robot_type": f"g1_dex3_arena_{kind}_sonic",
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": 1,
        "total_videos": total_videos,
        "total_chunks": max(1, math.ceil(total_episodes / CHUNKS_SIZE)),
        "chunks_size": CHUNKS_SIZE,
        "fps": FPS,
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            VIDEO_KEY: {
                "dtype": "video",
                "shape": [480, 640, 3],
                "names": ["height", "width", "channel"],
                "info": {
                    "video.height": 480,
                    "video.width": 640,
                    "video.codec": video_codec,
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "video.fps": FPS,
                    "video.channels": 3,
                    "has_audio": False,
                },
            },
            STATE_KEY: {
                "dtype": "float32",
                "shape": [46],
                "names": OFFICIAL_STATE_NAMES,
            },
            ACTION_KEY: {
                "dtype": "float32",
                "shape": [78],
                "names": ACTION_NAMES,
            },
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        },
        "script_config": {
            "source": source_name,
            "schema_version": schema_version,
            "kind": kind,
            "source_root": str(source_root),
            "source_selection": (
                "native SONIC NPZ files with an existing front RGB video"
                if source_family == SOURCE_FAMILY_SONIC
                else (
                    "all TWIST2/tw and TWIST2/yb NPZ files"
                    if kind == "football"
                    else (
                        "twist2/zz NPZ files with an existing front RGB video"
                        if kind == "open_door"
                        else "twist2/yb NPZ files with an existing front RGB video"
                    )
                )
            ),
            "skipped_missing_video": len(skipped),
            "fps": FPS,
            "drop_final_frame": (
                motion_token_source == MOTION_TOKEN_SOURCE_FUTURE
                and not keep_final_frame
            ),
            "source_family": source_family,
            "state": f"{STATE_KEY}(46=body29+hand14+projected_gravity3)",
            "action": f"{ACTION_KEY}(78=motion_token64+target_hand14)",
            "motion_token_encoder": str(encoder_path),
            "encoder_layout": "release G1 mode, obs_dict[1,1762] -> token[1,64]",
            "future_observation_indices": (
                None if recorded_tokens else "t+1+5*k, k=0..9, clamp=end"
            ),
            "future_reference_seconds": None if recorded_tokens else [0.02, 0.92],
            "motion_token_source": motion_token_source,
            "realized_body_source": (
                "robot_qpos_before_decimation + robot_qvel_before_decimation + robot_root_orientation"
                if motion_token_source == MOTION_TOKEN_SOURCE_FUTURE
                else None
            ),
            "source_body29_order": (
                "SONIC IsaacLab interleaved -> reordered to UNITREE_G1_SONIC grouped order"
                if source_family == SOURCE_FAMILY_SONIC
                else "TWIST2/MuJoCo grouped order"
            ),
            "state_hand_order": "index,middle,thumb",
            "action_hand_order": "thumb,index,middle",
            "source_hand_order": (
                "SONIC provider thumb,middle,index; reordered for state and action"
                if source_family == SOURCE_FAMILY_SONIC
                else "TWIST2 thumb,index,middle"
            ),
            "video_source": (
                f"reused from {reused_video_dataset}"
                if reused_video_dataset is not None
                else "converted directly from the selected raw recording"
            ),
            "task": task,
        },
    }


def schema(
    motion_token_source: str,
    source_family: str = SOURCE_FAMILY_AUTO,
    keep_final_frame: bool = False,
) -> dict[str, Any]:
    source_family = resolve_source_family(source_family, motion_token_source)
    recorded_tokens = motion_token_source == MOTION_TOKEN_SOURCE_RECORDED
    return {
        STATE_KEY: {
            "dim": 46,
            "layout": "left_leg6 + right_leg6 + waist3 + left_arm7 + left_hand7 + right_arm7 + right_hand7 + projected_gravity3",
            "names": OFFICIAL_STATE_NAMES,
        },
        ACTION_KEY: {
            "dim": 78,
            "layout": "motion_token64 + left_hand7 + right_hand7",
            "names": ACTION_NAMES,
        },
        "motion_token": {
            "encoder_input": "obs_dict[1,1762]",
            "source": motion_token_source,
            "source_family": source_family,
            "keep_final_frame": keep_final_frame,
            "future_frames": (
                None
                if recorded_tokens
                else "t+1,t+6,...,t+46 at 50 Hz, clamped at episode end"
            ),
            "body_order": (
                None
                if recorded_tokens
                else (
                    "SONIC IsaacLab interleaved -> grouped -> MUJOCO_TO_ISAACLAB"
                    if source_family == SOURCE_FAMILY_SONIC
                    else "MuJoCo-order qpos/qvel indexed by MUJOCO_TO_ISAACLAB"
                )
            ),
            "anchor_orientation": (
                None
                if recorded_tokens
                else "R(root_t)^T @ R(root_future), first two columns flattened row-major"
            ),
        },
    }


def read_video_reuse_map(dataset_root: Path) -> dict[str, Path]:
    episodes_path = dataset_root / "meta/episodes.jsonl"
    if not episodes_path.is_file():
        raise FileNotFoundError(f"Missing reuse dataset episodes metadata: {episodes_path}")
    result: dict[str, Path] = {}
    for line in episodes_path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        source_npz = str(Path(row["source_npz"]).resolve())
        episode_index = int(row["episode_index"])
        chunk = episode_index // CHUNKS_SIZE
        video = (
            dataset_root
            / "videos"
            / f"chunk-{chunk:03d}"
            / VIDEO_KEY
            / f"episode_{episode_index:06d}.mp4"
        )
        if not video.is_file():
            raise FileNotFoundError(f"Missing reusable video: {video}")
        result[source_npz] = video.resolve()
    return result


def read_reused_video_codec(dataset_root: Path) -> str:
    info_path = dataset_root / "meta/info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"Missing reuse dataset info metadata: {info_path}")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    return str(info["features"][VIDEO_KEY]["info"]["video.codec"])


def link_video(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(os.path.relpath(source, destination.parent))


def encode_external_video(source: Path, destination: Path, output_length: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-frames:v",
        str(output_length),
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
        str(destination),
    ]
    subprocess.run(cmd, check=True)


def source_length(data: np.lib.npyio.NpzFile) -> int:
    length = pp_box.scalar_int(data, "num_frames")
    required = (
        "robot_qpos_before_decimation",
        "robot_qvel_before_decimation",
        "robot_root_orientation",
        pp_box.hand_key(data, "left"),
        pp_box.hand_key(data, "right"),
    )
    return min(length, *(len(data[key]) for key in required))


def discover_sources(
    kind: str, source_root: Path
) -> tuple[list[Path], list[dict[str, Any]]]:
    files = sorted(
        source_root.rglob("*.npz") if kind == "football" else source_root.glob("*.npz")
    )
    if not files:
        raise FileNotFoundError(f"No NPZ files found under {source_root}")
    if kind == "football":
        return files, []

    selected: list[Path] = []
    skipped: list[dict[str, Any]] = []
    for npz_path in files:
        with np.load(npz_path, allow_pickle=True) as data:
            video = pp_box.video_path_for_episode(npz_path, data)
        if not video.is_file():
            skipped.append(
                {
                    "source_npz": str(npz_path),
                    "reason": "missing_front_video",
                    "expected_video": str(video),
                }
            )
            continue
        selected.append(npz_path)
    return selected, skipped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kind", required=True, choices=("football", "pp_box", "open_door")
    )
    parser.add_argument("--src", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--encoder", type=Path, default=DEFAULT_ENCODER)
    parser.add_argument("--provider", default="CPUExecutionProvider")
    parser.add_argument("--onnx-threads", type=int, default=8)
    parser.add_argument("--task")
    parser.add_argument("--expected-episodes", type=int, default=100)
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--reuse-videos-from", type=Path)
    parser.add_argument(
        "--encode-videos-from-source",
        action="store_true",
        help=(
            "Ignore the task's default reusable video dataset and encode the "
            "external video paired with each selected source NPZ."
        ),
    )
    parser.add_argument(
        "--motion-token-source",
        choices=MOTION_TOKEN_SOURCES,
        default=MOTION_TOKEN_SOURCE_FUTURE,
        help=(
            "Re-encode future body motion (TWIST2 default), or preserve the "
            "source NPZ encoder_latent array (native SONIC recordings)."
        ),
    )
    parser.add_argument(
        "--source-family",
        choices=SOURCE_FAMILIES,
        default=SOURCE_FAMILY_AUTO,
        help=(
            "Recording joint/hand layout. auto preserves legacy behavior: "
            "recorded tokens imply sonic, future re-encoding implies twist2."
        ),
    )
    parser.add_argument(
        "--keep-final-frame",
        action="store_true",
        help=(
            "Keep the final frame when future-reencoding and clamp all future "
            "references to the episode end."
        ),
    )
    parser.add_argument("--skip-videos", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    defaults = {
        "football": {
            "src": DEFAULT_FOOTBALL_SRC,
            "out": DEFAULT_FOOTBALL_OUT,
            "task": "Move toward the football and kick it.",
            "reuse": DEFAULT_FOOTBALL_VIDEO_DATASET,
        },
        "pp_box": {
            "src": DEFAULT_PP_BOX_SRC,
            "out": DEFAULT_PP_BOX_OUT,
            "task": "Pick up the box and place it on the shelf.",
            "reuse": DEFAULT_PP_BOX_VIDEO_DATASET,
        },
        "open_door": {
            "src": DEFAULT_OPEN_DOOR_SRC,
            "out": DEFAULT_OPEN_DOOR_OUT,
            "task": "Open the door.",
            "reuse": DEFAULT_OPEN_DOOR_VIDEO_DATASET,
        },
    }[args.kind]
    src = (args.src or defaults["src"]).resolve()
    out = (args.out or defaults["out"]).resolve()
    encoder_path = args.encoder.resolve()
    task = args.task or defaults["task"]
    source_family = resolve_source_family(
        args.source_family, args.motion_token_source
    )
    reuse_root = (
        args.reuse_videos_from.resolve()
        if args.reuse_videos_from is not None
        else Path(defaults["reuse"]).resolve()
    )
    if args.reuse_videos_from is not None and args.encode_videos_from_source:
        raise ValueError(
            "--reuse-videos-from and --encode-videos-from-source are mutually exclusive"
        )
    if args.skip_videos or args.encode_videos_from_source:
        reuse_root = None

    if not src.is_dir():
        raise FileNotFoundError(f"Source directory does not exist: {src}")
    if (
        args.motion_token_source == MOTION_TOKEN_SOURCE_FUTURE
        and not encoder_path.is_file()
    ):
        raise FileNotFoundError(f"SONIC encoder does not exist: {encoder_path}")
    if out.exists():
        if not args.overwrite:
            raise FileExistsError(f"{out} exists; pass --overwrite to replace it")
        shutil.rmtree(out)

    npz_files, skipped = discover_sources(args.kind, src)
    if args.max_episodes is None and len(npz_files) != args.expected_episodes:
        raise ValueError(
            f"Expected exactly {args.expected_episodes} selected {args.kind} episodes, "
            f"got {len(npz_files)} (skipped={len(skipped)})"
        )
    if args.max_episodes is not None:
        npz_files = npz_files[: args.max_episodes]
    if not npz_files:
        raise FileNotFoundError("No episodes selected")

    reuse_map = read_video_reuse_map(reuse_root) if reuse_root is not None else {}
    video_codec = (
        read_reused_video_codec(reuse_root) if reuse_root is not None else "h264"
    )
    encoder = (
        SonicEncoder(encoder_path, args.provider, args.onnx_threads)
        if args.motion_token_source == MOTION_TOKEN_SOURCE_FUTURE
        else None
    )
    stats_parts: dict[str, list[np.ndarray]] = {
        STATE_KEY: [],
        ACTION_KEY: [],
        "timestamp": [],
        "frame_index": [],
        "episode_index": [],
        "index": [],
        "task_index": [],
    }
    episode_rows: list[dict[str, Any]] = []
    episode_stats_rows: list[dict[str, Any]] = []
    total_frames = 0

    for episode_index, npz_path in enumerate(npz_files):
        with np.load(npz_path, allow_pickle=True) as data:
            raw_length = source_length(data)
            if args.max_frames is not None:
                raw_length = min(raw_length, args.max_frames)
            try:
                output_length = output_length_for_episode(
                    raw_length,
                    args.motion_token_source,
                    args.keep_final_frame,
                )
            except ValueError as exc:
                raise ValueError(f"{npz_path}: {exc}") from exc
            arrays = load_episode_arrays(
                data,
                raw_length,
                output_length,
                encoder,
                label=f"episode_{episode_index:06d}",
                motion_token_source=args.motion_token_source,
                source_family=source_family,
            )

            chunk = episode_index // CHUNKS_SIZE
            parquet = (
                out
                / "data"
                / f"chunk-{chunk:03d}"
                / f"episode_{episode_index:06d}.parquet"
            )
            parquet.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(build_table(arrays, episode_index, total_frames), parquet)

            output_video = (
                out
                / "videos"
                / f"chunk-{chunk:03d}"
                / VIDEO_KEY
                / f"episode_{episode_index:06d}.mp4"
            )
            source_video: str | None = None
            if not args.skip_videos:
                source_key = str(npz_path.resolve())
                if reuse_root is not None:
                    if source_key not in reuse_map:
                        raise KeyError(
                            f"No reusable video mapped to source episode {npz_path} "
                            f"in {reuse_root}"
                        )
                    link_video(reuse_map[source_key], output_video)
                    source_video = str(reuse_map[source_key])
                elif args.kind == "football":
                    arena.write_video(data, output_video, output_length)
                    source_video = "embedded vision_rgb"
                else:
                    raw_video = pp_box.video_path_for_episode(npz_path, data)
                    encode_external_video(raw_video, output_video, output_length)
                    source_video = str(raw_video)

            frame_indices = np.arange(output_length, dtype=np.int64)
            stats_parts[STATE_KEY].append(arrays[STATE_KEY])
            stats_parts[ACTION_KEY].append(arrays[ACTION_KEY])
            stats_parts["timestamp"].append(frame_indices.astype(np.float32) / FPS)
            stats_parts["frame_index"].append(frame_indices)
            stats_parts["episode_index"].append(
                np.full(output_length, episode_index, dtype=np.int64)
            )
            stats_parts["index"].append(total_frames + frame_indices)
            stats_parts["task_index"].append(
                np.zeros(output_length, dtype=np.int64)
            )
            episode_rows.append(
                {
                    "episode_index": episode_index,
                    "tasks": [task],
                    "length": output_length,
                    "source_npz": str(npz_path),
                    "source_video": source_video,
                }
            )
            episode_stats_rows.append(
                episode_stats(arrays, episode_index, total_frames)
            )
            total_frames += output_length
            print(
                f"[{episode_index + 1}/{len(npz_files)}] {npz_path.name}: "
                f"source_frames={raw_length} output_frames={output_length}"
            )

    meta = out / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    arena.write_jsonl(meta / "tasks.jsonl", [{"task_index": 0, "task": task}])
    arena.write_jsonl(meta / "episodes.jsonl", episode_rows)
    arena.write_jsonl(meta / "episodes_stats.jsonl", episode_stats_rows)
    arena.write_json(
        meta / "stats.json",
        {
            key: feature_stats(np.concatenate(parts, axis=0))
            for key, parts in stats_parts.items()
        },
    )
    arena.write_json(meta / "relative_stats.json", {"__fingerprints__": {}})
    arena.write_json(meta / "modality.json", modality())
    arena.write_json(
        meta / "sonic_schema.json",
        schema(
            args.motion_token_source,
            source_family=source_family,
            keep_final_frame=args.keep_final_frame,
        ),
    )
    arena.write_json(
        meta / "info.json",
        build_info(
            kind=args.kind,
            total_episodes=len(npz_files),
            total_frames=total_frames,
            total_videos=0 if args.skip_videos else len(npz_files),
            task=task,
            source_root=src,
            encoder_path=encoder_path,
            skipped=skipped,
            reused_video_dataset=reuse_root,
            motion_token_source=args.motion_token_source,
            video_codec=video_codec,
            source_family=source_family,
            keep_final_frame=args.keep_final_frame,
        ),
    )
    if skipped:
        arena.write_json(meta / "conversion_report.json", {"skipped": skipped})

    print(f"kind={args.kind}")
    print(f"source={src}")
    print(f"output={out}")
    print(f"episodes={len(npz_files)} frames={total_frames} fps={FPS}")
    print(f"skipped_missing_video={len(skipped)}")
    print(f"motion_token_source={args.motion_token_source}")
    print(f"source_family={source_family}")
    print(f"keep_final_frame={args.keep_final_frame}")
    print(f"{STATE_KEY}=46D {ACTION_KEY}=78D")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
