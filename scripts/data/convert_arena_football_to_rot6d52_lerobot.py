#!/usr/bin/env python3
"""Convert HumanoidArena football npz recordings to rot6d59 LeRobot data.

The exported dataset follows the modified GR00T VLA path used in this repo:

    observation.images.ego_view
    observation.full_state_rot6d: hand14 + body29 + root9 = 52D
    action.policy_action_rot6d59:
        hand14 + root9 + left/right hand pose9 + left/right foot pose9 = 59D

HumanoidArena npz files do not store the four end-effector poses directly.
They are reconstructed from root pose + 29D body qpos with a lightweight MJCF
forward-kinematics helper for the four body frames used by Kimodo/TextOp.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SRC = Path(
    "/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/dataset/"
    "HumanoidArena_football/HOI_football_v2"
)
DEFAULT_OUT = REPO_ROOT / "data/output/arena_football_rot6d59"
DEFAULT_MJCF = REPO_ROOT / "third_party/SIMPLE/data/robots/g1/g1_29dof_with_dex3.xml"

STATE_KEY = "observation.full_state_rot6d"
ROOT_KEY = "observation.root_pose_rot6d"
ACTION_KEY = "action.policy_action_rot6d59"
REF_BODY_KEY = "action.ref_body29"
VIDEO_KEY = "observation.images.ego_view"
FPS = 50
CHUNKS_SIZE = 1000

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

POSE9_NAMES = ["x", "y", "z", "r00", "r01", "r10", "r11", "r20", "r21"]
ROOT9_NAMES = [f"root.{name}" for name in POSE9_NAMES]
EE_BODY_NAMES = [
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
]
EE9_NAMES = [
    f"{prefix}.{name}"
    for prefix in ("left_hand_pose", "right_hand_pose", "left_foot_pose", "right_foot_pose")
    for name in POSE9_NAMES
]
STATE52_NAMES = [f"hand.{name}" for name in HAND14_ORDER] + [
    f"body.{name}" for name in BODY29_ORDER
] + ROOT9_NAMES
ACTION59_NAMES = [f"hand.{name}" for name in HAND14_ORDER] + ROOT9_NAMES + EE9_NAMES
REF_BODY29_NAMES = [f"body.{name}" for name in BODY29_ORDER]

# TWIST2 hand streams are thumb, index, middle for each hand. The rot6d59
# schema stores left hand as thumb, middle, index and right hand as thumb,
# index, middle.
LEFT_HAND_TO_ROT59 = [0, 1, 2, 5, 6, 3, 4]
RIGHT_HAND_TO_ROT59 = [0, 1, 2, 3, 4, 5, 6]

# Native SONIC stores robot_qpos_before_decimation in the interleaved Isaac
# Lab controller order, while rot6d59/Kimodo/TextOp use the grouped MuJoCo
# order above.  This semantic difference cannot be detected from the 29D
# shape, so conversion must make the source family explicit.
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
    [SONIC_ISAACLAB_BODY29_ORDER.index(name) for name in BODY29_ORDER],
    dtype=np.int64,
)

# Native SONIC hand targets are in provider order thumb, middle, index.
# rot6d59 keeps that order on the left but uses thumb, index, middle on the
# right.
SONIC_LEFT_TO_ROT59 = np.arange(7, dtype=np.int64)
SONIC_RIGHT_TO_ROT59 = np.asarray([0, 1, 2, 5, 6, 3, 4], dtype=np.int64)


@dataclass(frozen=True)
class MjcfJoint:
    name: str
    kind: str
    pos: np.ndarray
    axis: np.ndarray


@dataclass(frozen=True)
class MjcfBody:
    name: str
    pos: np.ndarray
    quat_wxyz: np.ndarray
    joints: tuple[MjcfJoint, ...]
    children: tuple["MjcfBody", ...]


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=4, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def parse_vec(text: str | None, default: tuple[float, ...]) -> np.ndarray:
    if text is None:
        return np.asarray(default, dtype=np.float64)
    return np.asarray([float(item) for item in text.split()], dtype=np.float64)


def quat_wxyz_to_matrix(quat_wxyz: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat_wxyz, dtype=np.float64)
    if quat.ndim == 1:
        quat = quat[None, :]
    quat = quat / np.maximum(np.linalg.norm(quat, axis=1, keepdims=True), 1e-12)
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

    mat = np.empty((quat.shape[0], 3, 3), dtype=np.float64)
    mat[:, 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    mat[:, 0, 1] = 2.0 * (x * y - z * w)
    mat[:, 0, 2] = 2.0 * (x * z + y * w)
    mat[:, 1, 0] = 2.0 * (x * y + z * w)
    mat[:, 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    mat[:, 1, 2] = 2.0 * (y * z - x * w)
    mat[:, 2, 0] = 2.0 * (x * z - y * w)
    mat[:, 2, 1] = 2.0 * (y * z + x * w)
    mat[:, 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return mat[0] if np.asarray(quat_wxyz).ndim == 1 else mat


def matrix_to_rot6d(matrix: np.ndarray) -> np.ndarray:
    mat = np.asarray(matrix, dtype=np.float32)
    if mat.ndim == 2:
        return mat[:, :2].reshape(6).astype(np.float32)
    return mat[:, :, :2].reshape(mat.shape[0], 6).astype(np.float32)


def quat_wxyz_to_rot6d(quat_wxyz: np.ndarray) -> np.ndarray:
    return matrix_to_rot6d(quat_wxyz_to_matrix(quat_wxyz))


def transform_from_pos_rot(pos: np.ndarray, rot: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rot
    transform[:3, 3] = pos
    return transform


def axis_angle_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    norm = float(np.linalg.norm(axis))
    if norm <= 1e-12:
        return np.eye(3, dtype=np.float64)
    x, y, z = axis / norm
    c = math.cos(angle)
    s = math.sin(angle)
    one_c = 1.0 - c
    return np.asarray(
        [
            [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
            [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
            [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
        ],
        dtype=np.float64,
    )


def joint_transform(joint: MjcfJoint, value: float) -> np.ndarray:
    if joint.kind == "free":
        return np.eye(4, dtype=np.float64)

    if joint.kind == "slide":
        transform = np.eye(4, dtype=np.float64)
        transform[:3, 3] = joint.axis * value
        return transform

    if joint.kind not in {"hinge", ""}:
        return np.eye(4, dtype=np.float64)

    to_joint = np.eye(4, dtype=np.float64)
    to_joint[:3, 3] = joint.pos
    from_joint = np.eye(4, dtype=np.float64)
    from_joint[:3, 3] = -joint.pos
    return to_joint @ transform_from_pos_rot(np.zeros(3), axis_angle_matrix(joint.axis, value)) @ from_joint


def parse_body(element: ET.Element) -> MjcfBody:
    joints = []
    for joint_element in element.findall("joint") + element.findall("freejoint"):
        if joint_element.tag == "freejoint":
            kind = "free"
        else:
            kind = joint_element.attrib.get("type", "hinge")
        joints.append(
            MjcfJoint(
                name=joint_element.attrib.get("name", ""),
                kind=kind,
                pos=parse_vec(joint_element.attrib.get("pos"), (0.0, 0.0, 0.0)),
                axis=parse_vec(joint_element.attrib.get("axis"), (1.0, 0.0, 0.0)),
            )
        )

    return MjcfBody(
        name=element.attrib["name"],
        pos=parse_vec(element.attrib.get("pos"), (0.0, 0.0, 0.0)),
        quat_wxyz=parse_vec(element.attrib.get("quat"), (1.0, 0.0, 0.0, 0.0)),
        joints=tuple(joints),
        children=tuple(parse_body(child) for child in element.findall("body")),
    )


class G1MjcfFK:
    def __init__(self, mjcf_path: Path) -> None:
        root = ET.parse(mjcf_path).getroot()
        worldbody = root.find("worldbody")
        if worldbody is None:
            raise ValueError(f"{mjcf_path} does not contain <worldbody>")
        bodies = [parse_body(body) for body in worldbody.findall("body")]
        if not bodies:
            raise ValueError(f"{mjcf_path} contains no root bodies")
        self.root_bodies = tuple(bodies)
        self.joint_to_index = {name: idx for idx, name in enumerate(BODY29_ORDER)}
        missing = set(BODY29_ORDER) - self._joint_names()
        if missing:
            raise ValueError(f"{mjcf_path} missing expected joints: {sorted(missing)}")

    def _joint_names(self) -> set[str]:
        names = set()

        def visit(body: MjcfBody) -> None:
            for joint in body.joints:
                if joint.name:
                    names.add(joint.name)
            for child in body.children:
                visit(child)

        for body in self.root_bodies:
            visit(body)
        return names

    def poses(self, body29: np.ndarray, root_pos: np.ndarray, root_quat_wxyz: np.ndarray) -> np.ndarray:
        body29 = np.asarray(body29, dtype=np.float64)
        root_transform = transform_from_pos_rot(
            np.asarray(root_pos, dtype=np.float64),
            quat_wxyz_to_matrix(root_quat_wxyz),
        )
        targets: dict[str, np.ndarray] = {}

        def visit(body: MjcfBody, parent_transform: np.ndarray) -> None:
            transform = parent_transform @ transform_from_pos_rot(body.pos, quat_wxyz_to_matrix(body.quat_wxyz))
            for joint in body.joints:
                idx = self.joint_to_index.get(joint.name)
                value = 0.0 if idx is None else float(body29[idx])
                transform = transform @ joint_transform(joint, value)
            if body.name in EE_BODY_NAMES:
                targets[body.name] = transform.copy()
            for child in body.children:
                visit(child, transform)

        for body in self.root_bodies:
            visit(body, root_transform)

        missing = [name for name in EE_BODY_NAMES if name not in targets]
        if missing:
            raise RuntimeError(f"FK did not produce target bodies: {missing}")

        parts = []
        for name in EE_BODY_NAMES:
            transform = targets[name]
            parts.append(np.concatenate([transform[:3, 3], matrix_to_rot6d(transform[:3, :3])]))
        return np.concatenate(parts).astype(np.float32)


def validate_rot6d(rot6d: np.ndarray, label: str, atol: float = 1e-5) -> None:
    rot6d = np.asarray(rot6d, dtype=np.float32)
    c0 = rot6d[:, [0, 2, 4]]
    c1 = rot6d[:, [1, 3, 5]]
    error = max(
        float(np.max(np.abs(np.linalg.norm(c0, axis=1) - 1.0))),
        float(np.max(np.abs(np.linalg.norm(c1, axis=1) - 1.0))),
        float(np.max(np.abs(np.sum(c0 * c1, axis=1)))),
    )
    if not np.isfinite(error) or error > atol:
        raise ValueError(f"{label} is not valid rot6d; max error={error:.6g}")


def find_npz_files(src: Path, npz_index: int | None, max_episodes: int | None) -> list[Path]:
    files = sorted(src.rglob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz files found under {src}")
    if npz_index is not None:
        if npz_index < 0 or npz_index >= len(files):
            raise IndexError(f"--npz-index {npz_index} out of range; found {len(files)} files")
        files = [files[npz_index]]
    if max_episodes is not None:
        files = files[: int(max_episodes)]
    return files


def scalar_text(data: np.lib.npyio.NpzFile, key: str, default: str = "") -> str:
    if key not in data.files:
        return default
    value = data[key]
    return str(value.item()) if value.shape == () else str(value)


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


def detect_source_family(
    npz_path: Path,
    data: np.lib.npyio.NpzFile,
    requested_family: str,
) -> str:
    if requested_family != "auto":
        return requested_family
    schema_version = scalar_text(data, "schema_version").lower()
    path_parts = {part.lower() for part in npz_path.parts}
    if "sonic" in schema_version or "sonic" in path_parts:
        return "sonic"
    if "twist2" in schema_version or "twist2" in path_parts:
        return "twist2"
    raise ValueError(
        f"Cannot infer source family for {npz_path}; pass --source-family explicitly"
    )


def reorder_source_qpos_to_body29(
    qpos: np.ndarray,
    *,
    source_family: str,
) -> np.ndarray:
    qpos = np.asarray(qpos, dtype=np.float32)
    if qpos.ndim != 2 or qpos.shape[1] != len(BODY29_ORDER):
        raise ValueError(f"Expected body qpos [T, 29], got {qpos.shape}")
    if source_family == "sonic":
        return qpos[:, SONIC_TO_BODY29].copy()
    if source_family == "twist2":
        return qpos.copy()
    raise ValueError(f"Unsupported source family: {source_family}")


def hand14_from_arena(left_hand: np.ndarray, right_hand: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [left_hand[:, LEFT_HAND_TO_ROT59], right_hand[:, RIGHT_HAND_TO_ROT59]],
        axis=1,
    ).astype(np.float32)


def hand14_from_source(
    left_hand: np.ndarray,
    right_hand: np.ndarray,
    *,
    source_family: str,
) -> np.ndarray:
    if source_family == "sonic":
        return np.concatenate(
            [
                left_hand[:, SONIC_LEFT_TO_ROT59],
                right_hand[:, SONIC_RIGHT_TO_ROT59],
            ],
            axis=1,
        ).astype(np.float32)
    if source_family == "twist2":
        return hand14_from_arena(left_hand, right_hand)
    raise ValueError(f"Unsupported source family: {source_family}")


def shift_future(values: np.ndarray, shift: int) -> np.ndarray:
    values = np.asarray(values)
    if shift < 0:
        raise ValueError(f"Shift must be >= 0, got {shift}")
    if shift == 0 or values.shape[0] == 0:
        return values.copy()
    if shift >= values.shape[0]:
        return np.repeat(values[-1:], values.shape[0], axis=0)
    return np.concatenate([values[shift:], np.repeat(values[-1:], shift, axis=0)], axis=0)


def load_episode_arrays(
    data: np.lib.npyio.NpzFile,
    length: int,
    fk: G1MjcfFK,
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
    root9 = np.concatenate([root_pos, quat_wxyz_to_rot6d(root_quat)], axis=1).astype(np.float32)
    state52 = np.concatenate([hand14, qpos, root9], axis=1).astype(np.float32)
    if state52.shape != (length, 52):
        raise ValueError(f"Built invalid {STATE_KEY} shape {state52.shape}")

    ee36 = np.stack(
        [fk.poses(qpos[i], root_pos[i], root_quat[i]) for i in range(length)],
        axis=0,
    ).astype(np.float32)
    hand_action = shift_future(hand14, hand_target_shift).astype(np.float32)
    root_action = shift_future(root9, target_shift).astype(np.float32)
    ee_action = shift_future(ee36, target_shift).astype(np.float32)
    ref_body29 = shift_future(qpos, target_shift).astype(np.float32)
    action59 = np.concatenate([hand_action, root_action, ee_action], axis=1).astype(np.float32)

    validate_rot6d(root9[:, 3:9], STATE_KEY)
    validate_rot6d(action59[:, 17:23], f"{ACTION_KEY}.root")
    validate_rot6d(action59[:, 26:32], f"{ACTION_KEY}.left_hand_pose")
    validate_rot6d(action59[:, 35:41], f"{ACTION_KEY}.right_hand_pose")
    validate_rot6d(action59[:, 44:50], f"{ACTION_KEY}.left_foot_pose")
    validate_rot6d(action59[:, 53:59], f"{ACTION_KEY}.right_foot_pose")

    return {
        STATE_KEY: state52,
        ROOT_KEY: root9,
        ACTION_KEY: action59,
        REF_BODY_KEY: ref_body29,
    }


def feature_stats(values: np.ndarray) -> dict[str, list[Any]]:
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[:, None]
    return {
        "mean": arr.mean(axis=0).astype(float).tolist(),
        "std": arr.std(axis=0).astype(float).tolist(),
        "min": arr.min(axis=0).astype(float).tolist(),
        "max": arr.max(axis=0).astype(float).tolist(),
        "q01": np.quantile(arr, 0.01, axis=0).astype(float).tolist(),
        "q99": np.quantile(arr, 0.99, axis=0).astype(float).tolist(),
        "count": [int(arr.shape[0])],
    }


def build_table(arrays: dict[str, np.ndarray], episode_index: int, global_offset: int) -> pa.Table:
    length = arrays[STATE_KEY].shape[0]
    timestamps = np.arange(length, dtype=np.float32) / float(FPS)
    frame_indices = np.arange(length, dtype=np.int64)
    global_indices = global_offset + frame_indices
    fields = [
        pa.field(STATE_KEY, pa.list_(pa.float32(), 52)),
        pa.field(ROOT_KEY, pa.list_(pa.float32(), 9)),
        pa.field(ACTION_KEY, pa.list_(pa.float32(), 59)),
        pa.field(REF_BODY_KEY, pa.list_(pa.float32(), 29)),
        pa.field("timestamp", pa.float32()),
        pa.field("frame_index", pa.int64()),
        pa.field("episode_index", pa.int64()),
        pa.field("index", pa.int64()),
        pa.field("task_index", pa.int64()),
    ]
    parquet_arrays = [
        pa.array(arrays[STATE_KEY].tolist(), type=pa.list_(pa.float32(), 52)),
        pa.array(arrays[ROOT_KEY].tolist(), type=pa.list_(pa.float32(), 9)),
        pa.array(arrays[ACTION_KEY].tolist(), type=pa.list_(pa.float32(), 59)),
        pa.array(arrays[REF_BODY_KEY].tolist(), type=pa.list_(pa.float32(), 29)),
        pa.array(timestamps, type=pa.float32()),
        pa.array(frame_indices, type=pa.int64()),
        pa.array(np.full(length, episode_index, dtype=np.int64), type=pa.int64()),
        pa.array(global_indices, type=pa.int64()),
        pa.array(np.zeros(length, dtype=np.int64), type=pa.int64()),
    ]
    return pa.Table.from_arrays(parquet_arrays, schema=pa.schema(fields))


def build_modality() -> dict[str, Any]:
    return {
        "state": {
            "rot59_left_hand": {"start": 0, "end": 7, "original_key": STATE_KEY},
            "rot59_right_hand": {"start": 7, "end": 14, "original_key": STATE_KEY},
            "rot59_left_leg": {"start": 14, "end": 20, "original_key": STATE_KEY},
            "rot59_right_leg": {"start": 20, "end": 26, "original_key": STATE_KEY},
            "rot59_waist": {"start": 26, "end": 29, "original_key": STATE_KEY},
            "rot59_left_arm": {"start": 29, "end": 36, "original_key": STATE_KEY},
            "rot59_right_arm": {"start": 36, "end": 43, "original_key": STATE_KEY},
            "rot59_root": {"start": 43, "end": 52, "original_key": STATE_KEY},
            "rot59_hand": {"start": 0, "end": 14, "original_key": STATE_KEY},
            "rot59_body": {"start": 14, "end": 43, "original_key": STATE_KEY},
        },
        "action": {
            "rot59_hand": {"start": 0, "end": 14, "original_key": ACTION_KEY},
            "rot59_root": {"start": 14, "end": 23, "original_key": ACTION_KEY},
            "rot59_left_hand_pose": {"start": 23, "end": 32, "original_key": ACTION_KEY},
            "rot59_right_hand_pose": {"start": 32, "end": 41, "original_key": ACTION_KEY},
            "rot59_left_foot_pose": {"start": 41, "end": 50, "original_key": ACTION_KEY},
            "rot59_right_foot_pose": {"start": 50, "end": 59, "original_key": ACTION_KEY},
            "ref_body29": {"start": 0, "end": 29, "original_key": REF_BODY_KEY},
        },
        "video": {"ego_view": {"original_key": VIDEO_KEY}},
        "annotation": {"human.task_description": {"original_key": "task_index"}},
    }


def build_info(
    *,
    total_episodes: int,
    total_frames: int,
    task: str,
    source_root: Path,
    mjcf_path: Path,
    target_shift: int,
    hand_target_shift: int,
    source_families: tuple[str, ...],
) -> dict[str, Any]:
    sonic_only = source_families == ("sonic",)
    return {
        "codebase_version": "v2.1",
        "robot_type": "g1_dex3_arena",
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": 1,
        "total_videos": total_episodes,
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
                    "video.codec": "h264",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "video.fps": FPS,
                    "video.channels": 3,
                    "has_audio": False,
                },
            },
            STATE_KEY: {"dtype": "float32", "shape": [52], "names": STATE52_NAMES},
            ROOT_KEY: {"dtype": "float32", "shape": [9], "names": ROOT9_NAMES},
            ACTION_KEY: {"dtype": "float32", "shape": [59], "names": ACTION59_NAMES},
            REF_BODY_KEY: {"dtype": "float32", "shape": [29], "names": REF_BODY29_NAMES},
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        },
        "script_config": {
            "source": (
                "humanoid_arena_sonic_football_to_rot6d59"
                if sonic_only
                else "humanoid_arena_npz_football_to_rot6d59"
            ),
            "schema_version": (
                "arena_football_sonic_rot6d59_v2"
                if sonic_only
                else "arena_football_rot6d59_v1"
            ),
            "source_root": str(source_root),
            "source_families": list(source_families),
            "mjcf_path": str(mjcf_path),
            "fps": FPS,
            "target_shift": target_shift,
            "hand_target_shift": hand_target_shift,
            "state": "observation.full_state_rot6d(52=hand14+body29+root9)",
            "action": "action.policy_action_rot6d59(59=hand14+next_root9+next_world_FK_ee36)",
            "hand14_source": (
                "source-family-aware provider hand targets reordered into the "
                "asymmetric rot6d59 hand order"
            ),
            "body29_source": (
                "robot_qpos_before_decimation; native SONIC IsaacLab order is "
                "reordered into rot6d59/TWIST2 MuJoCo order"
            ),
            "root9_source": "robot_root_position + robot_root_orientation(wxyz)->rot6d",
            "ee36_source": "FK on left/right wrist_yaw_link and left/right ankle_roll_link",
            "task": task,
        },
    }


def build_schema(target_shift: int, hand_target_shift: int) -> dict[str, Any]:
    return {
        ACTION_KEY: {
            "dim": 59,
            "layout": "hand14 + root9(xyz,rot6d) + 4 ee poses * 9(xyz,rot6d)",
            "names": ACTION59_NAMES,
        },
        STATE_KEY: {
            "dim": 52,
            "layout": "hand14 + body29 + root9(xyz,rot6d)",
            "names": STATE52_NAMES,
        },
        "coordinate_notes": {
            "coordinates": "Arena/SIMPLE/MuJoCo z-up world coordinates.",
            "rotation_6d": "First two rotation-matrix columns flattened as [r00,r01,r10,r11,r20,r21].",
            "action_target_timing": (
                f"root and EE targets are from frame t+{target_shift}; "
                f"hand targets are from frame t+{hand_target_shift}; tails repeat the last frame."
            ),
            "ee_body_sources": EE_BODY_NAMES,
        },
    }


def write_video(data: np.lib.npyio.NpzFile, out: Path, length: int) -> None:
    video = np.asarray(data["vision_rgb"][:length])
    if video.ndim != 4 or video.shape[1:] != (480, 640, 3):
        raise ValueError(f"Expected vision_rgb shape (T,480,640,3), got {video.shape}")
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        "640x480",
        "-r",
        str(FPS),
        "-i",
        "-",
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
        str(out),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    assert proc.stdin is not None
    try:
        for start in range(0, length, 32):
            batch = np.ascontiguousarray(video[start : start + 32])
            proc.stdin.write(batch.tobytes())
    finally:
        proc.stdin.close()
    return_code = proc.wait()
    if return_code != 0:
        raise RuntimeError(f"ffmpeg failed while writing {out}")


def episode_stats(arrays: dict[str, np.ndarray], episode_index: int, global_offset: int) -> dict[str, Any]:
    length = arrays[STATE_KEY].shape[0]
    frame_indices = np.arange(length, dtype=np.int64)
    return {
        "episode_index": episode_index,
        "stats": {
            STATE_KEY: feature_stats(arrays[STATE_KEY]),
            ROOT_KEY: feature_stats(arrays[ROOT_KEY]),
            ACTION_KEY: feature_stats(arrays[ACTION_KEY]),
            REF_BODY_KEY: feature_stats(arrays[REF_BODY_KEY]),
            "timestamp": feature_stats(frame_indices.astype(np.float32) / float(FPS)),
            "frame_index": feature_stats(frame_indices),
            "episode_index": feature_stats(np.full(length, episode_index, dtype=np.int64)),
            "index": feature_stats(global_offset + frame_indices),
            "task_index": feature_stats(np.zeros(length, dtype=np.int64)),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC, help="HumanoidArena football dataset root")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output LeRobot dataset directory")
    parser.add_argument("--mjcf", type=Path, default=DEFAULT_MJCF, help="G1 MJCF used for FK")
    parser.add_argument("--npz-index", type=int, default=None, help="Convert one sorted npz index")
    parser.add_argument("--max-episodes", type=int, default=None, help="Convert only the first N episodes")
    parser.add_argument("--max-frames", type=int, default=None, help="Cap frames per episode, useful for smoke tests")
    parser.add_argument("--target-shift", type=int, default=1, help="Future shift for root/EE/body targets")
    parser.add_argument("--hand-target-shift", type=int, default=0, help="Future shift for hand target commands")
    parser.add_argument(
        "--source-family",
        choices=("auto", "sonic", "twist2"),
        default="auto",
        help="Source controller/order; auto infers it from schema/path",
    )
    parser.add_argument("--task", default="Move toward the football and kick it.")
    parser.add_argument("--skip-videos", action="store_true", help="Write parquet/meta only")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    src = args.src.resolve()
    out = args.out.resolve()
    mjcf = args.mjcf.resolve()
    if out.exists():
        if not args.overwrite:
            raise FileExistsError(f"{out} exists; pass --overwrite to replace it")
        shutil.rmtree(out)
    if not mjcf.exists():
        raise FileNotFoundError(f"G1 MJCF does not exist: {mjcf}")

    npz_files = find_npz_files(src, args.npz_index, args.max_episodes)
    fk = G1MjcfFK(mjcf)

    all_stats_arrays: dict[str, list[np.ndarray]] = {
        STATE_KEY: [],
        ROOT_KEY: [],
        ACTION_KEY: [],
        REF_BODY_KEY: [],
        "timestamp": [],
        "frame_index": [],
        "episode_index": [],
        "index": [],
        "task_index": [],
    }
    episodes_rows: list[dict[str, Any]] = []
    episodes_stats_rows: list[dict[str, Any]] = []
    source_families: set[str] = set()
    total_frames = 0

    for episode_index, npz_path in enumerate(npz_files):
        with np.load(npz_path, allow_pickle=True) as data:
            source_family = detect_source_family(
                npz_path,
                data,
                args.source_family,
            )
            source_families.add(source_family)
            length = int(data["num_frames"])
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

            chunk = episode_index // CHUNKS_SIZE
            parquet_path = out / "data" / f"chunk-{chunk:03d}" / f"episode_{episode_index:06d}.parquet"
            parquet_path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(build_table(arrays, episode_index, total_frames), parquet_path)

            if not args.skip_videos:
                video_path = out / "videos" / f"chunk-{chunk:03d}" / VIDEO_KEY / f"episode_{episode_index:06d}.mp4"
                write_video(data, video_path, length)

            frame_indices = np.arange(length, dtype=np.int64)
            all_stats_arrays[STATE_KEY].append(arrays[STATE_KEY])
            all_stats_arrays[ROOT_KEY].append(arrays[ROOT_KEY])
            all_stats_arrays[ACTION_KEY].append(arrays[ACTION_KEY])
            all_stats_arrays[REF_BODY_KEY].append(arrays[REF_BODY_KEY])
            all_stats_arrays["timestamp"].append(frame_indices.astype(np.float32) / float(FPS))
            all_stats_arrays["frame_index"].append(frame_indices)
            all_stats_arrays["episode_index"].append(np.full(length, episode_index, dtype=np.int64))
            all_stats_arrays["index"].append(total_frames + frame_indices)
            all_stats_arrays["task_index"].append(np.zeros(length, dtype=np.int64))
            episodes_rows.append(
                {
                    "episode_index": episode_index,
                    "tasks": [args.task],
                    "length": length,
                    "source_npz": str(npz_path),
                    "source_family": source_family,
                    "source_schema_version": scalar_text(data, "schema_version"),
                }
            )
            episodes_stats_rows.append(episode_stats(arrays, episode_index, total_frames))
            total_frames += length
            print(f"[{episode_index + 1}/{len(npz_files)}] {npz_path.name}: frames={length}")

    meta = out / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    write_jsonl(meta / "tasks.jsonl", [{"task_index": 0, "task": args.task}])
    write_jsonl(meta / "episodes.jsonl", episodes_rows)
    write_jsonl(meta / "episodes_stats.jsonl", episodes_stats_rows)

    stats = {
        key: feature_stats(np.concatenate(values, axis=0))
        for key, values in all_stats_arrays.items()
    }
    write_json(meta / "stats.json", stats)
    write_json(meta / "relative_stats.json", {"__fingerprints__": {}})
    write_json(meta / "modality.json", build_modality())
    write_json(meta / "rot6d59_schema.json", build_schema(args.target_shift, args.hand_target_shift))
    write_json(
        meta / "info.json",
        build_info(
            total_episodes=len(npz_files),
            total_frames=total_frames,
            task=args.task,
            source_root=src,
            mjcf_path=mjcf,
            target_shift=args.target_shift,
            hand_target_shift=args.hand_target_shift,
            source_families=tuple(sorted(source_families)),
        ),
    )

    print(f"source={src}")
    print(f"out={out}")
    print(f"episodes={len(npz_files)} frames={total_frames} fps={FPS}")
    print(f"{STATE_KEY}=52D {ACTION_KEY}=59D")
    if args.skip_videos:
        print("videos=skipped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
