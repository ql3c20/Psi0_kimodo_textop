#!/usr/bin/env python3
"""Convert real SONIC+mocap data to the canonical fullstate rot6d59 contract.

The canonical contract is taken directly from ``--reference`` (by default the
20260615 task1 rot6d59 dataset).  The task1/task3/task4 reference datasets have
identical feature, modality, and rot6d59 schemas.

For every output row t:

* observation is the measured real-robot state at t;
* hand action is the logged WBC hand target at t;
* root action target is the measured mocap root at t+1;
* body target is selected by ``--body-target-source``: either measured body
  state at t+1 (the backward-compatible default) or logged WBC body qpos at t;
* all four world-frame EE targets are FK(body_target, measured_root[t+1])
  using the canonical G1 body29+hand14 URDF.  The hand14 values are carried as
  a separate action term and do not affect the wrist/ankle FK frames;
* the final source sample is used as the last target and is not emitted as an
  observation, matching ``mujoco_csv2lerobot_fullstate.py``.

The source FPS is preserved by default.  If ``--output-fps`` explicitly asks
for another rate, low-dimensional streams and video are resampled before the
one-frame target shift.  Quaternion streams use SLERP; other continuous
streams use linear interpolation.
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation, Slerp


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SRC = REPO_ROOT / "data/output/lqb_20260831_to_20260901_realtask2"
DEFAULT_OUT = REPO_ROOT / "data/output/lqb_20260831_to_20260901_realtask2_rot6d59"
DEFAULT_REFERENCE = REPO_ROOT / "data/output/lqb_fullstate_20260615_task1_rot6d59"
DEFAULT_URDF = (
    REPO_ROOT
    / "third_party/SIMPLE/third_party/gear_sonic/data/robot_model/model_data/g1"
    / "g1_29dof_with_hand.urdf"
)

VIDEO_KEY = "observation.images.ego_view"
STATE_KEY = "observation.full_state_rot6d"
ACTION_KEY = "action.policy_action_rot6d59"
REF_BODY_KEY = "action.ref_body29"
SONIC_STATE_KEY = "observation.sonic_state"
SONIC_ACTION_KEY = "action.sonic"
JOINT_KEY = "observation.joint43"
JOINT_REF_KEY = "action.joint43_ref"
JOINT_LAST_KEY = "action.joint43_last"
PSI_STATE_KEY = "states"
PSI_ACTION_KEY = "action"
PSI_LAST_ACTION_KEY = "last_action"
ROOT_KEY = "observation.root_pose_rot6d"

BODY_TARGET_MEASURED_NEXT = "measured-next"
BODY_TARGET_WBC = "wbc"
BODY_TARGET_SOURCES = (BODY_TARGET_MEASURED_NEXT, BODY_TARGET_WBC)
FK_LIMIT_REJECT = "reject"
FK_LIMIT_ALLOW = "allow"
FK_LIMIT_POLICIES = (FK_LIMIT_REJECT, FK_LIMIT_ALLOW)

SOURCE_STATE_KEY = "observation.state"
SOURCE_ACTION_KEY = "action.wbc"
SOURCE_ROOT_POS_KEY = "observation.mocap_root_position"
SOURCE_ROOT_QUAT_KEY = "observation.mocap_root_orientation_xyzw"
SOURCE_GRAVITY_KEY = "observation.projected_gravity"
SOURCE_MOTION_TOKEN_KEY = "action.motion_token"

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

JOINT43_ORDER = HAND14_ORDER + BODY29_ORDER
ARM14_ORDER = BODY29_ORDER[15:22] + BODY29_ORDER[22:29]
EE_FRAME_TO_JOINT = {
    "left_hand_pose": "left_wrist_yaw_joint",
    "right_hand_pose": "right_wrist_yaw_joint",
    "left_foot_pose": "left_ankle_roll_joint",
    "right_foot_pose": "right_ankle_roll_joint",
}

# Optional legacy compatibility: ``RobotModel.cache_forward_kinematics`` clips
# every configuration with a 1e-6 margin.  Most supplemental limits equal the
# SONIC URDF limits; these are the deliberate G1SupplementalInfo overrides.
# Physical real-data mode does not apply them and instead rejects violations of
# the wider mechanical URDF limits.
CANONICAL_LIMIT_OVERRIDES = {
    "left_shoulder_roll_joint": (0.19, 2.2515),
    "right_shoulder_roll_joint": (-2.2515, -0.19),
    "right_hand_thumb_1_joint": (-0.72431163, 1.04719755),
    "right_hand_thumb_2_joint": (0.0, 1.74532925),
    "right_hand_index_0_joint": (-1.57079632, 0.0),
    "right_hand_index_1_joint": (-1.74532925, 0.0),
    "right_hand_middle_0_joint": (-1.57079632, 0.0),
    "right_hand_middle_1_joint": (-1.74532925, 0.0),
}
FK_CLIP_MARGIN = 1e-6

FLOAT32_KEYS = {STATE_KEY, ACTION_KEY, PSI_STATE_KEY, PSI_ACTION_KEY, PSI_LAST_ACTION_KEY, ROOT_KEY}
FLOAT64_WIDTHS = {
    REF_BODY_KEY: 29,
    SONIC_STATE_KEY: 46,
    SONIC_ACTION_KEY: 78,
    JOINT_KEY: 43,
    JOINT_REF_KEY: 43,
    JOINT_LAST_KEY: 43,
}
FLOAT32_WIDTHS = {
    STATE_KEY: 52,
    ACTION_KEY: 59,
    PSI_STATE_KEY: 32,
    PSI_ACTION_KEY: 36,
    PSI_LAST_ACTION_KEY: 36,
    ROOT_KEY: 9,
}
LOWDIM_ORDER = [
    STATE_KEY,
    ACTION_KEY,
    REF_BODY_KEY,
    SONIC_STATE_KEY,
    SONIC_ACTION_KEY,
    JOINT_KEY,
    JOINT_REF_KEY,
    JOINT_LAST_KEY,
    PSI_STATE_KEY,
    PSI_ACTION_KEY,
    PSI_LAST_ACTION_KEY,
    "timestamp",
    "frame_index",
    "episode_index",
    "index",
    "task_index",
    ROOT_KEY,
]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=4, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def stack_column(table: pa.Table, key: str, width: int) -> np.ndarray:
    values = np.asarray(table[key].to_pylist(), dtype=np.float64)
    if values.shape != (table.num_rows, width):
        raise ValueError(f"{key}: expected {(table.num_rows, width)}, got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError(f"{key}: contains NaN or Inf")
    return values


def indices_for_names(info: dict[str, Any], key: str, names: list[str]) -> np.ndarray:
    source_names = info["features"][key].get("names")
    if not isinstance(source_names, list):
        raise ValueError(f"{key}: metadata names must be a list")
    if len(source_names) != len(set(source_names)):
        raise ValueError(f"{key}: duplicate metadata names")
    by_name = {name: index for index, name in enumerate(source_names)}
    missing = [name for name in names if name not in by_name]
    if missing:
        raise ValueError(f"{key}: missing canonical joints: {missing}")
    return np.asarray([by_name[name] for name in names], dtype=np.int64)


def matrix_to_rot6d(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim == 2:
        return matrix[:, :2].reshape(6)
    return matrix[:, :, :2].reshape(len(matrix), 6)


def rot6d_to_matrix(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    single = values.ndim == 1
    values = np.atleast_2d(values)
    c0 = values[:, [0, 2, 4]]
    c1 = values[:, [1, 3, 5]]
    c0 /= np.maximum(np.linalg.norm(c0, axis=1, keepdims=True), 1e-12)
    c1 -= np.sum(c0 * c1, axis=1, keepdims=True) * c0
    c1 /= np.maximum(np.linalg.norm(c1, axis=1, keepdims=True), 1e-12)
    matrix = np.stack([c0, c1, np.cross(c0, c1)], axis=-1)
    return matrix[0] if single else matrix


def validate_rot6d(values: np.ndarray, label: str, atol: float = 1e-5) -> None:
    values = np.asarray(values, dtype=np.float64)
    c0 = values[:, [0, 2, 4]]
    c1 = values[:, [1, 3, 5]]
    error = max(
        float(np.max(np.abs(np.linalg.norm(c0, axis=1) - 1.0))),
        float(np.max(np.abs(np.linalg.norm(c1, axis=1) - 1.0))),
        float(np.max(np.abs(np.sum(c0 * c1, axis=1)))),
    )
    if not np.isfinite(error) or error > atol:
        raise ValueError(f"{label}: invalid rot6d, max error={error:.6g}")


class CanonicalG1UrdfFK:
    """Self-contained FK using the SONIC G1 URDF and canonical EE frames."""

    def __init__(self, urdf_path: Path, clipping: str = "none") -> None:
        if clipping not in {"none", "urdf", "robot-model"}:
            raise ValueError(f"Unsupported FK clipping mode: {clipping}")
        self.clipping = clipping
        root = ET.parse(urdf_path).getroot()
        links = {element.attrib["name"] for element in root.findall("link")}
        child_links: set[str] = set()
        self.children: dict[str, list[dict[str, Any]]] = {}
        self.urdf_joint_limits: dict[str, tuple[float, float]] = {}
        for joint in root.findall("joint"):
            parent = joint.find("parent")
            child = joint.find("child")
            if parent is None or child is None:
                continue
            origin = joint.find("origin")
            axis = joint.find("axis")
            xyz = np.fromstring(origin.attrib.get("xyz", "0 0 0"), sep=" ") if origin is not None else np.zeros(3)
            rpy = np.fromstring(origin.attrib.get("rpy", "0 0 0"), sep=" ") if origin is not None else np.zeros(3)
            axis_xyz = np.fromstring(axis.attrib.get("xyz", "1 0 0"), sep=" ") if axis is not None else np.array([1.0, 0.0, 0.0])
            origin_transform = np.eye(4, dtype=np.float64)
            origin_transform[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
            origin_transform[:3, 3] = xyz
            parent_link, child_link = parent.attrib["link"], child.attrib["link"]
            limit = joint.find("limit")
            if limit is not None and "lower" in limit.attrib and "upper" in limit.attrib:
                self.urdf_joint_limits[joint.attrib["name"]] = (
                    float(limit.attrib["lower"]),
                    float(limit.attrib["upper"]),
                )
            child_links.add(child_link)
            self.children.setdefault(parent_link, []).append(
                {
                    "name": joint.attrib["name"],
                    "type": joint.attrib.get("type", "fixed"),
                    "child": child_link,
                    "origin": origin_transform,
                    "axis": axis_xyz,
                }
            )
        candidates = links - child_links
        self.root_link = "pelvis" if "pelvis" in links else sorted(candidates)[0]
        self.fk_joint_limits = dict(self.urdf_joint_limits)
        if clipping == "robot-model":
            self.fk_joint_limits.update(CANONICAL_LIMIT_OVERRIDES)

    def validate_body_limits(self, ref_joint43: np.ndarray) -> None:
        body29 = np.asarray(ref_joint43, dtype=np.float64)[:, 14:43]
        violations = []
        for index, name in enumerate(BODY29_ORDER):
            lower, upper = self.urdf_joint_limits[name]
            values = body29[:, index]
            mask = (values < lower) | (values > upper)
            if np.any(mask):
                violations.append(
                    {
                        "joint": name,
                        "count": int(np.count_nonzero(mask)),
                        "range": [float(values.min()), float(values.max())],
                        "urdf_limits": [lower, upper],
                    }
                )
        if violations:
            raise ValueError(
                "Reference body joints violate physical URDF limits; refusing to hide "
                f"the source error with clipping: {violations}"
            )

    def frame_transforms(self, joint_values: dict[str, float]) -> dict[str, np.ndarray]:
        wanted = set(EE_FRAME_TO_JOINT.values())
        found: dict[str, np.ndarray] = {}

        def visit(link: str, link_transform: np.ndarray) -> None:
            for joint in self.children.get(link, []):
                value = float(joint_values.get(joint["name"], 0.0))
                if self.clipping in {"urdf", "robot-model"} and joint["name"] in self.fk_joint_limits:
                    lower, upper = self.fk_joint_limits[joint["name"]]
                    value = float(
                        np.clip(value, lower + FK_CLIP_MARGIN, upper - FK_CLIP_MARGIN)
                    )
                motion = np.eye(4, dtype=np.float64)
                if joint["type"] in {"revolute", "continuous"}:
                    axis = joint["axis"] / max(float(np.linalg.norm(joint["axis"])), 1e-12)
                    motion[:3, :3] = Rotation.from_rotvec(axis * value).as_matrix()
                elif joint["type"] == "prismatic":
                    motion[:3, 3] = joint["axis"] * value
                current = link_transform @ joint["origin"] @ motion
                if joint["name"] in wanted:
                    found[joint["name"]] = current.copy()
                visit(joint["child"], current)

        visit(self.root_link, np.eye(4, dtype=np.float64))
        missing = wanted - set(found)
        if missing:
            raise RuntimeError(f"URDF FK missing canonical frames: {sorted(missing)}")
        return found


def resample_linear(values: np.ndarray, old_time: np.ndarray, new_time: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return np.stack([np.interp(new_time, old_time, values[:, index]) for index in range(values.shape[1])], axis=1)


def resample_quaternion_xyzw(values: np.ndarray, old_time: np.ndarray, new_time: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    values /= np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)
    return Slerp(old_time, Rotation.from_quat(values))(new_time).as_quat()


def load_resampled_source(table: pa.Table, input_fps: int, output_fps: int) -> dict[str, np.ndarray]:
    source = {
        SOURCE_STATE_KEY: stack_column(table, SOURCE_STATE_KEY, 43),
        SOURCE_ACTION_KEY: stack_column(table, SOURCE_ACTION_KEY, 43),
        SOURCE_ROOT_POS_KEY: stack_column(table, SOURCE_ROOT_POS_KEY, 3),
        SOURCE_ROOT_QUAT_KEY: stack_column(table, SOURCE_ROOT_QUAT_KEY, 4),
        SOURCE_GRAVITY_KEY: stack_column(table, SOURCE_GRAVITY_KEY, 3),
        SOURCE_MOTION_TOKEN_KEY: stack_column(table, SOURCE_MOTION_TOKEN_KEY, 64),
    }
    quaternion_norm_error = float(
        np.max(np.abs(np.linalg.norm(source[SOURCE_ROOT_QUAT_KEY], axis=1) - 1.0))
    )
    if quaternion_norm_error > 1e-4:
        raise ValueError(
            f"{SOURCE_ROOT_QUAT_KEY}: non-unit quaternion, max norm error="
            f"{quaternion_norm_error:.6g}"
        )

    # The capture metadata names the upstream Motive convention, but the
    # exported xyz/xyzw streams must already form the z-up world_R_pelvis pose
    # consumed by the canonical MuJoCo/SIMPLE contract.  Check that convention
    # independently against the robot's projected-gravity observation.  A raw
    # y-up quaternion or an inverse/body-to-world convention fails by ~90 deg.
    root_rotation = Rotation.from_quat(source[SOURCE_ROOT_QUAT_KEY]).as_matrix()
    predicted_gravity = np.einsum(
        "nji,j->ni", root_rotation, np.array([0.0, 0.0, -1.0], dtype=np.float64)
    )
    measured_gravity = source[SOURCE_GRAVITY_KEY]
    cosine = np.sum(predicted_gravity * measured_gravity, axis=1) / np.maximum(
        np.linalg.norm(predicted_gravity, axis=1)
        * np.linalg.norm(measured_gravity, axis=1),
        1e-12,
    )
    median_gravity_error_deg = float(
        np.degrees(np.median(np.arccos(np.clip(cosine, -1.0, 1.0))))
    )
    if median_gravity_error_deg > 20.0:
        raise ValueError(
            "Mocap xyzw orientation is inconsistent with z-up projected gravity: "
            f"median error={median_gravity_error_deg:.3f} deg"
        )
    if input_fps == output_fps:
        source["task_index"] = np.asarray(table["task_index"].to_numpy(), dtype=np.int64)
        return source
    old_time = np.arange(table.num_rows, dtype=np.float64) / input_fps
    new_count = int(np.floor(old_time[-1] * output_fps + 1e-9)) + 1
    new_time = np.arange(new_count, dtype=np.float64) / output_fps
    result = {
        key: resample_linear(values, old_time, new_time)
        for key, values in source.items()
        if key != SOURCE_ROOT_QUAT_KEY
    }
    result[SOURCE_ROOT_QUAT_KEY] = resample_quaternion_xyzw(
        source[SOURCE_ROOT_QUAT_KEY], old_time, new_time
    )
    gravity = result[SOURCE_GRAVITY_KEY]
    result[SOURCE_GRAVITY_KEY] = gravity / np.maximum(
        np.linalg.norm(gravity, axis=1, keepdims=True), 1e-12
    )
    result["task_index"] = np.full(new_count, int(np.asarray(table["task_index"])[0]), dtype=np.int64)
    return result


def compute_reference_ee36(
    fk: CanonicalG1UrdfFK,
    ref_joint43: np.ndarray,
    ref_root_position: np.ndarray,
    ref_root_rotation: np.ndarray,
    limit_policy: str = FK_LIMIT_REJECT,
) -> np.ndarray:
    if limit_policy not in FK_LIMIT_POLICIES:
        raise ValueError(f"Unsupported FK limit policy: {limit_policy}")
    if fk.clipping == "none" and limit_policy == FK_LIMIT_REJECT:
        fk.validate_body_limits(ref_joint43)
    output = np.empty((len(ref_joint43), 36), dtype=np.float64)
    for frame in range(len(ref_joint43)):
        local = fk.frame_transforms(
            {name: float(value) for name, value in zip(JOINT43_ORDER, ref_joint43[frame])}
        )
        parts = []
        for joint_name in EE_FRAME_TO_JOINT.values():
            placement = local[joint_name]
            world_position = ref_root_position[frame] + ref_root_rotation[frame] @ placement[:3, 3]
            world_rotation = ref_root_rotation[frame] @ placement[:3, :3]
            parts.append(np.concatenate([world_position, matrix_to_rot6d(world_rotation)]))
        output[frame] = np.concatenate(parts)
    return output


def derive_loco4(root9: np.ndarray, fps: int) -> np.ndarray:
    rotation = rot6d_to_matrix(root9[:, 3:9])
    yaw = Rotation.from_matrix(rotation).as_euler("xyz")[:, 2]
    delta_world = root9[1:, :2] - root9[:-1, :2]
    current_yaw = yaw[:-1]
    vx = np.cos(current_yaw) * delta_world[:, 0] + np.sin(current_yaw) * delta_world[:, 1]
    vy = -np.sin(current_yaw) * delta_world[:, 0] + np.cos(current_yaw) * delta_world[:, 1]
    yaw_delta = np.arctan2(np.sin(yaw[1:] - current_yaw), np.cos(yaw[1:] - current_yaw))
    return np.stack([vx * fps, vy * fps, yaw_delta * fps, yaw[1:]], axis=1)


def convert_episode(
    table: pa.Table,
    input_fps: int,
    output_fps: int,
    fk: CanonicalG1UrdfFK,
    state_joint43_indices: np.ndarray,
    action_hand_indices: np.ndarray,
    action_body_indices: np.ndarray,
    body_target_source: str = BODY_TARGET_MEASURED_NEXT,
    fk_limit_policy: str = FK_LIMIT_REJECT,
) -> tuple[dict[str, np.ndarray], int]:
    source = load_resampled_source(table, input_fps, output_fps)
    samples = len(source[SOURCE_STATE_KEY])
    if samples < 2:
        raise ValueError("Episode has fewer than two samples after resampling")

    joint43 = source[SOURCE_STATE_KEY][:, state_joint43_indices]
    hand_target = source[SOURCE_ACTION_KEY][:, action_hand_indices]
    if body_target_source == BODY_TARGET_MEASURED_NEXT:
        body_target = joint43[1:, 14:43]
    elif body_target_source == BODY_TARGET_WBC:
        body_target = source[SOURCE_ACTION_KEY][:-1, action_body_indices]
    else:
        raise ValueError(f"Unsupported body target source: {body_target_source}")
    root_position = source[SOURCE_ROOT_POS_KEY]
    root_rotation = Rotation.from_quat(source[SOURCE_ROOT_QUAT_KEY]).as_matrix()
    root9 = np.concatenate([root_position, matrix_to_rot6d(root_rotation)], axis=1)
    ref_joint43 = np.concatenate([hand_target[:-1], body_target], axis=1)
    ref_root9 = root9[1:]
    ref_ee36 = compute_reference_ee36(
        fk,
        ref_joint43,
        root_position[1:],
        root_rotation[1:],
        limit_policy=fk_limit_policy,
    )

    state52 = np.concatenate([joint43[:-1], root9[:-1]], axis=1)
    action59 = np.concatenate([hand_target[:-1], ref_root9, ref_ee36], axis=1)
    sonic_state = np.concatenate(
        [source[SOURCE_STATE_KEY][:-1], source[SOURCE_GRAVITY_KEY][:-1]], axis=1
    )
    sonic_action = np.concatenate(
        [source[SOURCE_MOTION_TOKEN_KEY][:-1], hand_target[:-1]], axis=1
    )
    current_rpy = Rotation.from_matrix(root_rotation[:-1]).as_euler("xyz")
    next_rpy = Rotation.from_matrix(root_rotation[1:]).as_euler("xyz")
    arms_current = joint43[:-1, 29:43]
    arms_target = ref_joint43[:, 29:43]
    psi_state = np.concatenate(
        [joint43[:-1, :14], arms_current, current_rpy, root_position[:-1, 2:3]], axis=1
    )
    psi_action = np.concatenate(
        [
            hand_target[:-1],
            arms_target,
            next_rpy,
            root_position[1:, 2:3],
            derive_loco4(root9, output_fps),
        ],
        axis=1,
    )

    arrays = {
        STATE_KEY: state52.astype(np.float32),
        ACTION_KEY: action59.astype(np.float32),
        REF_BODY_KEY: ref_joint43[:, 14:43].astype(np.float64),
        SONIC_STATE_KEY: sonic_state.astype(np.float64),
        SONIC_ACTION_KEY: sonic_action.astype(np.float64),
        JOINT_KEY: joint43[:-1].astype(np.float64),
        JOINT_REF_KEY: ref_joint43.astype(np.float64),
        JOINT_LAST_KEY: ref_joint43.astype(np.float64),
        PSI_STATE_KEY: psi_state.astype(np.float32),
        PSI_ACTION_KEY: psi_action.astype(np.float32),
        PSI_LAST_ACTION_KEY: psi_action.astype(np.float32),
        ROOT_KEY: root9[:-1].astype(np.float32),
        "timestamp": (np.arange(samples - 1, dtype=np.float32) / output_fps),
        "task_index": source["task_index"][:-1].astype(np.int64),
    }
    validate_rot6d(arrays[STATE_KEY][:, 46:52], f"{STATE_KEY}.root")
    for label, start in (
        ("root", 17),
        ("left hand", 26),
        ("right hand", 35),
        ("left foot", 44),
        ("right foot", 53),
    ):
        validate_rot6d(arrays[ACTION_KEY][:, start : start + 6], f"{ACTION_KEY}.{label}")
    if not np.allclose(arrays[ACTION_KEY][:-1, 14:23], arrays[STATE_KEY][1:, 43:52], atol=1e-6):
        raise AssertionError("Internal canonical t+1 root alignment failure")
    for key, values in arrays.items():
        if not np.isfinite(values).all():
            raise ValueError(f"Converted {key} contains NaN or Inf")
    return arrays, samples


def vector_array(values: np.ndarray, dtype: pa.DataType) -> pa.Array:
    return pa.array(values.tolist(), type=pa.list_(dtype))


def build_table(arrays: dict[str, np.ndarray], episode: int, offset: int) -> pa.Table:
    length = len(arrays[STATE_KEY])
    values: dict[str, pa.Array] = {}
    for key, width in FLOAT32_WIDTHS.items():
        if arrays[key].shape != (length, width):
            raise ValueError(f"{key}: expected {(length, width)}, got {arrays[key].shape}")
        values[key] = vector_array(arrays[key], pa.float32())
    for key, width in FLOAT64_WIDTHS.items():
        if arrays[key].shape != (length, width):
            raise ValueError(f"{key}: expected {(length, width)}, got {arrays[key].shape}")
        values[key] = vector_array(arrays[key], pa.float64())
    values.update(
        {
            "timestamp": pa.array(arrays["timestamp"], type=pa.float32()),
            "frame_index": pa.array(np.arange(length), type=pa.int64()),
            "episode_index": pa.array(np.full(length, episode), type=pa.int64()),
            "index": pa.array(offset + np.arange(length), type=pa.int64()),
            "task_index": pa.array(arrays["task_index"], type=pa.int64()),
        }
    )
    return pa.Table.from_arrays([values[key] for key in LOWDIM_ORDER], names=LOWDIM_ORDER)


def feature_stats(values: np.ndarray) -> dict[str, list[Any]]:
    values = np.asarray(values)
    if values.ndim == 1:
        values = values[:, None]
    return {
        "mean": values.mean(axis=0).astype(float).tolist(),
        "std": values.std(axis=0).astype(float).tolist(),
        "min": values.min(axis=0).astype(float).tolist(),
        "max": values.max(axis=0).astype(float).tolist(),
        "q01": np.quantile(values, 0.01, axis=0).astype(float).tolist(),
        "q99": np.quantile(values, 0.99, axis=0).astype(float).tolist(),
        "count": [int(len(values))],
    }


def convert_video(src: Path, dst: Path, frames: int, output_fps: int) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(src),
            "-vf",
            f"fps={output_fps},trim=end_frame={frames},setpts=PTS-STARTPTS",
            "-an",
            "-r",
            str(output_fps),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(dst),
        ],
        check=True,
    )
    probe = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=r_frame_rate,nb_frames",
            "-of",
            "csv=p=0",
            str(dst),
        ],
        text=True,
    ).strip().split(",")
    if probe != [f"{output_fps}/1", str(frames)]:
        raise ValueError(f"{dst}: video probe {probe}, expected {output_fps} FPS / {frames} frames")


def validate_contracts(
    source_info: dict[str, Any], reference_info: dict[str, Any], reference_modality: dict[str, Any]
) -> None:
    required = {
        SOURCE_STATE_KEY: 43,
        SOURCE_ACTION_KEY: 43,
        SOURCE_ROOT_POS_KEY: 3,
        SOURCE_ROOT_QUAT_KEY: 4,
        SOURCE_GRAVITY_KEY: 3,
        SOURCE_MOTION_TOKEN_KEY: 64,
    }
    missing = [key for key in [VIDEO_KEY, *required] if key not in source_info.get("features", {})]
    if missing:
        raise ValueError(f"Source is missing required terms: {missing}")
    for key, width in required.items():
        if source_info["features"][key].get("shape") != [width]:
            raise ValueError(f"{key}: expected shape [{width}]")
    expected_features = [VIDEO_KEY, *LOWDIM_ORDER]
    if list(reference_info["features"]) != expected_features:
        raise ValueError("Reference feature order differs from the canonical task1/task3/task4 contract")
    required_groups = {
        "rot59_left_hand",
        "rot59_right_hand",
        "rot59_root",
        "groot_left_leg",
        "groot_projected_gravity",
    }
    if not required_groups.issubset(reference_modality.get("state", {})):
        raise ValueError("Reference modality is not the canonical multichain rot6d59 contract")
    config = source_info.get("script_config", {})
    expected_coordinates = {
        "mocap_position_order": "xyz",
        "mocap_quaternion_order": "xyzw",
        "mocap_position_mode": "episode-local-xy",
    }
    mismatch = {
        key: (config.get(key), expected)
        for key, expected in expected_coordinates.items()
        if config.get(key) != expected
    }
    if mismatch:
        raise ValueError(f"Unsupported source coordinate contract: {mismatch}")


def patch_reference_features(reference_info: dict[str, Any], output_fps: int) -> dict[str, Any]:
    features = copy.deepcopy(reference_info["features"])
    features[VIDEO_KEY]["info"]["video.fps"] = output_fps
    return features


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument(
        "--fk-clipping",
        choices=("none", "urdf", "robot-model"),
        default="none",
        help=(
            "Default none: do not clip; --fk-limit-policy controls handling of violations. "
            "urdf clips to the mechanical limits declared by the selected URDF. "
            "robot-model reproduces the legacy bundle's supplemental safety clipping."
        ),
    )
    parser.add_argument(
        "--body-target-source",
        choices=BODY_TARGET_SOURCES,
        default=BODY_TARGET_MEASURED_NEXT,
        help=(
            "Body29 used for action/FK: measured-next uses observation.state[t+1] "
            "(backward-compatible); wbc uses the logged action.wbc body qpos at t. "
            "Both modes use measured mocap root[t+1]."
        ),
    )
    parser.add_argument(
        "--fk-limit-policy",
        choices=FK_LIMIT_POLICIES,
        default=FK_LIMIT_REJECT,
        help=(
            "With --fk-clipping none, reject (default) fails on body targets outside "
            "mechanical URDF limits; allow preserves those raw targets for mathematical "
            "FK. WBC qpos targets in real logs can require allow."
        ),
    )
    parser.add_argument("--output-fps", type=int, default=None, help="Default: source dataset FPS (no resampling).")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--skip-videos", action="store_true", help="Testing only; output is not trainable.")
    args = parser.parse_args()

    src, out = args.src.resolve(), args.out.resolve()
    reference, urdf = args.reference.resolve(), args.urdf.resolve()
    if src == out:
        raise ValueError("--src and --out must differ")
    for path in (src, reference):
        if not path.is_dir():
            raise FileNotFoundError(path)
    if not urdf.is_file():
        raise FileNotFoundError(urdf)
    if out.exists():
        if not args.overwrite:
            raise FileExistsError(f"{out} exists; pass --overwrite to replace it")
        shutil.rmtree(out)

    source_info = read_json(src / "meta/info.json")
    reference_info = read_json(reference / "meta/info.json")
    reference_modality = read_json(reference / "meta/modality.json")
    validate_contracts(source_info, reference_info, reference_modality)
    input_fps = int(source_info["fps"])
    output_fps = int(args.output_fps or input_fps)
    if input_fps <= 0 or output_fps <= 0:
        raise ValueError("FPS must be positive")
    if args.fk_clipping != "none" and args.fk_limit_policy != FK_LIMIT_REJECT:
        raise ValueError(
            "--fk-limit-policy allow is only meaningful with --fk-clipping none"
        )
    if args.fk_limit_policy == FK_LIMIT_ALLOW:
        print(
            "WARNING: preserving body targets outside mechanical URDF limits for FK; "
            "no joint clipping will be applied."
        )

    state_joint43_indices = indices_for_names(source_info, SOURCE_STATE_KEY, JOINT43_ORDER)
    action_hand_indices = indices_for_names(source_info, SOURCE_ACTION_KEY, HAND14_ORDER)
    action_body_indices = indices_for_names(source_info, SOURCE_ACTION_KEY, BODY29_ORDER)
    fk = CanonicalG1UrdfFK(urdf, clipping=args.fk_clipping)
    files = sorted((src / "data").glob("chunk-*/episode_*.parquet"))
    if len(files) != int(source_info["total_episodes"]):
        raise ValueError(
            f"Found {len(files)} episodes; source info.json declares {source_info['total_episodes']}"
        )
    if args.max_episodes is not None:
        if args.max_episodes <= 0:
            raise ValueError("--max-episodes must be positive")
        files = files[: args.max_episodes]

    (out / "meta").mkdir(parents=True)
    source_episodes = {int(row["episode_index"]): row for row in read_jsonl(src / "meta/episodes.jsonl")}
    episodes_meta: list[dict[str, Any]] = []
    episodes_stats: list[dict[str, Any]] = []
    lighting_meta: list[dict[str, Any]] = []
    all_values: dict[str, list[np.ndarray]] = {
        key: []
        for key in [
            STATE_KEY,
            ACTION_KEY,
            REF_BODY_KEY,
            SONIC_STATE_KEY,
            SONIC_ACTION_KEY,
            JOINT_KEY,
            JOINT_REF_KEY,
            JOINT_LAST_KEY,
            PSI_STATE_KEY,
            PSI_ACTION_KEY,
            PSI_LAST_ACTION_KEY,
            "timestamp",
            ROOT_KEY,
        ]
    }
    offset = 0
    for episode, source_path in enumerate(files):
        table = pq.read_table(source_path)
        arrays, resampled_samples = convert_episode(
            table,
            input_fps,
            output_fps,
            fk,
            state_joint43_indices,
            action_hand_indices,
            action_body_indices,
            args.body_target_source,
            args.fk_limit_policy,
        )
        output_table = build_table(arrays, episode, offset)
        destination = out / "data" / f"chunk-{episode // 1000:03d}" / f"episode_{episode:06d}.parquet"
        destination.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(output_table, destination)

        source_episode = int(source_path.stem.rsplit("_", 1)[1])
        if not args.skip_videos:
            source_video = (
                src
                / "videos"
                / f"chunk-{source_episode // 1000:03d}"
                / VIDEO_KEY
                / f"episode_{source_episode:06d}.mp4"
            )
            output_video = (
                out
                / "videos"
                / f"chunk-{episode // 1000:03d}"
                / VIDEO_KEY
                / f"episode_{episode:06d}.mp4"
            )
            convert_video(source_video, output_video, output_table.num_rows, output_fps)

        episodes_meta.append(
            {
                "episode_index": episode,
                "tasks": source_episodes[source_episode]["tasks"],
                "length": output_table.num_rows,
            }
        )
        episode_stats = {key: feature_stats(arrays[key]) for key in all_values}
        episodes_stats.append({"episode_index": episode, "stats": episode_stats})
        lighting_meta.append(
            {
                "episode_index": episode,
                "lighting": {"randomized": False, "rig": "recorded_real"},
                "recording_name": f"source_episode_{source_episode:06d}",
            }
        )
        for key in all_values:
            all_values[key].append(arrays[key])
        offset += output_table.num_rows
        print(
            f"[{episode + 1}/{len(files)}] {source_path.name}: "
            f"{table.num_rows}@{input_fps}Hz -> {resampled_samples}@{output_fps}Hz "
            f"-> {output_table.num_rows} training rows"
        )

    shutil.copy2(src / "meta/tasks.jsonl", out / "meta/tasks.jsonl")
    write_jsonl(out / "meta/episodes.jsonl", episodes_meta)
    write_jsonl(out / "meta/episodes_stats.jsonl", episodes_stats)
    write_jsonl(out / "meta/lighting.jsonl", lighting_meta)
    write_json(out / "meta/modality.json", reference_modality)
    write_json(out / "meta/relative_stats.json", {"__fingerprints__": {}})
    write_json(
        out / "meta/stats.json",
        {key: feature_stats(np.concatenate(values)) for key, values in all_values.items()},
    )
    schema = copy.deepcopy(read_json(reference / "meta/rot6d59_schema.json"))
    schema["reference_contract"] = str(reference)
    schema["sources_read_only"] = [str(src)]
    schema["coordinate_notes"] = {
        "coordinates": (
            "Episode-local mocap xyz with xyzw world_R_pelvis orientation; z-up convention "
            "validated against observation.projected_gravity."
        ),
        "rotation_6d": "First two matrix columns flattened [r00,r01,r10,r11,r20,r21].",
        "action_target_timing": (
            "hand=WBC target[t]; body=measured state[t+1]; root=mocap[t+1]; "
            "4EE=FK(body, root)"
            if args.body_target_source == BODY_TARGET_MEASURED_NEXT
            else "hand/body=WBC qpos target[t]; root=mocap[t+1]; 4EE=FK(body, root)"
        ),
        "fk": f"SONIC G1 URDF with clipping={args.fk_clipping}: {urdf}",
        "fk_limit_policy": args.fk_limit_policy,
        "resampling": (
            f"disabled; native {input_fps}Hz preserved"
            if input_fps == output_fps
            else f"{input_fps}Hz source -> {output_fps}Hz output"
        ),
        "terminal_handling": "last resampled source is target only; output has N-1 rows",
    }
    write_json(out / "meta/rot6d59_schema.json", schema)

    info = copy.deepcopy(source_info)
    info.update(
        {
            "fps": output_fps,
            "total_episodes": len(files),
            "total_frames": offset,
            "total_videos": 0 if args.skip_videos else len(files),
            "total_chunks": (len(files) + 999) // 1000,
            "splits": {"train": f"0:{len(files)}"},
            "features": patch_reference_features(reference_info, output_fps),
        }
    )
    info["script_config"] = {
        "source": "real_sonic_mocap_canonical_fullstate",
        "schema_version": "real_rot6d59_v5",
        "source_dataset": str(src),
        "reference_contract": str(reference),
        "source_fps": input_fps,
        "output_fps": output_fps,
        "urdf": str(urdf),
        "chains": reference_info["script_config"].get("chains", {}),
        "body_target_source": args.body_target_source,
        "fk_limit_policy": args.fk_limit_policy,
        "action_joint43_source": (
            "action.wbc hand14[t] + measured body29[t+1]"
            if args.body_target_source == BODY_TARGET_MEASURED_NEXT
            else "action.wbc hand14[t] + action.wbc body29[t]"
        ),
        "action_root_source": "mocap root9[t+1]",
        "action_ee_source": (
            f"SONIC URDF FK of body29 source={args.body_target_source} "
            "anchored at mocap root9[t+1]; hand14 is not used by wrist/ankle FK; "
            f"fk_clipping={args.fk_clipping}; fk_limit_policy={args.fk_limit_policy}"
        ),
        "rotation_6d": "[r00,r01,r10,r11,r20,r21]",
        "drop_terminal_source_observation": True,
    }
    write_json(out / "meta/info.json", info)
    print(f"Saved canonical dataset: {len(files)} episodes / {offset} frames / {output_fps}Hz -> {out}")
    if args.skip_videos:
        print("WARNING: --skip-videos output is validation-only and not trainable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
