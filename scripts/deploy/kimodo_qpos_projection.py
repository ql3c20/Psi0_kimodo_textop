#!/usr/bin/env python3
"""Optional G1 inverse-kinematics projection for exported Kimodo qpos.

Kimodo predicts unconstrained 3D joint rotations.  Exporting those rotations to
the 29 one-DoF joints of the G1 can move wrists and ankles away from the sparse
world-space constraints.  This module corrects only the exported hinge angles;
it does not modify the Kimodo sample, root trajectory, sampler, or checkpoint.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares


EE_FIELDS = (
    ("left_hand_pose", "left_wrist_yaw_link"),
    ("right_hand_pose", "right_wrist_yaw_link"),
    ("left_foot_pose", "left_ankle_roll_link"),
    ("right_foot_pose", "right_ankle_roll_link"),
)


@dataclass(frozen=True)
class ProjectionStats:
    constrained_frames: int
    before_mean_m: float
    before_max_m: float
    after_mean_m: float
    after_max_m: float
    max_joint_correction_rad: float
    elapsed_s: float


def _kimodo_xyz_to_mujoco(xyz: np.ndarray) -> np.ndarray:
    # Kimodo is y-up/z-forward; MuJoCo is z-up/x-forward.
    return np.asarray(xyz, dtype=np.float64)[..., [2, 0, 1]]


def _load_position_constraints(path: str | Path, num_frames: int) -> tuple[np.ndarray, np.ndarray]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    item = next((value for value in payload if value.get("type") == "ee-pose"), None)
    if item is None:
        raise ValueError(f"No ee-pose constraint found in {path}")

    frame_indices = np.asarray(item.get("frame_indices", []), dtype=np.int64)
    if frame_indices.ndim != 1 or frame_indices.size == 0:
        raise ValueError("ee-pose frame_indices must be a non-empty 1D array")
    if np.any(frame_indices < 0) or np.any(frame_indices >= int(num_frames)):
        raise ValueError(
            f"ee-pose frame indices {frame_indices.tolist()} are outside qpos length {num_frames}"
        )
    if np.any(frame_indices[1:] <= frame_indices[:-1]):
        raise ValueError("ee-pose frame_indices must be strictly increasing")

    positions = []
    for field, _ in EE_FIELDS:
        poses = np.asarray(item.get(field, []), dtype=np.float64)
        if poses.shape != (frame_indices.size, 6):
            raise ValueError(
                f"ee-pose {field} must have shape {(frame_indices.size, 6)}, got {poses.shape}"
            )
        positions.append(_kimodo_xyz_to_mujoco(poses[:, :3]))
    return frame_indices, np.stack(positions, axis=1)


class G1QposConstraintProjector:
    """Project sparse wrist/ankle positions onto G1 hinge-joint qpos."""

    def __init__(
        self,
        xml_path: str | Path,
        *,
        target_scale_m: float = 0.01,
        joint_regularization_rad: float = 0.35,
        max_nfev: int = 10,
        optimizer_tolerance: float = 1e-3,
    ) -> None:
        if target_scale_m <= 0.0:
            raise ValueError("target_scale_m must be positive")
        if joint_regularization_rad <= 0.0:
            raise ValueError("joint_regularization_rad must be positive")
        if max_nfev <= 0:
            raise ValueError("max_nfev must be positive")
        if optimizer_tolerance <= 0.0:
            raise ValueError("optimizer_tolerance must be positive")

        self.xml_path = Path(xml_path).expanduser().resolve()
        if not self.xml_path.is_file():
            raise FileNotFoundError(f"G1 MuJoCo XML not found: {self.xml_path}")
        self.model = mujoco.MjModel.from_xml_path(str(self.xml_path))
        if self.model.nq != 36:
            raise ValueError(f"Expected G1 nq=36, got {self.model.nq} from {self.xml_path}")
        self.data = mujoco.MjData(self.model)
        self.target_scale_m = float(target_scale_m)
        self.joint_regularization_rad = float(joint_regularization_rad)
        self.max_nfev = int(max_nfev)
        self.optimizer_tolerance = float(optimizer_tolerance)

        body_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            for _, body_name in EE_FIELDS
        ]
        if any(body_id < 0 for body_id in body_ids):
            raise ValueError(f"G1 XML is missing an end-effector body: {self.xml_path}")
        self.body_ids = np.asarray(body_ids, dtype=np.int32)
        self.lower, self.upper = self._joint_bounds()

    def _joint_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        lower = np.full(29, -np.inf, dtype=np.float64)
        upper = np.full(29, np.inf, dtype=np.float64)
        for joint_id in range(self.model.njnt):
            qpos_address = int(self.model.jnt_qposadr[joint_id])
            if 7 <= qpos_address < 36 and int(self.model.jnt_limited[joint_id]):
                lower[qpos_address - 7], upper[qpos_address - 7] = self.model.jnt_range[joint_id]
        return lower, upper

    def _positions(self, qpos: np.ndarray) -> np.ndarray:
        self.data.qpos[:] = qpos
        mujoco.mj_forward(self.model, self.data)
        return self.data.xpos[self.body_ids].copy()

    def _position_jacobian(self, qpos: np.ndarray) -> np.ndarray:
        self.data.qpos[:] = qpos
        mujoco.mj_forward(self.model, self.data)
        blocks = []
        for body_id in self.body_ids:
            jacobian_position = np.zeros((3, self.model.nv), dtype=np.float64)
            jacobian_rotation = np.zeros((3, self.model.nv), dtype=np.float64)
            mujoco.mj_jacBody(
                self.model,
                self.data,
                jacobian_position,
                jacobian_rotation,
                int(body_id),
            )
            # The free root occupies qvel[0:6]; the 29 hinge velocities follow.
            blocks.append(jacobian_position[:, 6:35])
        return np.concatenate(blocks, axis=0)

    def project(
        self,
        qpos: np.ndarray,
        constraints_path: str | Path,
    ) -> tuple[np.ndarray, ProjectionStats]:
        started = time.perf_counter()
        original = np.asarray(qpos)
        original_shape = original.shape
        if original.ndim == 3:
            if original.shape[0] != 1:
                raise ValueError(f"Qpos projection supports batch size 1, got {original.shape}")
            sequence = np.asarray(original[0], dtype=np.float64).copy()
        elif original.ndim == 2:
            sequence = np.asarray(original, dtype=np.float64).copy()
        else:
            raise ValueError(f"Expected qpos shape (T,36) or (1,T,36), got {original.shape}")
        if sequence.shape[1] != 36:
            raise ValueError(f"Expected qpos dimension 36, got {sequence.shape}")

        frame_indices, targets = _load_position_constraints(constraints_path, sequence.shape[0])
        corrections = np.zeros((frame_indices.size, 29), dtype=np.float64)
        before_errors = []

        for constraint_index, frame_index in enumerate(frame_indices):
            qpos_frame = sequence[frame_index].copy()
            reference_joints = np.clip(qpos_frame[7:36], self.lower, self.upper)
            before_errors.append(np.linalg.norm(self._positions(qpos_frame) - targets[constraint_index], axis=1))

            def residual(joints: np.ndarray) -> np.ndarray:
                candidate = qpos_frame.copy()
                candidate[7:36] = joints
                position_residual = (
                    self._positions(candidate) - targets[constraint_index]
                ).reshape(-1) / self.target_scale_m
                regularization = (
                    joints - reference_joints
                ) / self.joint_regularization_rad
                return np.concatenate((position_residual, regularization))

            def jacobian(joints: np.ndarray) -> np.ndarray:
                candidate = qpos_frame.copy()
                candidate[7:36] = joints
                position_jacobian = self._position_jacobian(candidate) / self.target_scale_m
                regularization_jacobian = np.eye(29, dtype=np.float64) / self.joint_regularization_rad
                return np.concatenate((position_jacobian, regularization_jacobian), axis=0)

            result = least_squares(
                residual,
                reference_joints,
                jac=jacobian,
                bounds=(self.lower, self.upper),
                max_nfev=self.max_nfev,
                xtol=self.optimizer_tolerance,
                ftol=self.optimizer_tolerance,
                gtol=self.optimizer_tolerance,
            )
            corrections[constraint_index] = result.x - qpos_frame[7:36]

        timeline = np.arange(sequence.shape[0], dtype=np.float64)
        interpolated = np.empty((sequence.shape[0], 29), dtype=np.float64)
        for joint_index in range(29):
            interpolated[:, joint_index] = np.interp(
                timeline,
                frame_indices.astype(np.float64),
                corrections[:, joint_index],
            )
        sequence[:, 7:36] = np.clip(
            sequence[:, 7:36] + interpolated,
            self.lower[None, :],
            self.upper[None, :],
        )

        # Measure the actual interpolated result, including limit clipping.
        final_errors = np.asarray(
            [
                np.linalg.norm(self._positions(sequence[frame_index]) - targets[index], axis=1)
                for index, frame_index in enumerate(frame_indices)
            ]
        )
        before = np.asarray(before_errors)
        stats = ProjectionStats(
            constrained_frames=int(frame_indices.size),
            before_mean_m=float(before.mean()),
            before_max_m=float(before.max()),
            after_mean_m=float(final_errors.mean()),
            after_max_m=float(final_errors.max()),
            max_joint_correction_rad=float(np.max(np.abs(interpolated))),
            elapsed_s=float(time.perf_counter() - started),
        )
        result = sequence.astype(original.dtype, copy=False)
        if len(original_shape) == 3:
            result = result[None, ...]
        return result, stats
