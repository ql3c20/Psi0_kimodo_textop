#!/usr/bin/env python3
"""Optional root-heading alignment for exported MuJoCo qpos.

Kimodo represents global heading separately from its predicted pelvis rotation.
Sparse heading constraints can therefore be correct while the pelvis quaternion
used by the MuJoCo export points in a different direction.  This module rotates
only the free-root orientation about the MuJoCo world Z axis so its planar
heading matches Kimodo's explicit ``[cos, sin]`` heading.  Translation, tilt,
and all hinge-joint angles remain unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class RootHeadingAlignmentStats:
    frames: int
    mean_abs_correction_rad: float
    max_abs_correction_rad: float


def _wrap_angle(angle: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(angle), np.cos(angle))


def align_qpos_root_heading(
    qpos: np.ndarray,
    global_root_heading: np.ndarray,
    *,
    max_correction_rad: float,
) -> tuple[np.ndarray, RootHeadingAlignmentStats]:
    """Align MuJoCo root yaw to Kimodo ``[cos, sin]`` heading.

    Supports qpos shaped ``(T, 36)`` or ``(1, T, 36)``.  The batch form is
    deliberately limited to one sample because the online server exports one
    motion per request.
    """

    if not np.isfinite(max_correction_rad) or max_correction_rad <= 0.0:
        raise ValueError("max_correction_rad must be finite and positive")

    original = np.asarray(qpos)
    original_shape = original.shape
    if original.ndim == 3:
        if original.shape[0] != 1:
            raise ValueError(
                f"Root-heading alignment supports batch size 1, got {original.shape}"
            )
        sequence = np.asarray(original[0], dtype=np.float64).copy()
    elif original.ndim == 2:
        sequence = np.asarray(original, dtype=np.float64).copy()
    else:
        raise ValueError(f"Expected qpos shape (T,36) or (1,T,36), got {original.shape}")
    if sequence.shape[1] != 36:
        raise ValueError(f"Expected qpos dimension 36, got {sequence.shape}")
    if not np.all(np.isfinite(sequence)):
        raise ValueError("qpos contains NaN or Inf")

    heading = np.asarray(global_root_heading)
    if heading.ndim == 3:
        if heading.shape[0] != 1:
            raise ValueError(
                "Root-heading alignment supports one heading sample, "
                f"got {heading.shape}"
            )
        heading = heading[0]
    if heading.shape != (sequence.shape[0], 2):
        raise ValueError(
            "global_root_heading must have shape "
            f"{(sequence.shape[0], 2)}, got {heading.shape}"
        )
    heading = np.asarray(heading, dtype=np.float64)
    if not np.all(np.isfinite(heading)):
        raise ValueError("global_root_heading contains NaN or Inf")
    heading_norm = np.linalg.norm(heading, axis=1)
    if np.any(heading_norm <= 1e-6):
        bad_frame = int(np.flatnonzero(heading_norm <= 1e-6)[0])
        raise ValueError(f"global_root_heading is degenerate at frame {bad_frame}")
    target_yaw = np.arctan2(heading[:, 1], heading[:, 0])

    root_quat_wxyz = sequence[:, 3:7]
    quat_norm = np.linalg.norm(root_quat_wxyz, axis=1)
    if np.any(quat_norm <= 1e-6):
        bad_frame = int(np.flatnonzero(quat_norm <= 1e-6)[0])
        raise ValueError(f"Root quaternion is degenerate at frame {bad_frame}")
    root_quat_wxyz = root_quat_wxyz / quat_norm[:, None]
    root_rotation = Rotation.from_quat(root_quat_wxyz, scalar_first=True).as_matrix()

    # MuJoCo uses X forward and Z up.  The projected forward axis supplies a
    # stable yaw except when the pelvis forward axis is almost vertical.
    forward_xy = root_rotation[:, :2, 0]
    forward_norm = np.linalg.norm(forward_xy, axis=1)
    if np.any(forward_norm <= 1e-6):
        bad_frame = int(np.flatnonzero(forward_norm <= 1e-6)[0])
        raise ValueError(f"Root forward axis is vertical at frame {bad_frame}")
    current_yaw = np.arctan2(forward_xy[:, 1], forward_xy[:, 0])
    correction = _wrap_angle(target_yaw - current_yaw)
    max_abs_correction = float(np.max(np.abs(correction), initial=0.0))
    if max_abs_correction > max_correction_rad + 1e-9:
        frame = int(np.argmax(np.abs(correction)))
        raise ValueError(
            "Root-heading correction exceeds safety limit: "
            f"frame={frame} correction={correction[frame]:.6f}rad "
            f"limit={max_correction_rad:.6f}rad"
        )

    world_yaw_rotation = Rotation.from_euler("z", correction).as_matrix()
    aligned_rotation = world_yaw_rotation @ root_rotation
    aligned_quat_wxyz = Rotation.from_matrix(aligned_rotation).as_quat(
        scalar_first=True
    )

    # q and -q encode the same rotation.  Keep signs continuous so downstream
    # interpolation or logging never sees artificial quaternion jumps.
    if np.dot(aligned_quat_wxyz[0], root_quat_wxyz[0]) < 0.0:
        aligned_quat_wxyz[0] *= -1.0
    for frame in range(1, aligned_quat_wxyz.shape[0]):
        if np.dot(aligned_quat_wxyz[frame - 1], aligned_quat_wxyz[frame]) < 0.0:
            aligned_quat_wxyz[frame] *= -1.0
    sequence[:, 3:7] = aligned_quat_wxyz

    result = sequence.astype(original.dtype, copy=False)
    if len(original_shape) == 3:
        result = result[None, ...]
    stats = RootHeadingAlignmentStats(
        frames=int(sequence.shape[0]),
        mean_abs_correction_rad=float(np.mean(np.abs(correction))),
        max_abs_correction_rad=max_abs_correction,
    )
    return result, stats
