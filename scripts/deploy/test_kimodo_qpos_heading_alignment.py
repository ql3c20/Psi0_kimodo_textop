#!/usr/bin/env python3
from __future__ import annotations

import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from kimodo_qpos_heading_alignment import align_qpos_root_heading


def make_qpos(euler_xyz: np.ndarray) -> np.ndarray:
    qpos = np.zeros((len(euler_xyz), 36), dtype=np.float32)
    qpos[:, 2] = 0.8
    qpos[:, 3:7] = Rotation.from_euler("xyz", euler_xyz).as_quat(
        scalar_first=True
    )
    qpos[:, 7:] = np.linspace(-0.2, 0.2, 29, dtype=np.float32)
    return qpos


def qpos_yaw(qpos: np.ndarray) -> np.ndarray:
    rotation = Rotation.from_quat(qpos[:, 3:7], scalar_first=True).as_matrix()
    return np.arctan2(rotation[:, 1, 0], rotation[:, 0, 0])


class RootHeadingAlignmentTest(unittest.TestCase):
    def test_aligns_yaw_and_preserves_translation_tilt_and_joints(self) -> None:
        euler = np.asarray(
            [[0.12, -0.08, -1.1], [-0.15, 0.07, -0.9], [0.03, 0.05, -0.6]]
        )
        target_yaw = np.asarray([-0.1, 0.0, 0.2])
        qpos = make_qpos(euler)
        heading = np.stack((np.cos(target_yaw), np.sin(target_yaw)), axis=1)

        aligned, stats = align_qpos_root_heading(
            qpos,
            heading,
            max_correction_rad=1.5,
        )

        np.testing.assert_allclose(qpos_yaw(aligned), target_yaw, atol=1e-6)
        np.testing.assert_array_equal(aligned[:, :3], qpos[:, :3])
        np.testing.assert_array_equal(aligned[:, 7:], qpos[:, 7:])
        before_euler = Rotation.from_quat(
            qpos[:, 3:7], scalar_first=True
        ).as_euler("xyz")
        after_euler = Rotation.from_quat(
            aligned[:, 3:7], scalar_first=True
        ).as_euler("xyz")
        np.testing.assert_allclose(after_euler[:, :2], before_euler[:, :2], atol=1e-6)
        np.testing.assert_allclose(np.linalg.norm(aligned[:, 3:7], axis=1), 1.0, atol=1e-6)
        self.assertEqual(stats.frames, 3)
        self.assertGreater(stats.max_abs_correction_rad, 0.9)

    def test_handles_pi_wraparound_with_small_correction(self) -> None:
        qpos = make_qpos(np.asarray([[0.0, 0.0, np.pi - 0.02]]))
        target = -np.pi + 0.03
        heading = np.asarray([[np.cos(target), np.sin(target)]])

        aligned, stats = align_qpos_root_heading(
            qpos,
            heading,
            max_correction_rad=0.1,
        )

        np.testing.assert_allclose(qpos_yaw(aligned), [target], atol=1e-6)
        self.assertAlmostEqual(stats.max_abs_correction_rad, 0.05, places=6)

    def test_preserves_batch_shape_and_dtype(self) -> None:
        qpos = make_qpos(np.asarray([[0.0, 0.0, -0.4], [0.0, 0.0, -0.2]]))[None]
        heading = np.asarray([[[1.0, 0.0], [1.0, 0.0]]], dtype=np.float32)

        aligned, _ = align_qpos_root_heading(
            qpos,
            heading,
            max_correction_rad=1.0,
        )

        self.assertEqual(aligned.shape, qpos.shape)
        self.assertEqual(aligned.dtype, qpos.dtype)

    def test_rejects_unsafe_or_invalid_inputs(self) -> None:
        qpos = make_qpos(np.asarray([[0.0, 0.0, -1.0]]))
        with self.assertRaisesRegex(ValueError, "exceeds safety limit"):
            align_qpos_root_heading(
                qpos,
                np.asarray([[1.0, 0.0]]),
                max_correction_rad=0.5,
            )
        with self.assertRaisesRegex(ValueError, "degenerate"):
            align_qpos_root_heading(
                qpos,
                np.asarray([[0.0, 0.0]]),
                max_correction_rad=2.0,
            )


if __name__ == "__main__":
    unittest.main()
