#!/usr/bin/env python3
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import mujoco
import numpy as np

from kimodo_qpos_projection import EE_FIELDS, G1QposConstraintProjector


KIMODO_ROOT = Path("/home/ubuntu/yzh/kimodo_my")
G1_XML = KIMODO_ROOT / "kimodo/assets/skeletons/g1skel34/xml/g1.xml"


def mujoco_xyz_to_kimodo(xyz: np.ndarray) -> np.ndarray:
    return np.asarray(xyz)[..., [1, 2, 0]]


class G1QposConstraintProjectorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.projector = G1QposConstraintProjector(G1_XML)
        self.model = mujoco.MjModel.from_xml_path(str(G1_XML))
        self.data = mujoco.MjData(self.model)
        self.body_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body)
            for _, body in EE_FIELDS
        ]

    def positions(self, qpos: np.ndarray) -> np.ndarray:
        self.data.qpos[:] = qpos
        mujoco.mj_forward(self.model, self.data)
        return self.data.xpos[self.body_ids].copy()

    def write_constraints(self, path: Path, qpos: np.ndarray) -> None:
        frames = [0, 2, 4]
        item: dict[str, object] = {
            "type": "ee-pose",
            "frame_indices": frames,
            "root_xyzyaw": [[0.0, 0.8, 0.0, 0.0] for _ in frames],
        }
        positions = np.stack([self.positions(qpos[index]) for index in frames])
        positions[:, 0, 0] += 0.03
        for body_index, (field, _) in enumerate(EE_FIELDS):
            values = []
            for xyz in mujoco_xyz_to_kimodo(positions[:, body_index]):
                values.append([*xyz.tolist(), 0.0, 0.0, 0.0])
            item[field] = values
        path.write_text(json.dumps([item]), encoding="utf-8")

    def test_projection_reduces_error_and_preserves_root(self) -> None:
        qpos = np.zeros((1, 5, 36), dtype=np.float32)
        qpos[:, :, 2] = 0.8
        qpos[:, :, 3] = 1.0
        with tempfile.TemporaryDirectory() as tmp:
            constraints = Path(tmp) / "constraints.json"
            self.write_constraints(constraints, qpos[0])
            projected, stats = self.projector.project(qpos, constraints)

        self.assertEqual(projected.shape, qpos.shape)
        self.assertEqual(projected.dtype, qpos.dtype)
        np.testing.assert_array_equal(projected[..., :7], qpos[..., :7])
        self.assertGreater(stats.before_mean_m, 0.005)
        self.assertLess(stats.after_mean_m, stats.before_mean_m * 0.2)
        self.assertLess(stats.after_max_m, 0.005)

    def test_rejects_multi_sample_batch(self) -> None:
        qpos = np.zeros((2, 5, 36), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "batch size 1"):
            self.projector.project(qpos, "unused.json")


if __name__ == "__main__":
    unittest.main()
