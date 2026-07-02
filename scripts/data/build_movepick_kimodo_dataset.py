#!/usr/bin/env python3
"""Build a Kimodo-augmented MovePick dataset.

The source MovePick parquet files keep Psi0/SIMPLE actions, but they do not
store a dense root pose trajectory. This script reconstructs a root trajectory
from the episode reset pose plus planar velocity commands, then adds:

  - observation.full_state: 14 hand + 29 body joints + 6 root
  - action.ee_constraints: root_xyzyaw + 4 EE poses = 28D in SIMPLE/MuJoCo z-up coordinates
  - action.policy_action: hand joints + root xyz/rpy + 4 EE poses = 44D
  - action.direct49: hand joints + body joints + root xyz/rpy = 49D
  - action.direct73: hand joints + body joints + root xyz/rpy + 4 EE poses = 73D

The original dataset is not modified.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[2]
VIZ_DIR = REPO_ROOT / "scripts" / "viz"
if str(VIZ_DIR) not in sys.path:
    sys.path.insert(0, str(VIZ_DIR))

try:
    from fk import G1FK as PinocchioG1FK  # noqa: E402
except ModuleNotFoundError:
    PinocchioG1FK = None
from g1 import ARM_JOINT_NAMES, HAND_JOINT_NAMES, LEG_JOINT_NAMES  # noqa: E402


DEFAULT_DATASET = REPO_ROOT / "data" / "simple" / "G1WholebodyXMovePickTeleop-v0"
DEFAULT_URDF = REPO_ROOT / "real" / "assets" / "g1" / "g1_body29_hand14.urdf"

MUJOCO_TO_KIMODO = np.array(
    [
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
    ],
    dtype=np.float64,
)

EE_CONSTRAINT_NAMES = (
    [f"root_xyzyaw.{name}" for name in ("x", "y", "z", "yaw")]
    + [f"left_hand_pose.{name}" for name in ("x", "y", "z", "roll", "pitch", "yaw")]
    + [f"right_hand_pose.{name}" for name in ("x", "y", "z", "roll", "pitch", "yaw")]
    + [f"left_foot_pose.{name}" for name in ("x", "y", "z", "roll", "pitch", "yaw")]
    + [f"right_foot_pose.{name}" for name in ("x", "y", "z", "roll", "pitch", "yaw")]
)

EE_POSE_NAMES = EE_CONSTRAINT_NAMES[4:]

POLICY_ACTION_NAMES = (
    [f"hand.{i}" for i in range(14)]
    + [f"root_state_6.{name}" for name in ("x", "y", "z", "roll", "pitch", "yaw")]
    + [f"ee_pose.{name}" for name in EE_POSE_NAMES]
)

DIRECT49_ACTION_NAMES = (
    [f"hand.{i}" for i in range(14)]
    + [f"body29.{i}" for i in range(29)]
    + [f"root_state_6.{name}" for name in ("x", "y", "z", "roll", "pitch", "yaw")]
)

DIRECT73_ACTION_NAMES = DIRECT49_ACTION_NAMES + [f"ee_pose.{name}" for name in EE_POSE_NAMES]


def load_episode_configs(meta_dir: Path) -> dict[int, dict[str, Any]]:
    configs: dict[int, dict[str, Any]] = {}
    with (meta_dir / "episodes.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            ep = json.loads(line)
            cfg = json.loads(ep["environment_config"])
            configs[int(ep["episode_index"])] = cfg
    return configs


def robot_reset_pose(cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    spatial = cfg.get("dr_state_dict", {}).get("spatial", {})
    robot = spatial.get("g1_sonic") or spatial.get("g1")
    if robot is None:
        raise ValueError("environment_config missing dr_state_dict.spatial.g1_sonic")
    pos = np.asarray(robot["position"], dtype=np.float64)
    quat_wxyz = np.asarray(robot["quaternion"], dtype=np.float64)
    return pos, quat_wxyz


def quat_wxyz_to_yaw(quat_wxyz: np.ndarray) -> float:
    quat_xyzw = np.asarray([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=np.float64)
    return float(R.from_quat(quat_xyzw).as_euler("xyz")[2])


def parse_xyz(text: str | None, default: tuple[float, float, float]) -> np.ndarray:
    if text is None:
        return np.asarray(default, dtype=np.float64)
    return np.asarray([float(v) for v in text.split()], dtype=np.float64)


def transform_from_xyz_rpy(xyz: np.ndarray, rpy: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = R.from_euler("xyz", rpy).as_matrix()
    transform[:3, 3] = xyz
    return transform


def axis_angle_transform(axis: np.ndarray, angle: float) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    norm = np.linalg.norm(axis)
    if norm > 0:
        transform[:3, :3] = R.from_rotvec(axis / norm * angle).as_matrix()
    return transform


class UrdfG1FK:
    """Small URDF FK fallback for the four frames needed by Kimodo constraints."""

    FRAME_TO_JOINT = {
        "l_ee": "left_wrist_yaw_joint",
        "r_ee": "right_wrist_yaw_joint",
        "l_foot": "left_ankle_roll_joint",
        "r_foot": "right_ankle_roll_joint",
    }

    def __init__(self, urdf_path: str, mode: str = "default") -> None:
        del mode
        root = ET.parse(urdf_path).getroot()
        links = {link.attrib["name"] for link in root.findall("link")}
        child_links: set[str] = set()
        self.children: dict[str, list[dict[str, Any]]] = {}
        self.joint_names: set[str] = set()

        for joint in root.findall("joint"):
            parent = joint.find("parent")
            child = joint.find("child")
            if parent is None or child is None:
                continue
            origin = joint.find("origin")
            axis = joint.find("axis")
            parent_link = parent.attrib["link"]
            child_link = child.attrib["link"]
            child_links.add(child_link)
            joint_info = {
                "name": joint.attrib["name"],
                "type": joint.attrib.get("type", "fixed"),
                "child": child_link,
                "origin": transform_from_xyz_rpy(
                    parse_xyz(origin.attrib.get("xyz") if origin is not None else None, (0.0, 0.0, 0.0)),
                    parse_xyz(origin.attrib.get("rpy") if origin is not None else None, (0.0, 0.0, 0.0)),
                ),
                "axis": parse_xyz(axis.attrib.get("xyz") if axis is not None else None, (1.0, 0.0, 0.0)),
            }
            self.children.setdefault(parent_link, []).append(joint_info)
            self.joint_names.add(joint_info["name"])

        root_candidates = links - child_links
        self.root_link = "pelvis" if "pelvis" in root_candidates or "pelvis" in links else sorted(root_candidates)[0]

    def fk(self, joint_dict: dict[str, float], base_frame: str = "world") -> dict[str, dict[str, Any]]:
        if base_frame != "world":
            raise ValueError("UrdfG1FK fallback only supports base_frame='world'")

        targets = set(self.FRAME_TO_JOINT.values())
        found: dict[str, np.ndarray] = {}

        def visit(link: str, link_transform: np.ndarray) -> None:
            for joint in self.children.get(link, []):
                angle = 0.0
                if joint["type"] in {"revolute", "continuous", "prismatic"}:
                    angle = float(joint_dict.get(joint["name"], 0.0))
                if joint["type"] == "prismatic":
                    motion = np.eye(4, dtype=np.float64)
                    motion[:3, 3] = joint["axis"] * angle
                elif joint["type"] in {"revolute", "continuous"}:
                    motion = axis_angle_transform(joint["axis"], angle)
                else:
                    motion = np.eye(4, dtype=np.float64)

                joint_transform = link_transform @ joint["origin"] @ motion
                if joint["name"] in targets:
                    found[joint["name"]] = joint_transform.copy()
                visit(joint["child"], joint_transform)

        visit(self.root_link, np.eye(4, dtype=np.float64))
        missing = targets - set(found)
        if missing:
            raise RuntimeError(f"URDF FK missing target joints: {sorted(missing)}")

        out = {}
        for frame, joint_name in self.FRAME_TO_JOINT.items():
            transform = found[joint_name]
            out[frame] = {
                "matrix": transform.tolist(),
                "position": transform[:3, 3].tolist(),
            }
        return out


def make_fk(urdf_path: Path):
    if PinocchioG1FK is not None:
        return PinocchioG1FK(str(urdf_path), mode="default")
    print("[warn] pinocchio is not installed; using built-in URDF FK fallback.", file=sys.stderr)
    return UrdfG1FK(str(urdf_path), mode="default")


def mujoco_xyz_to_kimodo(xyz_m: np.ndarray) -> np.ndarray:
    return np.asarray([xyz_m[1], xyz_m[2], xyz_m[0]], dtype=np.float64)


def rot_mujoco_to_kimodo(rot_m: np.ndarray) -> np.ndarray:
    return MUJOCO_TO_KIMODO @ rot_m @ MUJOCO_TO_KIMODO.T


def yaw_from_rot_kimodo(rot_k: np.ndarray) -> float:
    forward = rot_k @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return float(np.arctan2(forward[0], forward[2]))


def pose_mujoco_to_kimodo(xyz_m: np.ndarray, rot_m: np.ndarray) -> np.ndarray:
    xyz_k = mujoco_xyz_to_kimodo(xyz_m)
    rot_k = rot_mujoco_to_kimodo(rot_m)
    rpy_k = R.from_matrix(rot_k).as_euler("xyz")
    return np.concatenate([xyz_k, rpy_k]).astype(np.float32)


def pose_mujoco_xyzrpy(xyz_m: np.ndarray, rot_m: np.ndarray) -> np.ndarray:
    rpy_m = R.from_matrix(rot_m).as_euler("xyz")
    return np.concatenate([xyz_m, rpy_m]).astype(np.float32)


def unwrap_yaw(yaw: np.ndarray, reset_yaw: float) -> np.ndarray:
    if len(yaw) == 0:
        return yaw.astype(np.float64)
    unwrapped = np.unwrap(yaw.astype(np.float64))
    return unwrapped + (reset_yaw - unwrapped[0])


def integrate_root_xyz(
    reset_pos: np.ndarray,
    height: np.ndarray,
    yaw: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    fps: float,
) -> np.ndarray:
    root = np.zeros((len(height), 3), dtype=np.float64)
    if len(height) == 0:
        return root
    root[0] = [reset_pos[0], reset_pos[1], float(height[0])]
    dt = 1.0 / fps
    for i in range(1, len(height)):
        prev_yaw = yaw[i - 1]
        v_world = np.array(
            [
                np.cos(prev_yaw) * vx[i - 1] - np.sin(prev_yaw) * vy[i - 1],
                np.sin(prev_yaw) * vx[i - 1] + np.cos(prev_yaw) * vy[i - 1],
            ],
            dtype=np.float64,
        )
        root[i, 0] = root[i - 1, 0] + v_world[0] * dt
        root[i, 1] = root[i - 1, 1] + v_world[1] * dt
        root[i, 2] = float(height[i])
    return root


def joint_dict_from_frame(row: pd.Series) -> dict[str, float]:
    action = np.asarray(row["action"], dtype=np.float64)
    joint_dict: dict[str, float] = {}
    joint_dict.update(zip(HAND_JOINT_NAMES, action[0:14]))
    joint_dict.update(zip(ARM_JOINT_NAMES, action[14:28]))
    joint_dict.update(zip(LEG_JOINT_NAMES, np.asarray(row["observation.leg_joints"], dtype=np.float64)))
    return {k: float(v) for k, v in joint_dict.items()}


def feature_stats(values: np.ndarray) -> dict[str, list[float]]:
    values = np.asarray(values, dtype=np.float32)
    return {
        "mean": values.mean(axis=0).astype(float).tolist(),
        "std": values.std(axis=0).astype(float).tolist(),
        "min": values.min(axis=0).astype(float).tolist(),
        "max": values.max(axis=0).astype(float).tolist(),
        "q01": np.quantile(values, 0.01, axis=0).astype(float).tolist(),
        "q99": np.quantile(values, 0.99, axis=0).astype(float).tolist(),
        "count": [int(values.shape[0])],
    }


def build_ee_constraint(
    fk: Any,
    row: pd.Series,
    root_xyz_m: np.ndarray,
    root_rpy_m: np.ndarray,
) -> np.ndarray:
    root_rot_m = R.from_euler("xyz", root_rpy_m).as_matrix()
    root_xyzyaw = np.asarray(
        [root_xyz_m[0], root_xyz_m[1], root_xyz_m[2], root_rpy_m[2]],
        dtype=np.float32,
    )

    fk_out = fk.fk(joint_dict_from_frame(row), base_frame="world")
    ee_parts = []
    for name in ("l_ee", "r_ee", "l_foot", "r_foot"):
        local_xyz = np.asarray(fk_out[name]["position"], dtype=np.float64)
        local_rot = np.asarray(fk_out[name]["matrix"], dtype=np.float64)[:3, :3]
        world_xyz = root_xyz_m + root_rot_m @ local_xyz
        world_rot = root_rot_m @ local_rot
        ee_parts.append(pose_mujoco_xyzrpy(world_xyz, world_rot))

    return np.concatenate([root_xyzyaw, *ee_parts]).astype(np.float32)


def augment_episode(
    data_path: Path,
    out_path: Path,
    cfg: dict[str, Any],
    fk: Any,
    fps: float,
) -> dict[str, np.ndarray]:
    df = pd.read_parquet(data_path)
    actions = np.stack(df["action"].to_numpy()).astype(np.float32)
    hand = np.stack(df["observation.hand_joints"].to_numpy()).astype(np.float32)
    arm = np.stack(df["observation.arm_joints"].to_numpy()).astype(np.float32)
    leg = np.stack(df["observation.leg_joints"].to_numpy()).astype(np.float32)

    reset_pos, reset_quat = robot_reset_pose(cfg)
    reset_yaw = quat_wxyz_to_yaw(reset_quat)

    root_rpy = actions[:, 28:31].astype(np.float64)
    root_yaw = unwrap_yaw(root_rpy[:, 2], reset_yaw)
    root_rpy[:, 2] = root_yaw
    root_height = actions[:, 31].astype(np.float64)
    root_vx = actions[:, 32].astype(np.float64)
    root_vy = actions[:, 33].astype(np.float64)
    root_xyz = integrate_root_xyz(reset_pos, root_height, root_yaw, root_vx, root_vy, fps)
    root_state_6 = np.concatenate(
        [
            root_xyz.astype(np.float32),
            root_rpy.astype(np.float32),
        ],
        axis=1,
    )
    body_29 = np.concatenate([leg, arm], axis=1).astype(np.float32)
    full_state = np.concatenate([hand, body_29, root_state_6], axis=1).astype(np.float32)

    constraints = np.stack(
        [
            build_ee_constraint(fk, row, root_xyz[i], root_rpy[i])
            for i, (_, row) in enumerate(df.iterrows())
        ],
        axis=0,
    ).astype(np.float32)
    policy_action = np.concatenate([actions[:, 0:14], root_state_6, constraints[:, 4:28]], axis=1).astype(np.float32)
    direct49 = np.concatenate([actions[:, 0:14], body_29, root_state_6], axis=1).astype(np.float32)
    direct73 = np.concatenate([direct49, constraints[:, 4:28]], axis=1).astype(np.float32)

    df["observation.full_state"] = list(full_state)
    df["observation.root_state_6"] = list(root_state_6.astype(np.float32))
    df["action.ee_constraints"] = list(constraints)
    df["action.ee_root_xyzyaw"] = list(constraints[:, :4].astype(np.float32))
    df["action.policy_action"] = list(policy_action)
    df["action.direct49"] = list(direct49)
    df["action.direct73"] = list(direct73)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, out_path)
    return {
        "observation.full_state": full_state,
        "observation.root_state_6": root_state_6.astype(np.float32),
        "action.ee_constraints": constraints,
        "action.ee_root_xyzyaw": constraints[:, :4].astype(np.float32),
        "action.policy_action": policy_action,
        "action.direct49": direct49,
        "action.direct73": direct73,
    }


def copy_and_patch_meta(src: Path, out: Path) -> None:
    src_meta = src / "meta"
    out_meta = out / "meta"
    out_meta.mkdir(parents=True, exist_ok=True)
    for item in src_meta.iterdir():
        if item.is_file():
            shutil.copy2(item, out_meta / item.name)

    info_path = out_meta / "info.json"
    info = json.loads(info_path.read_text())
    features = info.setdefault("features", {})
    features["observation.full_state"] = {
        "dtype": "float32",
        "shape": [49],
        "names": ["14 hand + 29 body joints + 6 root"],
    }
    features["observation.root_state_6"] = {
        "dtype": "float32",
        "shape": [6],
        "names": ["root_x", "root_y", "root_z", "roll", "pitch", "yaw"],
    }
    features["action.ee_constraints"] = {
        "dtype": "float32",
        "shape": [28],
        "names": EE_CONSTRAINT_NAMES,
    }
    features["action.ee_root_xyzyaw"] = {
        "dtype": "float32",
        "shape": [4],
        "names": ["x", "y", "z", "yaw"],
    }
    features["action.policy_action"] = {
        "dtype": "float32",
        "shape": [44],
        "names": POLICY_ACTION_NAMES,
    }
    features["action.direct49"] = {
        "dtype": "float32",
        "shape": [49],
        "names": DIRECT49_ACTION_NAMES,
    }
    features["action.direct73"] = {
        "dtype": "float32",
        "shape": [73],
        "names": DIRECT73_ACTION_NAMES,
    }
    info_path.write_text(json.dumps(info, indent=4) + "\n")

    schema = {
        "coordinate_notes": {
            "source": "SIMPLE/MuJoCo world root reconstructed from reset pose plus vx/vy command integration.",
            "dataset_coordinates": "SIMPLE/MuJoCo z-up coordinates. Positions are stored as [x_mujoco, y_mujoco, z_mujoco].",
            "kimodo_adapter": "Convert to Kimodo y-up right before generation with [x_k, y_k, z_k] = [y_mujoco, z_mujoco, x_mujoco].",
            "ee_constraints": "root_xyzyaw followed by left/right hand pose and left/right foot pose, each EE pose is xyz+rpy in SIMPLE/MuJoCo world coordinates.",
        },
        "observation.full_state": {
            "dim": 49,
            "layout": "hand14 + body29(leg15+arm14) + root6(xyz,rpy)",
        },
        "action.ee_constraints": {
            "dim": 28,
            "names": EE_CONSTRAINT_NAMES,
        },
        "action.policy_action": {
            "dim": 44,
            "layout": "hand14 + root6(xyz,rpy) + ee_poses24",
            "names": POLICY_ACTION_NAMES,
        },
        "action.direct49": {
            "dim": 49,
            "layout": "hand14 + body29(leg15+arm14) + root6(xyz,rpy)",
            "names": DIRECT49_ACTION_NAMES,
        },
        "action.direct73": {
            "dim": 73,
            "layout": "hand14 + body29(leg15+arm14) + root6(xyz,rpy) + ee_poses24",
            "names": DIRECT73_ACTION_NAMES,
        },
    }
    (out_meta / "kimodo_schema.json").write_text(json.dumps(schema, indent=2) + "\n")


def ensure_video_link(src: Path, out: Path) -> None:
    link = out / "videos"
    if link.exists() or link.is_symlink():
        return
    target = os.path.relpath(src / "videos", out)
    link.symlink_to(target)


def load_episode_stats(meta_dir: Path) -> dict[int, dict[str, Any]]:
    episode_stats: dict[int, dict[str, Any]] = {}
    path = meta_dir / "episodes_stats.jsonl"
    if not path.exists():
        return episode_stats
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            episode_stats[int(item["episode_index"])] = item
    return episode_stats


def write_updated_stats(
    src: Path,
    out: Path,
    per_episode_new: dict[int, dict[str, np.ndarray]],
) -> None:
    if not per_episode_new:
        return

    out_meta = out / "meta"
    stats_path = out_meta / "stats.json"
    stats = json.loads(stats_path.read_text())

    new_keys = sorted(next(iter(per_episode_new.values())).keys())
    for key in new_keys:
        stats[key] = feature_stats(np.concatenate([ep[key] for ep in per_episode_new.values()], axis=0))
    stats_path.write_text(json.dumps(stats, indent=4) + "\n")

    episodes_stats = load_episode_stats(src / "meta")
    for episode_index, arrays in per_episode_new.items():
        item = episodes_stats.setdefault(episode_index, {"episode_index": episode_index, "stats": {}})
        ep_stats = item.setdefault("stats", {})
        for key in new_keys:
            ep_stats[key] = feature_stats(arrays[key])

    with (out_meta / "episodes_stats.jsonl").open("w", encoding="utf-8") as f:
        for episode_index in sorted(episodes_stats):
            f.write(json.dumps(episodes_stats[episode_index], separators=(",", ":")) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, default=DEFAULT_DATASET, help="Source MovePick LeRobot dataset.")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output dataset dir. Default: <src>/kimodo_augmented_v2.",
    )
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF, help="G1 29+14 URDF for FK.")
    parser.add_argument("--fps", type=float, default=None, help="Override dataset fps.")
    parser.add_argument("--limit-episodes", type=int, default=None, help="Only convert the first N episodes.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    src = args.src.resolve()
    out = (args.out or (src / "kimodo_augmented_v2")).resolve()
    if out == src:
        raise ValueError("--out must not be the same as --src")
    if out.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {out}. Pass --overwrite to update it.")

    info = json.loads((src / "meta" / "info.json").read_text())
    fps = float(args.fps or info["fps"])
    episode_cfgs = load_episode_configs(src / "meta")
    fk = make_fk(args.urdf)

    copy_and_patch_meta(src, out)
    ensure_video_link(src, out)

    data_files = sorted((src / "data").glob("chunk-*/episode_*.parquet"))
    if args.limit_episodes is not None:
        data_files = data_files[: args.limit_episodes]

    per_episode_new: dict[int, dict[str, np.ndarray]] = {}
    for data_path in tqdm(data_files, desc="Augmenting MovePick"):
        episode_index = int(data_path.stem.split("_")[-1])
        rel = data_path.relative_to(src)
        out_path = out / rel
        if out_path.exists() and not args.overwrite:
            continue
        per_episode_new[episode_index] = augment_episode(data_path, out_path, episode_cfgs[episode_index], fk, fps)

    write_updated_stats(src, out, per_episode_new)

    print(f"Saved Kimodo-augmented dataset to: {out}")
    print("Added columns: observation.full_state, observation.root_state_6, action.ee_constraints, action.policy_action, action.direct49, action.direct73")


if __name__ == "__main__":
    main()
