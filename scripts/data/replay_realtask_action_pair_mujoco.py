#!/usr/bin/env python3
"""Replay measured states or WBC targets from two LeRobot episodes in MuJoCo."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import av
import mujoco
import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = REPO_ROOT / "real/assets/g1/g1_sonic_observation_replay_scene.xml"
DEFAULT_OUTPUT = REPO_ROOT / "outputs/visualizations/realtask2_episode_000000_mujoco"
LEGACY_FULL_STATE_KEY = "observation.full_state_rot6d"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    family = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{family}", size)


def find_episode(root: Path, episode: int) -> Path:
    paths = sorted((root / "data").glob(f"chunk-*/episode_{episode:06d}.parquet"))
    if not paths:
        raise FileNotFoundError(f"episode_{episode:06d}.parquet not found below {root}")
    return paths[0]


def feature_names(root: Path, key: str) -> list[str]:
    with (root / "meta/info.json").open() as file:
        info = json.load(file)
    names = info["features"][key]["names"]
    if not isinstance(names, list):
        raise TypeError(f"{key} must have a list of joint names")
    return names


def dataset_info(root: Path) -> dict:
    with (root / "meta/info.json").open() as file:
        return json.load(file)


def canonical_joint_name(name: str) -> str:
    for prefix in ("hand.", "body.", "joint43."):
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


def rot6d_columns_to_quaternion_xyzw(rot6d: np.ndarray) -> np.ndarray:
    """Convert [r00,r01,r10,r11,r20,r21] to normalized XYZW quaternions."""
    first = rot6d[:, [0, 2, 4]]
    second = rot6d[:, [1, 3, 5]]
    first /= np.maximum(np.linalg.norm(first, axis=1, keepdims=True), 1e-12)
    second -= np.sum(first * second, axis=1, keepdims=True) * first
    second /= np.maximum(np.linalg.norm(second, axis=1, keepdims=True), 1e-12)
    third = np.cross(first, second)
    matrices = np.stack([first, second, third], axis=2)
    return Rotation.from_matrix(matrices).as_quat()


def fixed_list_to_numpy(column: object) -> np.ndarray:
    values = column.combine_chunks().values.to_numpy(zero_copy_only=False)
    width = column.type.list_size
    return np.asarray(values, dtype=np.float64).reshape(-1, width)


def load_episode(
    root: Path, episode: int, joint_key: str = "action.wbc"
) -> dict[str, np.ndarray | list[str] | str]:
    info = dataset_info(root)
    features = info.get("features", {})
    if joint_key == "observation.state" and joint_key not in features:
        if LEGACY_FULL_STATE_KEY not in features:
            raise KeyError(f"{root} has neither {joint_key} nor {LEGACY_FULL_STATE_KEY}")
        names = feature_names(root, LEGACY_FULL_STATE_KEY)
        index_by_name = {name: index for index, name in enumerate(names)}
        root_position_names = ["root.x", "root.y", "root.z"]
        root_rotation_names = [
            "root.r00",
            "root.r01",
            "root.r10",
            "root.r11",
            "root.r20",
            "root.r21",
        ]
        missing = [
            name for name in root_position_names + root_rotation_names if name not in index_by_name
        ]
        if missing:
            raise ValueError(f"{LEGACY_FULL_STATE_KEY} is missing root fields: {missing}")
        joint_indices = [
            index for index, name in enumerate(names) if not name.startswith("root.")
        ]
        if len(joint_indices) != 43:
            raise ValueError(f"Expected 43 legacy robot joints, got {len(joint_indices)}")
        table = pq.read_table(
            find_episode(root, episode), columns=[LEGACY_FULL_STATE_KEY, "timestamp"]
        )
        full_state = fixed_list_to_numpy(table[LEGACY_FULL_STATE_KEY])
        position_indices = [index_by_name[name] for name in root_position_names]
        rotation_indices = [index_by_name[name] for name in root_rotation_names]
        return {
            "joint_source": LEGACY_FULL_STATE_KEY,
            "joint_names": [canonical_joint_name(names[index]) for index in joint_indices],
            "action": full_state[:, joint_indices],
            "root_pos": full_state[:, position_indices],
            "root_quat_xyzw": rot6d_columns_to_quaternion_xyzw(
                full_state[:, rotation_indices]
            ),
            "timestamp": np.asarray(table["timestamp"].to_numpy(), dtype=np.float64),
        }
    table = pq.read_table(
        find_episode(root, episode),
        columns=[
            joint_key,
            "observation.mocap_root_position",
            "observation.mocap_root_orientation_xyzw",
            "timestamp",
        ],
    )
    return {
        "joint_source": joint_key,
        "joint_names": feature_names(root, joint_key),
        "action": fixed_list_to_numpy(table[joint_key]),
        "root_pos": fixed_list_to_numpy(table["observation.mocap_root_position"]),
        "root_quat_xyzw": fixed_list_to_numpy(table["observation.mocap_root_orientation_xyzw"]),
        "timestamp": np.asarray(table["timestamp"].to_numpy(), dtype=np.float64),
    }


class G1ReplayRenderer:
    def __init__(self, model_path: Path, width: int = 640, height: int = 480) -> None:
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, width=width, height=height)
        self.camera = mujoco.MjvCamera()
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.camera.distance = 2.65
        self.camera.azimuth = 135.0
        self.camera.elevation = -12.0
        free_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "floating_base_joint"
        )
        if free_id < 0:
            raise ValueError("G1 model does not contain floating_base_joint")
        self.free_qpos_adr = int(self.model.jnt_qposadr[free_id])

    def set_pose(
        self,
        joint_names: list[str],
        joint_positions: np.ndarray,
        root_pos: np.ndarray,
        root_quat_xyzw: np.ndarray,
    ) -> None:
        adr = self.free_qpos_adr
        self.data.qpos[adr : adr + 3] = root_pos
        self.data.qpos[adr + 3 : adr + 7] = root_quat_xyzw[[3, 0, 1, 2]]
        for name, value in zip(joint_names, joint_positions):
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id < 0:
                raise KeyError(f"MuJoCo model is missing action joint: {name}")
            qpos_adr = int(self.model.jnt_qposadr[joint_id])
            self.data.qpos[qpos_adr] = value
        mujoco.mj_forward(self.model, self.data)

    def render(self, lookat: np.ndarray) -> np.ndarray:
        self.camera.lookat[:] = lookat
        self.renderer.update_scene(self.data, camera=self.camera)
        return self.renderer.render().copy()

    def close(self) -> None:
        self.renderer.close()


class VideoWriter:
    def __init__(self, path: Path, width: int, height: int, fps: int) -> None:
        self.container = av.open(str(path), mode="w")
        self.stream = self.container.add_stream("libx264", rate=fps)
        self.stream.width = width
        self.stream.height = height
        self.stream.pix_fmt = "yuv420p"
        self.stream.options = {"crf": "18", "preset": "medium"}

    def write(self, image: np.ndarray) -> None:
        frame = av.VideoFrame.from_ndarray(image, format="rgb24")
        for packet in self.stream.encode(frame):
            self.container.mux(packet)

    def close(self) -> None:
        for packet in self.stream.encode():
            self.container.mux(packet)
        self.container.close()


def labeled_frame(
    rendered: np.ndarray,
    label: str,
    frame: int,
    total: int,
    timestamp: float,
    root_pos: np.ndarray,
) -> np.ndarray:
    canvas = Image.new("RGB", (rendered.shape[1], rendered.shape[0] + 76), "#F3F5F8")
    canvas.paste(Image.fromarray(rendered), (0, 76))
    draw = ImageDraw.Draw(canvas)
    draw.text((18, 10), label, fill="#16233B", font=font(23, True))
    draw.text(
        (18, 43),
        f"frame {frame:03d}/{total - 1:03d}  t={timestamp:5.2f}s  "
        f"root=({root_pos[0]:+.3f}, {root_pos[1]:+.3f}, {root_pos[2]:+.3f}) m",
        fill="#536174",
        font=font(14),
    )
    return np.asarray(canvas)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--left-label", default="realtask2")
    parser.add_argument("--right-label", default="realtask2_new")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--joint-source",
        choices=("observation.state", "action.wbc"),
        default="observation.state",
        help="Measured collection pose by default; action.wbc is only the WBC command target.",
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--save-snapshots",
        action="store_true",
        help="Also save first, middle, and last frames as PNG files.",
    )
    args = parser.parse_args()

    left = load_episode(args.left.resolve(), args.episode, joint_key=args.joint_source)
    right = load_episode(args.right.resolve(), args.episode, joint_key=args.joint_source)
    left_action = np.asarray(left["action"])
    right_action = np.asarray(right["action"])
    if left_action.shape != right_action.shape:
        raise ValueError(f"episode lengths differ: {left_action.shape} vs {right_action.shape}")
    if left["joint_names"] != right["joint_names"]:
        raise ValueError("action.wbc joint orders differ")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_label = args.joint_source.replace(".", "_")
    safe_left = re.sub(r"[^A-Za-z0-9_.-]+", "_", args.left_label)
    safe_right = re.sub(r"[^A-Za-z0-9_.-]+", "_", args.right_label)
    prefix = f"episode_{args.episode:06d}"
    left_path = args.output_dir / f"{prefix}_{safe_left}_{source_label}_mujoco.mp4"
    right_path = args.output_dir / f"{prefix}_{safe_right}_{source_label}_mujoco.mp4"
    pair_path = args.output_dir / f"{prefix}_{safe_left}_vs_{safe_right}_{source_label}_mujoco.mp4"
    writers = (
        VideoWriter(left_path, 640, 556, args.fps),
        VideoWriter(right_path, 640, 556, args.fps),
        VideoWriter(pair_path, 1280, 556, args.fps),
    )
    replay = G1ReplayRenderer(args.model.resolve())
    total = left_action.shape[0]
    snapshots = {0, total // 2, total - 1} if args.save_snapshots else set()
    try:
        for frame in range(total):
            panels = []
            for label, motion in ((args.left_label, left), (args.right_label, right)):
                root_pos = np.asarray(motion["root_pos"])[frame]
                replay.set_pose(
                    list(motion["joint_names"]),
                    np.asarray(motion["action"])[frame],
                    root_pos,
                    np.asarray(motion["root_quat_xyzw"])[frame],
                )
                lookat = root_pos.copy()
                lookat[2] += 0.15
                image = replay.render(lookat)
                panel = labeled_frame(
                    image,
                    f"{label} [{args.joint_source}]",
                    frame,
                    total,
                    float(np.asarray(motion["timestamp"])[frame]),
                    root_pos,
                )
                panels.append(panel)
            writers[0].write(panels[0])
            writers[1].write(panels[1])
            pair = np.concatenate(panels, axis=1)
            writers[2].write(pair)
            if frame in snapshots:
                Image.fromarray(pair).save(
                    args.output_dir / f"{prefix}_mujoco_frame_{frame:03d}.png"
                )
    finally:
        replay.close()
        for writer in writers:
            writer.close()

    joint_value_diff = float(np.max(np.abs(left_action - right_action)))
    root_gap = float(
        np.max(
            np.linalg.norm(
                np.asarray(left["root_pos"]) - np.asarray(right["root_pos"]), axis=1
            )
        )
    )
    print(f"Saved: {left_path}")
    print(f"Saved: {right_path}")
    print(f"Saved: {pair_path}")
    print(f"Frames={total}, fps={args.fps}, duration={total / args.fps:.2f}s")
    print(f"Max {args.joint_source} difference={joint_value_diff:.6e} rad")
    print(f"Max root position gap={root_gap * 100:.3f} cm")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
