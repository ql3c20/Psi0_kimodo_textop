#!/usr/bin/env python3
"""Render a full-body URDF-FK comparison of measured-next and clipped WBC targets."""

from __future__ import annotations

import argparse
import importlib.util
from collections import deque
from pathlib import Path
from typing import Any

import av
import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation


REPO_ROOT = Path(__file__).resolve().parents[2]
CONVERTER_PATH = Path(__file__).with_name("build_real_rot6d59_dataset.py")
DEFAULT_SRC = REPO_ROOT / "data/output/lqb_20260831_to_20260901_realtask2_new"
DEFAULT_OUTPUT = REPO_ROOT / "outputs/analysis/episode_000075_fullbody_comparison.mp4"

COLORS = {
    "measured": "#3DAD68",
    "wbc": "#E25353",
    "ghost": "#AAB4C3",
    "left_hand": "#5BA3EC",
    "right_hand": "#F5A742",
    "left_foot": "#9B72CF",
    "right_foot": "#17A6A0",
    "pelvis": "#E7C447",
}

MAJOR_CHAINS = (
    ("pelvis", "left_hip_yaw_link", "left_knee_link", "left_ankle_roll_link"),
    ("pelvis", "right_hip_yaw_link", "right_knee_link", "right_ankle_roll_link"),
    ("pelvis", "torso_link", "head_link"),
    ("torso_link", "left_shoulder_yaw_link", "left_elbow_link", "left_wrist_yaw_link"),
    ("torso_link", "right_shoulder_yaw_link", "right_elbow_link", "right_wrist_yaw_link"),
)

TRAIL_LINKS = {
    "left_hand": "left_wrist_yaw_link",
    "right_hand": "right_wrist_yaw_link",
    "left_foot": "left_ankle_roll_link",
    "right_foot": "right_ankle_roll_link",
    "pelvis": "pelvis",
}


def load_converter() -> Any:
    spec = importlib.util.spec_from_file_location("real_rot6d59_converter", CONVERTER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import converter: {CONVERTER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    family = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{family}", size)


def all_link_transforms(
    converter: Any,
    fk: Any,
    joint_values: dict[str, float],
    root_position: np.ndarray,
    root_rotation: np.ndarray,
) -> dict[str, np.ndarray]:
    root_transform = np.eye(4, dtype=np.float64)
    root_transform[:3, :3] = root_rotation
    root_transform[:3, 3] = root_position
    found = {fk.root_link: root_transform}

    def visit(link: str, link_transform: np.ndarray) -> None:
        for joint in fk.children.get(link, []):
            value = float(joint_values.get(joint["name"], 0.0))
            if fk.clipping in {"urdf", "robot-model"} and joint["name"] in fk.fk_joint_limits:
                lower, upper = fk.fk_joint_limits[joint["name"]]
                value = float(
                    np.clip(
                        value,
                        lower + converter.FK_CLIP_MARGIN,
                        upper - converter.FK_CLIP_MARGIN,
                    )
                )
            motion = np.eye(4, dtype=np.float64)
            if joint["type"] in {"revolute", "continuous"}:
                axis = joint["axis"] / max(float(np.linalg.norm(joint["axis"])), 1e-12)
                motion[:3, :3] = Rotation.from_rotvec(axis * value).as_matrix()
            elif joint["type"] == "prismatic":
                motion[:3, 3] = joint["axis"] * value
            current = link_transform @ joint["origin"] @ motion
            found[joint["child"]] = current
            visit(joint["child"], current)

    visit(fk.root_link, root_transform)
    return found


class OrthographicCamera:
    def __init__(self, scale: float = 360.0) -> None:
        camera_offset = np.array([4.0, -6.0, 2.8], dtype=np.float64)
        forward = -camera_offset / np.linalg.norm(camera_offset)
        world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        self.right = np.cross(forward, world_up)
        self.right /= np.linalg.norm(self.right)
        self.up = np.cross(self.right, forward)
        self.up /= np.linalg.norm(self.up)
        self.scale = scale

    def project(
        self,
        point: np.ndarray,
        center: np.ndarray,
        panel_center: tuple[float, float],
    ) -> tuple[int, int]:
        relative = np.asarray(point) - center
        return (
            int(round(panel_center[0] + np.dot(relative, self.right) * self.scale)),
            int(round(panel_center[1] - np.dot(relative, self.up) * self.scale)),
        )


def draw_grid(
    draw: ImageDraw.ImageDraw,
    camera: OrthographicCamera,
    center: np.ndarray,
    panel_center: tuple[float, float],
) -> None:
    floor_z = 0.0
    grid_center = np.array([center[0], center[1], floor_z])
    extent = 1.0
    for value in np.arange(-extent, extent + 0.01, 0.25):
        shade = "#DDE3EB" if abs(value % 0.5) > 1e-6 else "#CED6E1"
        for p0, p1 in (
            (grid_center + [value, -extent, 0], grid_center + [value, extent, 0]),
            (grid_center + [-extent, value, 0], grid_center + [extent, value, 0]),
        ):
            draw.line(
                (camera.project(p0, center, panel_center), camera.project(p1, center, panel_center)),
                fill=shade,
                width=1,
            )


def draw_trails(
    draw: ImageDraw.ImageDraw,
    camera: OrthographicCamera,
    center: np.ndarray,
    panel_center: tuple[float, float],
    trails: dict[str, deque[np.ndarray]],
) -> None:
    for name, points in trails.items():
        if len(points) < 2:
            continue
        projected = [camera.project(point, center, panel_center) for point in points]
        draw.line(projected, fill=COLORS[name], width=3)


def draw_robot(
    draw: ImageDraw.ImageDraw,
    camera: OrthographicCamera,
    center: np.ndarray,
    panel_center: tuple[float, float],
    transforms: dict[str, np.ndarray],
    color: str,
) -> None:
    for chain in MAJOR_CHAINS:
        points = [
            camera.project(transforms[name][:3, 3], center, panel_center)
            for name in chain
        ]
        draw.line(points, fill="#FFFFFF", width=14, joint="curve")
        draw.line(points, fill=color, width=8, joint="curve")
    important = {name for chain in MAJOR_CHAINS for name in chain}
    for name in important:
        x, y = camera.project(transforms[name][:3, 3], center, panel_center)
        radius = 7 if name in {"pelvis", "torso_link", "head_link"} else 5
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill="#FFFFFF")
        draw.ellipse((x - radius + 2, y - radius + 2, x + radius - 2, y + radius - 2), fill=color)
    for trail_name, link_name in TRAIL_LINKS.items():
        point = transforms[link_name][:3, 3]
        x, y = camera.project(point, center, panel_center)
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=COLORS[trail_name], outline="#FFFFFF", width=2)
    # Small local x axes make wrist and ankle orientation differences visible.
    for link_name in ("left_wrist_yaw_link", "right_wrist_yaw_link", "left_ankle_roll_link", "right_ankle_roll_link"):
        transform = transforms[link_name]
        p0 = transform[:3, 3]
        p1 = p0 + transform[:3, :3] @ np.array([0.12, 0.0, 0.0])
        draw.line(
            (camera.project(p0, center, panel_center), camera.project(p1, center, panel_center)),
            fill="#29384F",
            width=3,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC)
    parser.add_argument("--episode", type=int, default=75)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    converter = load_converter()
    src = args.src.resolve()
    source_path = next((src / "data").glob(f"chunk-*/episode_{args.episode:06d}.parquet"))
    info = converter.read_json(src / "meta/info.json")
    source_fps = int(info["fps"])
    table = pq.read_table(source_path)
    source = converter.load_resampled_source(table, source_fps, args.fps)
    state_indices = converter.indices_for_names(info, converter.SOURCE_STATE_KEY, converter.JOINT43_ORDER)
    hand_indices = converter.indices_for_names(info, converter.SOURCE_ACTION_KEY, converter.HAND14_ORDER)
    body_indices = converter.indices_for_names(info, converter.SOURCE_ACTION_KEY, converter.BODY29_ORDER)

    joint43 = source[converter.SOURCE_STATE_KEY][:, state_indices]
    hand = source[converter.SOURCE_ACTION_KEY][:-1, hand_indices]
    measured = np.concatenate([hand, joint43[1:, 14:43]], axis=1)
    wbc = np.concatenate([hand, source[converter.SOURCE_ACTION_KEY][:-1, body_indices]], axis=1)
    roots = source[converter.SOURCE_ROOT_POS_KEY][1:]
    root_rotations = Rotation.from_quat(source[converter.SOURCE_ROOT_QUAT_KEY][1:]).as_matrix()
    measured_fk = converter.CanonicalG1UrdfFK(converter.DEFAULT_URDF.resolve(), "none")
    wbc_fk = converter.CanonicalG1UrdfFK(converter.DEFAULT_URDF.resolve(), "urdf")

    measured_frames = []
    wbc_frames = []
    differences_cm = []
    for frame in range(len(measured)):
        measured_tf = all_link_transforms(
            converter,
            measured_fk,
            dict(zip(converter.JOINT43_ORDER, measured[frame])),
            roots[frame],
            root_rotations[frame],
        )
        wbc_tf = all_link_transforms(
            converter,
            wbc_fk,
            dict(zip(converter.JOINT43_ORDER, wbc[frame])),
            roots[frame],
            root_rotations[frame],
        )
        measured_frames.append(measured_tf)
        wbc_frames.append(wbc_tf)
        differences_cm.append(
            max(
                np.linalg.norm(measured_tf[link][:3, 3] - wbc_tf[link][:3, 3]) * 100.0
                for link in TRAIL_LINKS.values()
                if link != "pelvis"
            )
        )

    width, height = 1920, 860
    panel_boxes = ((35, 105, 945, 770), (975, 105, 1885, 770))
    panel_centers = ((490, 455), (1430, 455))
    camera = OrthographicCamera()
    trail_length = args.fps
    trails = {
        mode: {name: deque(maxlen=trail_length) for name in TRAIL_LINKS}
        for mode in ("measured", "wbc")
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    container = av.open(str(args.output), mode="w")
    stream = container.add_stream("libx264", rate=args.fps)
    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": "18", "preset": "medium"}

    for frame, (measured_tf, wbc_tf) in enumerate(zip(measured_frames, wbc_frames)):
        canvas = Image.new("RGB", (width, height), "#F1F4F8")
        draw = ImageDraw.Draw(canvas)
        draw.text((42, 24), f"Episode {args.episode:06d}: full-body target comparison", fill="#14213D", font=font(30, True))
        draw.text(
            (43, 64),
            f"frame {frame:03d}/{len(measured_frames)-1:03d}   t={frame / args.fps:5.2f}s   max EE position gap={differences_cm[frame]:5.1f} cm",
            fill="#526071",
            font=font(18),
        )
        for box, title, subtitle, mode, transforms, panel_center, color in (
            (panel_boxes[0], "Measured next-state", "body29 = observation.state[t+1]", "measured", measured_tf, panel_centers[0], COLORS["measured"]),
            (panel_boxes[1], "WBC target + URDF clipping", "body29 = clip(action.wbc[t])", "wbc", wbc_tf, panel_centers[1], COLORS["wbc"]),
        ):
            draw.rounded_rectangle(box, radius=16, fill="#FFFFFF", outline="#D3DAE4", width=2)
            draw.text((box[0] + 22, box[1] + 16), title, fill=color, font=font(24, True))
            draw.text((box[0] + 22, box[1] + 49), subtitle, fill="#667386", font=font(15))
            center = roots[frame].copy()
            center[2] += 0.28
            draw_grid(draw, camera, center, panel_center)
            for name, link in TRAIL_LINKS.items():
                trails[mode][name].append(transforms[link][:3, 3].copy())
            draw_trails(draw, camera, center, panel_center, trails[mode])
            draw_robot(draw, camera, center, panel_center, transforms, color)
        draw.text((41, 803), "Trails (last 1 s):", fill="#536174", font=font(16, True))
        legend_x = 190
        for name in ("left_hand", "right_hand", "left_foot", "right_foot", "pelvis"):
            draw.line((legend_x, 815, legend_x + 30, 815), fill=COLORS[name], width=4)
            label = name.replace("_", " ")
            draw.text((legend_x + 38, 803), label, fill="#536174", font=font(15))
            legend_x += 170
        video_frame = av.VideoFrame.from_ndarray(np.asarray(canvas), format="rgb24")
        for packet in stream.encode(video_frame):
            container.mux(packet)
        if frame in {0, len(measured_frames) // 2, len(measured_frames) - 1}:
            canvas.save(args.output.with_name(f"{args.output.stem}_frame_{frame:03d}.png"))
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    print(f"Saved video: {args.output}")
    print(f"Frames: {len(measured_frames)}, FPS: {args.fps}, duration: {len(measured_frames) / args.fps:.2f}s")
    print(f"Max EE position gap: {max(differences_cm):.2f} cm")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
