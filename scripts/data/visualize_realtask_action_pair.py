#!/usr/bin/env python3
"""Render the same LeRobot episode from two real-task datasets side by side."""

from __future__ import annotations

import argparse
import importlib.util
from collections import deque
from pathlib import Path
from typing import Any

import av
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation


SCRIPT_DIR = Path(__file__).resolve().parent
FK_VIS_PATH = SCRIPT_DIR / "visualize_real_body_target_comparison.py"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[2] / "outputs/visualizations/realtask2_episode_000000"


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def episode_path(root: Path, episode: int) -> Path:
    matches = sorted((root / "data").glob(f"chunk-*/episode_{episode:06d}.parquet"))
    if not matches:
        raise FileNotFoundError(f"Episode {episode} not found under {root}")
    return matches[0]


def load_action_motion(root: Path, episode: int, fps: int, converter: Any) -> dict[str, np.ndarray]:
    info = converter.read_json(root / "meta/info.json")
    table = pq.read_table(episode_path(root, episode))
    source = converter.load_resampled_source(table, int(info["fps"]), fps)
    action_indices = converter.indices_for_names(
        info, converter.SOURCE_ACTION_KEY, converter.JOINT43_ORDER
    )
    return {
        "action": source[converter.SOURCE_ACTION_KEY][:, action_indices],
        "root_pos": source[converter.SOURCE_ROOT_POS_KEY],
        "root_quat": source[converter.SOURCE_ROOT_QUAT_KEY],
    }


def save_action_plot(
    output: Path,
    left: dict[str, np.ndarray],
    right: dict[str, np.ndarray],
    joint_names: list[str],
    fps: int,
) -> None:
    groups = (
        ("Hands (14)", joint_names[:14], range(0, 14)),
        ("Legs (12)", joint_names[14:26], range(14, 26)),
        ("Waist (3)", joint_names[26:29], range(26, 29)),
        ("Arms (14)", joint_names[29:43], range(29, 43)),
    )
    time = np.arange(left["action"].shape[0]) / fps
    fig, axes = plt.subplots(2, 2, figsize=(18, 10), sharex=True)
    for axis, (title, names, indices) in zip(axes.flat, groups):
        for name, index in zip(names, indices):
            line = axis.plot(time, left["action"][:, index], linewidth=1.15, label=name)[0]
            axis.plot(
                time,
                right["action"][:, index],
                linestyle="--",
                linewidth=0.8,
                color=line.get_color(),
                alpha=0.8,
            )
        axis.set_title(title)
        axis.set_ylabel("joint target (rad)")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7, ncol=2, loc="upper right")
    for axis in axes[-1]:
        axis.set_xlabel("time (s)")
    max_diff = float(np.max(np.abs(left["action"] - right["action"])))
    fig.suptitle(
        "Episode 000000 action.wbc comparison\n"
        "solid: realtask2, dashed: realtask2_new   "
        f"max absolute action difference = {max_diff:.3e} rad",
        fontsize=15,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output, dpi=150)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    vis = load_module(FK_VIS_PATH, "real_body_target_vis")
    converter = vis.load_converter()
    left = load_action_motion(args.left.resolve(), args.episode, args.fps, converter)
    right = load_action_motion(args.right.resolve(), args.episode, args.fps, converter)
    if left["action"].shape != right["action"].shape:
        raise ValueError(f"Episode shapes differ: {left['action'].shape} vs {right['action'].shape}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = args.output_dir / "episode_000000_action_wbc_comparison.png"
    video_path = args.output_dir / "episode_000000_robot_action_comparison.mp4"
    save_action_plot(plot_path, left, right, converter.JOINT43_ORDER, args.fps)

    fk = converter.CanonicalG1UrdfFK(converter.DEFAULT_URDF.resolve(), "urdf")
    count = left["action"].shape[0]
    datasets = (("realtask2", left, "#3DAD68"), ("realtask2_new", right, "#E25353"))
    transforms: dict[str, list[dict[str, np.ndarray]]] = {}
    for label, motion, _ in datasets:
        rotations = Rotation.from_quat(motion["root_quat"]).as_matrix()
        transforms[label] = [
            vis.all_link_transforms(
                converter,
                fk,
                dict(zip(converter.JOINT43_ORDER, motion["action"][frame])),
                motion["root_pos"][frame],
                rotations[frame],
            )
            for frame in range(count)
        ]

    width, height = 1920, 860
    boxes = ((35, 105, 945, 770), (975, 105, 1885, 770))
    panel_centers = ((490, 455), (1430, 455))
    camera = vis.OrthographicCamera()
    trails = {
        label: {name: deque(maxlen=args.fps) for name in vis.TRAIL_LINKS}
        for label, _, _ in datasets
    }
    container = av.open(str(video_path), mode="w")
    stream = container.add_stream("libx264", rate=args.fps)
    stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
    stream.options = {"crf": "18", "preset": "medium"}

    snapshots = {0, count // 2, count - 1}
    for frame in range(count):
        canvas = Image.new("RGB", (width, height), "#F1F4F8")
        draw = ImageDraw.Draw(canvas)
        draw.text(
            (42, 24),
            f"Episode {args.episode:06d}: action.wbc robot pose",
            fill="#14213D",
            font=vis.font(30, True),
        )
        action_diff = np.max(np.abs(left["action"][frame] - right["action"][frame]))
        root_gap = np.linalg.norm(left["root_pos"][frame] - right["root_pos"][frame]) * 100
        draw.text(
            (43, 64),
            f"frame {frame:03d}/{count - 1:03d}   t={frame / args.fps:5.2f}s   "
            f"action max diff={action_diff:.2e} rad   root gap={root_gap:.2f} cm",
            fill="#526071",
            font=vis.font(18),
        )
        shared_center = 0.5 * (left["root_pos"][frame] + right["root_pos"][frame])
        shared_center = shared_center.copy()
        shared_center[2] += 0.28
        for box, center, (label, motion, color) in zip(boxes, panel_centers, datasets):
            tf = transforms[label][frame]
            draw.rounded_rectangle(box, radius=16, fill="#FFFFFF", outline="#D3DAE4", width=2)
            draw.text((box[0] + 22, box[1] + 16), label, fill=color, font=vis.font(24, True))
            root = motion["root_pos"][frame]
            draw.text(
                (box[0] + 22, box[1] + 49),
                f"action.wbc[t], root xyz=({root[0]:+.3f}, {root[1]:+.3f}, {root[2]:+.3f}) m",
                fill="#667386",
                font=vis.font(15),
            )
            vis.draw_grid(draw, camera, shared_center, center)
            for name, link in vis.TRAIL_LINKS.items():
                trails[label][name].append(tf[link][:3, 3].copy())
            vis.draw_trails(draw, camera, shared_center, center, trails[label])
            vis.draw_robot(draw, camera, shared_center, center, tf, color)
        draw.text(
            (40, 805),
            "The 43-D WBC joint targets are identical; the new dataset changes only mocap root position/orientation.",
            fill="#536174",
            font=vis.font(17, True),
        )
        if frame in snapshots:
            canvas.save(args.output_dir / f"episode_000000_robot_action_frame_{frame:03d}.png")
        video_frame = av.VideoFrame.from_ndarray(np.asarray(canvas), format="rgb24")
        for packet in stream.encode(video_frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()

    print(f"Saved video: {video_path}")
    print(f"Saved plot: {plot_path}")
    print(f"Frames: {count}; FPS: {args.fps}; duration: {count / args.fps:.2f}s")
    print(f"Action max abs diff: {np.max(np.abs(left['action'] - right['action'])):.6e} rad")
    print(f"Root position max gap: {np.max(np.linalg.norm(left['root_pos'] - right['root_pos'], axis=1)) * 100:.3f} cm")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
