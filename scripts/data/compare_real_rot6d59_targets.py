#!/usr/bin/env python3
"""Compare WBC-qpos/URDF-clipped FK targets with measured-next FK targets."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation


REPO_ROOT = Path(__file__).resolve().parents[2]
CONVERTER_PATH = Path(__file__).with_name("build_real_rot6d59_dataset.py")
DEFAULT_SRC = REPO_ROOT / "data/output/lqb_20260831_to_20260901_realtask2_new"
DEFAULT_OUTPUT = REPO_ROOT / "outputs/analysis/wbc_urdfclip_vs_measured_next.png"

EE_LAYOUT = {
    "left hand": (23, 26),
    "right hand": (32, 35),
    "left foot": (41, 44),
    "right foot": (50, 53),
}
PERCENTILES = (50, 95, 99)


def load_converter() -> Any:
    spec = importlib.util.spec_from_file_location("real_rot6d59_converter", CONVERTER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import converter: {CONVERTER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rotation_distance_deg(converter: Any, lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    lhs_matrix = converter.rot6d_to_matrix(lhs)
    rhs_matrix = converter.rot6d_to_matrix(rhs)
    relative = np.einsum("nij,njk->nik", np.transpose(lhs_matrix, (0, 2, 1)), rhs_matrix)
    return np.degrees(Rotation.from_matrix(relative).magnitude())


def summarize(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(values.max()),
    }


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    family = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{family}", size)


def draw_panel_frame(
    draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], title: str, ylabel: str
) -> tuple[int, int, int, int]:
    left, top, right, bottom = box
    draw.rounded_rectangle(box, radius=12, fill="#ffffff", outline="#d7dde5", width=2)
    draw.text((left + 18, top + 14), title, fill="#172033", font=font(20, True))
    plot = (left + 70, top + 58, right - 20, bottom - 52)
    px0, py0, px1, py1 = plot
    draw.line((px0, py0, px0, py1), fill="#7b8798", width=2)
    draw.line((px0, py1, px1, py1), fill="#7b8798", width=2)
    draw.text((left + 8, top + 62), ylabel, fill="#526071", font=font(13))
    return plot


def draw_grouped_bars(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    title: str,
    ylabel: str,
    categories: list[str],
    series: list[tuple[str, list[float], str]],
) -> None:
    px0, py0, px1, py1 = draw_panel_frame(draw, box, title, ylabel)
    maximum = max(max(values) for _, values, _ in series) * 1.12 or 1.0
    for tick in range(5):
        value = maximum * tick / 4
        y = py1 - (py1 - py0) * tick / 4
        draw.line((px0, y, px1, y), fill="#e8ecf1", width=1)
        draw.text((px0 - 55, y - 8), f"{value:.1f}", fill="#6d7888", font=font(12))
    group_width = (px1 - px0) / len(categories)
    bar_width = group_width * 0.72 / len(series)
    for category_index, category in enumerate(categories):
        center = px0 + group_width * (category_index + 0.5)
        for series_index, (_, values, color) in enumerate(series):
            x0 = center - group_width * 0.36 + series_index * bar_width
            x1 = x0 + bar_width - 3
            y0 = py1 - (values[category_index] / maximum) * (py1 - py0)
            draw.rectangle((x0, y0, x1, py1), fill=color)
        label = category.replace(" ", "\n")
        bbox = draw.multiline_textbbox((0, 0), label, font=font(12), align="center")
        draw.multiline_text(
            (center - (bbox[2] - bbox[0]) / 2, py1 + 7),
            label,
            fill="#39475a",
            font=font(12),
            align="center",
            spacing=0,
        )
    legend_x = px1 - 10
    for name, _, color in reversed(series):
        width = draw.textbbox((0, 0), name, font=font(12))[2]
        legend_x -= width + 30
        draw.rectangle((legend_x, py0 + 5, legend_x + 14, py0 + 19), fill=color)
        draw.text((legend_x + 19, py0 + 3), name, fill="#39475a", font=font(12))


def draw_lines(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    title: str,
    ylabel: str,
    series: list[tuple[str, np.ndarray, str]],
) -> None:
    px0, py0, px1, py1 = draw_panel_frame(draw, box, title, ylabel)
    maximum = max(float(np.max(values)) for _, values, _ in series) * 1.08 or 1.0
    max_length = max(len(values) for _, values, _ in series)
    for tick in range(5):
        value = maximum * tick / 4
        y = py1 - (py1 - py0) * tick / 4
        draw.line((px0, y, px1, y), fill="#e8ecf1", width=1)
        draw.text((px0 - 55, y - 8), f"{value:.1f}", fill="#6d7888", font=font(12))
    for name, values, color in series:
        if len(values) < 2:
            continue
        points = [
            (
                px0 + index / (max_length - 1) * (px1 - px0),
                py1 - float(value) / maximum * (py1 - py0),
            )
            for index, value in enumerate(values)
        ]
        draw.line(points, fill=color, width=2)
    draw.text((px0, py1 + 20), "frame", fill="#526071", font=font(12))
    legend_x = px0 + 8
    for name, _, color in series:
        draw.line((legend_x, py0 + 11, legend_x + 18, py0 + 11), fill=color, width=3)
        draw.text((legend_x + 23, py0 + 3), name, fill="#39475a", font=font(11))
        legend_x += draw.textbbox((0, 0), name, font=font(11))[2] + 52


def draw_cdf(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    values: np.ndarray,
) -> None:
    px0, py0, px1, py1 = draw_panel_frame(
        draw, box, "All-EE position difference CDF", "CDF"
    )
    values = np.sort(values)
    xmax = float(np.percentile(values, 99.9)) * 1.08 or 1.0
    sampled = np.linspace(0, len(values) - 1, min(2000, len(values))).astype(int)
    points = [
        (
            px0 + min(float(values[index]), xmax) / xmax * (px1 - px0),
            py1 - (index + 1) / len(values) * (py1 - py0),
        )
        for index in sampled
    ]
    draw.line(points, fill="#4C78A8", width=3)
    for percentile, color in ((95, "#F58518"), (99, "#E45756")):
        value = float(np.percentile(values, percentile))
        x = px0 + value / xmax * (px1 - px0)
        draw.line((x, py0, x, py1), fill=color, width=2)
        draw.text((x + 4, py0 + 5), f"P{percentile}={value:.1f}cm", fill=color, font=font(12, True))
    for tick in range(5):
        value = xmax * tick / 4
        x = px0 + (px1 - px0) * tick / 4
        draw.text((x - 12, py1 + 8), f"{value:.1f}", fill="#6d7888", font=font(12))
    draw.text((px0, py1 + 28), "position difference (cm)", fill="#526071", font=font(12))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    converter = load_converter()
    src = args.src.resolve()
    output = args.output.resolve()
    info = converter.read_json(src / "meta/info.json")
    fps = int(info["fps"])
    state_indices = converter.indices_for_names(
        info, converter.SOURCE_STATE_KEY, converter.JOINT43_ORDER
    )
    action_hand_indices = converter.indices_for_names(
        info, converter.SOURCE_ACTION_KEY, converter.HAND14_ORDER
    )
    action_body_indices = converter.indices_for_names(
        info, converter.SOURCE_ACTION_KEY, converter.BODY29_ORDER
    )
    measured_fk = converter.CanonicalG1UrdfFK(converter.DEFAULT_URDF.resolve(), "none")
    clipped_wbc_fk = converter.CanonicalG1UrdfFK(converter.DEFAULT_URDF.resolve(), "urdf")

    errors: dict[str, dict[str, list[np.ndarray]]] = {
        name: {"position_m": [], "rotation_deg": []} for name in EE_LAYOUT
    }
    steps: dict[str, dict[str, dict[str, list[np.ndarray]]]] = {
        mode: {
            name: {"position_m": [], "rotation_deg": []} for name in EE_LAYOUT
        }
        for mode in ("measured-next", "wbc-urdfclip")
    }
    worst_episode: dict[str, Any] = {"max_position_m": -1.0}
    total_rows = 0

    files = sorted((src / "data").glob("chunk-*/episode_*.parquet"))
    for index, source_path in enumerate(files, start=1):
        table = pq.read_table(source_path)
        measured, _ = converter.convert_episode(
            table,
            fps,
            fps,
            measured_fk,
            state_indices,
            action_hand_indices,
            action_body_indices,
            converter.BODY_TARGET_MEASURED_NEXT,
            converter.FK_LIMIT_REJECT,
        )
        clipped_wbc, _ = converter.convert_episode(
            table,
            fps,
            fps,
            clipped_wbc_fk,
            state_indices,
            action_hand_indices,
            action_body_indices,
            converter.BODY_TARGET_WBC,
            converter.FK_LIMIT_REJECT,
        )
        measured_action = measured[converter.ACTION_KEY]
        wbc_action = clipped_wbc[converter.ACTION_KEY]
        total_rows += len(measured_action)

        episode_position: dict[str, np.ndarray] = {}
        episode_rotation: dict[str, np.ndarray] = {}
        for name, (position_start, rotation_start) in EE_LAYOUT.items():
            position_error = np.linalg.norm(
                wbc_action[:, position_start : position_start + 3]
                - measured_action[:, position_start : position_start + 3],
                axis=1,
            )
            rotation_error = rotation_distance_deg(
                converter,
                measured_action[:, rotation_start : rotation_start + 6],
                wbc_action[:, rotation_start : rotation_start + 6],
            )
            errors[name]["position_m"].append(position_error)
            errors[name]["rotation_deg"].append(rotation_error)
            episode_position[name] = position_error
            episode_rotation[name] = rotation_error

            for mode, action in (
                ("measured-next", measured_action),
                ("wbc-urdfclip", wbc_action),
            ):
                position_step = np.linalg.norm(
                    np.diff(action[:, position_start : position_start + 3], axis=0), axis=1
                )
                rotation_step = rotation_distance_deg(
                    converter,
                    action[:-1, rotation_start : rotation_start + 6],
                    action[1:, rotation_start : rotation_start + 6],
                )
                steps[mode][name]["position_m"].append(position_step)
                steps[mode][name]["rotation_deg"].append(rotation_step)

        episode_max = max(float(values.max()) for values in episode_position.values())
        if episode_max > worst_episode["max_position_m"]:
            worst_episode = {
                "name": source_path.stem,
                "max_position_m": episode_max,
                "position_m": episode_position,
                "rotation_deg": episode_rotation,
            }
        if index % 25 == 0 or index == len(files):
            print(f"Compared {index}/{len(files)} episodes")

    combined_errors = {
        name: {metric: np.concatenate(chunks) for metric, chunks in metrics.items()}
        for name, metrics in errors.items()
    }
    combined_steps = {
        mode: {
            name: {metric: np.concatenate(chunks) for metric, chunks in metrics.items()}
            for name, metrics in by_name.items()
        }
        for mode, by_name in steps.items()
    }

    report = {
        "source": str(src),
        "episodes": len(files),
        "rows": total_rows,
        "comparison": "WBC body qpos[t] clipped to mechanical URDF limits vs measured body[t+1]",
        "common_terms": "hand14[t] and processed mocap root9[t+1] are identical",
        "ee_error": {
            name: {metric: summarize(values) for metric, values in metrics.items()}
            for name, metrics in combined_errors.items()
        },
        "temporal_step": {
            mode: {
                name: {metric: summarize(values) for metric, values in metrics.items()}
                for name, metrics in by_name.items()
            }
            for mode, by_name in combined_steps.items()
        },
        "worst_position_episode": {
            "name": worst_episode["name"],
            "max_position_m": worst_episode["max_position_m"],
        },
    }

    names = list(EE_LAYOUT)
    percentile_colors = {50: "#4C78A8", 95: "#F58518", 99: "#E45756"}
    canvas = Image.new("RGB", (1800, 1120), "#f3f6fa")
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (55, 22),
        "Real task2: URDF-clipped WBC FK vs measured-next FK",
        fill="#14213d",
        font=font(32, True),
    )
    draw.text(
        (58, 62),
        "Common terms: hand14[t] + processed mocap root9[t+1]  |  124 episodes, 42,634 rows",
        fill="#536174",
        font=font(17),
    )
    boxes = [
        (35, 105, 585, 590),
        (625, 105, 1175, 590),
        (1215, 105, 1765, 590),
        (35, 615, 585, 1095),
        (625, 615, 1175, 1095),
        (1215, 615, 1765, 1095),
    ]
    draw_grouped_bars(
        draw,
        boxes[0],
        "WBC-clipped vs measured target",
        "position (cm)",
        names,
        [
            (
                f"P{percentile}",
                [
                    float(np.percentile(combined_errors[name]["position_m"], percentile)) * 100
                    for name in names
                ],
                percentile_colors[percentile],
            )
            for percentile in PERCENTILES
        ],
    )
    draw_grouped_bars(
        draw,
        boxes[1],
        "WBC-clipped vs measured target",
        "rotation (deg)",
        names,
        [
            (
                f"P{percentile}",
                [
                    float(np.percentile(combined_errors[name]["rotation_deg"], percentile))
                    for name in names
                ],
                percentile_colors[percentile],
            )
            for percentile in PERCENTILES
        ],
    )
    all_position_cm = (
        np.concatenate([combined_errors[name]["position_m"] for name in names]) * 100
    )
    draw_cdf(draw, boxes[2], all_position_cm)
    draw_grouped_bars(
        draw,
        boxes[3],
        "Temporal continuity at 30 FPS",
        "P99 step (cm)",
        names,
        [
            (
                mode,
                [
                    float(np.percentile(combined_steps[mode][name]["position_m"], 99)) * 100
                    for name in names
                ],
                color,
            )
            for mode, color in (("measured-next", "#54A24B"), ("wbc-urdfclip", "#E45756"))
        ],
    )
    worst_name = str(worst_episode["name"])
    line_colors = ("#4C78A8", "#F58518", "#54A24B", "#E45756")
    draw_lines(
        draw,
        boxes[4],
        f"Worst position episode: {worst_name}",
        "difference (cm)",
        [
            (name, worst_episode["position_m"][name] * 100, color)
            for name, color in zip(names, line_colors)
        ],
    )
    draw_lines(
        draw,
        boxes[5],
        f"Worst position episode: {worst_name}",
        "difference (deg)",
        [
            (name, worst_episode["rotation_deg"][name], color)
            for name, color in zip(names, line_colors)
        ],
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    report_path = output.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Saved plot: {output}")
    print(f"Saved report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
