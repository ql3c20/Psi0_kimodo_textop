#!/usr/bin/env python3
"""Open-loop evaluator for the Arena football GR00T rot6d59 checkpoint.

This script reads the converted LeRobot dataset, sends real ego images and
52D full-state observations to the existing GR00T rot6d59 HTTP bridge, and
compares the predicted 59D action chunk against dataset ground truth.
"""

from __future__ import annotations

import argparse
from base64 import b64decode, b64encode
import csv
import json
from pathlib import Path
from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.format import descr_to_dtype, dtype_to_descr
import pyarrow.parquet as pq
import requests
from tqdm import tqdm


STATE_KEY = "observation.full_state_rot6d"
ACTION_KEY = "action.policy_action_rot6d59"
VIDEO_KEY = "observation.images.ego_view"
DEFAULT_DATASET = Path(
    "/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Psi0/data/output/arena_football_rot6d59"
)
DEFAULT_OUT = Path(
    "/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Psi0/outputs/openloop_arena_football_gr00t_rot6d59"
)

ACTION_GROUPS = [
    ("hand14", 0, 14),
    ("root_xyz", 14, 17),
    ("root_rot6d", 17, 23),
    ("left_hand_xyz", 23, 26),
    ("left_hand_rot6d", 26, 32),
    ("right_hand_xyz", 32, 35),
    ("right_hand_rot6d", 35, 41),
    ("left_foot_xyz", 41, 44),
    ("left_foot_rot6d", 44, 50),
    ("right_foot_xyz", 50, 53),
    ("right_foot_rot6d", 53, 59),
]


def numpy_encode(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: numpy_encode(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [numpy_encode(item) for item in value]
    if isinstance(value, (np.ndarray, np.generic)):
        arr = np.asarray(value)
        data = arr.data if arr.flags["C_CONTIGUOUS"] else arr.tobytes()
        return {
            "__numpy__": b64encode(data).decode(),
            "dtype": dtype_to_descr(arr.dtype),
            "shape": arr.shape,
        }
    return value


def numpy_decode(value: Any) -> Any:
    if isinstance(value, dict) and "__numpy__" in value:
        arr = np.frombuffer(
            b64decode(value["__numpy__"]),
            descr_to_dtype(value["dtype"]),
        )
        return arr.reshape(value["shape"]) if value["shape"] else arr[0]
    if isinstance(value, dict):
        return {key: numpy_decode(item) for key, item in value.items()}
    if isinstance(value, list):
        return [numpy_decode(item) for item in value]
    return value


def fixed_list_column(table: pq.Table, key: str) -> np.ndarray:
    if key not in table.column_names:
        raise KeyError(f"{key!r} is missing from {table.column_names}")
    return np.asarray(table[key].to_pylist(), dtype=np.float32)


def load_info(dataset_dir: Path) -> dict[str, Any]:
    info_path = dataset_dir / "meta/info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"Missing LeRobot metadata: {info_path}")
    return json.loads(info_path.read_text())


def episode_chunk(episode_index: int, chunks_size: int) -> int:
    return int(episode_index) // max(1, int(chunks_size))


def episode_paths(dataset_dir: Path, episode_index: int, info: dict[str, Any]) -> tuple[Path, Path]:
    chunk = episode_chunk(episode_index, int(info.get("chunks_size", 1000)))
    parquet_rel = info.get(
        "data_path",
        "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
    ).format(episode_chunk=chunk, episode_index=episode_index)
    video_rel = info.get(
        "video_path",
        "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
    ).format(
        episode_chunk=chunk,
        episode_index=episode_index,
        video_key=VIDEO_KEY,
    )
    return dataset_dir / parquet_rel, dataset_dir / video_rel


def read_video_frame(video_path: Path, frame_index: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame_bgr = cap.read()
        if not ok or frame_bgr is None:
            raise RuntimeError(f"Could not read frame {frame_index} from {video_path}")
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()


def query_policy(
    *,
    server_url: str,
    image: np.ndarray,
    state52: np.ndarray,
    instruction: str,
    reset_history: bool,
    timeout_s: float,
) -> np.ndarray:
    payload = {
        "image": {"ego_view": np.asarray(image, dtype=np.uint8)},
        "state": {"states": np.asarray(state52, dtype=np.float32)},
        "instruction": instruction,
        "history": {"reset": True} if reset_history else {},
    }
    response = requests.post(
        f"{server_url.rstrip('/')}/act",
        json=numpy_encode(payload),
        timeout=timeout_s,
    )
    if response.status_code != 200:
        raise RuntimeError(f"Policy server returned HTTP {response.status_code}: {response.text}")
    data = numpy_decode(response.json())
    if "status" in data:
        raise RuntimeError(f"Policy server error: {data['status']}")
    action = np.asarray(data["action"], dtype=np.float32)
    if action.ndim != 2 or action.shape[1] != 59:
        raise ValueError(f"Expected action chunk (T, 59), got {action.shape}")
    return action


def summarize_error(errors: np.ndarray) -> dict[str, Any]:
    flat = errors.reshape(-1, errors.shape[-1])
    summary: dict[str, Any] = {
        "mae": float(np.mean(np.abs(flat))),
        "rmse": float(np.sqrt(np.mean(flat**2))),
        "max_abs": float(np.max(np.abs(flat))),
        "num_compared_action_frames": int(flat.shape[0]),
    }
    groups = {}
    for name, start, end in ACTION_GROUPS:
        group = flat[:, start:end]
        groups[name] = {
            "mae": float(np.mean(np.abs(group))),
            "rmse": float(np.sqrt(np.mean(group**2))),
            "mean_l2": float(np.mean(np.linalg.norm(group, axis=-1))),
        }
    summary["groups"] = groups
    return summary


def plot_group_mae(per_query_rows: list[dict[str, Any]], out_path: Path) -> None:
    if not per_query_rows:
        return
    x = np.arange(len(per_query_rows))
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    plot_sets = [
        ("root", ["root_xyz_mae", "root_rot6d_mae"]),
        ("hands", ["hand14_mae", "left_hand_xyz_mae", "right_hand_xyz_mae"]),
        ("feet", ["left_foot_xyz_mae", "right_foot_xyz_mae"]),
    ]
    for ax, (title, keys) in zip(axes, plot_sets):
        for key in keys:
            values = [float(row[key]) for row in per_query_rows]
            ax.plot(x, values, label=key)
        ax.set_title(title)
        ax.set_ylabel("MAE")
        ax.grid(True, alpha=0.3)
        ax.legend()
    axes[-1].set_xlabel("query index")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=22096)
    parser.add_argument("--instruction", default="Kick the football into the goal.")
    parser.add_argument("--episodes", type=int, nargs="*", default=[0])
    parser.add_argument("--num-episodes", type=int, default=None)
    parser.add_argument("--stride", type=int, default=34)
    parser.add_argument("--max-queries-per-episode", type=int, default=20)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument(
        "--use-rtc-continuity",
        action="store_true",
        help="Only reset policy history at the first query of each episode.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    info = load_info(dataset_dir)

    total_episodes = int(info.get("total_episodes", 0))
    episodes = args.episodes
    if args.num_episodes is not None:
        episodes = list(range(args.num_episodes))
    if not episodes:
        raise ValueError("No episodes selected")
    for episode in episodes:
        if episode < 0 or episode >= total_episodes:
            raise ValueError(f"Episode {episode} is outside 0..{total_episodes - 1}")

    server_url = f"http://{args.host}:{args.port}"
    health = requests.get(f"{server_url}/health", timeout=10.0)
    health.raise_for_status()

    all_errors = []
    per_query_rows: list[dict[str, Any]] = []

    for episode in episodes:
        parquet_path, video_path = episode_paths(dataset_dir, episode, info)
        table = pq.read_table(parquet_path)
        states = fixed_list_column(table, STATE_KEY)
        actions = fixed_list_column(table, ACTION_KEY)
        episode_len = min(len(states), len(actions))
        if episode_len == 0:
            continue

        query_indices = list(range(0, episode_len - 1, max(1, args.stride)))
        if args.max_queries_per_episode > 0:
            query_indices = query_indices[: args.max_queries_per_episode]

        for query_i, frame_index in enumerate(
            tqdm(query_indices, desc=f"episode {episode}", unit="query")
        ):
            image = read_video_frame(video_path, frame_index)
            pred = query_policy(
                server_url=server_url,
                image=image,
                state52=states[frame_index],
                instruction=args.instruction,
                reset_history=(query_i == 0 or not args.use_rtc_continuity),
                timeout_s=args.timeout_s,
            )
            compare_len = min(pred.shape[0], episode_len - frame_index)
            gt = actions[frame_index : frame_index + compare_len]
            pred = pred[:compare_len]
            err = pred - gt
            all_errors.append(err)

            row: dict[str, Any] = {
                "episode": episode,
                "frame_index": int(frame_index),
                "compare_len": int(compare_len),
                "mae": float(np.mean(np.abs(err))),
                "rmse": float(np.sqrt(np.mean(err**2))),
            }
            flat_err = err.reshape(-1, err.shape[-1])
            for name, start, end in ACTION_GROUPS:
                group = flat_err[:, start:end]
                row[f"{name}_mae"] = float(np.mean(np.abs(group)))
                row[f"{name}_rmse"] = float(np.sqrt(np.mean(group**2)))
            per_query_rows.append(row)

    if not all_errors:
        raise RuntimeError("No action chunks were evaluated")

    metrics = {
        "dataset": str(dataset_dir),
        "server_url": server_url,
        "episodes": episodes,
        "stride": args.stride,
        "max_queries_per_episode": args.max_queries_per_episode,
        "use_rtc_continuity": bool(args.use_rtc_continuity),
        "summary": summarize_error(np.concatenate(all_errors, axis=0)),
    }
    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))

    csv_path = out_dir / "per_query_metrics.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_query_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_query_rows)

    plot_path = out_dir / "group_mae.png"
    plot_group_mae(per_query_rows, plot_path)

    print(json.dumps(metrics["summary"], indent=2))
    print(f"wrote {metrics_path}")
    print(f"wrote {csv_path}")
    print(f"wrote {plot_path}")


if __name__ == "__main__":
    main()
