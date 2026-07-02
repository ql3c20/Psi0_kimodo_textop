#!/usr/bin/env python3
"""Merge per-level LeRobot datasets into one flat LeRobot dataset."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def replace_column(table: pa.Table, name: str, values: pa.Array) -> pa.Table:
    idx = table.schema.get_field_index(name)
    if idx < 0:
        return table
    return table.set_column(idx, table.schema.field(idx), values)


def rewrite_episode_parquet(src: Path, dst: Path, episode_index: int, start_index: int) -> int:
    table = pq.read_table(src)
    nrows = table.num_rows
    table = replace_column(table, "episode_index", pa.array([episode_index] * nrows, type=pa.int64()))
    table = replace_column(table, "frame_index", pa.array(range(nrows), type=pa.int64()))
    table = replace_column(table, "index", pa.array(range(start_index, start_index + nrows), type=pa.int64()))
    dst.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, dst)
    return nrows


def copy_episode_videos(src_root: Path, dst_root: Path, old_ep: int, new_ep: int, chunk: int) -> None:
    src_video_root = src_root / "videos"
    if not src_video_root.exists():
        return
    pattern = f"episode_{old_ep:06d}.mp4"
    for src in src_video_root.rglob(pattern):
        rel = src.relative_to(src_video_root)
        parts = list(rel.parts)
        parts[-1] = f"episode_{new_ep:06d}.mp4"
        if parts and parts[0].startswith("chunk-"):
            parts[0] = f"chunk-{chunk:03d}"
        dst = dst_root / "videos" / Path(*parts)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--levels", nargs="+", default=["level-0", "level-1", "level-2"])
    parser.add_argument("--chunks-size", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"{args.output_root} exists; pass --overwrite to replace it")
        shutil.rmtree(args.output_root)

    first_level = args.input_root / args.levels[0]
    first_info_path = first_level / "meta" / "info.json"
    if not first_info_path.is_file():
        raise FileNotFoundError(first_info_path)

    out_meta = args.output_root / "meta"
    out_meta.mkdir(parents=True, exist_ok=True)
    (args.output_root / "data").mkdir(parents=True, exist_ok=True)
    (args.output_root / "videos").mkdir(parents=True, exist_ok=True)

    info = json.loads(first_info_path.read_text(encoding="utf-8"))
    chunks_size = int(args.chunks_size or info.get("chunks_size") or 1000)

    for meta_name in ["modality.json", "tasks.jsonl"]:
        src = first_level / "meta" / meta_name
        if src.is_file():
            shutil.copy2(src, out_meta / meta_name)

    merged_episodes: list[dict] = []
    merged_episode_stats: list[dict] = []
    global_ep = 0
    global_frame_index = 0

    for level in args.levels:
        level_root = args.input_root / level
        meta_root = level_root / "meta"
        episodes = read_jsonl(meta_root / "episodes.jsonl")
        episode_stats = {
            int(row["episode_index"]): row for row in read_jsonl(meta_root / "episodes_stats.jsonl")
        }

        for row in sorted(episodes, key=lambda r: int(r["episode_index"])):
            old_ep = int(row["episode_index"])
            chunk = global_ep // chunks_size
            src_parquet = level_root / "data" / f"chunk-{old_ep // chunks_size:03d}" / f"episode_{old_ep:06d}.parquet"
            dst_parquet = args.output_root / "data" / f"chunk-{chunk:03d}" / f"episode_{global_ep:06d}.parquet"
            nrows = rewrite_episode_parquet(src_parquet, dst_parquet, global_ep, global_frame_index)

            new_row = dict(row)
            new_row["episode_index"] = global_ep
            new_row["length"] = nrows
            new_row["source_level"] = level
            new_row["source_episode_index"] = old_ep
            merged_episodes.append(new_row)

            stats_row = dict(episode_stats[old_ep])
            stats_row["episode_index"] = global_ep
            stats_row["source_level"] = level
            stats_row["source_episode_index"] = old_ep
            merged_episode_stats.append(stats_row)

            copy_episode_videos(level_root, args.output_root, old_ep, global_ep, chunk)
            global_frame_index += nrows
            global_ep += 1

    write_jsonl(out_meta / "episodes.jsonl", merged_episodes)
    write_jsonl(out_meta / "episodes_stats.jsonl", merged_episode_stats)

    info["total_episodes"] = len(merged_episodes)
    info["total_frames"] = global_frame_index
    info["total_videos"] = len(merged_episodes)
    info["total_chunks"] = math.ceil(len(merged_episodes) / chunks_size) if chunks_size else 1
    info["chunks_size"] = chunks_size
    info["splits"] = {"train": f"0:{len(merged_episodes)}"}
    (out_meta / "info.json").write_text(json.dumps(info, indent=4, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Merged {len(merged_episodes)} episodes / {global_frame_index} frames into {args.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
