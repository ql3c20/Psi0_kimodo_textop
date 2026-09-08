#!/usr/bin/env bash
set -euo pipefail

PSI0_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
RAW_ROOT="${RAW_ROOT:-/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/dataset/HumanoidArena_open_door/HSI_open_door}"
RAW_TWIST2="${RAW_TWIST2:-$RAW_ROOT/twist2/zz}"
OUTPUT="${OUTPUT:-$PSI0_ROOT/data/output/arena_open_door_twist2_rot6d59_v1}"
DATA_PYTHON="${DATA_PYTHON:-$PSI0_ROOT/third_party/SIMPLE/.venv/bin/python}"
EXPECTED_EPISODES="${EXPECTED_EPISODES:-100}"
OVERWRITE="${OVERWRITE:-0}"
CONVERTER="$PSI0_ROOT/scripts/data/convert_arena_pp_box_to_rot6d59_lerobot.py"

for required in "$DATA_PYTHON" "$CONVERTER"; do
  if [[ ! -f "$required" ]]; then
    echo "Missing required file: $required" >&2
    exit 2
  fi
done
if [[ ! -d "$RAW_TWIST2" ]]; then
  echo "Missing raw OpenDoor TWIST2 recordings: $RAW_TWIST2" >&2
  exit 2
fi
if [[ "$OVERWRITE" != "0" && "$OVERWRITE" != "1" ]]; then
  echo "OVERWRITE must be 0 or 1; got $OVERWRITE" >&2
  exit 2
fi

raw_episode_count="$(find "$RAW_TWIST2" -maxdepth 1 -type f -name '*.npz' | wc -l)"
raw_video_count="$(find "$RAW_TWIST2" -type f -name '*.mp4' | wc -l)"
if [[ "$raw_episode_count" -ne "$EXPECTED_EPISODES" ]]; then
  echo "Expected $EXPECTED_EPISODES raw TWIST2 episodes; found $raw_episode_count." >&2
  exit 2
fi
if [[ "$raw_video_count" -ne "$EXPECTED_EPISODES" ]]; then
  echo "Expected $EXPECTED_EPISODES raw TWIST2 videos; found $raw_video_count." >&2
  exit 2
fi

overwrite_args=()
if [[ "$OVERWRITE" == "1" ]]; then
  overwrite_args+=(--overwrite)
fi

echo "Converting HumanoidArena OpenDoor TWIST2 demonstrations"
echo "  raw:      $RAW_TWIST2"
echo "  output:   $OUTPUT"
echo "  episodes: $EXPECTED_EPISODES"
echo "  videos:   re-encoded H.264"

"$DATA_PYTHON" "$CONVERTER" \
  --kind open_door \
  --src "$RAW_ROOT" \
  --branches twist2/zz \
  --out "$OUTPUT" \
  "${overwrite_args[@]}"

"$DATA_PYTHON" - "$OUTPUT" "$EXPECTED_EPISODES" <<'PY'
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

root = Path(sys.argv[1]).resolve()
expected = int(sys.argv[2])
info = json.loads((root / "meta/info.json").read_text(encoding="utf-8"))
episodes = [
    json.loads(line)
    for line in (root / "meta/episodes.jsonl").read_text(encoding="utf-8").splitlines()
    if line.strip()
]
videos = sorted((root / "videos").rglob("*.mp4"))
parquets = sorted((root / "data").rglob("*.parquet"))
config = info["script_config"]

assert config["schema_version"] == "arena_open_door_twist2_rot6d59_v1"
assert config["source_families"] == ["twist2"]
assert info["features"]["observation.full_state_rot6d"]["shape"] == [52]
assert info["features"]["action.policy_action_rot6d59"]["shape"] == [59]
assert len(episodes) == len(videos) == len(parquets) == expected
assert all("/twist2/zz/" in row["source_npz"] for row in episodes)
assert all(video.is_file() and not video.is_symlink() for video in videos)

sample = pq.read_table(parquets[0])
assert sample.column("observation.full_state_rot6d").type.list_size == 52
assert sample.column("action.policy_action_rot6d59").type.list_size == 59
print(
    f"Validated OpenDoor TWIST2 rot6d59: episodes={expected} "
    f"frames={info['total_frames']} state=52D action=59D videos=H.264"
)
PY
