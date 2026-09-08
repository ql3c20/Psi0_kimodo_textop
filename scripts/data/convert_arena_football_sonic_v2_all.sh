#!/usr/bin/env bash
set -euo pipefail

PSI0_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
RAW_ROOT="${RAW_ROOT:-/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/dataset/HumanoidArena_football}"
RAW_SONIC="${RAW_SONIC:-$RAW_ROOT/HOI_football_v2/SONIC}"
DATA_PYTHON="${DATA_PYTHON:-$PSI0_ROOT/third_party/SIMPLE/.venv/bin/python}"
ROT6D_OUTPUT="${ROT6D_OUTPUT:-$PSI0_ROOT/data/output/arena_football_sonic_rot6d59_v2}"
NATIVE_OUTPUT="${NATIVE_OUTPUT:-$PSI0_ROOT/data/output/arena_football_sonic_native_realized_v1}"
EXPECTED_EPISODES="${EXPECTED_EPISODES:-100}"
OVERWRITE="${OVERWRITE:-0}"

ROT6D_CONVERTER="$PSI0_ROOT/scripts/data/convert_arena_football_to_rot6d52_lerobot.py"
NATIVE_CONVERTER="$PSI0_ROOT/scripts/data/convert_arena_twist2_to_sonic_lerobot.py"

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Missing required file: $1" >&2
    exit 2
  fi
}

require_file "$DATA_PYTHON"
require_file "$ROT6D_CONVERTER"
require_file "$NATIVE_CONVERTER"
if [[ ! -d "$RAW_SONIC" ]]; then
  echo "Missing raw football SONIC recordings: $RAW_SONIC" >&2
  echo "Download HOI_football_v2/SONIC from ModelScope Twang2026/HumanoidArena_raw first." >&2
  exit 2
fi

raw_episode_count="$(find "$RAW_SONIC" -type f -name '*.npz' | wc -l)"
if [[ "$raw_episode_count" -ne "$EXPECTED_EPISODES" ]]; then
  echo "Expected $EXPECTED_EPISODES raw SONIC episodes under $RAW_SONIC; found $raw_episode_count." >&2
  exit 2
fi

"$DATA_PYTHON" -c \
  'import sys, mujoco, numpy, onnxruntime, pyarrow; assert sys.version_info[:2] == (3, 10)'

overwrite_args=()
if [[ "$OVERWRITE" == "1" ]]; then
  overwrite_args+=(--overwrite)
fi

echo "Converting HumanoidArena football SONIC demonstrations"
echo "  raw:            $RAW_SONIC"
echo "  rot6d59 output: $ROT6D_OUTPUT"
echo "  native output:  $NATIVE_OUTPUT"
echo "  episodes:       $EXPECTED_EPISODES"

# Build rot6d59 first. Its normalized H.264 videos are reused by the native
# dataset through relative symlinks, avoiding a second physical video copy.
"$DATA_PYTHON" "$ROT6D_CONVERTER" \
  --src "$RAW_SONIC" \
  --source-family sonic \
  --out "$ROT6D_OUTPUT" \
  "${overwrite_args[@]}"

"$DATA_PYTHON" "$NATIVE_CONVERTER" \
  --kind football \
  --src "$RAW_SONIC" \
  --out "$NATIVE_OUTPUT" \
  --expected-episodes "$EXPECTED_EPISODES" \
  --motion-token-source future_reencode \
  --source-family sonic \
  --keep-final-frame \
  --reuse-videos-from "$ROT6D_OUTPUT" \
  "${overwrite_args[@]}"

"$DATA_PYTHON" - "$ROT6D_OUTPUT" "$NATIVE_OUTPUT" "$EXPECTED_EPISODES" <<'PY'
import json
import sys
from pathlib import Path

rot6d_root = Path(sys.argv[1]).resolve()
native_root = Path(sys.argv[2]).resolve()
expected_episodes = int(sys.argv[3])


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


rot6d_info = load_json(rot6d_root / "meta/info.json")
native_info = load_json(native_root / "meta/info.json")
rot6d_episodes = load_jsonl(rot6d_root / "meta/episodes.jsonl")
native_episodes = load_jsonl(native_root / "meta/episodes.jsonl")

assert rot6d_info["script_config"]["schema_version"] == "arena_football_sonic_rot6d59_v2"
assert native_info["script_config"]["schema_version"] == "arena_sonic_native_realized_v1"
assert native_info["script_config"]["source_family"] == "sonic"
assert native_info["script_config"]["motion_token_source"] == "future_reencode"
assert native_info["script_config"]["drop_final_frame"] is False
assert "robot_qpos_before_decimation" in native_info["script_config"]["realized_body_source"]
assert rot6d_info["features"]["observation.full_state_rot6d"]["shape"] == [52]
assert rot6d_info["features"]["action.policy_action_rot6d59"]["shape"] == [59]
assert native_info["features"]["observation.sonic_state"]["shape"] == [46]
assert native_info["features"]["action.sonic"]["shape"] == [78]
assert len(rot6d_episodes) == expected_episodes
assert len(native_episodes) == expected_episodes
assert rot6d_info["total_frames"] == native_info["total_frames"]
assert [row["source_npz"] for row in rot6d_episodes] == [
    row["source_npz"] for row in native_episodes
]

for root in (rot6d_root, native_root):
    videos = sorted((root / "videos").rglob("*.mp4"))
    assert len(videos) == expected_episodes, (root, len(videos))
    assert all(video.is_file() for video in videos), root

print(
    "Validated converted football SONIC datasets: "
    f"episodes={expected_episodes} frames={rot6d_info['total_frames']} "
    "rot6d59=52D/59D native_sonic_realized=46D/78D"
)
PY

echo "Both football SONIC-derived datasets are ready; native tokens encode realized robot motion."
