#!/usr/bin/env bash
set -euo pipefail

PSI0_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
RAW_ROOT="${RAW_ROOT:-/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/dataset/HumanoidArena_open_door/HSI_open_door}"
RAW_SONIC="${RAW_SONIC:-$RAW_ROOT/sonic/zz}"
DATA_PYTHON="${DATA_PYTHON:-$PSI0_ROOT/third_party/SIMPLE/.venv/bin/python}"
ROT6D_OUTPUT="${ROT6D_OUTPUT:-$PSI0_ROOT/data/output/arena_open_door_sonic_rot6d59_v1}"
NATIVE_OUTPUT="${NATIVE_OUTPUT:-$PSI0_ROOT/data/output/arena_open_door_sonic_native_realized_v1}"
EXPECTED_EPISODES="${EXPECTED_EPISODES:-100}"
OVERWRITE="${OVERWRITE:-0}"
RUN_ROT6D="${RUN_ROT6D:-1}"
RUN_NATIVE="${RUN_NATIVE:-1}"

ROT6D_CONVERTER="$PSI0_ROOT/scripts/data/convert_arena_pp_box_to_rot6d59_lerobot.py"
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
for flag_name in OVERWRITE RUN_ROT6D RUN_NATIVE; do
  flag_value="${!flag_name}"
  if [[ "$flag_value" != "0" && "$flag_value" != "1" ]]; then
    echo "$flag_name must be 0 or 1; got $flag_value" >&2
    exit 2
  fi
done
if [[ "$RUN_ROT6D" == "0" && "$RUN_NATIVE" == "0" ]]; then
  echo "At least one of RUN_ROT6D or RUN_NATIVE must be 1." >&2
  exit 2
fi
if [[ ! -d "$RAW_SONIC" ]]; then
  echo "Missing raw OpenDoor SONIC recordings: $RAW_SONIC" >&2
  exit 2
fi

raw_episode_count="$(find "$RAW_SONIC" -type f -name '*.npz' | wc -l)"
raw_video_count="$(find "$RAW_SONIC" -type f -name '*.mp4' | wc -l)"
if [[ "$raw_episode_count" -ne "$EXPECTED_EPISODES" ]]; then
  echo "Expected $EXPECTED_EPISODES raw SONIC episodes; found $raw_episode_count." >&2
  exit 2
fi
if [[ "$raw_video_count" -ne "$EXPECTED_EPISODES" ]]; then
  echo "Expected $EXPECTED_EPISODES raw SONIC videos; found $raw_video_count." >&2
  exit 2
fi

"$DATA_PYTHON" -c \
  'import sys, mujoco, numpy, onnxruntime, pyarrow; assert sys.version_info[:2] == (3, 10)'

overwrite_args=()
if [[ "$OVERWRITE" == "1" ]]; then
  overwrite_args+=(--overwrite)
fi

echo "Converting HumanoidArena OpenDoor SONIC demonstrations"
echo "  raw:            $RAW_SONIC"
echo "  rot6d59 output: $ROT6D_OUTPUT"
echo "  native output:  $NATIVE_OUTPUT"
echo "  episodes:       $EXPECTED_EPISODES"
echo "  videos:         relative symlinks to raw MP4 files"
echo "  stages:         rot6d59=$RUN_ROT6D native_realized=$RUN_NATIVE"

# The rot6d59 view links directly to the raw MP4. The native SONIC view then
# links to the matching rot6d59 video, so neither converted dataset duplicates
# video bytes.
if [[ "$RUN_ROT6D" == "1" ]]; then
  "$DATA_PYTHON" "$ROT6D_CONVERTER" \
    --kind open_door \
    --src "$RAW_ROOT" \
    --branches sonic/zz \
    --out "$ROT6D_OUTPUT" \
    --link-videos \
    "${overwrite_args[@]}"
fi

if [[ "$RUN_NATIVE" == "1" ]]; then
  "$DATA_PYTHON" "$NATIVE_CONVERTER" \
    --kind open_door \
    --src "$RAW_SONIC" \
    --out "$NATIVE_OUTPUT" \
    --expected-episodes "$EXPECTED_EPISODES" \
    --motion-token-source future_reencode \
    --source-family sonic \
    --keep-final-frame \
    --reuse-videos-from "$ROT6D_OUTPUT" \
    "${overwrite_args[@]}"
fi

"$DATA_PYTHON" - "$RAW_SONIC" "$ROT6D_OUTPUT" "$NATIVE_OUTPUT" "$EXPECTED_EPISODES" <<'PY'
import json
import sys
from pathlib import Path

raw_root = Path(sys.argv[1]).resolve()
rot6d_root = Path(sys.argv[2]).resolve()
native_root = Path(sys.argv[3]).resolve()
expected_episodes = int(sys.argv[4])


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

assert rot6d_info["script_config"]["schema_version"] == "arena_open_door_sonic_rot6d59_v1"
native_config = native_info["script_config"]
assert native_config["schema_version"] == "arena_sonic_native_realized_v1"
assert native_config["source_family"] == "sonic"
assert native_config["motion_token_source"] == "future_reencode"
assert native_config["drop_final_frame"] is False
assert "robot_qpos_before_decimation" in native_config["realized_body_source"]
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
    assert all(video.is_symlink() for video in videos), root
    assert all(video.is_file() for video in videos), root
    assert all(video.resolve().is_relative_to(raw_root) for video in videos), root

print(
    "Validated OpenDoor SONIC datasets: "
    f"episodes={expected_episodes} frames={rot6d_info['total_frames']} "
    "rot6d59=52D/59D native_sonic_realized=46D/78D videos=symlinked"
)
PY

echo "Requested OpenDoor SONIC-derived datasets are ready."
