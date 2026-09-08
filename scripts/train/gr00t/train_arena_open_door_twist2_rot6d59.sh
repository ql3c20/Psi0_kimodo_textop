#!/usr/bin/env bash
set -euo pipefail

PSI0_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
GR00T_ROOT="${GR00T_ROOT:-/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Isaac-GR00T}"
DATASET_PATH="${DATASET_PATH:-$PSI0_ROOT/data/output/arena_open_door_twist2_rot6d59_v1}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-$PSI0_ROOT/checkpoints/GR00T-N1.7-3B}"
MODALITY_CONFIG_PATH="${MODALITY_CONFIG_PATH:-$GR00T_ROOT/examples/unitree_g1_rot6d59_config.py}"
BACKBONE_PATH="${GR00T_BACKBONE_PATH:-$PSI0_ROOT/huggingface/hub/models--nvidia--Cosmos-Reason2-2B/snapshots/9ce19a195e423419c349abfc86fd07178b230561}"
PREFIX_RTC_MODULE="${GR00T_PREFIX_RTC_MODULE:-$PSI0_ROOT/scripts/deploy/gr00t_n17_prefix_rtc.py}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
VISIBLE_GPU_COUNT="$(awk -F, '{print NF}' <<< "$CUDA_VISIBLE_DEVICES")"
NUM_GPUS="${NUM_GPUS:-$VISIBLE_GPU_COUNT}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-512}"
MAX_STEPS="${MAX_STEPS:-10000}"
SAVE_STEPS="${SAVE_STEPS:-5000}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
MASTER_PORT="${MASTER_PORT:-29547}"
USE_WANDB="${USE_WANDB:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-gr00t-n1.7}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-$GR00T_ROOT/outputs/arena-open-door-twist2-gr00t-n17-rot6d59-prefixrtc-delay0to12-${NUM_GPUS}gpu-bs${GLOBAL_BATCH_SIZE}-step${MAX_STEPS}}"

DEFAULT_PYTHON="$GR00T_ROOT/.venv/bin/python"
FALLBACK_PYTHON="/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/miniconda3/envs/kimodo/bin/python3.10"
GR00T_SITE_PACKAGES="$GR00T_ROOT/.venv/lib/python3.10/site-packages"
EXTRA_PYTHONPATH="${GR00T_PYTHONPATH:-}"
if [[ -z "${GR00T_PYTHON:-}" ]]; then
  if [[ -x "$DEFAULT_PYTHON" ]] && "$DEFAULT_PYTHON" -c \
      'import sys, torch, transformers, gr00t, wandb; assert sys.version_info[:2] == (3, 10)' \
      >/dev/null 2>&1; then
    GR00T_PYTHON="$DEFAULT_PYTHON"
  else
    GR00T_PYTHON="$FALLBACK_PYTHON"
    EXTRA_PYTHONPATH="$GR00T_SITE_PACKAGES${EXTRA_PYTHONPATH:+:$EXTRA_PYTHONPATH}"
  fi
fi
EFFECTIVE_PYTHONPATH="$GR00T_ROOT${EXTRA_PYTHONPATH:+:$EXTRA_PYTHONPATH}${PYTHONPATH:+:$PYTHONPATH}"

for required in \
  "$GR00T_PYTHON" \
  "$DATASET_PATH/meta/info.json" \
  "$BASE_MODEL_PATH/config.json" \
  "$MODALITY_CONFIG_PATH" \
  "$BACKBONE_PATH/config.json" \
  "$BACKBONE_PATH/model.safetensors" \
  "$PREFIX_RTC_MODULE" \
  "$GR00T_ROOT/gr00t/experiment/launch_finetune.py"; do
  if [[ ! -f "$required" ]]; then
    echo "Missing required file: $required" >&2
    exit 2
  fi
done
for boolean_name in USE_WANDB PREFLIGHT_ONLY; do
  boolean_value="${!boolean_name}"
  if [[ "$boolean_value" != "0" && "$boolean_value" != "1" ]]; then
    echo "$boolean_name must be 0 or 1; got $boolean_value" >&2
    exit 2
  fi
done
if [[ "$VISIBLE_GPU_COUNT" -ne "$NUM_GPUS" ]]; then
  echo "NUM_GPUS=$NUM_GPUS does not match CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" >&2
  exit 2
fi
if ((GLOBAL_BATCH_SIZE % NUM_GPUS != 0)); then
  echo "GLOBAL_BATCH_SIZE=$GLOBAL_BATCH_SIZE must be divisible by NUM_GPUS=$NUM_GPUS" >&2
  exit 2
fi

PYTHONPATH="$EFFECTIVE_PYTHONPATH" "$GR00T_PYTHON" - "$DATASET_PATH" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
info = json.loads((root / "meta/info.json").read_text(encoding="utf-8"))
config = info["script_config"]
assert config["schema_version"] == "arena_open_door_twist2_rot6d59_v1"
assert config["source_families"] == ["twist2"]
assert info["features"]["observation.full_state_rot6d"]["shape"] == [52]
assert info["features"]["action.policy_action_rot6d59"]["shape"] == [59]
assert info["total_episodes"] == info["total_videos"] == 100
print(f"Dataset validated: episodes=100 frames={info['total_frames']} state=52D action=59D")
PY

CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
PYTHONPATH="$EFFECTIVE_PYTHONPATH" \
  "$GR00T_PYTHON" -c \
    "import sys, torch, transformers, gr00t, wandb; assert sys.version_info[:2] == (3, 10); assert torch.cuda.is_available(); assert torch.cuda.device_count() == $NUM_GPUS"

echo "Arena OpenDoor TWIST2 rot6d59 Prefix-RTC training"
echo "  GPUs:    $CUDA_VISIBLE_DEVICES"
echo "  batch:   $GLOBAL_BATCH_SIZE (per GPU $((GLOBAL_BATCH_SIZE / NUM_GPUS)))"
echo "  steps:   $MAX_STEPS; save every $SAVE_STEPS"
echo "  RTC:     prefix, delay 0..12, timestep groot_clean"
echo "  python:  $GR00T_PYTHON"
echo "  dataset: $DATASET_PATH"
echo "  output:  $OUTPUT_DIR"
if [[ "$PREFLIGHT_ONLY" == "1" ]]; then
  echo "Preflight completed; training was not started."
  exit 0
fi

latest_step=0
shopt -s nullglob
for checkpoint in "$OUTPUT_DIR"/checkpoint-*; do
  step="${checkpoint##*/checkpoint-}"
  if [[ -d "$checkpoint" && "$step" =~ ^[0-9]+$ ]] && ((10#$step > latest_step)); then
    latest_step=$((10#$step))
  fi
done
shopt -u nullglob
if ((latest_step >= MAX_STEPS)); then
  echo "checkpoint-$latest_step already reached MAX_STEPS=$MAX_STEPS; nothing to do."
  exit 0
fi

resume_args=()
if ((latest_step > 0)); then
  resume_args+=(--resume_from_checkpoint)
  echo "Resuming from checkpoint-$latest_step."
fi
wandb_args=()
if [[ "$USE_WANDB" == "1" ]]; then
  wandb_args+=(--use_wandb --wandb_project "$WANDB_PROJECT")
fi

mkdir -p "$OUTPUT_DIR"
cd "$GR00T_ROOT"
CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
GR00T_BACKBONE_PATH="$BACKBONE_PATH" \
GR00T_PREFIX_RTC_MODULE="$PREFIX_RTC_MODULE" \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
PYTHONPATH="$EFFECTIVE_PYTHONPATH" \
  "$GR00T_PYTHON" -m torch.distributed.run \
    --nproc_per_node="$NUM_GPUS" \
    --master_port="$MASTER_PORT" \
    gr00t/experiment/launch_finetune.py \
    --base_model_path "$BASE_MODEL_PATH" \
    --dataset_path "$DATASET_PATH" \
    --modality_config_path "$MODALITY_CONFIG_PATH" \
    --embodiment_tag NEW_EMBODIMENT \
    --num_gpus "$NUM_GPUS" \
    --output_dir "$OUTPUT_DIR" \
    --save_steps "$SAVE_STEPS" \
    --save_total_limit 5 \
    --max_steps "$MAX_STEPS" \
    --warmup_ratio 0.05 \
    --weight_decay 1e-5 \
    --learning_rate 1e-4 \
    --global_batch_size "$GLOBAL_BATCH_SIZE" \
    --color_jitter_params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
    --dataloader_num_workers "$DATALOADER_NUM_WORKERS" \
    --shard_size 1024 \
    --num_shards_per_epoch 100000 \
    --episode_sampling_rate 0.1 \
    --train_prefix_rtc \
    --prefix_rtc_timestep_mode groot_clean \
    --train_rtc \
    --train_rtc_min_delay 0 \
    --train_rtc_max_delay 12 \
    "${wandb_args[@]}" \
    "${resume_args[@]}"
