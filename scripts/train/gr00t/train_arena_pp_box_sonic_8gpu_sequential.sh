#!/usr/bin/env bash
set -euo pipefail

PSI0_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
NATIVE_PSI0_ROOT="${NATIVE_PSI0_ROOT:-/pfs/pfs-oHNwH0/mnt/pfs/humanoid/wzl/Psi0}"
ROT6D_GR00T_ROOT="${ROT6D_GR00T_ROOT:-/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Isaac-GR00T}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
NUM_GPUS=8
GLOBAL_BATCH_SIZE=256
MAX_STEPS=20000
SAVE_STEPS="${SAVE_STEPS:-10000}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
USE_WANDB=1
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"
NATIVE_MASTER_PORT="${NATIVE_MASTER_PORT:-29531}"
ROT6D_MASTER_PORT="${ROT6D_MASTER_PORT:-29532}"

NATIVE_DATASET="${NATIVE_DATASET:-$PSI0_ROOT/data/output/arena_pp_box_sonic_native_realized_v1}"
ROT6D_DATASET="${ROT6D_DATASET:-$PSI0_ROOT/data/output/arena_pp_box_sonic_rot6d59_v3}"
NATIVE_PRESET="${NATIVE_PRESET:-$NATIVE_PSI0_ROOT/baselines/gr00t-n1.7/presets/train/finetune_arena_pp_box_sonic_native_v2_8gpu_bs256_step20000.yaml}"
NATIVE_LAUNCHER="$NATIVE_PSI0_ROOT/baselines/gr00t-n1.7/train_gr00t_n17_sonic_gr00t_new_4gpu.sh"
NATIVE_OUTPUT_ROOT="${NATIVE_OUTPUT_ROOT:-$NATIVE_PSI0_ROOT/checkpoints}"
NATIVE_EXPERIMENT_NAME="gr00t-n17-sonic-arena-pp-box-sonic-native-realized-v1-8gpu-bs${GLOBAL_BATCH_SIZE}-step${MAX_STEPS}"
NATIVE_RUN_DIR="$NATIVE_OUTPUT_ROOT/$NATIVE_EXPERIMENT_NAME"

ROT6D_BASE_MODEL="${ROT6D_BASE_MODEL:-$PSI0_ROOT/checkpoints/GR00T-N1.7-3B}"
ROT6D_MODALITY_CONFIG="${ROT6D_MODALITY_CONFIG:-$ROT6D_GR00T_ROOT/examples/unitree_g1_rot6d59_config.py}"
ROT6D_BACKBONE="${ROT6D_BACKBONE:-$PSI0_ROOT/huggingface/hub/models--nvidia--Cosmos-Reason2-2B/snapshots/9ce19a195e423419c349abfc86fd07178b230561}"
ROT6D_OUTPUT_DIR="${ROT6D_OUTPUT_DIR:-$ROT6D_GR00T_ROOT/outputs/arena-pp-box-sonic-gr00t-n17-rot6d59-v3-prefixrtc-delay0to12-8gpu-bs${GLOBAL_BATCH_SIZE}-step${MAX_STEPS}}"
ROT6D_PYTHON="${ROT6D_PYTHON:-$ROT6D_GR00T_ROOT/.venv/bin/python}"

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "Missing required file: $path" >&2
    exit 2
  fi
}

latest_checkpoint_step() {
  local run_dir="$1"
  local latest=0
  local checkpoint
  local checkpoint_name
  shopt -s nullglob
  for checkpoint in "$run_dir"/checkpoint-*; do
    [[ -d "$checkpoint" ]] || continue
    checkpoint_name="${checkpoint##*/checkpoint-}"
    if [[ "$checkpoint_name" =~ ^[0-9]+$ ]] && ((10#$checkpoint_name > latest)); then
      latest=$((10#$checkpoint_name))
    fi
  done
  shopt -u nullglob
  printf '%s\n' "$latest"
}

validate_common_settings() {
  local visible_gpu_count
  visible_gpu_count="$(awk -F, '{print NF}' <<< "$CUDA_VISIBLE_DEVICES")"
  if [[ "$visible_gpu_count" -ne "$NUM_GPUS" ]]; then
    echo "NUM_GPUS=$NUM_GPUS does not match CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" >&2
    exit 2
  fi
  if ((GLOBAL_BATCH_SIZE % NUM_GPUS != 0)); then
    echo "GLOBAL_BATCH_SIZE=$GLOBAL_BATCH_SIZE must be divisible by NUM_GPUS=$NUM_GPUS" >&2
    exit 2
  fi
  require_file "$NATIVE_DATASET/meta/info.json"
  require_file "$ROT6D_DATASET/meta/info.json"
  require_file "$NATIVE_PRESET"
  require_file "$NATIVE_LAUNCHER"
  require_file "$ROT6D_GR00T_ROOT/examples/finetune.sh"
  require_file "$ROT6D_BASE_MODEL/config.json"
  require_file "$ROT6D_MODALITY_CONFIG"
  require_file "$ROT6D_BACKBONE/config.json"
  require_file "$ROT6D_BACKBONE/model.safetensors"
  require_file "$ROT6D_PYTHON"
  "$ROT6D_PYTHON" -c \
    'import sys, torch, transformers, gr00t, wandb; assert sys.version_info[:2] == (3, 10)'

  echo "Arena PP-box SONIC sequential training"
  echo "  GPUs:                 $CUDA_VISIBLE_DEVICES"
  echo "  num GPUs:             $NUM_GPUS"
  echo "  global batch:         $GLOBAL_BATCH_SIZE"
  echo "  per-GPU batch:        $((GLOBAL_BATCH_SIZE / NUM_GPUS))"
  echo "  max steps:            $MAX_STEPS"
  echo "  order:                native SONIC -> rot6d59 Prefix-RTC"
  echo "  native output:        $NATIVE_RUN_DIR"
  echo "  rot6d59 output:       $ROT6D_OUTPUT_DIR"
}

run_native_sonic() {
  local latest_step
  local -a resume_args=()
  latest_step="$(latest_checkpoint_step "$NATIVE_RUN_DIR")"
  if ((latest_step >= MAX_STEPS)); then
    echo "Skipping native SONIC: checkpoint-$latest_step already reached target $MAX_STEPS."
    return 0
  fi
  if ((latest_step > 0)); then
    resume_args+=(--resume-from-checkpoint)
    echo "Resuming native SONIC from checkpoint-$latest_step."
  fi

  echo "Starting native SONIC training."
  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
  NUM_GPUS="$NUM_GPUS" \
  MASTER_PORT="$NATIVE_MASTER_PORT" \
  PRESET="$NATIVE_PRESET" \
  DATASET_PATH="$NATIVE_DATASET" \
  OUTPUT_DIR="$NATIVE_OUTPUT_ROOT" \
    bash "$NATIVE_LAUNCHER" \
      --experiment-name "$NATIVE_EXPERIMENT_NAME" \
      --max-steps "$MAX_STEPS" \
      --save-steps "$SAVE_STEPS" \
      --global-batch-size "$GLOBAL_BATCH_SIZE" \
      "${resume_args[@]}"
  echo "Native SONIC training completed successfully."
}

run_rot6d59_prefix_rtc() {
  local latest_step
  local -a resume_args=()
  latest_step="$(latest_checkpoint_step "$ROT6D_OUTPUT_DIR")"
  if ((latest_step >= MAX_STEPS)); then
    echo "Skipping rot6d59 Prefix-RTC: checkpoint-$latest_step already reached target $MAX_STEPS."
    return 0
  fi
  if ((latest_step > 0)); then
    resume_args+=(--resume-from-checkpoint)
    echo "Resuming rot6d59 Prefix-RTC from checkpoint-$latest_step."
  fi

  echo "Starting rot6d59 Prefix-RTC training with delay range [0, 12]."
  (
    cd "$ROT6D_GR00T_ROOT"
    export PATH="$ROT6D_GR00T_ROOT/.venv/bin:$PATH"
    export VIRTUAL_ENV="$ROT6D_GR00T_ROOT/.venv"
    CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
    NUM_GPUS="$NUM_GPUS" \
    MASTER_PORT="$ROT6D_MASTER_PORT" \
    MAX_STEPS="$MAX_STEPS" \
    SAVE_STEPS="$SAVE_STEPS" \
    GLOBAL_BATCH_SIZE="$GLOBAL_BATCH_SIZE" \
    DATALOADER_NUM_WORKERS="$DATALOADER_NUM_WORKERS" \
    TRAIN_PREFIX_RTC=1 \
    TRAIN_RTC_MIN_DELAY=0 \
    TRAIN_RTC_MAX_DELAY=12 \
    GR00T_PREFIX_RTC_TIMESTEP_MODE=groot_clean \
    GR00T_BACKBONE_PATH="$ROT6D_BACKBONE" \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    USE_WANDB="$USE_WANDB" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      bash examples/finetune.sh \
        --base-model-path "$ROT6D_BASE_MODEL" \
        --dataset-path "$ROT6D_DATASET" \
        --modality-config-path "$ROT6D_MODALITY_CONFIG" \
        --embodiment-tag NEW_EMBODIMENT \
        --output-dir "$ROT6D_OUTPUT_DIR" \
        --wandb-project gr00t-n1.7 \
        "${resume_args[@]}"
  )
  echo "rot6d59 Prefix-RTC training completed successfully."
}

validate_common_settings
if [[ "$PREFLIGHT_ONLY" == "1" ]]; then
  echo "Preflight completed; PREFLIGHT_ONLY=1, training was not started."
  exit 0
fi
run_native_sonic
run_rot6d59_prefix_rtc
echo "Both Arena PP-box training runs completed successfully."
