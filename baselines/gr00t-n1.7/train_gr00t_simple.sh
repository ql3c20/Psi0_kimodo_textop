#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PSI0_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
ISAAC_GR00T_ROOT="${ISAAC_GR00T_ROOT:-/pfs/pfs-ilWc5D/yzh/Isaac-GR00T}"
export HF_HOME="${HF_HOME:-$PSI0_ROOT/huggingface}"
export GR00T_BACKBONE_PATH="${GR00T_BACKBONE_PATH:-$HF_HOME/hub/models--nvidia--Cosmos-Reason2-2B/snapshots/9ce19a195e423419c349abfc86fd07178b230561}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

DATASET_PATH="${GR00T_DATASET_PATH:-$PSI0_ROOT/data/simple/G1WholebodyXMovePickTeleop-v0}"
BASE_MODEL_PATH="${PRETRAINED_MODEL_PATH:-$PSI0_ROOT/checkpoints/GR00T-N1.7-3B}"
OUTPUT_DIR="${OUTPUT_DIR:-$PSI0_ROOT/checkpoints/gr00t_n1d7_movepick_decoupled_wbc}"
MODALITY_CONFIG_PATH="${MODALITY_CONFIG_PATH:-$SCRIPT_DIR/g1_decoupled_wbc_config.py}"

for path in "$ISAAC_GR00T_ROOT" "$DATASET_PATH" "$BASE_MODEL_PATH" "$MODALITY_CONFIG_PATH" "$GR00T_BACKBONE_PATH"; do
  if [[ ! -e "$path" ]]; then
    echo "Missing required path: $path" >&2
    exit 1
  fi
done

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export NUM_GPUS="${NUM_GPUS:-1}"
export MASTER_PORT="${MASTER_PORT:-29531}"
export MAX_STEPS="${MAX_STEPS:-10000}"
export SAVE_STEPS="${SAVE_STEPS:-1000}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-32}"
export DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
export GR00T_ACTION_HORIZON="${GR00T_ACTION_HORIZON:-40}"
export USE_WANDB="${GR00T_USE_WANDB:-0}"
WANDB_PROJECT_NAME="${WANDB_PROJECT:-psi}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-gr00t_movepick_teleop_n17}"

cd "$ISAAC_GR00T_ROOT"

# Statistics depend on the modality config and chosen action horizon. Set
# RECOMPUTE_STATS=0 only after they have already been generated successfully.
if [[ "${RECOMPUTE_STATS:-1}" == "1" ]]; then
  uv run python gr00t/data/stats.py \
    --dataset-path "$DATASET_PATH" \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path "$MODALITY_CONFIG_PATH"
fi

exec uv run bash examples/finetune.sh \
  --base-model-path "$BASE_MODEL_PATH" \
  --dataset-path "$DATASET_PATH" \
  --modality-config-path "$MODALITY_CONFIG_PATH" \
  --embodiment-tag NEW_EMBODIMENT \
  --output-dir "$OUTPUT_DIR" \
  --experiment-name "$EXPERIMENT_NAME" \
  --wandb-project "$WANDB_PROJECT_NAME" \
  "$@"
