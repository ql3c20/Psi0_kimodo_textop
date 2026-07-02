#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

extra_args=()
if [[ "${GR00T_USE_WANDB:-1}" != "0" ]]; then
  extra_args+=(--use-wandb)
fi

export WANDB_PROJECT="${WANDB_PROJECT:-psi}"

exec python3 "$SCRIPT_DIR/finetune_gr00t.py" \
  --preset finetune_simple \
  --dataset-path "${GR00T_DATASET_PATH:-/pfs/pfs-ilWc5D/yzh/Psi0/data/simple/G1WholebodyXMovePickTeleop-v0}" \
  --base-model-path "${PRETRAINED_MODEL_PATH:-/pfs/pfs-ilWc5D/yzh/Psi0/checkpoints/nvidia_GR00T-N1.6-3B}" \
  --output-dir "${OUTPUT_DIR:-/pfs/pfs-ilWc5D/yzh/Psi0/checkpoints/gr00t_movepick_teleop}" \
  --cuda-visible-devices "${CUDA_VISIBLE_DEVICES:-1,2,3,4}" \
  --num-gpus "${NUM_GPUS:-4}" \
  --master-port "${MASTER_PORT:-29501}" \
  "${extra_args[@]}" \
  "$@"
