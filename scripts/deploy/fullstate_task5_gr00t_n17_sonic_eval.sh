#!/usr/bin/env bash
set -euo pipefail

# GR00T N1.7 native SONIC -> SONIC on the 2026-08-25 Task5 drawer scene.

export PSI0_ROOT="${PSI0_ROOT:-/home/ubuntu/yzh/Psi0_kimodo_textop}"
export SIMPLE_ROOT="${SIMPLE_ROOT:-${PSI0_ROOT}/third_party/SIMPLE}"
export GR00T_ROOT="${GR00T_ROOT:-/home/ubuntu/yzh/Isaac-GR00T-rtc}"
BASE_SCRIPT="${PSI0_ROOT}/scripts/deploy/fullstate_task1_gr00t_n17_sonic_eval.sh"

export GR00T_MODEL_PATH="${GR00T_MODEL_PATH:-/home/ubuntu/yzh/ckpt/gr00tn17/task5_newlight_sonic}"
export GR00T_BACKBONE_PATH="${GR00T_BACKBONE_PATH:-${GR00T_ROOT}/huggingface/Cosmos-Reason2-2B}"
export GR00T_PREFIX_RTC="${GR00T_PREFIX_RTC:-0}"
export GR00T_EXECUTION_HORIZON="${GR00T_EXECUTION_HORIZON:-40}"
export GR00T_PORT="${GR00T_PORT:-22095}"

export GR00T_SONIC_TASK="G1Fullstate20260825Task5-v0"
export GR00T_SONIC_EVAL_DIR="${GR00T_SONIC_EVAL_DIR:-data/evals_task5_scq_gr00t_n17_sonic_newlight_ckpt80k}"
export SONIC_DECODER_ONNX="${SONIC_DECODER_ONNX:-/home/ubuntu/yzh/GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/model_decoder.onnx}"
export MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-500}"

export HUMANOID_VLA_MJ_ROOT="${HUMANOID_VLA_MJ_ROOT:-/home/ubuntu/yzh/HumanoidVLA_MJ_backup/HumanoidVLA_MJ}"
export TASK5_RECORDINGS_DIR="${TASK5_RECORDINGS_DIR:-/home/ubuntu/yzh/mujoco_recordings/copy/20260825_task5_scq}"
export TASK5_RECORDING_SEED="${TASK5_RECORDING_SEED:-42}"
export TASK5_INSTRUCTION="${TASK5_INSTRUCTION:-Walk forward and pull open the upper drawer.}"

if [[ -n "${TASK5_RECORDING_INDEX:-}" ]]; then
  export TASK5_RECORDING_INDEX
else
  unset TASK5_RECORDING_INDEX 2>/dev/null || true
fi

exec bash "$BASE_SCRIPT" "${1:-}"
