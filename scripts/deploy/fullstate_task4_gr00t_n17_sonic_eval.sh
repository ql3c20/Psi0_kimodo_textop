#!/usr/bin/env bash
set -euo pipefail

# GR00T N1.7 Prefix-RTC -> SONIC on the 2026-07-29 Task4 bottle-box scene.

export PSI0_ROOT="${PSI0_ROOT:-/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Psi0}"
export SIMPLE_ROOT="${SIMPLE_ROOT:-${PSI0_ROOT}/third_party/SIMPLE}"
export GR00T_ROOT="${GR00T_ROOT:-/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Isaac-GR00T}"
BASE_SCRIPT="${PSI0_ROOT}/scripts/deploy/fullstate_task1_gr00t_n17_sonic_eval.sh"

# This must be the 46-state/78-action UNITREE_G1_SONIC checkpoint.  The
# rot6d59 Kimodo/TextOp checkpoint is a different action schema.
export GR00T_MODEL_PATH="${GR00T_MODEL_PATH:-${GR00T_ROOT}/outputs/task4-gr00t-n17-sonic-prefix-rtc-groot-clean/checkpoint-80000}"
export GR00T_BACKBONE_PATH="${GR00T_BACKBONE_PATH:-${PSI0_ROOT}/huggingface/hub/models--nvidia--Cosmos-Reason2-2B/snapshots/9ce19a195e423419c349abfc86fd07178b230561}"
export GR00T_PREFIX_RTC="${GR00T_PREFIX_RTC:-1}"
export GR00T_PREFIX_RTC_TIMESTEP_MODE="${GR00T_PREFIX_RTC_TIMESTEP_MODE:-groot_clean}"
export GR00T_EXECUTION_HORIZON="${GR00T_EXECUTION_HORIZON:-34}"
export GR00T_PORT="${GR00T_PORT:-22095}"

export GR00T_SONIC_TASK="G1Fullstate20260729Task4-v0"
export GR00T_SONIC_EVAL_DIR="${GR00T_SONIC_EVAL_DIR:-data/evals_task4_gr00t_n17_sonic_prefix_rtc_groot_clean_ckpt80000}"
export SONIC_DECODER_ONNX="${SONIC_DECODER_ONNX:-/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/SONIC_my/gear_sonic_deploy/policy/release/model_decoder.onnx}"
export MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-500}"

export HUMANOID_VLA_MJ_ROOT="${HUMANOID_VLA_MJ_ROOT:-/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/HumanoidVLA_MJ_1}"
export TASK4_INIT_FROM_RECORDINGS="${TASK4_INIT_FROM_RECORDINGS:-1}"
export TASK4_RECORDINGS_DIR="${TASK4_RECORDINGS_DIR:-${HUMANOID_VLA_MJ_ROOT}/output/20260729_task4}"
export TASK4_RECORDING_SEED="${TASK4_RECORDING_SEED:-0}"
export TASK4_ALLOW_INSTRUCTION_OVERRIDE="${TASK4_ALLOW_INSTRUCTION_OVERRIDE:-0}"
export TASK4_INSTRUCTION="${TASK4_INSTRUCTION:-Walk forward and put the bottle into the box.}"

if [[ -n "${TASK4_RECORDING_INDEX:-}" ]]; then
  export TASK4_RECORDING_INDEX
else
  unset TASK4_RECORDING_INDEX
fi

exec bash "$BASE_SCRIPT" "${1:-}"
