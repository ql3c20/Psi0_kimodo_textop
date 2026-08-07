#!/usr/bin/env bash
set -euo pipefail

# GR00T N1.7 prefix RTC -> Kimodo -> TextOp on the 2026-07-29 task4 scene.

PSI0_ROOT="${PSI0_ROOT:-/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Psi0}"
GR00T_ROOT="${GR00T_ROOT:-/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Isaac-GR00T}"
BASE_SCRIPT="${PSI0_ROOT}/scripts/deploy/fullstate_task1_gr00t_rot6d59_kimodo_textop_eval.sh"

export GR00T_MODEL_PATH="${GR00T_MODEL_PATH:-${GR00T_ROOT}/outputs/task4-gr00t-n17-rot6d59-kimodo-textop-prefix-rtc-groot-clean/checkpoint-160000}"
export GR00T_PREFIX_RTC="${GR00T_PREFIX_RTC:-1}"
export GR00T_USE_RTC="${GR00T_USE_RTC:-1}"
export GR00T_PREFIX_RTC_TIMESTEP_MODE="${GR00T_PREFIX_RTC_TIMESTEP_MODE:-groot_clean}"
export GR00T_EXECUTION_HORIZON="${GR00T_EXECUTION_HORIZON:-34}"
export GR00T_PORT="${GR00T_PORT:-22096}"

export FULLSTATE_GR00T_TASK="G1Fullstate20260729Task4-v0"
export MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-900}"

export HUMANOID_VLA_MJ_ROOT="${HUMANOID_VLA_MJ_ROOT:-/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/HumanoidVLA_MJ_1}"
export TASK4_INIT_FROM_RECORDINGS="${TASK4_INIT_FROM_RECORDINGS:-1}"
export TASK4_RECORDINGS_DIR="${TASK4_RECORDINGS_DIR:-${HUMANOID_VLA_MJ_ROOT}/output/20260729_task4}"
export TASK4_RECORDING_SEED="${TASK4_RECORDING_SEED:-0}"
export TASK4_BOTTLE_X_OFFSET="${TASK4_BOTTLE_X_OFFSET:-0}"
export TASK4_BOTTLE_Y_OFFSET="${TASK4_BOTTLE_Y_OFFSET:-0}"
export TASK4_ALLOW_INSTRUCTION_OVERRIDE="${TASK4_ALLOW_INSTRUCTION_OVERRIDE:-0}"
export TASK4_INSTRUCTION="${TASK4_INSTRUCTION:-Walk forward and put the bottle into the box.}"

# Leave TASK4_RECORDING_INDEX unset for a seeded shuffled pass over all 203
# recordings. Set a zero-based index to fix every reset for single-episode debug.
if [[ -n "${TASK4_RECORDING_INDEX:-}" ]]; then
  export TASK4_RECORDING_INDEX
else
  unset TASK4_RECORDING_INDEX
fi

if [[ -n "${TASK4_RECORDING_INDEX:-}" ]]; then
  TASK4_INIT_TAG="recording${TASK4_RECORDING_INDEX}"
else
  TASK4_INIT_TAG="recordingseed${TASK4_RECORDING_SEED}"
fi
TASK4_CHECKPOINT_TAG="${TASK4_CHECKPOINT_TAG:-${GR00T_MODEL_PATH##*/}}"
TASK4_VARIANT_TAG="${TASK4_VARIANT_TAG:-${TASK4_CHECKPOINT_TAG}_${TASK4_INIT_TAG}_xoff${TASK4_BOTTLE_X_OFFSET}_yoff${TASK4_BOTTLE_Y_OFFSET}}"
export FULLSTATE_GR00T_EVAL_DIR="${FULLSTATE_TASK4_GR00T_EVAL_DIR:-data/evals_fullstate_20260729_task4_gr00t_rot6d59_kimodo_textop_prefixrtc_grootclean_${TASK4_VARIANT_TAG}}"
export FULLSTATE_GR00T_KIMODO_WORK_DIR="${FULLSTATE_TASK4_GR00T_KIMODO_WORK_DIR:-${PSI0_ROOT}/outputs/kimodo_fullstate_20260729_task4_gr00t_rot6d59_prefixrtc_grootclean_${TASK4_VARIANT_TAG}}"

exec bash "$BASE_SCRIPT" "${1:-}"
