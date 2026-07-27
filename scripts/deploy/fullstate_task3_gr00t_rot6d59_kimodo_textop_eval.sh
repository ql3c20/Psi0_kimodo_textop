#!/usr/bin/env bash
set -euo pipefail

# GR00T N1.7 prefix RTC -> Kimodo -> TextOp on the 2026-06-12 task3 pedal-bin scene.

PSI0_ROOT="${PSI0_ROOT:-/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Psi0}"
GR00T_ROOT="${GR00T_ROOT:-/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Isaac-GR00T}"
BASE_SCRIPT="${PSI0_ROOT}/scripts/deploy/fullstate_task1_gr00t_rot6d59_kimodo_textop_eval.sh"

export GR00T_MODEL_PATH="${GR00T_MODEL_PATH:-${GR00T_ROOT}/outputs/task3-gr00t-n17-rot6d59-kimodo-textop-prefix-rtc-round2}"
export GR00T_PREFIX_RTC="${GR00T_PREFIX_RTC:-1}"
export GR00T_USE_RTC="${GR00T_USE_RTC:-1}"
export GR00T_EXECUTION_HORIZON="${GR00T_EXECUTION_HORIZON:-34}"
export GR00T_PORT="${GR00T_PORT:-22096}"

export FULLSTATE_GR00T_TASK="G1Fullstate20260612Task3-v0"
export FULLSTATE_GR00T_EVAL_DIR="${FULLSTATE_TASK3_GR00T_EVAL_DIR:-data/evals_fullstate_20260612_task3_gr00t_rot6d59_kimodo_textop_prefixrtc_round2}"
export FULLSTATE_GR00T_KIMODO_WORK_DIR="${FULLSTATE_TASK3_GR00T_KIMODO_WORK_DIR:-${PSI0_ROOT}/outputs/kimodo_fullstate_20260612_task3_gr00t_rot6d59_prefixrtc_round2}"
export MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-800}"

export HUMANOID_VLA_MJ_ROOT="${HUMANOID_VLA_MJ_ROOT:-/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/HumanoidVLA_MJ}"
export TASK3_INIT_FROM_RECORDINGS="${TASK3_INIT_FROM_RECORDINGS:-1}"
export TASK3_RECORDINGS_DIR="${TASK3_RECORDINGS_DIR:-${HUMANOID_VLA_MJ_ROOT}/output/20260612_task3}"
# Leave TASK3_RECORDING_INDEX unset to use a seeded shuffled pass over all
# recordings. Set it to a zero-based index to fix every episode to one reset.
if [[ -n "${TASK3_RECORDING_INDEX:-}" ]]; then
  export TASK3_RECORDING_INDEX
else
  unset TASK3_RECORDING_INDEX
fi

exec bash "$BASE_SCRIPT" "${1:-}"
