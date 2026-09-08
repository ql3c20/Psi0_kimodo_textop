#!/usr/bin/env bash
set -euo pipefail

# GR00T N1.7 prefix RTC -> Kimodo -> TextOp on the 2026-06-25 task2 MuJoCo scene.

PSI0_ROOT="${PSI0_ROOT:-/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Psi0}"
GR00T_ROOT="${GR00T_ROOT:-/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Isaac-GR00T}"
BASE_SCRIPT="${PSI0_ROOT}/scripts/deploy/fullstate_task1_gr00t_rot6d59_kimodo_textop_eval.sh"

export GR00T_MODEL_PATH="${GR00T_MODEL_PATH:-${GR00T_ROOT}/outputs/task2-gr00t-n17-rot6d59-kimodo-textop-prefix-rtc-67ep/checkpoint-120000}"
export GR00T_PREFIX_RTC="${GR00T_PREFIX_RTC:-1}"
export GR00T_USE_RTC="${GR00T_USE_RTC:-1}"
export GR00T_EXECUTION_HORIZON="${GR00T_EXECUTION_HORIZON:-34}"
export GR00T_PORT="${GR00T_PORT:-22096}"

export FULLSTATE_GR00T_TASK="G1Fullstate20260625Task2-v0"
export FULLSTATE_GR00T_EVAL_DIR="${FULLSTATE_TASK2_GR00T_EVAL_DIR:-data/evals_fullstate_20260625_task2_gr00t_rot6d59_kimodo_textop_prefixrtc}"
export FULLSTATE_GR00T_KIMODO_WORK_DIR="${FULLSTATE_TASK2_GR00T_KIMODO_WORK_DIR:-${PSI0_ROOT}/outputs/kimodo_fullstate_20260625_task2_gr00t_rot6d59_prefixrtc}"
export MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-900}"

export HUMANOID_VLA_MJ_ROOT="${HUMANOID_VLA_MJ_ROOT:-/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/HumanoidVLA_MJ}"
export TASK2_INIT_FROM_RECORDINGS="${TASK2_INIT_FROM_RECORDINGS:-1}"
export TASK2_RECORDINGS_DIR="${TASK2_RECORDINGS_DIR:-${HUMANOID_VLA_MJ_ROOT}/output/20260625_task2_67}"
# Leave TASK2_RECORDING_INDEX unset to use the task's seeded shuffled order.
# Set it explicitly to a zero-based index (for example, 2) to fix every
# episode to one recording.
if [[ -n "${TASK2_RECORDING_INDEX:-}" ]]; then
  export TASK2_RECORDING_INDEX
else
  unset TASK2_RECORDING_INDEX
fi

# Use the Task2 SIMPLE scene with MuJoCo physics and Isaac Ego rendering while
# keeping the GR00T -> Kimodo -> TextOp controller chain unchanged.
if [[ "${TASK234_STRICT_ISAAC_EVAL:-0}" == "1" ]]; then
  export FULLSTATE_GR00T_TASK="G1Fullstate20260805Task2IsaacEval-v0"
  export FULLSTATE_GR00T_SIM_MODE="mujoco_isaac"
  export SIMPLE_TASK_ASSETS_ROOT="${SIMPLE_TASK_ASSETS_ROOT:-/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/assets}"
  export TASK2_INIT_FROM_RECORDINGS=0
  export FULLSTATE_GR00T_EVAL_DIR="${FULLSTATE_TASK2_GR00T_EVAL_DIR:-data/evals_task2_20260805_strict_isaac_gr00t_kimodo_textop}"
fi

exec bash "$BASE_SCRIPT" "${1:-}"
