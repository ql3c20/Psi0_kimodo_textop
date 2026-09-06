#!/usr/bin/env bash
set -euo pipefail

export TASK_NUMBER=2
export FULLSTATE_GR00T_TASK=G1Fullstate20260805Task2-v0
export TASK_RECORDINGS_DIR="${TASK_RECORDINGS_DIR:-/home/ubuntu/yzh/mujoco_recordings/20260805_task2_new}"
export TASK_HSSD_USD="${TASK_HSSD_USD:-/home/ubuntu/yzh/Psi0_kimodo_textop/third_party/SIMPLE/data/scenes/hssd/102344280/102344280.usd}"
export TASK_EGO_EYE="${TASK_EGO_EYE:-0.06 0.06 0.45}"
export TASK_TRANSLATE="${TASK_TRANSLATE:-0.0 0.0 0.0}"
export TASK2_TRASH_TRANSLATE="${TASK2_TRASH_TRANSLATE:-0.0 0.0 0.0}"
export ISAAC_TRASH_TRANSLATE="$TASK2_TRASH_TRANSLATE"
export GR00T_MODEL_PATH="${GR00T_MODEL_PATH:-/home/ubuntu/yzh/ckpt/gr00tn17/task2_newbg}"
export ISAAC_RANDOMIZE_LIGHTING="${ISAAC_RANDOMIZE_LIGHTING:-1}"
export ISAAC_LIGHTING_SEED="${ISAAC_LIGHTING_SEED:-42}"
export MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-900}"
export GR00T_PREFIX_RTC="${GR00T_PREFIX_RTC:-1}"
export GR00T_USE_RTC="${GR00T_USE_RTC:-1}"
exec bash "${PSI0_ROOT:-/home/ubuntu/yzh/Psi0_kimodo_textop}/scripts/deploy/fullstate_task1_gr00t_rot6d59_kimodo_textop_eval.sh" "${1:-}"
