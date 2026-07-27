#!/usr/bin/env bash
set -euo pipefail

# GR00T N1.7 Prefix-RTC -> SONIC on the 2026-06-12 Task3 pedal-bin scene.

export PSI0_ROOT="${PSI0_ROOT:-/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Psi0}"
export SIMPLE_ROOT="${SIMPLE_ROOT:-${PSI0_ROOT}/third_party/SIMPLE}"
export GR00T_ROOT="${GR00T_ROOT:-/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Isaac-GR00T}"
BASE_SCRIPT="${PSI0_ROOT}/scripts/deploy/fullstate_task1_gr00t_n17_sonic_eval.sh"

export GR00T_MODEL_PATH="${GR00T_MODEL_PATH:-${GR00T_ROOT}/outputs/task3-gr00t-n17-sonic-prefix-rtc-round2}"
export GR00T_BACKBONE_PATH="${GR00T_BACKBONE_PATH:-${PSI0_ROOT}/huggingface/hub/models--nvidia--Cosmos-Reason2-2B/snapshots/9ce19a195e423419c349abfc86fd07178b230561}"
export GR00T_PREFIX_RTC="${GR00T_PREFIX_RTC:-1}"
export GR00T_EXECUTION_HORIZON="${GR00T_EXECUTION_HORIZON:-34}"
export GR00T_PORT="${GR00T_PORT:-22095}"

export GR00T_SONIC_TASK="G1Fullstate20260612Task3-v0"
export GR00T_SONIC_EVAL_DIR="${GR00T_SONIC_EVAL_DIR:-data/evals_task3_gr00t_n17_sonic_prefix_rtc_round2}"
export SONIC_DECODER_ONNX="${SONIC_DECODER_ONNX:-/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/SONIC_my/gear_sonic_deploy/policy/release/model_decoder.onnx}"
export MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-500}"

export HUMANOID_VLA_MJ_ROOT="${HUMANOID_VLA_MJ_ROOT:-/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/HumanoidVLA_MJ}"
export TASK3_INIT_FROM_RECORDINGS="${TASK3_INIT_FROM_RECORDINGS:-1}"
export TASK3_RECORDINGS_DIR="${TASK3_RECORDINGS_DIR:-${HUMANOID_VLA_MJ_ROOT}/output/20260612_task3}"
export TASK3_RECORDING_SEED="${TASK3_RECORDING_SEED:-0}"
# Preserve the annotation accidentally used by the gr00t_new training dataset.
export TASK3_INSTRUCTION="${TASK3_INSTRUCTION:-Move forward, pick up the bottle, then step on the trash can and throw the bottle inside.}"

if [[ -n "${TASK3_RECORDING_INDEX:-}" ]]; then
  export TASK3_RECORDING_INDEX
else
  unset TASK3_RECORDING_INDEX
fi

exec bash "$BASE_SCRIPT" "${1:-}"
