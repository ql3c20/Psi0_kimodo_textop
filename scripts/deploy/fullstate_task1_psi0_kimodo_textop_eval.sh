#!/usr/bin/env bash
set -euo pipefail

PSI0_ROOT="${PSI0_ROOT:-/pfs/pfs-ilWc5D/yzh/Psi0}"
BASE_SCRIPT="${PSI0_ROOT}/scripts/deploy/simple_psi0_kimodo_eval_commands.sh"
RUN_DIR="${FULLSTATE_TASK1_RUN_DIR:-${PSI0_ROOT}/.runs/finetune/fullstate-20260615-task1-rot6d59-80k.simple.flow1000.cosine.lr1.0e-04.b64.gpus4.2607011204}"

case "${1:-}" in
  serve)
    POLICYHAND_ROT6D59_RUN_DIR="$RUN_DIR" \
    CKPT_STEP="${CKPT_STEP:-80000}" \
    bash "$BASE_SCRIPT" serve-policyhand-rot6d59
    ;;
  kimodo-serve)
    bash "$BASE_SCRIPT" kimodo-serve
    ;;
  eval)
    TASK=G1Fullstate20260615Task1-v0 \
    SIM_MODE=mujoco \
    DATA_FORMAT=fixed \
    DATA_DIR=unused \
    TEXTOP_POLICY_ROOT_EE="${TEXTOP_POLICY_ROOT_EE:-1}" \
    ROT6D59_RGZ_ONESTEP_TEXTOP_EVAL_DIR="${FULLSTATE_TASK1_EVAL_DIR:-data/evals_fullstate_20260615_task1_rgz}" \
    ROT6D59_RGZ_ONESTEP_KIMODO_WORK_DIR="${FULLSTATE_TASK1_KIMODO_WORK_DIR:-${PSI0_ROOT}/outputs/kimodo_fullstate_20260615_task1_rgz}" \
    bash "$BASE_SCRIPT" eval-rot6d59-textop-onestep-initref-rgz
    ;;
  *)
    echo "Usage: $0 {serve|kimodo-serve|eval}"
    echo "  SERVE_GPU=1 bash $0 serve"
    echo "  KIMODO_GPU=2 bash $0 kimodo-serve"
    echo "  EVAL_GPU=3 NUM_EPISODES=10 bash $0 eval"
    exit 2
    ;;
esac
