#!/usr/bin/env bash
set -euo pipefail

PSI0_ROOT="${PSI0_ROOT:-/pfs/pfs-ilWc5D/yzh/Psi0}"
BASE_SCRIPT="${PSI0_ROOT}/scripts/deploy/simple_psi0_kimodo_eval_commands.sh"
RUN_DIR="${FULLSTATE_TASK1_RUN_DIR:-${PSI0_ROOT}/.runs/finetune/fullstate-20260615-task1-rot6d59-80k.simple.flow1000.cosine.lr1.0e-04.b64.gpus4.2607011204}"

case "${1:-}" in
  serve)
    RETURN_FULL_ACTION_CHUNK="${RETURN_FULL_ACTION_CHUNK:-1}" \
    ACTION_EXEC_HORIZON="${ACTION_EXEC_HORIZON:-24}" \
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
    VLA_PROPRIO_SOURCE="${VLA_PROPRIO_SOURCE:-${BRIDGE_VLA_PROPRIO_SOURCE:-actual}}" \
    POLICY_EXECUTION_HORIZON="${POLICY_EXECUTION_HORIZON:-24}" \
    SKIP_STABILIZE="${SKIP_STABILIZE:-1}" \
    TEXTOP_INIT_TO_REF="${PSI0_TEXTOP_INIT_TO_REF:-0}" \
    TEXTOP_ONESTEP_TRACKER_RUN="${TEXTOP_RGZ_TRACKER_RUN:-/pfs/pfs-ilWc5D/yzh/textop/2026-06-11_11-39-47_rgz_loco_manip_obj_transf_vae_1step_ddp_4gpu_gear_sonic_ads_naug}" \
    TEXTOP_ONESTEP_POLICY_ONNX="${TEXTOP_RGZ_POLICY_ONNX:-/pfs/pfs-ilWc5D/yzh/textop/2026-06-11_11-39-47_rgz_loco_manip_obj_transf_vae_1step_ddp_4gpu_gear_sonic_ads_naug/latest.onnx}" \
    TEXTOP_ONESTEP_VAE_RUN="${TEXTOP_RGZ_VAE_RUN:-/pfs/pfs-ilWc5D/yzh/textop/2026-05-30_04-21-30_npz_rgz_filtered_ddp_save}" \
    TEXTOP_ONESTEP_VAE_ONNX="${TEXTOP_RGZ_VAE_ONNX:-/pfs/pfs-ilWc5D/yzh/textop/2026-05-30_04-21-30_npz_rgz_filtered_ddp_save/artifacts/motion_transformer_vae_encoder_z_c.onnx}" \
    TEXTOP_ONESTEP_VAE_STATS="${TEXTOP_RGZ_VAE_STATS:-/pfs/pfs-ilWc5D/yzh/textop/2026-05-30_04-21-30_npz_rgz_filtered_ddp_save/artifacts/stats.npz}" \
    TEXTOP_ONESTEP_VAE_WINDOW_STEPS="${TEXTOP_RGZ_VAE_WINDOW_STEPS:-10}" \
    ROT6D59_ONESTEP_TEXTOP_EVAL_DIR="${FULLSTATE_TASK1_EVAL_DIR:-data/evals_fullstate_20260615_task1_rgz_full30_exec24_noinitref}" \
    ROT6D59_ONESTEP_KIMODO_WORK_DIR="${FULLSTATE_TASK1_KIMODO_WORK_DIR:-${PSI0_ROOT}/outputs/kimodo_fullstate_20260615_task1_rgz_full30_exec24_noinitref}" \
    bash "$BASE_SCRIPT" eval-rot6d59-textop-onestep
    ;;
  *)
    echo "Usage: $0 {serve|kimodo-serve|eval}"
    echo "  SERVE_GPU=1 bash $0 serve"
    echo "  KIMODO_GPU=2 bash $0 kimodo-serve"
    echo "  EVAL_GPU=3 NUM_EPISODES=10 bash $0 eval"
    exit 2
    ;;
esac
