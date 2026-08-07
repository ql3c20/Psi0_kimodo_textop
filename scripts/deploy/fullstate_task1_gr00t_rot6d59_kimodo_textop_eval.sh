#!/usr/bin/env bash
set -euo pipefail

# GR00T N1.7 rot6d59 -> Kimodo -> TextOp Tracker on the fullstate task1 MuJoCo scene.
#
# Default receding-horizon rhythm (RTC / Real-Time Chunking, mirrors the Psi0 path):
#   1. GR00T predicts 40 frames of rot6d59 policy actions.
#   2. Kimodo consumes the full 40-frame root/EE plan and generates a 40-frame full-body qpos reference.
#   3. TextOp consumes the full 40-frame full-body reference plus the VLA root/EE plan.
#   4. SIMPLE executes only the first 34 tracker steps, then replans.
#   5. The GR00T serve keeps the unexecuted tail (40-34 = 6 frames) as an RTC
#      continuity prefix so the *next* 40-frame chunk starts from it. Kimodo/TextOp
#      always receive the full 40-frame chunk; only execution is trimmed to 34.
#
# RTC modes (server side):
#   GR00T_USE_RTC=1              official soft vel_strength freeze (default)
#   GR00T_PREFIX_RTC=1           Hard-rewrite prefix + per-token timestep AdaLN
#                                (implies RTC continuity)
#   GR00T_PREFIX_RTC_TIMESTEP_MODE=legacy_zero|groot_clean
#                                Optional override; otherwise read from checkpoint.
# TensorRT first-stage acceleration:
#   GR00T_TRT_ENGINE_DIR=/path/to/engines
#   GR00T_TRT_MODE=vit_llm_only   # ViT + LLM TRT, RTC action head stays PyTorch.
# Keep POLICY_EXECUTION_HORIZON == GR00T_EXECUTION_HORIZON so the server's
# carry-over aligns with what the client actually executed.
#
# This wrapper intentionally does NOT use the historical "*initref*" eval entry:
# the robot state stays whatever the MuJoCo task reset produced, while only the
# TextOp reference comes from Kimodo.  Set GR00T_TEXTOP_INIT_TO_REF=1 only for
# explicit teleport-to-reference ablations.

PSI0_ROOT="${PSI0_ROOT:-/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Psi0}"
SIMPLE_ROOT="${SIMPLE_ROOT:-${PSI0_ROOT}/third_party/SIMPLE}"
GR00T_ROOT="${GR00T_ROOT:-/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Isaac-GR00T}"
BASE_SCRIPT="${PSI0_ROOT}/scripts/deploy/simple_psi0_kimodo_eval_commands.sh"

GR00T_MODEL_PATH="${GR00T_MODEL_PATH:-${GR00T_ROOT}/outputs/task1-gr00t-n17-rot6d59-kimodo-textop/task1-gr00t-n17-rot6d59-kimodo-textop/checkpoint-120000}"
GR00T_BACKBONE_PATH="${GR00T_BACKBONE_PATH:-${PSI0_ROOT}/huggingface/hub/models--nvidia--Cosmos-Reason2-2B/snapshots/9ce19a195e423419c349abfc86fd07178b230561}"
GR00T_MODALITY_CONFIG_PATH="${GR00T_MODALITY_CONFIG_PATH:-${GR00T_ROOT}/examples/unitree_g1_rot6d59_config.py}"
GR00T_PYTHON="${GR00T_PYTHON:-${GR00T_ROOT}/.venv/bin/python}"
GR00T_PORT="${GR00T_PORT:-22096}"
# RTC: predict 40, execute 34, carry over 40-34 = 6 frames to the next chunk.
GR00T_EXECUTION_HORIZON="${GR00T_EXECUTION_HORIZON:-34}"
GR00T_USE_RTC="${GR00T_USE_RTC:-1}"
GR00T_PREFIX_RTC="${GR00T_PREFIX_RTC:-0}"
GR00T_RTC_FROZEN_STEPS="${GR00T_RTC_FROZEN_STEPS:-2}"
GR00T_RTC_RAMP_RATE="${GR00T_RTC_RAMP_RATE:-2.0}"

case "${1:-}" in
  serve)
    cd "$GR00T_ROOT"
    export CUDA_VISIBLE_DEVICES="${SERVE_GPU:-1}"
    export PYTHONPATH="${GR00T_ROOT}:${PYTHONPATH:-}"
    export HF_HOME="${HF_HOME:-${PSI0_ROOT}/huggingface}"
    export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
    export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
    export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
    export GR00T_BACKBONE_PATH
    export NO_ALBUMENTATIONS_UPDATE=1
    serve_args=(
      --model-path "$GR00T_MODEL_PATH"
      --backbone-path "$GR00T_BACKBONE_PATH"
      --modality-config-path "$GR00T_MODALITY_CONFIG_PATH"
      --host 0.0.0.0
      --port "$GR00T_PORT"
      --device cuda
      --action-exec-horizon "$GR00T_EXECUTION_HORIZON"
      --rtc-frozen-steps "$GR00T_RTC_FROZEN_STEPS"
      --rtc-ramp-rate "$GR00T_RTC_RAMP_RATE"
    )
    if [[ -n "${GR00T_TRT_ENGINE_DIR:-}" ]]; then
      serve_args+=(
        --trt-engine-dir "$GR00T_TRT_ENGINE_DIR"
        --trt-mode "${GR00T_TRT_MODE:-vit_llm_only}"
      )
      if [[ -n "${GR00T_TRT_DEPLOY_DIR:-}" ]]; then
        serve_args+=(--trt-deploy-dir "$GR00T_TRT_DEPLOY_DIR")
      fi
    fi
    if [[ "$GR00T_PREFIX_RTC" == "1" ]]; then
      serve_args+=(--prefix-rtc)
      if [[ -n "${GR00T_PREFIX_RTC_TIMESTEP_MODE:-}" ]]; then
        serve_args+=(
          --prefix-rtc-timestep-mode
          "$GR00T_PREFIX_RTC_TIMESTEP_MODE"
        )
      fi
    fi
    if [[ "$GR00T_USE_RTC" == "1" || "$GR00T_PREFIX_RTC" == "1" ]]; then
      serve_args+=(--enable-rtc)
    else
      serve_args+=(--no-enable-rtc)
    fi
    exec "$GR00T_PYTHON" \
      "${PSI0_ROOT}/scripts/deploy/gr00t_n17_rot6d59_server.py" \
      "${serve_args[@]}"
    ;;
  kimodo-serve)
    bash "$BASE_SCRIPT" kimodo-serve
    ;;
  eval)
    TASK="${FULLSTATE_GR00T_TASK:-G1Fullstate20260615Task1-v0}" \
    SIM_MODE=mujoco \
    DATA_FORMAT=fixed \
    DATA_DIR=unused \
    PORT="$GR00T_PORT" \
    TEXTOP_POLICY_ROOT_EE="${TEXTOP_POLICY_ROOT_EE:-1}" \
    VLA_PROPRIO_SOURCE="${VLA_PROPRIO_SOURCE:-${BRIDGE_VLA_PROPRIO_SOURCE:-actual}}" \
    POLICY_EXECUTION_HORIZON="${POLICY_EXECUTION_HORIZON:-$GR00T_EXECUTION_HORIZON}" \
    SKIP_STABILIZE="${SKIP_STABILIZE:-1}" \
    MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-500}" \
    KIMODO_ANCHOR_MODE="${KIMODO_ANCHOR_MODE:-policy_only}" \
    TEXTOP_INIT_TO_REF="${GR00T_TEXTOP_INIT_TO_REF:-0}" \
    TEXTOP_ONESTEP_TRACKER_RUN="${TEXTOP_RGZ_TRACKER_RUN:-/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/textop/2026-06-11_11-39-47_rgz_loco_manip_obj_transf_vae_1step_ddp_4gpu_gear_sonic_ads_naug}" \
    TEXTOP_ONESTEP_POLICY_ONNX="${TEXTOP_RGZ_POLICY_ONNX:-/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/textop/2026-06-11_11-39-47_rgz_loco_manip_obj_transf_vae_1step_ddp_4gpu_gear_sonic_ads_naug/latest.onnx}" \
    TEXTOP_ONESTEP_VAE_RUN="${TEXTOP_RGZ_VAE_RUN:-/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/textop/2026-05-30_04-21-30_npz_rgz_filtered_ddp_save}" \
    TEXTOP_ONESTEP_VAE_ONNX="${TEXTOP_RGZ_VAE_ONNX:-/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/textop/2026-05-30_04-21-30_npz_rgz_filtered_ddp_save/artifacts/motion_transformer_vae_encoder_z_c.onnx}" \
    TEXTOP_ONESTEP_VAE_STATS="${TEXTOP_RGZ_VAE_STATS:-/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/textop/2026-05-30_04-21-30_npz_rgz_filtered_ddp_save/artifacts/stats.npz}" \
    TEXTOP_ONESTEP_VAE_WINDOW_STEPS="${TEXTOP_RGZ_VAE_WINDOW_STEPS:-10}" \
    ROT6D59_RGZ_ONESTEP_TEXTOP_EVAL_DIR="${FULLSTATE_GR00T_EVAL_DIR:-${FULLSTATE_TASK1_GR00T_EVAL_DIR:-data/evals_fullstate_20260615_task1_gr00t_rot6d59_kimodo_textop_ckpt120k_exec20}}" \
    ROT6D59_RGZ_ONESTEP_KIMODO_WORK_DIR="${FULLSTATE_GR00T_KIMODO_WORK_DIR:-${FULLSTATE_TASK1_GR00T_KIMODO_WORK_DIR:-${PSI0_ROOT}/outputs/kimodo_fullstate_20260615_task1_gr00t_rot6d59_ckpt120k_exec20}}" \
    ROT6D59_ONESTEP_TEXTOP_EVAL_DIR="${FULLSTATE_GR00T_EVAL_DIR:-${FULLSTATE_TASK1_GR00T_EVAL_DIR:-data/evals_fullstate_20260615_task1_gr00t_rot6d59_kimodo_textop_ckpt120k_exec20}}" \
    ROT6D59_ONESTEP_KIMODO_WORK_DIR="${FULLSTATE_GR00T_KIMODO_WORK_DIR:-${FULLSTATE_TASK1_GR00T_KIMODO_WORK_DIR:-${PSI0_ROOT}/outputs/kimodo_fullstate_20260615_task1_gr00t_rot6d59_ckpt120k_exec20}}" \
    bash "$BASE_SCRIPT" eval-rot6d59-textop-onestep
    ;;
  *)
    echo "Usage: $0 {serve|kimodo-serve|eval}"
    echo "  SERVE_GPU=1 bash $0 serve"
    echo "  KIMODO_GPU=2 bash $0 kimodo-serve"
    echo "  EVAL_GPU=3 NUM_EPISODES=1 bash $0 eval"
    echo "  GR00T_PREFIX_RTC=1 GR00T_USE_RTC=1 SERVE_GPU=1 bash $0 serve"
    exit 2
    ;;
esac
