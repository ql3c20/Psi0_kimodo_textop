#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/deploy/simple_psi0_kimodo_eval_commands.sh serve
#   bash scripts/deploy/simple_psi0_kimodo_eval_commands.sh serve-rot6d59
#   bash scripts/deploy/simple_psi0_kimodo_eval_commands.sh serve-policyhand-rot6d59-qguided
#   bash scripts/deploy/simple_psi0_kimodo_eval_commands.sh kimodo-serve
#   bash scripts/deploy/simple_psi0_kimodo_eval_commands.sh health
#   bash scripts/deploy/simple_psi0_kimodo_eval_commands.sh download-data
#   bash scripts/deploy/simple_psi0_kimodo_eval_commands.sh eval
#   bash scripts/deploy/simple_psi0_kimodo_eval_commands.sh eval-one
#   bash scripts/deploy/simple_psi0_kimodo_eval_commands.sh stitch
#   bash scripts/deploy/simple_psi0_kimodo_eval_commands.sh stitch30
#
# This is the Kimodo-in-the-loop version of simple_psi0_eval_commands.sh:
#   Psi0 checkpoint returns 44D policy actions
#   SIMPLE agent converts 44D -> Kimodo -> original SIMPLE 36D -> WBC.

PSI0_ROOT="${PSI0_ROOT:-/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Psi0}"
SIMPLE_ROOT="${SIMPLE_ROOT:-${PSI0_ROOT}/third_party/SIMPLE}"
KIMODO_ROOT="${KIMODO_ROOT:-/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/kimodo_my}"

export HF_HOME="${HF_HOME:-${PSI0_ROOT}/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export HUGGINGFACE_CACHE_DIR="${HUGGINGFACE_CACHE_DIR:-${HF_HOME}/hub}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

TASK="${TASK:-G1WholebodyXMovePickTeleop-v0}"
DR="${DR:-level-0}"
ENTRY="${ENTRY:-eval_decoupled_wbc.py}"
AGENT="${AGENT:-psi0_kimodo_decoupled_wbc}"
EVAL_DIR="${EVAL_DIR:-data/evals_decoupled_wbc_ep0_v2_44d}"

RUN_DIR="${RUN_DIR:-${PSI0_ROOT}/.runs/finetune/movepick-kimodo-v2.simple.flow1000.cosine.lr1.0e-04.b64.gpus4.2606091729}"
ROT6D59_RUN_DIR="${ROT6D59_RUN_DIR:-${PSI0_ROOT}/.runs/finetune/movepick-visualdr-rootfix-rot6d59.simple.flow1000.cosine.lr1.0e-04.b64.gpus4.2606212252}"
POLICYHAND_ROT6D59_RUN_DIR="${POLICYHAND_ROT6D59_RUN_DIR:-${PSI0_ROOT}/.runs/finetune/movepick-policyraw-new-rot6d59-policyhand.hand14x5.0.simple.flow1000.cosine.lr1.0e-04.b64.gpus4.2606251040}"
CKPT_STEP="${CKPT_STEP:-40000}"

SERVE_GPU="${SERVE_GPU:-1}"
KIMODO_GPU="${KIMODO_GPU:-2}"
EVAL_GPU="${EVAL_GPU:-3}"
HOST="${HOST:-localhost}"
PORT="${PORT:-22085}"
ACTION_EXEC_HORIZON="${ACTION_EXEC_HORIZON:-24}"
USE_RTC="${USE_RTC:-1}"
RETURN_FULL_ACTION_CHUNK="${RETURN_FULL_ACTION_CHUNK:-0}"
NUM_EPISODES="${NUM_EPISODES:-20}"
SAVE_VIDEO="${SAVE_VIDEO:-1}"
EPISODE_START="${EPISODE_START:-0}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-800}"
SIM_MODE="${SIM_MODE:-mujoco_isaac}"
SIMPLE_MUJOCO_GUI="${SIMPLE_MUJOCO_GUI:-0}"
export SIMPLE_TRACKER_GHOST="${SIMPLE_TRACKER_GHOST:-0}"
DATA_FORMAT="${DATA_FORMAT:-lerobot}"
DATA_DIR="${DATA_DIR:-data/evals/simple-eval/${TASK}/${DR}}"

export KIMODO_ROOT
export KIMODO_PYTHON="${KIMODO_PYTHON:-/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/miniconda3/envs/kimodo/bin/python}"
export KIMODO_DISTILL_CONFIG="${KIMODO_DISTILL_CONFIG:-${KIMODO_ROOT}/outputs/g1_distill_16to8_100to20_dagger_teacher_gt03_100k_bs4x4_k3_cosine_selfacc50_rootacc10_headingacc10_gtbranch/resolved_config.yaml}"
export KIMODO_DISTILL_CKPT="${KIMODO_DISTILL_CKPT:-${KIMODO_ROOT}/outputs/g1_distill_16to8_100to20_dagger_teacher_gt03_100k_bs4x4_k3_cosine_selfacc50_rootacc10_headingacc10_gtbranch/ema_final.pt}"
export KIMODO_DIFFUSION_STEPS="${KIMODO_DIFFUSION_STEPS:-20}"
export KIMODO_KEYFRAME_STEP="${KIMODO_KEYFRAME_STEP:-10}"
export KIMODO_BLEND_FRAMES="${KIMODO_BLEND_FRAMES:-4}"
export KIMODO_BASE_CMD_SMOOTH_WINDOW="${KIMODO_BASE_CMD_SMOOTH_WINDOW:-5}"
export KIMODO_MAX_ABS_VX="${KIMODO_MAX_ABS_VX:-0.5}"
export KIMODO_MAX_ABS_VY="${KIMODO_MAX_ABS_VY:-0.08}"
export KIMODO_MAX_ABS_VYAW="${KIMODO_MAX_ABS_VYAW:-0.25}"
export KIMODO_ANCHOR_MODE="${KIMODO_ANCHOR_MODE:-policy_only}"
export KIMODO_WORK_DIR="${KIMODO_WORK_DIR:-${PSI0_ROOT}/outputs/kimodo_eval_ep0_v2_44d}"
export KIMODO_KEEP_WORK="${KIMODO_KEEP_WORK:-0}"
export KIMODO_EPISODE_SUBDIR="${KIMODO_EPISODE_SUBDIR:-1}"
export KIMODO_SERVER_HOST="${KIMODO_SERVER_HOST:-127.0.0.1}"
export KIMODO_SERVER_PORT="${KIMODO_SERVER_PORT:-22185}"
export KIMODO_SERVER_URL="${KIMODO_SERVER_URL:-http://${KIMODO_SERVER_HOST}:${KIMODO_SERVER_PORT}}"

export TEXTOP_ROOT="${TEXTOP_ROOT:-/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh}"
export TEXTOP_TRACKER_RUN="${TEXTOP_TRACKER_RUN:-${TEXTOP_ROOT}/textop/2026-05-18_19-55-31_transformer_vae_eeobs_g1_before_2023}"
export TEXTOP_VAE_RUN="${TEXTOP_VAE_RUN:-${TEXTOP_ROOT}/textop/2026-05-12_16-34-08_optitrack_npz_soma_before_2023}"
export TEXTOP_POLICY_ONNX="${TEXTOP_POLICY_ONNX:-${TEXTOP_TRACKER_RUN}/exported/policy.onnx}"
export TEXTOP_VAE_ONNX="${TEXTOP_VAE_ONNX:-${TEXTOP_VAE_RUN}/artifacts/motion_transformer_vae_encoder_z_c.onnx}"
export TEXTOP_VAE_STATS="${TEXTOP_VAE_STATS:-${TEXTOP_VAE_RUN}/artifacts/stats.npz}"
export TEXTOP_FUTURE_STEPS="${TEXTOP_FUTURE_STEPS:-10}"
export TEXTOP_TASK="${TEXTOP_TASK:-Tracking-Flat-G1-ProjGravAnchorEEObs-TransformerVAE-NMMLP-v0}"
# PSI0 -> Kimodo -> TextOp default:
#   - hand14 and root + four EE poses come directly from the VLA policy action;
#   - Kimodo qpos50 remains the full-body future reference encoded by the VAE to 128D z/c.
# Set TEXTOP_POLICY_ROOT_EE=0 explicitly to recover Kimodo-FK root/EE references.
export TEXTOP_POLICY_ROOT_EE="${TEXTOP_POLICY_ROOT_EE:-1}"

TEXTOP_CKPT_ROOT="${TEXTOP_CKPT_ROOT:-/home/ubuntu/yzh/ckpt}"
TEXTOP_ONESTEP_TASK="${TEXTOP_ONESTEP_TASK:-Tracking-Flat-G1-ProjGravAnchorEEObsOneStep-TransformerVAE-NMMLP-v0}"
TEXTOP_ONESTEP_RUN_DIR="${TEXTOP_ONESTEP_RUN_DIR:-$POLICYHAND_ROT6D59_RUN_DIR}"
TEXTOP_ONESTEP_TRACKER_RUN="${TEXTOP_ONESTEP_TRACKER_RUN:-${TEXTOP_CKPT_ROOT}/textop/2026-06-11_11-39-47_rgz_loco_manip_obj_transf_vae_1step_ddp_4gpu_gear_sonic_ads_naug}"
TEXTOP_ONESTEP_POLICY_ONNX="${TEXTOP_ONESTEP_POLICY_ONNX:-${TEXTOP_ONESTEP_TRACKER_RUN}/latest.onnx}"
TEXTOP_ONESTEP_VAE_RUN="${TEXTOP_ONESTEP_VAE_RUN:-${TEXTOP_CKPT_ROOT}/vae/2026-05-30_04-21-30_npz_rgz_filtered_ddp_save}"
TEXTOP_ONESTEP_VAE_ONNX="${TEXTOP_ONESTEP_VAE_ONNX:-${TEXTOP_ONESTEP_VAE_RUN}/artifacts/motion_transformer_vae_encoder_z_c.onnx}"
TEXTOP_ONESTEP_VAE_STATS="${TEXTOP_ONESTEP_VAE_STATS:-${TEXTOP_ONESTEP_VAE_RUN}/artifacts/stats.npz}"
TEXTOP_ONESTEP_VAE_WINDOW_STEPS="${TEXTOP_ONESTEP_VAE_WINDOW_STEPS:-10}"

TEXTOP_RGZ_TRACKER_RUN="${TEXTOP_RGZ_TRACKER_RUN:-${TEXTOP_CKPT_ROOT}/textop/2026-06-11_11-39-47_rgz_loco_manip_obj_transf_vae_1step_ddp_4gpu_gear_sonic_ads_naug}"
TEXTOP_RGZ_POLICY_ONNX="${TEXTOP_RGZ_POLICY_ONNX:-${TEXTOP_RGZ_TRACKER_RUN}/latest.onnx}"
TEXTOP_RGZ_VAE_RUN="${TEXTOP_RGZ_VAE_RUN:-${TEXTOP_CKPT_ROOT}/vae/2026-05-30_04-21-30_npz_rgz_filtered_ddp_save}"
TEXTOP_RGZ_VAE_ONNX="${TEXTOP_RGZ_VAE_ONNX:-${TEXTOP_RGZ_VAE_RUN}/artifacts/motion_transformer_vae_encoder_z_c.onnx}"
TEXTOP_RGZ_VAE_STATS="${TEXTOP_RGZ_VAE_STATS:-${TEXTOP_RGZ_VAE_RUN}/artifacts/stats.npz}"
TEXTOP_RGZ_VAE_WINDOW_STEPS="${TEXTOP_RGZ_VAE_WINDOW_STEPS:-10}"

Q_GUIDANCE_CKPT="${Q_GUIDANCE_CKPT:-${TEXTOP_ROOT}/textop/horizon_clip_mlp_psi0_rot6d59_chunk30_horizon32_q400_lowreg_wide/horizon_clip_mlp_psi0_rot6d59_chunk30_horizon32_q400_lowreg_wide/checkpoints/best.pt}"
Q_GUIDANCE_BETA="${Q_GUIDANCE_BETA:-0.03}"
Q_GUIDANCE_START_T="${Q_GUIDANCE_START_T:-0.3}"
Q_GUIDANCE_MAX_GRAD_NORM="${Q_GUIDANCE_MAX_GRAD_NORM:-0.3}"
Q_GUIDANCE_MASK="${Q_GUIDANCE_MASK:-position}"

TEXTOP_ALLFILTER_TRACKER_RUN="${TEXTOP_ALLFILTER_TRACKER_RUN:-${TEXTOP_ROOT}/textop/2026-06-15_16-42-21_npz_all_transf_vae_1step_ddp_4gpu_gear_sonic_ads_save}"
TEXTOP_ALLFILTER_POLICY_ONNX="${TEXTOP_ALLFILTER_POLICY_ONNX:-${TEXTOP_ALLFILTER_TRACKER_RUN}/exported/model_30000.onnx}"
TEXTOP_ALLFILTER_VAE_RUN="${TEXTOP_ALLFILTER_VAE_RUN:-${TEXTOP_ROOT}/textop/2026-05-18_18-59-39_1gpu_npz_all_filtered_resume_save}"
TEXTOP_ALLFILTER_VAE_ONNX="${TEXTOP_ALLFILTER_VAE_ONNX:-${TEXTOP_ALLFILTER_VAE_RUN}/artifacts/motion_transformer_vae_encoder_z_c.onnx}"
TEXTOP_ALLFILTER_VAE_STATS="${TEXTOP_ALLFILTER_VAE_STATS:-${TEXTOP_ALLFILTER_VAE_RUN}/artifacts/stats.npz}"
TEXTOP_ALLFILTER_VAE_WINDOW_STEPS="${TEXTOP_ALLFILTER_VAE_WINDOW_STEPS:-10}"

STITCH_START_CHUNK="${STITCH_START_CHUNK:-}"
STITCH_END_CHUNK="${STITCH_END_CHUNK:-}"
STITCH_NUM_CHUNKS="${STITCH_NUM_CHUNKS:-}"
STITCH_EPISODE="${STITCH_EPISODE:-}"
STITCH_EXEC_FRAMES="${STITCH_EXEC_FRAMES:-$ACTION_EXEC_HORIZON}"
STITCH_BLEND_FRAMES="${STITCH_BLEND_FRAMES:-$KIMODO_BLEND_FRAMES}"
STITCH_OUTPUT="${STITCH_OUTPUT:-${KIMODO_WORK_DIR}/stitched_executed_qpos50.csv}"
STITCH_PLOT="${STITCH_PLOT:-${KIMODO_WORK_DIR}/stitched_root.png}"
STITCH30_FRAMES_PER_CHUNK="${STITCH30_FRAMES_PER_CHUNK:-15}"
STITCH30_BLEND_FRAMES="${STITCH30_BLEND_FRAMES:-2}"
STITCH30_OUTPUT="${STITCH30_OUTPUT:-${KIMODO_WORK_DIR}/stitched_executed_qpos30.csv}"
STITCH30_PLOT="${STITCH30_PLOT:-${KIMODO_WORK_DIR}/stitched_root_30hz.png}"

serve() {
  cd "$PSI0_ROOT"
  source .venv-psi/bin/activate

  export CUDA_VISIBLE_DEVICES="$SERVE_GPU"
  export run_dir="$RUN_DIR"
  export ckpt_step="$CKPT_STEP"

  serve_args=(
    --host 0.0.0.0 \
    --port "$PORT" \
    --run-dir="$run_dir" \
    --ckpt-step="$ckpt_step" \
    --action-exec-horizon="$ACTION_EXEC_HORIZON"
  )
  if [[ "$USE_RTC" == "1" ]]; then
    serve_args+=(--rtc)
  fi
  if [[ "$RETURN_FULL_ACTION_CHUNK" == "1" ]]; then
    serve_args+=(--return-full-action-chunk)
  fi
  if [[ "${ENABLE_Q_GUIDANCE:-0}" == "1" ]]; then
    serve_args+=(
      --q-guidance-checkpoint "$Q_GUIDANCE_CKPT"
      --q-guidance-beta "$Q_GUIDANCE_BETA"
      --q-guidance-start-t "$Q_GUIDANCE_START_T"
      --q-guidance-max-grad-norm "$Q_GUIDANCE_MAX_GRAD_NORM"
      --q-guidance-mask "$Q_GUIDANCE_MASK"
    )
  fi

  serve_psi0 "${serve_args[@]}"
}

health() {
  export NO_PROXY="${NO_PROXY:-localhost,127.0.0.1,0.0.0.0,::1}"
  export no_proxy="${no_proxy:-localhost,127.0.0.1,0.0.0.0,::1}"
  curl -i "http://${HOST}:${PORT}/health"
}

kimodo_serve() {
  cd "$KIMODO_ROOT"

  export CUDA_VISIBLE_DEVICES="$KIMODO_GPU"
  export HF_HOME="${KIMODO_ROOT}/huggingface"
  export HF_HUB_CACHE="${KIMODO_ROOT}/huggingface/hub"
  export HUGGINGFACE_HUB_CACHE="${KIMODO_ROOT}/huggingface/hub"
  export HUGGINGFACE_CACHE_DIR="${KIMODO_ROOT}/huggingface/hub"
  export TRANSFORMERS_OFFLINE=1
  export HF_HUB_OFFLINE=1
  export PYTHONNOUSERSITE=1
  export LOCAL_CACHE="${LOCAL_CACHE:-true}"
  export TEXT_ENCODER_MODE="${TEXT_ENCODER_MODE:-local}"

  "$KIMODO_PYTHON" "${PSI0_ROOT}/scripts/deploy/kimodo_generation_server.py"
}

download_data() {
  cd "$SIMPLE_ROOT"
  source .venv/bin/activate

  hf download USC-PSI-Lab/psi-data \
    "simple-eval/${TASK}.zip" \
    --local-dir=data/evals \
    --repo-type=dataset

  unzip -o "data/evals/simple-eval/${TASK}.zip" -d data/evals/simple-eval
}

eval_simple() {
  cd "$SIMPLE_ROOT"
  source .venv/bin/activate

  export CUDA_VISIBLE_DEVICES="$EVAL_GPU"
  # uv may place the large ORT CUDA provider under .venv/lib64 while Python
  # imports the package from .venv/lib.  Add the real provider directory and
  # pip-installed NVIDIA runtime libraries before the eval Python process is
  # created; changing LD_LIBRARY_PATH after Python starts is too late for
  # dlopen("libonnxruntime_providers_cuda.so").
  ort_cuda_provider="$(
    find "$SIMPLE_ROOT/.venv" -type f \
      -path '*/site-packages/onnxruntime/capi/libonnxruntime_providers_cuda.so' \
      -print -quit
  )"
  if [[ -z "$ort_cuda_provider" ]]; then
    echo "Missing libonnxruntime_providers_cuda.so below $SIMPLE_ROOT/.venv" >&2
    exit 1
  fi
  ort_cuda_dir="$(dirname "$ort_cuda_provider")"
  ort_nvidia_libs=""
  shopt -s nullglob
  nvidia_roots=("$SIMPLE_ROOT"/.venv/lib*/python*/site-packages/nvidia)
  shopt -u nullglob
  while IFS= read -r nvidia_lib; do
    ort_nvidia_libs="${ort_nvidia_libs}:${nvidia_lib}"
  done < <(
    for nvidia_root in "${nvidia_roots[@]}"; do
      find "$nvidia_root" -mindepth 2 -maxdepth 2 -type d -name lib -print
    done
  )
  export LD_LIBRARY_PATH="${ort_cuda_dir}${ort_nvidia_libs}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

  EVAL_GPU="$EVAL_GPU" python - <<'PY'
import os
from pathlib import Path

import onnxruntime as ort

models = {
    "VAE": Path(os.environ["TEXTOP_VAE_ONNX"]),
    "policy": Path(os.environ["TEXTOP_POLICY_ONNX"]),
}
for label, path in models.items():
    if not path.is_file():
        raise FileNotFoundError(f"TextOp {label} ONNX is missing: {path}")
    session = ort.InferenceSession(
        str(path), providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    active = session.get_providers()
    if not active or active[0] != "CUDAExecutionProvider":
        raise RuntimeError(
            f"TextOp {label} CUDA preflight failed with EVAL_GPU="
            f"{os.environ['EVAL_GPU']}; active providers={active}"
        )
    print(
        f"[preflight] TextOp {label} providers={active} "
        f"EVAL_GPU={os.environ['EVAL_GPU']}",
        flush=True,
    )
    del session
PY
  export OMNI_KIT_ACCEPT_EULA=Y
  headless_flag="--headless"
  if [[ "$SIMPLE_MUJOCO_GUI" == "1" ]]; then
    headless_flag="--no-headless"
    unset MUJOCO_GL PYOPENGL_PLATFORM
    echo "[eval] MuJoCo GUI enabled; tracker_ghost=${SIMPLE_TRACKER_GHOST}"
  else
    export MUJOCO_GL=egl
    export PYOPENGL_PLATFORM=egl
  fi
  if [[ "$SIMPLE_TRACKER_GHOST" == "1" && "$SIMPLE_MUJOCO_GUI" != "1" ]]; then
    echo "SIMPLE_TRACKER_GHOST=1 requires SIMPLE_MUJOCO_GUI=1" >&2
    exit 1
  fi
  export NO_PROXY="${NO_PROXY:-localhost,127.0.0.1,0.0.0.0,::1}"
  export no_proxy="${no_proxy:-localhost,127.0.0.1,0.0.0.0,::1}"

  VIDEO_FLAG="--save-video"
  if [[ "$SAVE_VIDEO" == "0" ]]; then
    VIDEO_FLAG="--no-save-video"
  fi

  eval_prefix=()
  if [[ -n "${EVAL_CPUSET:-}" ]]; then
    eval_prefix=(taskset -c "$EVAL_CPUSET")
  fi
  "${eval_prefix[@]}" python "src/simple/cli/${ENTRY}" \
    "simple/${TASK}" \
    "$AGENT" \
    "$DR" \
    --eval-dir="$EVAL_DIR" \
    --host="$HOST" \
    --port="$PORT" \
    --sim-mode="$SIM_MODE" \
    "$headless_flag" \
    --data-format="$DATA_FORMAT" \
    --data-dir="$DATA_DIR" \
    --num-episodes="$NUM_EPISODES" \
    --episode-start="$EPISODE_START" \
    "$VIDEO_FLAG" \
	    ${MAX_EPISODE_STEPS:+--max-episode-steps="$MAX_EPISODE_STEPS"}
}

eval_one() {
  NUM_EPISODES="${EVAL_ONE_NUM_EPISODES:-1}"
  KIMODO_KEEP_WORK="${EVAL_ONE_KIMODO_KEEP_WORK:-1}"
  export NUM_EPISODES
  export KIMODO_KEEP_WORK
  eval_simple
}

configure_textop_onestep_defaults() {
  export TEXTOP_TASK="$TEXTOP_ONESTEP_TASK"
  export TEXTOP_FUTURE_STEPS=1
  export TEXTOP_TRACKER_RUN="$TEXTOP_ONESTEP_TRACKER_RUN"
  export TEXTOP_POLICY_ONNX="$TEXTOP_ONESTEP_POLICY_ONNX"
  export TEXTOP_VAE_RUN="$TEXTOP_ONESTEP_VAE_RUN"
  export TEXTOP_VAE_ONNX="$TEXTOP_ONESTEP_VAE_ONNX"
  export TEXTOP_VAE_STATS="$TEXTOP_ONESTEP_VAE_STATS"
  export TEXTOP_VAE_WINDOW_STEPS="$TEXTOP_ONESTEP_VAE_WINDOW_STEPS"
}

configure_textop_onestep_rgz_defaults() {
  export TEXTOP_TASK="$TEXTOP_ONESTEP_TASK"
  export TEXTOP_FUTURE_STEPS=1
  export TEXTOP_TRACKER_RUN="$TEXTOP_RGZ_TRACKER_RUN"
  export TEXTOP_POLICY_ONNX="$TEXTOP_RGZ_POLICY_ONNX"
  export TEXTOP_VAE_RUN="$TEXTOP_RGZ_VAE_RUN"
  export TEXTOP_VAE_ONNX="$TEXTOP_RGZ_VAE_ONNX"
  export TEXTOP_VAE_STATS="$TEXTOP_RGZ_VAE_STATS"
  export TEXTOP_VAE_WINDOW_STEPS="$TEXTOP_RGZ_VAE_WINDOW_STEPS"
}

configure_textop_onestep_allfilter_defaults() {
  export TEXTOP_TASK="$TEXTOP_ONESTEP_TASK"
  export TEXTOP_FUTURE_STEPS=1
  export TEXTOP_TRACKER_RUN="$TEXTOP_ALLFILTER_TRACKER_RUN"
  export TEXTOP_POLICY_ONNX="$TEXTOP_ALLFILTER_POLICY_ONNX"
  export TEXTOP_VAE_RUN="$TEXTOP_ALLFILTER_VAE_RUN"
  export TEXTOP_VAE_ONNX="$TEXTOP_ALLFILTER_VAE_ONNX"
  export TEXTOP_VAE_STATS="$TEXTOP_ALLFILTER_VAE_STATS"
  export TEXTOP_VAE_WINDOW_STEPS="$TEXTOP_ALLFILTER_VAE_WINDOW_STEPS"
}

stitch_kimodo_chunks() {
  cd "$PSI0_ROOT"
  export PYTHONPATH="${KIMODO_ROOT}:${PYTHONPATH:-}"
  stitch_work_dir="$KIMODO_WORK_DIR"
  if [[ -n "$STITCH_EPISODE" ]]; then
    if [[ "$STITCH_EPISODE" == ep* ]]; then
      stitch_work_dir="${KIMODO_WORK_DIR}/${STITCH_EPISODE}"
    else
      stitch_work_dir="$(printf "%s/ep%04d" "$KIMODO_WORK_DIR" "$STITCH_EPISODE")"
    fi
  fi
  stitch_output="$STITCH_OUTPUT"
  stitch_plot="$STITCH_PLOT"
  if [[ -n "$STITCH_EPISODE" && "$STITCH_OUTPUT" == "${KIMODO_WORK_DIR}/stitched_executed_qpos50.csv" ]]; then
    stitch_output="${stitch_work_dir}/stitched_executed_qpos50.csv"
  fi
  if [[ -n "$STITCH_EPISODE" && "$STITCH_PLOT" == "${KIMODO_WORK_DIR}/stitched_root.png" ]]; then
    stitch_plot="${stitch_work_dir}/stitched_root.png"
  fi

  stitch_args=(
    --work-dir "$stitch_work_dir"
    --output "$stitch_output"
    --exec-frames "$STITCH_EXEC_FRAMES"
    --blend-frames "$STITCH_BLEND_FRAMES"
    --plot "$stitch_plot"
  )
  if [[ -n "$STITCH_START_CHUNK" ]]; then
    stitch_args+=(--start-chunk "$STITCH_START_CHUNK")
  fi
  if [[ -n "$STITCH_END_CHUNK" ]]; then
    stitch_args+=(--end-chunk "$STITCH_END_CHUNK")
  fi
  if [[ -n "$STITCH_NUM_CHUNKS" ]]; then
    stitch_args+=(--num-chunks "$STITCH_NUM_CHUNKS")
  fi

  "$KIMODO_PYTHON" scripts/deploy/stitch_kimodo_chunks.py "${stitch_args[@]}"
}

stitch_kimodo_chunks_30hz() {
  cd "$PSI0_ROOT"
  export PYTHONPATH="${KIMODO_ROOT}:${PYTHONPATH:-}"
  stitch_work_dir="$KIMODO_WORK_DIR"
  if [[ -n "$STITCH_EPISODE" ]]; then
    if [[ "$STITCH_EPISODE" == ep* ]]; then
      stitch_work_dir="${KIMODO_WORK_DIR}/${STITCH_EPISODE}"
    else
      stitch_work_dir="$(printf "%s/ep%04d" "$KIMODO_WORK_DIR" "$STITCH_EPISODE")"
    fi
  fi
  stitch30_output="$STITCH30_OUTPUT"
  stitch30_plot="$STITCH30_PLOT"
  if [[ -n "$STITCH_EPISODE" && "$STITCH30_OUTPUT" == "${KIMODO_WORK_DIR}/stitched_executed_qpos30.csv" ]]; then
    stitch30_output="${stitch_work_dir}/stitched_executed_qpos30.csv"
  fi
  if [[ -n "$STITCH_EPISODE" && "$STITCH30_PLOT" == "${KIMODO_WORK_DIR}/stitched_root_30hz.png" ]]; then
    stitch30_plot="${stitch_work_dir}/stitched_root_30hz.png"
  fi

  stitch_args=(
    --work-dir "$stitch_work_dir"
    --output "$stitch30_output"
    --source-fps 30
    --output-fps 30
    --chunk-policy-frames "$STITCH30_FRAMES_PER_CHUNK"
    --exec-frames "$STITCH30_FRAMES_PER_CHUNK"
    --blend-frames "$STITCH30_BLEND_FRAMES"
    --plot "$stitch30_plot"
  )
  if [[ -n "$STITCH_START_CHUNK" ]]; then
    stitch_args+=(--start-chunk "$STITCH_START_CHUNK")
  fi
  if [[ -n "$STITCH_END_CHUNK" ]]; then
    stitch_args+=(--end-chunk "$STITCH_END_CHUNK")
  fi
  if [[ -n "$STITCH_NUM_CHUNKS" ]]; then
    stitch_args+=(--num-chunks "$STITCH_NUM_CHUNKS")
  fi

  "$KIMODO_PYTHON" scripts/deploy/stitch_kimodo_chunks.py "${stitch_args[@]}"
}

case "${1:-}" in
  serve)
    serve
    ;;
  serve-rot6d59)
    RUN_DIR="$ROT6D59_RUN_DIR"
    export RUN_DIR
    serve
    ;;
  serve-policyhand-rot6d59)
    RUN_DIR="$POLICYHAND_ROT6D59_RUN_DIR"
    export RUN_DIR
    serve
    ;;
  serve-policyhand-rot6d59-qguided)
    RUN_DIR="$POLICYHAND_ROT6D59_RUN_DIR"
    ENABLE_Q_GUIDANCE=1
    export RUN_DIR ENABLE_Q_GUIDANCE
    serve
    ;;
  health)
    health
    ;;
  kimodo-serve)
    kimodo_serve
    ;;
  download-data)
    download_data
    ;;
	  eval)
	    eval_simple
	    ;;
	  eval-one)
	    eval_one
	    ;;
	  eval-textop)
	    AGENT=psi0_kimodo_textop_tracker
	    EVAL_DIR="${TEXTOP_EVAL_DIR:-data/evals_kimodo_textop_ep0_v2_44d}"
	    KIMODO_WORK_DIR="${TEXTOP_KIMODO_WORK_DIR:-${PSI0_ROOT}/outputs/kimodo_textop_ep0_v2_44d}"
	    export FALL_STOP="${FALL_STOP:-1}"
	    if [[ "${TEXTOP_SKIP_STABILIZE:-0}" == "1" ]]; then
	      export SKIP_STABILIZE=1
	    fi
	    eval_simple
	    ;;
	  eval-textop-one)
	    AGENT=psi0_kimodo_textop_tracker
	    EVAL_DIR="${TEXTOP_EVAL_DIR:-data/evals_kimodo_textop_ep0_v2_44d}"
	    KIMODO_WORK_DIR="${TEXTOP_KIMODO_WORK_DIR:-${PSI0_ROOT}/outputs/kimodo_textop_ep0_v2_44d}"
	    export FALL_STOP="${FALL_STOP:-1}"
	    if [[ "${TEXTOP_SKIP_STABILIZE:-0}" == "1" ]]; then
	      export SKIP_STABILIZE=1
	    fi
	    eval_one
	    ;;
	  eval-textop-policy-ee)
	    AGENT=psi0_kimodo_textop_tracker
	    EVAL_DIR="${TEXTOP_EVAL_DIR:-data/evals_kimodo_textop_policy_ee_ep0_v2_44d}"
	    KIMODO_WORK_DIR="${TEXTOP_KIMODO_WORK_DIR:-${PSI0_ROOT}/outputs/kimodo_textop_policy_ee_ep0_v2_44d}"
	    export FALL_STOP="${FALL_STOP:-1}"
	    export TEXTOP_POLICY_ROOT_EE=1
	    if [[ "${TEXTOP_SKIP_STABILIZE:-0}" == "1" ]]; then
	      export SKIP_STABILIZE=1
	    fi
	    eval_simple
	    ;;
	  eval-textop-policy-ee-one)
	    AGENT=psi0_kimodo_textop_tracker
	    EVAL_DIR="${TEXTOP_EVAL_DIR:-data/evals_kimodo_textop_policy_ee_ep0_v2_44d}"
	    KIMODO_WORK_DIR="${TEXTOP_KIMODO_WORK_DIR:-${PSI0_ROOT}/outputs/kimodo_textop_policy_ee_ep0_v2_44d}"
	    export FALL_STOP="${FALL_STOP:-1}"
	    export TEXTOP_POLICY_ROOT_EE=1
	    if [[ "${TEXTOP_SKIP_STABILIZE:-0}" == "1" ]]; then
	      export SKIP_STABILIZE=1
	    fi
	    eval_one
	    ;;
	  eval-textop-direct)
	    AGENT=psi0_kimodo_textop_tracker
	    EVAL_DIR="${TEXTOP_EVAL_DIR:-data/evals_kimodo_textop_direct_ep0_v2_44d}"
	    KIMODO_WORK_DIR="${TEXTOP_KIMODO_WORK_DIR:-${PSI0_ROOT}/outputs/kimodo_textop_direct_ep0_v2_44d}"
	    export FALL_STOP="${FALL_STOP:-1}"
	    export SKIP_STABILIZE=1
	    export TEXTOP_TARGET_RATE_LIMIT=0
	    eval_simple
	    ;;
	  eval-textop-direct-one)
	    AGENT=psi0_kimodo_textop_tracker
	    EVAL_DIR="${TEXTOP_EVAL_DIR:-data/evals_kimodo_textop_direct_ep0_v2_44d}"
	    KIMODO_WORK_DIR="${TEXTOP_KIMODO_WORK_DIR:-${PSI0_ROOT}/outputs/kimodo_textop_direct_ep0_v2_44d}"
	    export FALL_STOP="${FALL_STOP:-1}"
	    export SKIP_STABILIZE=1
	    export TEXTOP_TARGET_RATE_LIMIT=0
	    eval_one
	    ;;
	  eval-textop-initref)
	    AGENT=psi0_kimodo_textop_tracker
	    EVAL_DIR="${TEXTOP_EVAL_DIR:-data/evals_kimodo_textop_initref_ep0_v2_44d}"
	    KIMODO_WORK_DIR="${TEXTOP_KIMODO_WORK_DIR:-${PSI0_ROOT}/outputs/kimodo_textop_initref_ep0_v2_44d}"
	    export FALL_STOP="${FALL_STOP:-1}"
	    export SKIP_STABILIZE=1
	    export TEXTOP_TARGET_RATE_LIMIT=0
	    export TEXTOP_INIT_TO_REF=1
	    eval_simple
	    ;;
	  eval-textop-initref-one)
	    AGENT=psi0_kimodo_textop_tracker
	    EVAL_DIR="${TEXTOP_EVAL_DIR:-data/evals_kimodo_textop_initref_ep0_v2_44d}"
	    KIMODO_WORK_DIR="${TEXTOP_KIMODO_WORK_DIR:-${PSI0_ROOT}/outputs/kimodo_textop_initref_ep0_v2_44d}"
	    export FALL_STOP="${FALL_STOP:-1}"
	    export SKIP_STABILIZE=1
	    export TEXTOP_TARGET_RATE_LIMIT=0
	    export TEXTOP_INIT_TO_REF=1
	    eval_one
	    ;;
	  eval-rot6d59-textop)
	    AGENT=psi0_kimodo_textop_tracker
	    RUN_DIR="$ROT6D59_RUN_DIR"
	    EVAL_DIR="${ROT6D59_TEXTOP_EVAL_DIR:-data/evals_kimodo_textop_rot6d59}"
	    KIMODO_WORK_DIR="${ROT6D59_KIMODO_WORK_DIR:-${PSI0_ROOT}/outputs/kimodo_textop_rot6d59}"
	    export RUN_DIR
	    export FALL_STOP="${FALL_STOP:-1}"
	    export SKIP_STABILIZE="${SKIP_STABILIZE:-0}"
	    export TEXTOP_TARGET_RATE_LIMIT="${TEXTOP_TARGET_RATE_LIMIT:-0}"
	    eval_simple
	    ;;
	  eval-rot6d59-textop-one)
	    AGENT=psi0_kimodo_textop_tracker
	    RUN_DIR="$ROT6D59_RUN_DIR"
	    EVAL_DIR="${ROT6D59_TEXTOP_EVAL_DIR:-data/evals_kimodo_textop_rot6d59}"
	    KIMODO_WORK_DIR="${ROT6D59_KIMODO_WORK_DIR:-${PSI0_ROOT}/outputs/kimodo_textop_rot6d59}"
	    export RUN_DIR
	    export FALL_STOP="${FALL_STOP:-1}"
	    export SKIP_STABILIZE="${SKIP_STABILIZE:-0}"
	    export TEXTOP_TARGET_RATE_LIMIT="${TEXTOP_TARGET_RATE_LIMIT:-0}"
	    eval_one
	    ;;
	  eval-rot6d59-textop-initref)
	    AGENT=psi0_kimodo_textop_tracker
	    RUN_DIR="$ROT6D59_RUN_DIR"
	    EVAL_DIR="${ROT6D59_TEXTOP_EVAL_DIR:-data/evals_kimodo_textop_rot6d59_initref}"
	    KIMODO_WORK_DIR="${ROT6D59_KIMODO_WORK_DIR:-${PSI0_ROOT}/outputs/kimodo_textop_rot6d59_initref}"
	    export RUN_DIR
	    export FALL_STOP="${FALL_STOP:-1}"
	    export SKIP_STABILIZE="${SKIP_STABILIZE:-0}"
	    export TEXTOP_TARGET_RATE_LIMIT="${TEXTOP_TARGET_RATE_LIMIT:-0}"
	    export TEXTOP_INIT_TO_REF=1
	    eval_simple
	    ;;
	  eval-rot6d59-textop-initref-one)
	    AGENT=psi0_kimodo_textop_tracker
	    RUN_DIR="$ROT6D59_RUN_DIR"
	    EVAL_DIR="${ROT6D59_TEXTOP_EVAL_DIR:-data/evals_kimodo_textop_rot6d59_initref}"
	    KIMODO_WORK_DIR="${ROT6D59_KIMODO_WORK_DIR:-${PSI0_ROOT}/outputs/kimodo_textop_rot6d59_initref}"
	    export RUN_DIR
	    export FALL_STOP="${FALL_STOP:-1}"
	    export SKIP_STABILIZE="${SKIP_STABILIZE:-0}"
	    export TEXTOP_TARGET_RATE_LIMIT="${TEXTOP_TARGET_RATE_LIMIT:-0}"
	    export TEXTOP_INIT_TO_REF=1
	    eval_one
	    ;;
	  eval-rot6d59-textop-onestep)
	    configure_textop_onestep_defaults
	    AGENT=psi0_kimodo_textop_tracker
	    RUN_DIR="$TEXTOP_ONESTEP_RUN_DIR"
	    EVAL_DIR="${ROT6D59_ONESTEP_TEXTOP_EVAL_DIR:-data/evals_kimodo_textop_policyhand_rot6d59_onestep}"
	    KIMODO_WORK_DIR="${ROT6D59_ONESTEP_KIMODO_WORK_DIR:-${PSI0_ROOT}/outputs/kimodo_textop_policyhand_rot6d59_onestep}"
	    export RUN_DIR
	    export FALL_STOP="${FALL_STOP:-1}"
	    export SKIP_STABILIZE="${SKIP_STABILIZE:-0}"
	    export TEXTOP_TARGET_RATE_LIMIT="${TEXTOP_TARGET_RATE_LIMIT:-0}"
	    eval_simple
	    ;;
	  eval-rot6d59-textop-onestep-one)
	    configure_textop_onestep_defaults
	    AGENT=psi0_kimodo_textop_tracker
	    RUN_DIR="$TEXTOP_ONESTEP_RUN_DIR"
	    EVAL_DIR="${ROT6D59_ONESTEP_TEXTOP_EVAL_DIR:-data/evals_kimodo_textop_policyhand_rot6d59_onestep}"
	    KIMODO_WORK_DIR="${ROT6D59_ONESTEP_KIMODO_WORK_DIR:-${PSI0_ROOT}/outputs/kimodo_textop_policyhand_rot6d59_onestep}"
	    export RUN_DIR
	    export FALL_STOP="${FALL_STOP:-1}"
	    export SKIP_STABILIZE="${SKIP_STABILIZE:-0}"
	    export TEXTOP_TARGET_RATE_LIMIT="${TEXTOP_TARGET_RATE_LIMIT:-0}"
	    eval_one
	    ;;
	  eval-rot6d59-textop-onestep-initref)
	    configure_textop_onestep_defaults
	    AGENT=psi0_kimodo_textop_tracker
	    RUN_DIR="$TEXTOP_ONESTEP_RUN_DIR"
	    EVAL_DIR="${ROT6D59_ONESTEP_TEXTOP_EVAL_DIR:-data/evals_kimodo_textop_policyhand_rot6d59_onestep_initref}"
	    KIMODO_WORK_DIR="${ROT6D59_ONESTEP_KIMODO_WORK_DIR:-${PSI0_ROOT}/outputs/kimodo_textop_policyhand_rot6d59_onestep_initref}"
	    export RUN_DIR
	    export FALL_STOP="${FALL_STOP:-1}"
	    export SKIP_STABILIZE="${SKIP_STABILIZE:-0}"
	    export TEXTOP_TARGET_RATE_LIMIT="${TEXTOP_TARGET_RATE_LIMIT:-0}"
	    export TEXTOP_INIT_TO_REF=1
	    eval_simple
	    ;;
	  eval-rot6d59-textop-onestep-initref-one)
	    configure_textop_onestep_defaults
	    AGENT=psi0_kimodo_textop_tracker
	    RUN_DIR="$TEXTOP_ONESTEP_RUN_DIR"
	    EVAL_DIR="${ROT6D59_ONESTEP_TEXTOP_EVAL_DIR:-data/evals_kimodo_textop_policyhand_rot6d59_onestep_initref}"
	    KIMODO_WORK_DIR="${ROT6D59_ONESTEP_KIMODO_WORK_DIR:-${PSI0_ROOT}/outputs/kimodo_textop_policyhand_rot6d59_onestep_initref}"
	    export RUN_DIR
	    export FALL_STOP="${FALL_STOP:-1}"
	    export SKIP_STABILIZE="${SKIP_STABILIZE:-0}"
	    export TEXTOP_TARGET_RATE_LIMIT="${TEXTOP_TARGET_RATE_LIMIT:-0}"
	    export TEXTOP_INIT_TO_REF=1
	    eval_one
	    ;;
	  eval-rot6d59-textop-onestep-initref-rgz)
	    configure_textop_onestep_rgz_defaults
	    AGENT=psi0_kimodo_textop_tracker
	    RUN_DIR="$TEXTOP_ONESTEP_RUN_DIR"
	    EVAL_DIR="${ROT6D59_RGZ_ONESTEP_TEXTOP_EVAL_DIR:-data/evals_kimodo_textop_rgz_policyhand_rot6d59_onestep_initref}"
	    KIMODO_WORK_DIR="${ROT6D59_RGZ_ONESTEP_KIMODO_WORK_DIR:-${PSI0_ROOT}/outputs/kimodo_textop_rgz_policyhand_rot6d59_onestep_initref}"
	    export RUN_DIR
	    export FALL_STOP="${FALL_STOP:-1}"
	    export SKIP_STABILIZE="${SKIP_STABILIZE:-0}"
	    export TEXTOP_TARGET_RATE_LIMIT="${TEXTOP_TARGET_RATE_LIMIT:-0}"
	    export TEXTOP_INIT_TO_REF=1
	    eval_simple
	    ;;
	  eval-rot6d59-textop-onestep-initref-one-rgz)
	    configure_textop_onestep_rgz_defaults
	    AGENT=psi0_kimodo_textop_tracker
	    RUN_DIR="$TEXTOP_ONESTEP_RUN_DIR"
	    EVAL_DIR="${ROT6D59_RGZ_ONESTEP_TEXTOP_EVAL_DIR:-data/evals_kimodo_textop_rgz_policyhand_rot6d59_onestep_initref}"
	    KIMODO_WORK_DIR="${ROT6D59_RGZ_ONESTEP_KIMODO_WORK_DIR:-${PSI0_ROOT}/outputs/kimodo_textop_rgz_policyhand_rot6d59_onestep_initref}"
	    export RUN_DIR
	    export FALL_STOP="${FALL_STOP:-1}"
	    export SKIP_STABILIZE="${SKIP_STABILIZE:-0}"
	    export TEXTOP_TARGET_RATE_LIMIT="${TEXTOP_TARGET_RATE_LIMIT:-0}"
	    export TEXTOP_INIT_TO_REF=1
	    eval_one
	    ;;
	  eval-rot6d59-textop-onestep-initref-rgz-qguided)
	    configure_textop_onestep_rgz_defaults
	    AGENT=psi0_kimodo_textop_tracker
	    RUN_DIR="$TEXTOP_ONESTEP_RUN_DIR"
	    EVAL_DIR="${ROT6D59_RGZ_QGUIDED_EVAL_DIR:-data/evals_kimodo_textop_rgz_policyhand_rot6d59_onestep_initref_qguided}"
	    KIMODO_WORK_DIR="${ROT6D59_RGZ_QGUIDED_KIMODO_WORK_DIR:-${PSI0_ROOT}/outputs/kimodo_textop_rgz_policyhand_rot6d59_onestep_initref_qguided}"
	    export RUN_DIR
	    export FALL_STOP="${FALL_STOP:-1}"
	    export SKIP_STABILIZE="${SKIP_STABILIZE:-0}"
	    export TEXTOP_TARGET_RATE_LIMIT="${TEXTOP_TARGET_RATE_LIMIT:-0}"
	    export TEXTOP_INIT_TO_REF=1
	    eval_simple
	    ;;
	  eval-rot6d59-textop-onestep-initref-one-rgz-qguided)
	    configure_textop_onestep_rgz_defaults
	    AGENT=psi0_kimodo_textop_tracker
	    RUN_DIR="$TEXTOP_ONESTEP_RUN_DIR"
	    EVAL_DIR="${ROT6D59_RGZ_QGUIDED_EVAL_DIR:-data/evals_kimodo_textop_rgz_policyhand_rot6d59_onestep_initref_qguided}"
	    KIMODO_WORK_DIR="${ROT6D59_RGZ_QGUIDED_KIMODO_WORK_DIR:-${PSI0_ROOT}/outputs/kimodo_textop_rgz_policyhand_rot6d59_onestep_initref_qguided}"
	    export RUN_DIR
	    export FALL_STOP="${FALL_STOP:-1}"
	    export SKIP_STABILIZE="${SKIP_STABILIZE:-0}"
	    export TEXTOP_TARGET_RATE_LIMIT="${TEXTOP_TARGET_RATE_LIMIT:-0}"
	    export TEXTOP_INIT_TO_REF=1
	    eval_one
	    ;;
	  eval-rot6d59-textop-onestep-initref-allfilter)
	    configure_textop_onestep_allfilter_defaults
	    AGENT=psi0_kimodo_textop_tracker
	    RUN_DIR="$TEXTOP_ONESTEP_RUN_DIR"
	    EVAL_DIR="${ROT6D59_ALLFILTER_ONESTEP_TEXTOP_EVAL_DIR:-data/evals_kimodo_textop_allfilter_policyhand_rot6d59_onestep_initref}"
	    KIMODO_WORK_DIR="${ROT6D59_ALLFILTER_ONESTEP_KIMODO_WORK_DIR:-${PSI0_ROOT}/outputs/kimodo_textop_allfilter_policyhand_rot6d59_onestep_initref}"
	    export RUN_DIR
	    export FALL_STOP="${FALL_STOP:-1}"
	    export SKIP_STABILIZE="${SKIP_STABILIZE:-0}"
	    export TEXTOP_TARGET_RATE_LIMIT="${TEXTOP_TARGET_RATE_LIMIT:-0}"
	    export TEXTOP_INIT_TO_REF=1
	    eval_simple
	    ;;
	  eval-rot6d59-textop-onestep-initref-one-allfilter)
	    configure_textop_onestep_allfilter_defaults
	    AGENT=psi0_kimodo_textop_tracker
	    RUN_DIR="$TEXTOP_ONESTEP_RUN_DIR"
	    EVAL_DIR="${ROT6D59_ALLFILTER_ONESTEP_TEXTOP_EVAL_DIR:-data/evals_kimodo_textop_allfilter_policyhand_rot6d59_onestep_initref}"
	    KIMODO_WORK_DIR="${ROT6D59_ALLFILTER_ONESTEP_KIMODO_WORK_DIR:-${PSI0_ROOT}/outputs/kimodo_textop_allfilter_policyhand_rot6d59_onestep_initref}"
	    export RUN_DIR
	    export FALL_STOP="${FALL_STOP:-1}"
	    export SKIP_STABILIZE="${SKIP_STABILIZE:-0}"
	    export TEXTOP_TARGET_RATE_LIMIT="${TEXTOP_TARGET_RATE_LIMIT:-0}"
	    export TEXTOP_INIT_TO_REF=1
	    eval_one
	    ;;
	  eval-direct49-textop)
	    AGENT=psi0_direct49_textop_tracker
	    EVAL_DIR="${DIRECT49_TEXTOP_EVAL_DIR:-data/evals_direct49_textop}"
	    export FALL_STOP="${FALL_STOP:-1}"
	    eval_simple
	    ;;
	  eval-direct49-textop-one)
	    AGENT=psi0_direct49_textop_tracker
	    EVAL_DIR="${DIRECT49_TEXTOP_EVAL_DIR:-data/evals_direct49_textop}"
	    export FALL_STOP="${FALL_STOP:-1}"
	    eval_one
	    ;;
	  eval-ik-textop)
	    AGENT=psi0_ik_textop_tracker
	    EVAL_DIR="${IK_TEXTOP_EVAL_DIR:-data/evals_ik_textop}"
	    export FALL_STOP="${FALL_STOP:-1}"
	    eval_simple
	    ;;
	  eval-ik-textop-one)
	    AGENT=psi0_ik_textop_tracker
	    EVAL_DIR="${IK_TEXTOP_EVAL_DIR:-data/evals_ik_textop}"
	    export FALL_STOP="${FALL_STOP:-1}"
	    eval_one
	    ;;
	  eval-direct73-textop)
	    AGENT=psi0_direct73_textop_tracker
	    EVAL_DIR="${DIRECT73_TEXTOP_EVAL_DIR:-data/evals_direct73_textop}"
	    export FALL_STOP="${FALL_STOP:-1}"
	    eval_simple
	    ;;
	  eval-direct73-textop-one)
	    AGENT=psi0_direct73_textop_tracker
	    EVAL_DIR="${DIRECT73_TEXTOP_EVAL_DIR:-data/evals_direct73_textop}"
	    export FALL_STOP="${FALL_STOP:-1}"
	    eval_one
	    ;;
	  stitch)
	    stitch_kimodo_chunks
	    ;;
	  stitch30)
	    stitch_kimodo_chunks_30hz
	    ;;
	  *)
	    echo "Usage: $0 {serve|serve-rot6d59|serve-policyhand-rot6d59|serve-policyhand-rot6d59-qguided|kimodo-serve|health|download-data|eval|eval-one|eval-textop|eval-textop-one|eval-textop-direct|eval-textop-direct-one|eval-textop-initref|eval-textop-initref-one|eval-rot6d59-textop|eval-rot6d59-textop-one|eval-rot6d59-textop-onestep|eval-rot6d59-textop-onestep-one|eval-rot6d59-textop-onestep-initref|eval-rot6d59-textop-onestep-initref-one|eval-rot6d59-textop-onestep-initref-rgz|eval-rot6d59-textop-onestep-initref-one-rgz|eval-rot6d59-textop-onestep-initref-rgz-qguided|eval-rot6d59-textop-onestep-initref-one-rgz-qguided|eval-rot6d59-textop-onestep-initref-allfilter|eval-rot6d59-textop-onestep-initref-one-allfilter|eval-direct49-textop|eval-direct49-textop-one|eval-ik-textop|eval-ik-textop-one|eval-direct73-textop|eval-direct73-textop-one|stitch|stitch30}"
	    echo
	    echo "Examples:"
	    echo "  SERVE_GPU=1 bash $0 serve"
	    echo "  SERVE_GPU=1 bash $0 serve-policyhand-rot6d59"
	    echo "  SERVE_GPU=1 Q_GUIDANCE_BETA=0.03 bash $0 serve-policyhand-rot6d59-qguided"
	    echo "  KIMODO_GPU=2 bash $0 kimodo-serve"
	    echo "  bash $0 health"
	    echo "  EVAL_GPU=3 KIMODO_KEEP_WORK=1 bash $0 eval"
	    echo "  EVAL_GPU=3 bash $0 eval-rot6d59-textop-onestep-initref-one-rgz"
	    echo "  EVAL_GPU=3 NUM_EPISODES=10 bash $0 eval-rot6d59-textop-onestep-initref-rgz-qguided"
	    echo "  EVAL_GPU=3 KIMODO_WORK_DIR=/tmp/psi0_kimodo_eval_ep0 EVAL_DIR=data/evals_decoupled_wbc_ep0 bash $0 eval-one"
	    echo "  KIMODO_WORK_DIR=/tmp/psi0_kimodo_eval_ep0 bash $0 stitch"
	    echo "  KIMODO_WORK_DIR=/tmp/psi0_kimodo_eval_ep0 bash $0 stitch30"
	    exit 1
	    ;;
esac
