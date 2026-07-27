#!/usr/bin/env bash
set -euo pipefail

PSI0_ROOT="${PSI0_ROOT:-/pfs/pfs-ilWc5D/yzh/Psi0}"
SIMPLE_ROOT="${SIMPLE_ROOT:-${PSI0_ROOT}/third_party/SIMPLE}"
GR00T_ROOT="${GR00T_ROOT:-/pfs/pfs-ilWc5D/yzh/Isaac-GR00T}"
MODEL_PATH="${GR00T_MODEL_PATH:-${GR00T_ROOT}/outputs/task1-gr00t-n1.7-finetune/checkpoint-80000}"
BACKBONE_PATH="${GR00T_BACKBONE_PATH:-${PSI0_ROOT}/huggingface/hub/models--nvidia--Cosmos-Reason2-2B/snapshots/9ce19a195e423419c349abfc86fd07178b230561}"
GR00T_PORT="${GR00T_PORT:-22095}"

case "${1:-}" in
  serve)
    cd "$GR00T_ROOT"
    export CUDA_VISIBLE_DEVICES="${SERVE_GPU:-1}"
    export PYTHONPATH="${GR00T_ROOT}:${PYTHONPATH:-}"
    export HF_HOME="${HF_HOME:-${PSI0_ROOT}/huggingface}"
    export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
    export NO_ALBUMENTATIONS_UPDATE=1
    PREFIX_RTC_ARGS=()
    if [[ "${GR00T_PREFIX_RTC:-0}" == "1" ]]; then
      PREFIX_RTC_ARGS+=(
        --prefix-rtc
        --action-exec-horizon "${GR00T_EXECUTION_HORIZON:-34}"
      )
    fi
    exec .venv/bin/python \
      "${PSI0_ROOT}/scripts/deploy/gr00t_n17_sonic_server.py" \
      --model-path "$MODEL_PATH" \
      --backbone-path "$BACKBONE_PATH" \
      --host 0.0.0.0 \
      --port "$GR00T_PORT" \
      --device cuda \
      "${PREFIX_RTC_ARGS[@]}"
    ;;
  eval)
    cd "$SIMPLE_ROOT"
    source .venv/bin/activate
    export CUDA_VISIBLE_DEVICES="${EVAL_GPU:-3}"
    export MUJOCO_GL=egl
    export PYOPENGL_PLATFORM=egl
    export SONIC_DECODER_ONNX="${SONIC_DECODER_ONNX:-/pfs/pfs-ilWc5D/yzh/SONIC_my/gear_sonic_deploy/policy/release/model_decoder.onnx}"
    # This task was collected from a deterministic recorded initial pose.
    # By default, skip SIMPLE's generic stabilization phase so the first
    # policy observation stays close to the training distribution.  Override
    # with SKIP_STABILIZE=0 only for explicit stabilization ablations.
    export SKIP_STABILIZE="${SKIP_STABILIZE:-1}"
    export GR00T_INITIAL_POSE_STEPS="${GR00T_INITIAL_POSE_STEPS:-0}"
    SAVE_VIDEO="${SAVE_VIDEO:-1}"
    VIDEO_FLAG="--save-video"
    if [[ "$SAVE_VIDEO" == "0" ]]; then
      VIDEO_FLAG="--no-save-video"
    fi
    EVAL_DIR="${GR00T_SONIC_EVAL_DIR:-data/evals_fullstate_20260615_task1_gr00t_n17_sonic}"
    TASK="${GR00T_SONIC_TASK:-G1Fullstate20260615Task1-v0}"
    export GR00T_SONIC_DEBUG_DIR="${GR00T_SONIC_DEBUG_DIR:-${EVAL_DIR}/debug}"
    python src/simple/cli/eval_decoupled_wbc.py \
      "simple/${TASK}" \
      gr00t_n17_sonic \
      level-0 \
      --eval-dir="$EVAL_DIR" \
      --host="${GR00T_HOST:-localhost}" \
      --port="$GR00T_PORT" \
      --sim-mode=mujoco \
      --headless \
      "$VIDEO_FLAG" \
      --data-format=fixed \
      --data-dir=unused \
      --num-episodes="${NUM_EPISODES:-1}" \
      --episode-start="${EPISODE_START:-0}" \
      --max-episode-steps="${MAX_EPISODE_STEPS:-800}"
    ;;
  *)
    echo "Usage: $0 {serve|eval}"
    echo "  SERVE_GPU=1 bash $0 serve"
    echo "  EVAL_GPU=3 NUM_EPISODES=1 bash $0 eval"
    exit 2
    ;;
esac
