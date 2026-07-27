#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/deploy/simple_psi0_eval_commands.sh serve
#   bash scripts/deploy/simple_psi0_eval_commands.sh health
#   bash scripts/deploy/simple_psi0_eval_commands.sh download-data
#   bash scripts/deploy/simple_psi0_eval_commands.sh eval
#
# Terminal 1: run "serve" from Psi0.
# Terminal 2: run "health" first, then run "eval" from SIMPLE.

PSI0_ROOT="${PSI0_ROOT:-/pfs/pfs-ilWc5D/yzh/Psi0}"
SIMPLE_ROOT="${SIMPLE_ROOT:-${PSI0_ROOT}/third_party/SIMPLE}"

TASK="${TASK:-G1WholebodyXMovePickTeleop-v0}"
DR="${DR:-level-0}"
ENTRY="${ENTRY:-eval_decoupled_wbc.py}"
AGENT="${AGENT:-psi0_decoupled_wbc}"

RUN_DIR="${RUN_DIR:-${PSI0_ROOT}/.runs/finetune/g1wholebodyxmovepickteleop-v0.simple.flow1000.cosine.lr1.0e-04.b64.gpus4.2606031955}"
CKPT_STEP="${CKPT_STEP:-40000}"

SERVE_GPU="${SERVE_GPU:-1}"
EVAL_GPU="${EVAL_GPU:-2}"
HOST="${HOST:-localhost}"
PORT="${PORT:-22085}"
ACTION_EXEC_HORIZON="${ACTION_EXEC_HORIZON:-24}"
HF_HOME="${HF_HOME:-${PSI0_ROOT}/huggingface}"
HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
QWEN3VL_VARIANT="${QWEN3VL_VARIANT:-${HF_HUB_CACHE}/models--Qwen--Qwen3-VL-2B-Instruct/snapshots/89644892e4d85e24eaac8bacfd4f463576704203}"
USE_UV_SERVE="${USE_UV_SERVE:-0}"

serve() {
  cd "$PSI0_ROOT"
  source .venv-psi/bin/activate

  export CUDA_VISIBLE_DEVICES="$SERVE_GPU"
  export run_dir="$RUN_DIR"
  export ckpt_step="$CKPT_STEP"
  export HF_HOME
  export HF_HUB_CACHE
  export HUGGINGFACE_HUB_CACHE
  export QWEN3VL_VARIANT
  export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
  export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

  if [[ "$USE_UV_SERVE" == "1" ]]; then
    uv run --active --group psi --group serve serve_psi0 \
      --host 0.0.0.0 \
      --port "$PORT" \
      --run-dir="$run_dir" \
      --ckpt-step="$ckpt_step" \
      --action-exec-horizon="$ACTION_EXEC_HORIZON" \
      --rtc
  else
    serve_psi0 \
      --host 0.0.0.0 \
      --port "$PORT" \
      --run-dir="$run_dir" \
      --ckpt-step="$ckpt_step" \
      --action-exec-horizon="$ACTION_EXEC_HORIZON" \
      --rtc
  fi
}

health() {
  export NO_PROXY="${NO_PROXY:-localhost,127.0.0.1,0.0.0.0,::1}"
  export no_proxy="${no_proxy:-localhost,127.0.0.1,0.0.0.0,::1}"
  curl -i "http://${HOST}:${PORT}/health"
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
  export OMNI_KIT_ACCEPT_EULA=Y
  export MUJOCO_GL=egl
  export PYOPENGL_PLATFORM=egl
  export NO_PROXY="${NO_PROXY:-localhost,127.0.0.1,0.0.0.0,::1}"
  export no_proxy="${no_proxy:-localhost,127.0.0.1,0.0.0.0,::1}"

  python "src/simple/cli/${ENTRY}" \
    "simple/${TASK}" \
    "$AGENT" \
    "$DR" \
    --host="$HOST" \
    --port="$PORT" \
    --sim-mode=mujoco_isaac \
    --headless \
    --data-format=lerobot \
    --data-dir="data/evals/simple-eval/${TASK}/${DR}"
}

case "${1:-}" in
  serve)
    serve
    ;;
  health)
    health
    ;;
  download-data)
    download_data
    ;;
  eval)
    eval_simple
    ;;
  *)
    echo "Usage: $0 {serve|health|download-data|eval}"
    echo
    echo "Examples:"
    echo "  SERVE_GPU=1 bash $0 serve"
    echo "  bash $0 health"
    echo "  EVAL_GPU=2 bash $0 eval"
    exit 1
    ;;
esac
