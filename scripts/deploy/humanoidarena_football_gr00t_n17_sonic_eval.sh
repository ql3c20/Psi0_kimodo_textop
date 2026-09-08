#!/usr/bin/env bash
set -euo pipefail

# Native GR00T N1.7 UNITREE_G1_SONIC -> SONIC decoder -> HumanoidArena.

COMMAND="${1:-}"
PSI0_ROOT="${PSI0_ROOT:-/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Psi0}"
HUMANOID_ROOT="${HUMANOID_ROOT:-/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/HumanoidArena}"
ISAACLAB_ROOT="${HUMANOID_ROOT}/isaaclab_twist2_g1"
GR00T_ROOT="${GR00T_ROOT:-/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Isaac-GR00T}"
GR00T_SITE_PACKAGES="${GR00T_SITE_PACKAGES:-${GR00T_ROOT}/.venv/lib/python3.10/site-packages}"
SERVER_PYTHON="${SERVER_PYTHON:-/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/miniconda3/envs/kimodo/bin/python}"
SERVER_SCRIPT="${PSI0_ROOT}/scripts/deploy/gr00t_n17_sonic_server.py"
EVAL_PYTHON="${EVAL_PYTHON:-${PSI0_ROOT}/third_party/SIMPLE/.venv/bin/python}"
EVAL_RUNNER="${ISAACLAB_ROOT}/script/eval_scripts/sonic/run_vla_eval.sh"
ISAAC45_DEPS_DIR="${ISAAC45_DEPS_DIR:-/pfs/pfs-oHNwH0/mnt/pfs/humanoid/wzl/Psi0/.cache/humanoidarena-isaac45-compat}"
ISAACLAB_EXTERNAL_ROOT="${ISAACLAB_EXTERNAL_ROOT:-/pfs/pfs-oHNwH0/mnt/pfs/humanoid/wjs/tts/references/HumanoidArena/external/IsaacLab}"

GR00T_MODEL_PATH="${GR00T_MODEL_PATH:-/pfs/pfs-oHNwH0/mnt/pfs/humanoid/wzl/Psi0/checkpoints/gr00t-n17-sonic-arena-football-sonic-native-v2-8gpu-bs512-step10000/checkpoint-10000}"
GR00T_BACKBONE_PATH="${GR00T_BACKBONE_PATH:-${PSI0_ROOT}/huggingface/hub/models--nvidia--Cosmos-Reason2-2B/snapshots/9ce19a195e423419c349abfc86fd07178b230561}"
ENV_CONFIG_YAML="${ENV_CONFIG_YAML:-${ISAACLAB_ROOT}/tasks/common_test_config/base_test/football_single_sonic78_train_range_goalframe_only.yaml}"
RESULTS_DIR="${RESULTS_DIR:-${PSI0_ROOT}/evals/humanoidarena_football_sonic_native_v2_ckpt10000_goalframe_only_eval50-allvideo}"
SONIC_POLICY_ROOT="${SONIC_POLICY_ROOT:-/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/SONIC_my/gear_sonic_deploy/policy/release}"
SONIC_ENCODER_PATH="${SONIC_ENCODER_PATH:-${SONIC_POLICY_ROOT}/model_encoder.onnx}"
SONIC_DECODER_PATH="${SONIC_DECODER_PATH:-${SONIC_POLICY_ROOT}/model_decoder.onnx}"

TASK_NAME="Isaac-Move-Football-Single-G129-Dex3-Wholebody"
TASK_INSTRUCTION="Move toward the football and kick it."
SERVER_GPU="${SERVER_GPU:-6}"
EVAL_GPU="${EVAL_GPU:-7}"
SERVER_PORT="${SERVER_PORT:-22197}"
EVAL_SEEDS="${EVAL_SEEDS:-0 1 2 3 4}"
REPEATS_PER_SEED="${REPEATS_PER_SEED:-10}"
MAX_STEPS="${MAX_STEPS:-1300}"
VIDEO_FPS="${VIDEO_FPS:-50}"
SONIC_VLA_EXECUTION_HORIZON="${SONIC_VLA_EXECUTION_HORIZON:-40}"
ISAAC_KIT_ARGS="${ISAAC_KIT_ARGS:---/rtx/verifyDriverVersion/enabled=false --/rtx/post/aa/op=0 --/rtx-defaults/post/aa/op=0 --/rtx-transient/post/aa/limitedOps=false --/rtx-transient/dlssg/enabled=false}"

require_file() { [[ -f "$1" ]] || { echo "Required file does not exist: $1" >&2; exit 1; }; }
require_dir() { [[ -d "$1" ]] || { echo "Required directory does not exist: $1" >&2; exit 1; }; }

configure_isaac_runtime() {
  local lab_pythonpath
  lab_pythonpath="${ISAACLAB_EXTERNAL_ROOT}/source/isaaclab:${ISAACLAB_EXTERNAL_ROOT}/source/isaaclab_tasks:${ISAACLAB_EXTERNAL_ROOT}/source/isaaclab_assets"
  export PYTHONPATH="${ISAAC45_DEPS_DIR}:${lab_pythonpath}${PYTHONPATH:+:${PYTHONPATH}}"
  export LD_LIBRARY_PATH="${ISAAC45_DEPS_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y
}

preflight() {
  require_file "$SERVER_PYTHON"
  require_file "$SERVER_SCRIPT"
  require_file "$EVAL_PYTHON"
  require_file "$EVAL_RUNNER"
  require_file "$ENV_CONFIG_YAML"
  require_file "$SONIC_ENCODER_PATH"
  require_file "$SONIC_DECODER_PATH"
  require_dir "$GR00T_MODEL_PATH"
  require_dir "$GR00T_BACKBONE_PATH"
  require_dir "$GR00T_SITE_PACKAGES"
  require_dir "$ISAAC45_DEPS_DIR"
  require_dir "${ISAACLAB_EXTERNAL_ROOT}/source/isaaclab/isaaclab"
  PYTHONPATH="${GR00T_SITE_PACKAGES}:${GR00T_ROOT}" \
    "$SERVER_PYTHON" -c 'import fastapi, torch, tyro, uvicorn, gr00t' >/dev/null
  if ! [[ "$SERVER_GPU" =~ ^[0-7]$ && "$EVAL_GPU" =~ ^[0-7]$ ]]; then
    echo "SERVER_GPU and EVAL_GPU must be physical GPU indices 0-7" >&2
    exit 2
  fi
}

wait_health() {
  local start=$SECONDS
  until curl --noproxy '*' --fail --silent --max-time 2 "http://127.0.0.1:${SERVER_PORT}/health" >/dev/null 2>&1; do
    if (( SECONDS - start >= 600 )); then
      echo "Timed out waiting for native GR00T SONIC server" >&2
      return 1
    fi
    sleep 1
  done
}

serve() {
  preflight
  cd "$GR00T_ROOT"
  export CUDA_VISIBLE_DEVICES="$SERVER_GPU"
  export PYTHONPATH="${GR00T_SITE_PACKAGES}:${GR00T_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
  export HF_HOME="${HF_HOME:-${PSI0_ROOT}/huggingface}"
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 NO_ALBUMENTATIONS_UPDATE=1
  exec "$SERVER_PYTHON" "$SERVER_SCRIPT" \
    --model-path "$GR00T_MODEL_PATH" \
    --backbone-path "$GR00T_BACKBONE_PATH" \
    --host 0.0.0.0 --port "$SERVER_PORT" --device cuda
}

eval_checkpoint() {
  preflight
  configure_isaac_runtime
  wait_health
  export YZH_PSI0_ROOT="$PSI0_ROOT"
  export GR00T_TASK_INSTRUCTION="$TASK_INSTRUCTION"
  export SONIC_VLA_EXECUTION_HORIZON
  export NO_PROXY="${NO_PROXY:-localhost,127.0.0.1,0.0.0.0,::1}"
  export no_proxy="${no_proxy:-${NO_PROXY}}"
  cd "$ISAACLAB_ROOT"
  EVAL_PYTHON="$EVAL_PYTHON" \
  MODEL_PATH="$GR00T_MODEL_PATH" \
  ENV_CONFIG_YAML="$ENV_CONFIG_YAML" TASK_NAME="$TASK_NAME" \
  SONIC_VLA_ACTION_FORMAT=sonic78 \
  SONIC_ENCODER_PATH="$SONIC_ENCODER_PATH" SONIC_DECODER_PATH="$SONIC_DECODER_PATH" \
  EXTERNAL_SERVER=1 SERVER_HOST=127.0.0.1 SERVER_PORT="$SERVER_PORT" SERVER_SCHEME=http \
  ISAAC_DEVICE="cuda:${EVAL_GPU}" EVAL_SEEDS="$EVAL_SEEDS" \
  REPEATS_PER_SEED="$REPEATS_PER_SEED" RESULTS_DIR="$RESULTS_DIR" \
  MAX_STEPS="$MAX_STEPS" VIDEO_FPS="$VIDEO_FPS" PERSISTENT_SIM=1 HEADLESS=1 \
  RECORD_VIDEO_EVERY_N=1 LEROBOT_VLA_RECORD_OUTPUTS=0 \
  LEROBOT_SERVER_TIMEOUT="${LEROBOT_SERVER_TIMEOUT:-360}" ISAAC_KIT_ARGS="$ISAAC_KIT_ARGS" \
  bash "$EVAL_RUNNER"
}

run_all() {
  preflight
  mkdir -p "$RESULTS_DIR/logs"
  bash "$0" serve >"$RESULTS_DIR/logs/gr00t_sonic_server.log" 2>&1 &
  local server_pid=$!
  trap 'kill "$server_pid" 2>/dev/null || true' EXIT
  trap 'exit 130' INT TERM
  wait_health
  local health
  health="$(curl --noproxy '*' --fail --silent "http://127.0.0.1:${SERVER_PORT}/health")"
  if [[ "$health" != *"$(realpath "$GR00T_MODEL_PATH")"* || "$health" != *'"action_format":"sonic78"'* ]]; then
    echo "Unexpected native server health payload: $health" >&2
    exit 1
  fi
  eval_checkpoint
}

dry_run() {
  preflight
  echo "HumanoidArena native GR00T SONIC eval"
  echo "  checkpoint=$GR00T_MODEL_PATH"
  echo "  env_config=$ENV_CONFIG_YAML"
  echo "  GPUs: server=$SERVER_GPU isaac+sonic_decoder=$EVAL_GPU"
  echo "  horizon=predict40/execute${SONIC_VLA_EXECUTION_HORIZON}"
  echo "  seeds=$EVAL_SEEDS repeats=$REPEATS_PER_SEED max_steps=$MAX_STEPS fps=$VIDEO_FPS"
  echo "  results=$RESULTS_DIR"
  echo "  instruction=$TASK_INSTRUCTION"
}

case "$COMMAND" in
  serve) serve ;;
  eval) eval_checkpoint ;;
  all) run_all ;;
  dry-run) dry_run ;;
  *) echo "Usage: bash $0 {serve|eval|all|dry-run}" >&2; exit 2 ;;
esac
