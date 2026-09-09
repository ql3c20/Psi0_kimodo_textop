#!/usr/bin/env bash
set -euo pipefail

# Native GR00T N1.7 SONIC -> released SONIC decoder on the SIMPLE Arena football scene.

PSI0_ROOT="${PSI0_ROOT:-/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Psi0}"
SIMPLE_ROOT="${SIMPLE_ROOT:-${PSI0_ROOT}/third_party/SIMPLE}"
GR00T_ROOT="${GR00T_ROOT:-/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Isaac-GR00T}"
GR00T_SITE_PACKAGES="${GR00T_SITE_PACKAGES:-${GR00T_ROOT}/.venv/lib/python3.10/site-packages}"
KIMODO_PYTHON="${KIMODO_PYTHON:-/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/miniconda3/envs/kimodo/bin/python}"
CUROBO_SRC="${CUROBO_SRC:-${SIMPLE_ROOT}/third_party/curobo/src}"
CUDA_SHIM_DIR="${CUDA_SHIM_DIR:-${SIMPLE_ROOT}/.runtime/host-libcuda}"
# eval_checkpoint delegates to a legacy task-level launcher in a child shell.
# Export the resolved roots so that child does not fall back to its obsolete
# /pfs/pfs-ilWc5D defaults.
export PSI0_ROOT SIMPLE_ROOT GR00T_ROOT GR00T_SITE_PACKAGES KIMODO_PYTHON
BASE_SCRIPT="${PSI0_ROOT}/scripts/deploy/fullstate_task1_gr00t_n17_sonic_eval.sh"
SERVER_SCRIPT="${PSI0_ROOT}/scripts/deploy/gr00t_n17_sonic_server.py"

export GR00T_MODEL_PATH="${GR00T_MODEL_PATH:-/pfs/pfs-oHNwH0/mnt/pfs/humanoid/wzl/Psi0/checkpoints/gr00t-n17-sonic-arena-football-sonic-native-realized-v1-4gpu-bs512-step10000/checkpoint-10000}"
export GR00T_BACKBONE_PATH="${GR00T_BACKBONE_PATH:-${PSI0_ROOT}/huggingface/hub/models--nvidia--Cosmos-Reason2-2B/snapshots/9ce19a195e423419c349abfc86fd07178b230561}"
export SONIC_DECODER_ONNX="${SONIC_DECODER_ONNX:-/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/SONIC_my/gear_sonic_deploy/policy/release/model_decoder.onnx}"
export GR00T_PORT="${GR00T_PORT:-22097}"
export GR00T_EXECUTION_HORIZON="${GR00T_EXECUTION_HORIZON:-40}"
export GR00T_PREFIX_RTC="${GR00T_PREFIX_RTC:-0}"
export GR00T_PREFIX_RTC_TIMESTEP_MODE="${GR00T_PREFIX_RTC_TIMESTEP_MODE:-groot_clean}"
export SERVER_READY_TIMEOUT="${SERVER_READY_TIMEOUT:-600}"
export SERVE_GPU="${SERVE_GPU:-4}"
export EVAL_GPU="${EVAL_GPU:-5}"

export GR00T_SONIC_TASK="G1FullstateArenaFootball-v0"
export GR00T_SONIC_SIM_MODE="${GR00T_SONIC_SIM_MODE:-mujoco_isaac}"
export MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-1000}"
export NUM_EPISODES="${NUM_EPISODES:-50}"
export SKIP_STABILIZE="${SKIP_STABILIZE:-1}"
export GR00T_INITIAL_POSE_STEPS="${GR00T_INITIAL_POSE_STEPS:-0}"

export HUMANOID_ARENA_ROOT="${HUMANOID_ARENA_ROOT:-/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/HumanoidArena/isaaclab_twist2_g1}"
export HUMANOID_ARENA_ISAAC_ASSET_ROOT="${HUMANOID_ARENA_ISAAC_ASSET_ROOT:-${HUMANOID_ARENA_ROOT}/assets}"
export ARENA_FOOTBALL_USE_ARENA_USD_VISUALS="${ARENA_FOOTBALL_USE_ARENA_USD_VISUALS:-1}"
export ARENA_FOOTBALL_USE_GRASS_PBR="${ARENA_FOOTBALL_USE_GRASS_PBR:-1}"
export ARENA_FOOTBALL_RAW_DIR="${ARENA_FOOTBALL_RAW_DIR:-/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/dataset/HumanoidArena_football/HOI_football_v2/SONIC}"
export ARENA_FOOTBALL_INIT_FROM_DATA="${ARENA_FOOTBALL_INIT_FROM_DATA:-1}"
export ARENA_FOOTBALL_RECORDING_SEED="${ARENA_FOOTBALL_RECORDING_SEED:-0}"
export ARENA_FOOTBALL_RANDOMIZE_BALL="${ARENA_FOOTBALL_RANDOMIZE_BALL:-0}"
export ARENA_FOOTBALL_LOG_RESETS="${ARENA_FOOTBALL_LOG_RESETS:-1}"

MODEL_RUN_NAME="$(basename "$(dirname "$GR00T_MODEL_PATH")")"
CHECKPOINT_TAG="${ARENA_FOOTBALL_CHECKPOINT_TAG:-${MODEL_RUN_NAME}_${GR00T_MODEL_PATH##*/}}"
if [[ -n "${ARENA_FOOTBALL_RECORDING_INDEX:-}" ]]; then
  INIT_TAG="recording${ARENA_FOOTBALL_RECORDING_INDEX}"
elif [[ "$ARENA_FOOTBALL_INIT_FROM_DATA" == "1" ]]; then
  INIT_TAG="recordingseed${ARENA_FOOTBALL_RECORDING_SEED}"
elif [[ "$ARENA_FOOTBALL_RANDOMIZE_BALL" == "1" ]]; then
  INIT_TAG="randomseed${ARENA_FOOTBALL_OBJECT_SEED:-0}"
else
  INIT_TAG="default"
fi
export GR00T_SONIC_EVAL_DIR="${ARENA_FOOTBALL_SONIC_EVAL_DIR:-/pfs/pfs-oHNwH0/mnt/pfs/humanoid/wzl/Psi0/evals/simple_arena_football_gr00t_n17_sonic_${CHECKPOINT_TAG}_${INIT_TAG}}"
export GR00T_SONIC_DEBUG_DIR="${GR00T_SONIC_DEBUG_DIR:-${GR00T_SONIC_EVAL_DIR}/debug}"

if [[ -z "${GR00T_PYTHON:-}" ]]; then
  import_check='import fastapi, torch, tyro, uvicorn, gr00t'
  for candidate in \
    "${GR00T_ROOT}/.venv/bin/python" \
    "$KIMODO_PYTHON" \
    /usr/bin/python3.10 \
    "${GR00T_ROOT}"/.venv*/bin/python; do
    if [[ -x "$candidate" ]] && \
      PYTHONPATH="${GR00T_SITE_PACKAGES}:${GR00T_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
        "$candidate" -c "$import_check" >/dev/null 2>&1; then
      export GR00T_PYTHON="$candidate"
      break
    fi
  done
fi
if [[ -z "${GR00T_PYTHON:-}" ]]; then
  echo "No Python can import fastapi, torch, tyro, uvicorn, and gr00t." >&2
  exit 1
fi

require_file() { [[ -f "$1" ]] || { echo "Required file does not exist: $1" >&2; return 1; }; }
require_dir() { [[ -d "$1" ]] || { echo "Required directory does not exist: $1" >&2; return 1; }; }

preflight() {
  require_file "$BASE_SCRIPT"
  require_file "$SERVER_SCRIPT"
  require_file "$SONIC_DECODER_ONNX"
  require_file "${SIMPLE_ROOT}/.venv/bin/python"
  require_dir "$CUROBO_SRC/curobo/types"
  require_dir "$GR00T_MODEL_PATH"
  require_dir "$GR00T_BACKBONE_PATH"
  require_dir "$GR00T_SITE_PACKAGES"
  if [[ ! "$GR00T_PORT" =~ ^[0-9]+$ ]] || (( GR00T_PORT < 1 || GR00T_PORT > 65535 )); then
    echo "GR00T_PORT must be an integer in 1..65535; got $GR00T_PORT" >&2
    return 2
  fi
  if [[ ! "$SERVER_READY_TIMEOUT" =~ ^[0-9]+$ ]] || (( SERVER_READY_TIMEOUT < 1 )); then
    echo "SERVER_READY_TIMEOUT must be a positive integer; got $SERVER_READY_TIMEOUT" >&2
    return 2
  fi
}

server_health() {
  curl --noproxy '*' --fail --silent --max-time 5 "http://127.0.0.1:${GR00T_PORT}/health"
}

verify_server() {
  local payload
  local model_path
  payload="$(server_health)" || { echo "Native SONIC server is not healthy." >&2; return 1; }
  model_path="$(printf '%s' "$payload" | "$GR00T_PYTHON" -c 'import json,sys; print(json.load(sys.stdin).get("model_path", ""))')"
  if [[ "$model_path" != "$(realpath "$GR00T_MODEL_PATH")" ]] || [[ "$payload" != *'"action_format":"sonic78"'* ]]; then
    echo "Unexpected native SONIC health payload: $payload" >&2
    return 1
  fi
  echo "Verified native SONIC checkpoint: $model_path"
}

port_in_use() {
  "$GR00T_PYTHON" - "$GR00T_PORT" <<'PY'
import socket
import sys
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.settimeout(0.5)
    sys.exit(0 if sock.connect_ex(("127.0.0.1", int(sys.argv[1]))) == 0 else 1)
PY
}

serve() {
  preflight
  cd "$GR00T_ROOT"
  export CUDA_VISIBLE_DEVICES="$SERVE_GPU"
  export PYTHONPATH="${GR00T_SITE_PACKAGES}:${GR00T_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
  export HF_HOME="${HF_HOME:-${PSI0_ROOT}/huggingface}"
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 NO_ALBUMENTATIONS_UPDATE=1
  args=(
    --model-path "$GR00T_MODEL_PATH"
    --backbone-path "$GR00T_BACKBONE_PATH"
    --host 0.0.0.0 --port "$GR00T_PORT" --device cuda
    --action-exec-horizon "$GR00T_EXECUTION_HORIZON"
  )
  if [[ "$GR00T_PREFIX_RTC" == "1" ]]; then
    args+=(--prefix-rtc --prefix-rtc-timestep-mode "$GR00T_PREFIX_RTC_TIMESTEP_MODE")
  fi
  exec "$GR00T_PYTHON" "$SERVER_SCRIPT" "${args[@]}"
}

eval_checkpoint() {
  preflight
  verify_server
  export GR00T_HOST=127.0.0.1
  export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
  # CuRobo is vendored as source in SIMPLE and is not installed into the
  # evaluation venv.  Put it before any incomplete namespace package that
  # may be present in site-packages.
  export PYTHONPATH="${CUROBO_SRC}${PYTHONPATH:+:${PYTHONPATH}}"
  # Isaac/PhysX dlopens the unversioned libcuda.so name, while this node only
  # exposes the driver as libcuda.so.1.  Use the same local compatibility shim
  # as the managed rot6d59 launcher.
  mkdir -p "$CUDA_SHIM_DIR"
  ln -sfn /lib/x86_64-linux-gnu/libcuda.so.1 "$CUDA_SHIM_DIR/libcuda.so"
  export LD_LIBRARY_PATH="${CUDA_SHIM_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  exec bash "$BASE_SCRIPT" eval
}

SERVER_PID=""
cleanup_server() {
  local status=$?
  trap - EXIT INT TERM
  if [[ -n "$SERVER_PID" ]] && kill -0 -- "-$SERVER_PID" 2>/dev/null; then
    kill -TERM -- "-$SERVER_PID" 2>/dev/null || true
    for _ in $(seq 1 20); do
      kill -0 -- "-$SERVER_PID" 2>/dev/null || break
      sleep 0.25
    done
    kill -KILL -- "-$SERVER_PID" 2>/dev/null || true
  fi
  [[ -z "$SERVER_PID" ]] || wait "$SERVER_PID" 2>/dev/null || true
  exit "$status"
}

wait_health() {
  local log_path="$1"
  local start=$SECONDS
  until server_health >/dev/null 2>&1; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "Native SONIC server exited before becoming healthy." >&2
      tail -n 50 "$log_path" >&2 || true
      return 1
    fi
    if (( SECONDS - start >= SERVER_READY_TIMEOUT )); then
      echo "Timed out waiting for native SONIC server." >&2
      tail -n 50 "$log_path" >&2 || true
      return 1
    fi
    sleep 1
  done
}

run_all() {
  preflight
  command -v setsid >/dev/null || { echo "setsid is required." >&2; return 1; }
  command -v tee >/dev/null || { echo "tee is required." >&2; return 1; }
  if port_in_use; then
    echo "GR00T port $GR00T_PORT is already in use; refusing to reuse it." >&2
    return 1
  fi

  local run_log_dir="${GR00T_SONIC_EVAL_DIR}/logs/run_$(date +%Y%m%d_%H%M%S)_$$"
  local server_log="${run_log_dir}/gr00t_sonic_server.log"
  local eval_log="${run_log_dir}/eval.log"
  mkdir -p "$run_log_dir"
  echo "SIMPLE native SONIC football evaluation"
  echo "  checkpoint=$GR00T_MODEL_PATH"
  echo "  GPUs: server=$SERVE_GPU eval=$EVAL_GPU"
  echo "  episodes=$NUM_EPISODES recording_seed=$ARENA_FOOTBALL_RECORDING_SEED"
  echo "  ground friction=${ARENA_FOOTBALL_GROUND_FRICTION:-0.7,0.005,0.0001} restitution=${ARENA_FOOTBALL_GROUND_RESTITUTION:-MuJoCo-default}"
  echo "  foot friction=${ARENA_FOOTBALL_FOOT_FRICTION:-robot-MJCF-default-1.0,0.005,0.0001} restitution=${ARENA_FOOTBALL_FOOT_RESTITUTION:-MuJoCo-default}"
  echo "  results=$GR00T_SONIC_EVAL_DIR"
  echo "  logs=$run_log_dir"

  trap cleanup_server EXIT
  trap 'exit 130' INT TERM
  setsid env PYTHONUNBUFFERED=1 bash "$0" serve >"$server_log" 2>&1 &
  SERVER_PID=$!
  wait_health "$server_log"
  verify_server

  set +e
  PYTHONUNBUFFERED=1 SIMPLE_DISABLE_TUI=1 bash "$0" eval 2>&1 | tee "$eval_log"
  local eval_status=${PIPESTATUS[0]}
  set -e
  return "$eval_status"
}

dry_run() {
  preflight
  local port_state=free
  port_in_use && port_state=in-use
  echo "SIMPLE native SONIC football dry-run"
  echo "  checkpoint=$GR00T_MODEL_PATH"
  echo "  decoder=$SONIC_DECODER_ONNX"
  echo "  task=$GR00T_SONIC_TASK sim_mode=$GR00T_SONIC_SIM_MODE"
  echo "  GPUs: server=$SERVE_GPU eval=$EVAL_GPU"
  echo "  port=$GR00T_PORT ($port_state) horizon=$GR00T_EXECUTION_HORIZON"
  echo "  episodes=$NUM_EPISODES recording_seed=$ARENA_FOOTBALL_RECORDING_SEED"
  echo "  ground friction=${ARENA_FOOTBALL_GROUND_FRICTION:-0.7,0.005,0.0001} restitution=${ARENA_FOOTBALL_GROUND_RESTITUTION:-MuJoCo-default}"
  echo "  foot friction=${ARENA_FOOTBALL_FOOT_FRICTION:-robot-MJCF-default-1.0,0.005,0.0001} restitution=${ARENA_FOOTBALL_FOOT_RESTITUTION:-MuJoCo-default}"
  echo "  results=$GR00T_SONIC_EVAL_DIR"
}

case "${1:-}" in
  serve) serve ;;
  eval) eval_checkpoint ;;
  all) run_all ;;
  dry-run) dry_run ;;
  *) echo "Usage: bash $0 {serve|eval|all|dry-run}" >&2; exit 2 ;;
esac
