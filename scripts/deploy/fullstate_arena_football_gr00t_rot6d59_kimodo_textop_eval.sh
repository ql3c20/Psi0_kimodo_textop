#!/usr/bin/env bash
set -euo pipefail

# GR00T N1.7 rot6d59 -> Kimodo -> TextOp on the HumanoidArena football SIMPLE scene.

PSI0_ROOT="${PSI0_ROOT:-/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Psi0}"
SIMPLE_ROOT="${SIMPLE_ROOT:-${PSI0_ROOT}/third_party/SIMPLE}"
GR00T_ROOT="${GR00T_ROOT:-/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Isaac-GR00T}"
GR00T_SITE_PACKAGES="${GR00T_SITE_PACKAGES:-${GR00T_ROOT}/.venv/lib/python3.10/site-packages}"
export KIMODO_PYTHON="${KIMODO_PYTHON:-/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/miniconda3/envs/kimodo/bin/python}"
BASE_SCRIPT="${PSI0_ROOT}/scripts/deploy/fullstate_task1_gr00t_rot6d59_kimodo_textop_eval.sh"
EVAL_LABEL="${FULLSTATE_GR00T_EVAL_LABEL:-SIMPLE football}"

if [[ -z "${GR00T_PYTHON:-}" ]]; then
  GR00T_IMPORT_CHECK='import importlib.util, sys; modules = ("fastapi", "torch", "tyro", "uvicorn", "gr00t"); sys.exit(any(importlib.util.find_spec(module) is None for module in modules))'
  for candidate in \
    "${GR00T_ROOT}/.venv/bin/python" \
    "${GR00T_ROOT}/.venv.bak-py310-20260731/bin/python" \
    "$KIMODO_PYTHON" \
    /usr/bin/python3.10 \
    "${GR00T_ROOT}"/.venv*/bin/python; do
    if [[ -x "$candidate" ]] \
      && PYTHONPATH="${GR00T_SITE_PACKAGES}:${GR00T_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
        "$candidate" -c "$GR00T_IMPORT_CHECK" >/dev/null 2>&1; then
      export GR00T_PYTHON="$candidate"
      break
    fi
  done
  if [[ -z "${GR00T_PYTHON:-}" ]]; then
    echo "No complete GR00T server Python environment was found under $GR00T_ROOT." >&2
    echo "Set GR00T_PYTHON to an environment with fastapi, torch, tyro, uvicorn, and gr00t." >&2
    exit 1
  fi
fi

export MPLCONFIGDIR="${MPLCONFIGDIR:-${PSI0_ROOT}/.cache/matplotlib}"
export MESA_SHADER_CACHE_DIR="${MESA_SHADER_CACHE_DIR:-${PSI0_ROOT}/.cache/mesa_shader_cache}"
mkdir -p "$MPLCONFIGDIR" "$MESA_SHADER_CACHE_DIR"

export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"
export ACCEPT_EULA="${ACCEPT_EULA:-Y}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

export GR00T_MODEL_PATH="${GR00T_MODEL_PATH:-${GR00T_ROOT}/outputs/arena-football-gr00t-n17-rot6d59/checkpoint-160000}"
export GR00T_PREFIX_RTC="${GR00T_PREFIX_RTC:-1}"
export GR00T_USE_RTC="${GR00T_USE_RTC:-1}"
export GR00T_PREFIX_RTC_TIMESTEP_MODE="${GR00T_PREFIX_RTC_TIMESTEP_MODE:-groot_clean}"
export GR00T_EXECUTION_HORIZON="${GR00T_EXECUTION_HORIZON:-30}"
export GR00T_PORT="${GR00T_PORT:-22096}"
export KIMODO_SERVER_HOST="${KIMODO_SERVER_HOST:-127.0.0.1}"
export KIMODO_SERVER_PORT="${KIMODO_SERVER_PORT:-22185}"
export SERVER_READY_TIMEOUT="${SERVER_READY_TIMEOUT:-600}"
export GR00T_USE_TRT="${GR00T_USE_TRT:-0}"
if [[ "$GR00T_USE_TRT" != "1" ]]; then
  unset GR00T_TRT_ENGINE_DIR
  unset GR00T_TRT_MODE
fi

export FULLSTATE_GR00T_TASK="${FULLSTATE_GR00T_TASK:-G1FullstateArenaFootball-v0}"
export FULLSTATE_GR00T_SIM_MODE="${FULLSTATE_GR00T_SIM_MODE:-mujoco_isaac}"
export MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-1000}"
export NUM_EPISODES="${NUM_EPISODES:-50}"
export SAVE_VIDEO="${SAVE_VIDEO:-1}"

export HUMANOID_ARENA_ROOT="${HUMANOID_ARENA_ROOT:-/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/HumanoidArena/isaaclab_twist2_g1}"
export HUMANOID_ARENA_ISAAC_ASSET_ROOT="${HUMANOID_ARENA_ISAAC_ASSET_ROOT:-${HUMANOID_ARENA_ROOT}/assets}"
export ARENA_FOOTBALL_USE_ARENA_USD_VISUALS="${ARENA_FOOTBALL_USE_ARENA_USD_VISUALS:-1}"
export ARENA_FOOTBALL_USE_GRASS_PBR="${ARENA_FOOTBALL_USE_GRASS_PBR:-1}"

export ARENA_FOOTBALL_RAW_DIR="${ARENA_FOOTBALL_RAW_DIR:-/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/dataset/HumanoidArena_football/HOI_football_v2/SONIC}"
export ARENA_FOOTBALL_INIT_FROM_DATA="${ARENA_FOOTBALL_INIT_FROM_DATA:-1}"
export ARENA_FOOTBALL_RECORDING_SEED="${ARENA_FOOTBALL_RECORDING_SEED:-0}"
export ARENA_FOOTBALL_RANDOMIZE_BALL="${ARENA_FOOTBALL_RANDOMIZE_BALL:-0}"
export ARENA_FOOTBALL_LOG_RESETS="${ARENA_FOOTBALL_LOG_RESETS:-1}"
export ARENA_FOOTBALL_ALLOW_INSTRUCTION_OVERRIDE="${ARENA_FOOTBALL_ALLOW_INSTRUCTION_OVERRIDE:-0}"
export ARENA_FOOTBALL_INSTRUCTION="${ARENA_FOOTBALL_INSTRUCTION:-Kick the football into the goal.}"

# Keep the Arena-aligned 640x480 front camera as the sole VLA input.  Record a
# second camera from the same mount with unchanged vertical view and a wider
# horizontal aperture for undistorted 16:9 videos.
export ARENA_AUX_VIDEO_ENABLED="${ARENA_AUX_VIDEO_ENABLED:-1}"
export ARENA_AUX_VIDEO_WIDTH="${ARENA_AUX_VIDEO_WIDTH:-1280}"
export ARENA_AUX_VIDEO_HEIGHT="${ARENA_AUX_VIDEO_HEIGHT:-720}"
export ARENA_AUX_VIDEO_HORIZONTAL_APERTURE="${ARENA_AUX_VIDEO_HORIZONTAL_APERTURE:-26.666666666666668}"

if [[ -n "${ARENA_FOOTBALL_RECORDING_INDEX:-}" ]]; then
  export ARENA_FOOTBALL_RECORDING_INDEX
else
  unset ARENA_FOOTBALL_RECORDING_INDEX
fi

if [[ -n "${ARENA_FOOTBALL_RECORDING_INDEX:-}" ]]; then
  INIT_TAG="recording${ARENA_FOOTBALL_RECORDING_INDEX}"
elif [[ "$ARENA_FOOTBALL_INIT_FROM_DATA" == "1" ]]; then
  INIT_TAG="recordingseed${ARENA_FOOTBALL_RECORDING_SEED}"
elif [[ "$ARENA_FOOTBALL_RANDOMIZE_BALL" == "1" ]]; then
  INIT_TAG="randomseed${ARENA_FOOTBALL_OBJECT_SEED:-0}"
else
  INIT_TAG="default"
fi

CHECKPOINT_TAG="${ARENA_FOOTBALL_CHECKPOINT_TAG:-${GR00T_MODEL_PATH##*/}}"
VIDEO_TAG=""
if [[ "$ARENA_AUX_VIDEO_ENABLED" == "1" ]]; then
  VIDEO_TAG="_video${ARENA_AUX_VIDEO_WIDTH}x${ARENA_AUX_VIDEO_HEIGHT}_wide"
fi
VARIANT_TAG="${ARENA_FOOTBALL_VARIANT_TAG:-${CHECKPOINT_TAG}_${INIT_TAG}${VIDEO_TAG}}"
export FULLSTATE_GR00T_EVAL_DIR="${FULLSTATE_GR00T_EVAL_DIR:-${ARENA_FOOTBALL_GR00T_EVAL_DIR:-data/evals_arena_football_gr00t_rot6d59_kimodo_textop_${VARIANT_TAG}}}"
export FULLSTATE_GR00T_KIMODO_WORK_DIR="${FULLSTATE_GR00T_KIMODO_WORK_DIR:-${ARENA_FOOTBALL_GR00T_KIMODO_WORK_DIR:-${PSI0_ROOT}/outputs/kimodo_arena_football_gr00t_rot6d59_${VARIANT_TAG}}}"

if [[ "$FULLSTATE_GR00T_EVAL_DIR" == /* ]]; then
  EVAL_DIR_ABS="$FULLSTATE_GR00T_EVAL_DIR"
else
  EVAL_DIR_ABS="${SIMPLE_ROOT}/${FULLSTATE_GR00T_EVAL_DIR}"
fi

verify_gr00t_server() {
  local health_url="http://${HOST:-localhost}:${GR00T_PORT}/health"
  local health_payload
  local server_model_path
  local expected_model_path

  if ! health_payload="$(curl --noproxy '*' --fail --silent --show-error \
    --max-time 5 "$health_url")"; then
    echo "GR00T server is not healthy at $health_url." >&2
    echo "Start the ${EVAL_LABEL} server before running eval." >&2
    return 1
  fi

  if ! server_model_path="$(printf '%s' "$health_payload" | "$GR00T_PYTHON" -c \
    'import json, sys; print(json.load(sys.stdin).get("model_path", ""))')"; then
    echo "Invalid GR00T health response from $health_url: $health_payload" >&2
    return 1
  fi

  if [[ -z "$server_model_path" ]]; then
    if [[ "${GR00T_ALLOW_UNVERIFIED_SERVER:-0}" == "1" ]]; then
      echo "Warning: GR00T server does not report its model path; continuing by override." >&2
      return 0
    fi
    echo "GR00T server at $health_url does not report its model path." >&2
    echo "It may be an old or wrong-task server; restart it with this task wrapper." >&2
    echo "Set GR00T_ALLOW_UNVERIFIED_SERVER=1 only if you verified it manually." >&2
    return 1
  fi

  expected_model_path="$("$GR00T_PYTHON" -c \
    'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve())' \
    "$GR00T_MODEL_PATH")"
  if [[ "$server_model_path" != "$expected_model_path" ]]; then
    echo "Wrong GR00T checkpoint at $health_url." >&2
    echo "Expected: $expected_model_path" >&2
    echo "Running:  $server_model_path" >&2
    echo "Stop the existing server and start this task wrapper's serve command." >&2
    return 1
  fi

  echo "Verified GR00T checkpoint: $server_model_path"
}

validate_port() {
  local name="$1"
  local port="$2"
  if [[ ! "$port" =~ ^[0-9]+$ ]] || (( port < 1 || port > 65535 )); then
    echo "$name port must be an integer in 1..65535; got $port" >&2
    return 2
  fi
}

port_in_use() {
  local port="$1"
  "$GR00T_PYTHON" - "$port" <<'PY'
import socket
import sys

port = int(sys.argv[1])
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.settimeout(0.5)
    sys.exit(0 if sock.connect_ex(("127.0.0.1", port)) == 0 else 1)
PY
}

require_free_port() {
  local name="$1"
  local port="$2"
  validate_port "$name" "$port"
  if port_in_use "$port"; then
    echo "$name port $port is already in use; refusing to reuse or stop the existing process." >&2
    echo "Stop the existing service or select another port before running 'all'." >&2
    return 1
  fi
}

wait_for_health() {
  local url="$1"
  local name="$2"
  local pid="$3"
  local log_path="$4"
  local started_at=$SECONDS

  until curl --noproxy '*' --fail --silent --max-time 2 "$url" >/dev/null 2>&1; do
    if ! kill -0 "$pid" 2>/dev/null; then
      local process_status=0
      wait "$pid" || process_status=$?
      echo "$name exited before becoming healthy (status=$process_status)." >&2
      echo "Last 50 log lines from $log_path:" >&2
      tail -n 50 "$log_path" >&2 || true
      return 1
    fi
    if (( SECONDS - started_at >= SERVER_READY_TIMEOUT )); then
      echo "Timed out after ${SERVER_READY_TIMEOUT}s waiting for $name at $url." >&2
      echo "Last 50 log lines from $log_path:" >&2
      tail -n 50 "$log_path" >&2 || true
      return 1
    fi
    sleep 1
  done
  echo "$name is healthy at $url"
}

GR00T_SERVER_PID=""
KIMODO_SERVER_PID=""

managed_process_group_alive() {
  local group_leader_pid="$1"
  [[ -n "$group_leader_pid" ]] && kill -0 -- "-$group_leader_pid" 2>/dev/null
}

cleanup_managed_servers() {
  local exit_status=$?
  trap - EXIT INT TERM

  local pid
  for pid in "$GR00T_SERVER_PID" "$KIMODO_SERVER_PID"; do
    if managed_process_group_alive "$pid"; then
      kill -TERM -- "-$pid" 2>/dev/null || true
    fi
  done

  local attempt
  for attempt in $(seq 1 20); do
    local any_running=0
    for pid in "$GR00T_SERVER_PID" "$KIMODO_SERVER_PID"; do
      if managed_process_group_alive "$pid"; then
        any_running=1
      fi
    done
    (( any_running == 0 )) && break
    sleep 0.25
  done

  for pid in "$GR00T_SERVER_PID" "$KIMODO_SERVER_PID"; do
    if managed_process_group_alive "$pid"; then
      kill -KILL -- "-$pid" 2>/dev/null || true
    fi
    if [[ -n "$pid" ]]; then
      wait "$pid" 2>/dev/null || true
    fi
  done

  exit "$exit_status"
}

run_all() {
  command -v curl >/dev/null || { echo "curl is required for server health checks." >&2; return 1; }
  command -v setsid >/dev/null || { echo "setsid is required for managed server process groups." >&2; return 1; }
  command -v tee >/dev/null || { echo "tee is required for eval logging." >&2; return 1; }
  if [[ ! "$SERVER_READY_TIMEOUT" =~ ^[0-9]+$ ]] || (( SERVER_READY_TIMEOUT < 1 )); then
    echo "SERVER_READY_TIMEOUT must be a positive integer; got $SERVER_READY_TIMEOUT" >&2
    return 2
  fi
  require_free_port "GR00T" "$GR00T_PORT"
  require_free_port "Kimodo" "$KIMODO_SERVER_PORT"
  export HOST=127.0.0.1

  local run_id
  run_id="run_$(date +%Y%m%d_%H%M%S)_$$"
  local run_log_dir="${EVAL_DIR_ABS}/logs/${run_id}"
  local gr00t_log="${run_log_dir}/gr00t_server.log"
  local kimodo_log="${run_log_dir}/kimodo_server.log"
  local eval_log="${run_log_dir}/eval.log"
  mkdir -p "$run_log_dir"

  echo "${EVAL_LABEL} managed evaluation"
  echo "  checkpoint=$GR00T_MODEL_PATH"
  echo "  results=$EVAL_DIR_ABS"
  echo "  logs=$run_log_dir"
  echo "  GPUs: gr00t=${SERVE_GPU:-1} kimodo=${KIMODO_GPU:-2} eval=${EVAL_GPU:-3}"
  echo "  ports: gr00t=$GR00T_PORT kimodo=$KIMODO_SERVER_PORT"

  trap cleanup_managed_servers EXIT
  trap 'exit 130' INT TERM

  setsid env PYTHONUNBUFFERED=1 bash "$0" serve >"$gr00t_log" 2>&1 &
  GR00T_SERVER_PID=$!
  setsid env PYTHONUNBUFFERED=1 bash "$0" kimodo-serve >"$kimodo_log" 2>&1 &
  KIMODO_SERVER_PID=$!

  wait_for_health \
    "http://127.0.0.1:${GR00T_PORT}/health" \
    "GR00T server" "$GR00T_SERVER_PID" "$gr00t_log"
  verify_gr00t_server
  wait_for_health \
    "http://${KIMODO_SERVER_HOST}:${KIMODO_SERVER_PORT}/health" \
    "Kimodo server" "$KIMODO_SERVER_PID" "$kimodo_log"

  echo "Starting eval in the foreground; output is also saved to $eval_log"
  set +e
  PYTHONUNBUFFERED=1 SIMPLE_DISABLE_TUI=1 \
    bash "$0" eval 2>&1 | tee "$eval_log"
  local eval_status=${PIPESTATUS[0]}
  set -e
  if (( eval_status != 0 )); then
    echo "${EVAL_LABEL} eval failed with status $eval_status; see $eval_log" >&2
  else
    echo "${EVAL_LABEL} eval completed successfully; logs=$run_log_dir"
  fi
  return "$eval_status"
}

dry_run() {
  validate_port "GR00T" "$GR00T_PORT"
  validate_port "Kimodo" "$KIMODO_SERVER_PORT"
  local gr00t_port_state="free"
  local kimodo_port_state="free"
  port_in_use "$GR00T_PORT" && gr00t_port_state="in-use"
  port_in_use "$KIMODO_SERVER_PORT" && kimodo_port_state="in-use"

  echo "${EVAL_LABEL} managed evaluation dry-run"
  echo "  checkpoint=$GR00T_MODEL_PATH"
  echo "  results=$EVAL_DIR_ABS"
  echo "  logs=${EVAL_DIR_ABS}/logs/run_YYYYmmdd_HHMMSS_PID"
  echo "  GPUs: gr00t=${SERVE_GPU:-1} kimodo=${KIMODO_GPU:-2} eval=${EVAL_GPU:-3}"
  echo "  gr00t=http://127.0.0.1:${GR00T_PORT} ($gr00t_port_state)"
  echo "  kimodo=http://${KIMODO_SERVER_HOST}:${KIMODO_SERVER_PORT} ($kimodo_port_state)"
  echo "  server_ready_timeout=${SERVER_READY_TIMEOUT}s"
  echo "  task=$FULLSTATE_GR00T_TASK episodes=$NUM_EPISODES save_video=$SAVE_VIDEO"
  echo "  policy_camera=front_camera:640x480"
  if [[ "$ARENA_AUX_VIDEO_ENABLED" == "1" ]]; then
    echo "  video_camera=video_camera:${ARENA_AUX_VIDEO_WIDTH}x${ARENA_AUX_VIDEO_HEIGHT} horizontal_aperture=${ARENA_AUX_VIDEO_HORIZONTAL_APERTURE}"
  else
    echo "  video_camera=front_camera:640x480"
  fi
  if [[ "$FULLSTATE_GR00T_TASK" == "G1FullstateArenaOpenDoor-v0" ]]; then
    echo "  recording_seed=${ARENA_OPEN_DOOR_RECORDING_SEED:-0} raw_branch=${ARENA_OPEN_DOOR_RAW_BRANCH:-}"
  else
    echo "  recording_seed=$ARENA_FOOTBALL_RECORDING_SEED"
  fi
}

case "${1:-}" in
  serve)
    export PYTHONPATH="${GR00T_SITE_PACKAGES}:${GR00T_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
    exec bash "$BASE_SCRIPT" serve
    ;;
  kimodo-serve)
    exec bash "$BASE_SCRIPT" "$1"
    ;;
  eval)
    verify_gr00t_server
    exec bash "$BASE_SCRIPT" eval
    ;;
  all)
    run_all
    ;;
  dry-run)
    dry_run
    ;;
  *)
    echo "Usage: bash $0 {serve|kimodo-serve|eval|all|dry-run}" >&2
    exit 2
    ;;
esac
