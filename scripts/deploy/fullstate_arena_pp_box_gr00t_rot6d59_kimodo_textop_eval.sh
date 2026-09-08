#!/usr/bin/env bash
set -euo pipefail

# GR00T N1.7 rot6d59 -> Kimodo -> TextOp on the HumanoidArena twist2 P&P Box scene.

PSI0_ROOT="${PSI0_ROOT:-/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Psi0}"
SIMPLE_ROOT="${SIMPLE_ROOT:-${PSI0_ROOT}/third_party/SIMPLE}"
GR00T_ROOT="${GR00T_ROOT:-/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Isaac-GR00T}"
BASE_SCRIPT="${PSI0_ROOT}/scripts/deploy/fullstate_task1_gr00t_rot6d59_kimodo_textop_eval.sh"

select_gr00t_server_python() {
  if [[ -n "${GR00T_PYTHON:-}" ]]; then
    return 0
  fi
  GR00T_IMPORT_CHECK='import importlib.util, sys; modules = ("fastapi", "torch", "tyro", "uvicorn", "gr00t"); sys.exit(any(importlib.util.find_spec(module) is None for module in modules))'
  for candidate in \
    "${GR00T_ROOT}/.venv/bin/python" \
    "${GR00T_ROOT}/.venv.bak-py310-20260731/bin/python" \
    "${GR00T_ROOT}"/.venv*/bin/python; do
    if [[ -x "$candidate" ]] \
      && "$candidate" -c "$GR00T_IMPORT_CHECK" >/dev/null 2>&1; then
      export GR00T_PYTHON="$candidate"
      break
    fi
  done
  if [[ -z "${GR00T_PYTHON:-}" ]]; then
    echo "No complete GR00T server Python environment was found under $GR00T_ROOT." >&2
    echo "Set GR00T_PYTHON to an environment with the GR00T serve dependencies." >&2
    return 1
  fi
}

GR00T_JSON_PYTHON="${GR00T_JSON_PYTHON:-/usr/bin/python3.10}"

export MPLCONFIGDIR="${MPLCONFIGDIR:-${PSI0_ROOT}/.cache/matplotlib}"
export MESA_SHADER_CACHE_DIR="${MESA_SHADER_CACHE_DIR:-${PSI0_ROOT}/.cache/mesa_shader_cache}"
mkdir -p "$MPLCONFIGDIR" "$MESA_SHADER_CACHE_DIR"

export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"
export ACCEPT_EULA="${ACCEPT_EULA:-Y}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

export GR00T_MODEL_PATH="${GR00T_MODEL_PATH:-${GR00T_ROOT}/outputs/arena-pp-box-twist2-gr00t-n17-rot6d59/checkpoint-160000}"
export GR00T_PREFIX_RTC="${GR00T_PREFIX_RTC:-1}"
export GR00T_USE_RTC="${GR00T_USE_RTC:-1}"
export GR00T_PREFIX_RTC_TIMESTEP_MODE="${GR00T_PREFIX_RTC_TIMESTEP_MODE:-groot_clean}"
export GR00T_EXECUTION_HORIZON="${GR00T_EXECUTION_HORIZON:-34}"
export GR00T_PORT="${GR00T_PORT:-22196}"
export GR00T_USE_TRT="${GR00T_USE_TRT:-0}"
if [[ "$GR00T_USE_TRT" != "1" ]]; then
  unset GR00T_TRT_ENGINE_DIR
  unset GR00T_TRT_MODE
fi

export FULLSTATE_GR00T_TASK="G1FullstateArenaPPBox-v0"
export FULLSTATE_GR00T_SIM_MODE="${FULLSTATE_GR00T_SIM_MODE:-mujoco_isaac}"
export MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-1450}"

export HUMANOID_ARENA_ROOT="${HUMANOID_ARENA_ROOT:-/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/HumanoidArena/isaaclab_twist2_g1}"
export HUMANOID_ARENA_ISAAC_ASSET_ROOT="${HUMANOID_ARENA_ISAAC_ASSET_ROOT:-${HUMANOID_ARENA_ROOT}/assets1}"
export ARENA_PP_BOX_USE_ARENA_USD_VISUALS="${ARENA_PP_BOX_USE_ARENA_USD_VISUALS:-1}"

# Deliberately use only the original twist2 recordings with complete front MP4.
# The task filters missing videos and this count guard prevents accidental use
# of the 117-episode multicamera rerecord branch.
export ARENA_PP_BOX_RAW_DIR="${ARENA_PP_BOX_RAW_DIR:-/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/dataset/HumanoidArena_pp_box/twist2/yb}"
export ARENA_PP_BOX_EXPECTED_COMPLETE_RECORDINGS="${ARENA_PP_BOX_EXPECTED_COMPLETE_RECORDINGS:-100}"
export ARENA_PP_BOX_INIT_FROM_DATA="${ARENA_PP_BOX_INIT_FROM_DATA:-1}"
export ARENA_PP_BOX_RECORDING_SEED="${ARENA_PP_BOX_RECORDING_SEED:-0}"
export ARENA_PP_BOX_LOG_RESETS="${ARENA_PP_BOX_LOG_RESETS:-1}"
export ARENA_PP_BOX_ALLOW_INSTRUCTION_OVERRIDE="${ARENA_PP_BOX_ALLOW_INSTRUCTION_OVERRIDE:-0}"
export ARENA_PP_BOX_INSTRUCTION="${ARENA_PP_BOX_INSTRUCTION:-Pick up the box and place it on the shelf.}"
export ARENA_PP_BOX_BOX_X_OFFSET="${ARENA_PP_BOX_BOX_X_OFFSET:-0}"
export ARENA_PP_BOX_BOX_Y_OFFSET="${ARENA_PP_BOX_BOX_Y_OFFSET:-0}"
export ARENA_PP_BOX_SHELF_X_OFFSET="${ARENA_PP_BOX_SHELF_X_OFFSET:-0}"
export ARENA_PP_BOX_SHELF_Y_OFFSET="${ARENA_PP_BOX_SHELF_Y_OFFSET:-0}"

if [[ -n "${ARENA_PP_BOX_RECORDING_INDEX:-}" ]]; then
  export ARENA_PP_BOX_RECORDING_INDEX
  INIT_TAG="recording${ARENA_PP_BOX_RECORDING_INDEX}"
else
  unset ARENA_PP_BOX_RECORDING_INDEX
  INIT_TAG="recordingseed${ARENA_PP_BOX_RECORDING_SEED}"
fi

CHECKPOINT_TAG="${ARENA_PP_BOX_CHECKPOINT_TAG:-${GR00T_MODEL_PATH##*/}}"
VARIANT_TAG="${ARENA_PP_BOX_VARIANT_TAG:-${CHECKPOINT_TAG}_${INIT_TAG}}"
export FULLSTATE_GR00T_EVAL_DIR="${ARENA_PP_BOX_GR00T_EVAL_DIR:-data/evals_arena_pp_box_gr00t_rot6d59_kimodo_textop_${VARIANT_TAG}}"
export FULLSTATE_GR00T_KIMODO_WORK_DIR="${ARENA_PP_BOX_GR00T_KIMODO_WORK_DIR:-${PSI0_ROOT}/outputs/kimodo_arena_pp_box_gr00t_rot6d59_${VARIANT_TAG}}"

verify_gr00t_server() {
  local health_url="http://${HOST:-localhost}:${GR00T_PORT}/health"
  local health_payload
  local server_model_path
  local expected_model_path
  local prediction_horizon
  local execution_horizon
  local prefix_rtc

  if ! health_payload="$(curl --noproxy '*' --fail --silent --show-error \
    --max-time 5 "$health_url")"; then
    echo "GR00T server is not healthy at $health_url." >&2
    echo "Start the Arena P&P Box server before running eval." >&2
    return 1
  fi

  if ! readarray -t health_fields < <(printf '%s' "$health_payload" | "$GR00T_JSON_PYTHON" -c \
    'import json, sys; d=json.load(sys.stdin); print(d.get("model_path", "")); print(d.get("action_chunk_size", "")); print(d.get("action_exec_horizon", "")); print(int(bool(d.get("prefix_rtc", False))))'); then
    echo "Invalid GR00T health response from $health_url: $health_payload" >&2
    return 1
  fi
  server_model_path="${health_fields[0]:-}"
  prediction_horizon="${health_fields[1]:-}"
  execution_horizon="${health_fields[2]:-}"
  prefix_rtc="${health_fields[3]:-}"

  if [[ -z "$server_model_path" ]]; then
    if [[ "${GR00T_ALLOW_UNVERIFIED_SERVER:-0}" == "1" ]]; then
      echo "Warning: GR00T server does not report its model path; continuing by override." >&2
      return 0
    fi
    echo "GR00T server at $health_url does not report its model path." >&2
    echo "Restart it with this Arena P&P Box wrapper." >&2
    return 1
  fi

  expected_model_path="$("$GR00T_JSON_PYTHON" -c \
    'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve())' \
    "$GR00T_MODEL_PATH")"
  if [[ "$server_model_path" != "$expected_model_path" ]]; then
    echo "Wrong GR00T checkpoint at $health_url." >&2
    echo "Expected: $expected_model_path" >&2
    echo "Running:  $server_model_path" >&2
    echo "Stop the existing server and start this wrapper's serve command." >&2
    return 1
  fi

  if [[ "$prediction_horizon" != "40" || "$execution_horizon" != "$GR00T_EXECUTION_HORIZON" ]]; then
    echo "Wrong GR00T horizon at $health_url." >&2
    echo "Expected: predict=40 execute=$GR00T_EXECUTION_HORIZON" >&2
    echo "Running:  predict=${prediction_horizon:-<not reported>} execute=${execution_horizon:-<not reported>}" >&2
    return 1
  fi
  if [[ "$GR00T_PREFIX_RTC" == "1" && "$prefix_rtc" != "1" ]]; then
    echo "GR00T server at $health_url is not running Prefix-RTC." >&2
    return 1
  fi

  echo "Verified GR00T checkpoint: $server_model_path"
  echo "rot6d59 horizon: predict=$prediction_horizon execute=$execution_horizon carry_over=$((prediction_horizon - execution_horizon)) prefix_rtc=$prefix_rtc"
}

eval_dir_absolute() {
  if [[ "$FULLSTATE_GR00T_EVAL_DIR" = /* ]]; then
    printf '%s\n' "$FULLSTATE_GR00T_EVAL_DIR"
  else
    printf '%s/%s\n' "$SIMPLE_ROOT" "$FULLSTATE_GR00T_EVAL_DIR"
  fi
}

dry_run() {
  local output_dir
  output_dir="$(eval_dir_absolute)"
  echo "Environment: custom SIMPLE simple/G1FullstateArenaPPBox-v0 (mujoco_isaac)"
  echo "Checkpoint: $GR00T_MODEL_PATH"
  echo "Horizon: predict=40 execute=$GR00T_EXECUTION_HORIZON carry_over=$((40 - GR00T_EXECUTION_HORIZON)) prefix_rtc=$GR00T_PREFIX_RTC"
  echo "Init source: $ARENA_PP_BOX_RAW_DIR"
  echo "Init selection: complete=$ARENA_PP_BOX_EXPECTED_COMPLETE_RECORDINGS seed=$ARENA_PP_BOX_RECORDING_SEED tag=$INIT_TAG"
  echo "Offsets: box=($ARENA_PP_BOX_BOX_X_OFFSET,$ARENA_PP_BOX_BOX_Y_OFFSET) shelf=($ARENA_PP_BOX_SHELF_X_OFFSET,$ARENA_PP_BOX_SHELF_Y_OFFSET)"
  echo "Serve: SERVE_GPU=${SERVE_GPU:-1} GR00T_PORT=$GR00T_PORT bash $0 serve"
  echo "Kimodo: KIMODO_GPU=${KIMODO_GPU:-2} bash $0 kimodo-serve"
  echo "Eval: EVAL_GPU=${EVAL_GPU:-3} GR00T_PORT=$GR00T_PORT NUM_EPISODES=${NUM_EPISODES:-20} bash $0 eval"
  echo "Output: $output_dir"
  echo "Stats: $output_dir/eval_stats.txt"
}

case "${1:-}" in
  eval)
    verify_gr00t_server
    echo "Eval output: $(eval_dir_absolute)"
    echo "Eval stats: $(eval_dir_absolute)/eval_stats.txt"
    exec bash "$BASE_SCRIPT" eval
    ;;
  dry-run)
    dry_run
    ;;
  serve|kimodo-serve)
    if [[ "$1" == "serve" ]]; then
      select_gr00t_server_python
      export GR00T_PYTHON
    fi
    exec bash "$BASE_SCRIPT" "$1"
    ;;
  *)
    echo "Usage: $0 {serve|kimodo-serve|eval|dry-run}" >&2
    echo "  SERVE_GPU=1 GR00T_PORT=22196 bash $0 serve" >&2
    echo "  KIMODO_GPU=2 bash $0 kimodo-serve" >&2
    echo "  EVAL_GPU=3 GR00T_PORT=22196 NUM_EPISODES=1 bash $0 eval" >&2
    exit 2
    ;;
esac
