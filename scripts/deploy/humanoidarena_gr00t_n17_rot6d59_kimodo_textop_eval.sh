#!/usr/bin/env bash
set -euo pipefail

# Native HumanoidArena evaluation:
# GR00T N1.7 rot6d59 -> Kimodo -> TextOp -> Isaac Lab joint targets.

COMMAND="${1:-}"
TASK_KIND="${2:-${ARENA_TASK_KIND:-pp_box}}"
ARENA_EVAL_PROFILE="${ARENA_EVAL_PROFILE:-recording0}"

PSI0_ROOT="${PSI0_ROOT:-/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Psi0}"
HUMANOID_ROOT="${HUMANOID_ROOT:-/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/HumanoidArena}"
ISAACLAB_ROOT="${HUMANOID_ROOT}/isaaclab_twist2_g1"
GR00T_ROOT="${GR00T_ROOT:-/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Isaac-GR00T}"
KIMODO_ROOT="${KIMODO_ROOT:-/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/kimodo_my}"

GR00T_PYTHON="${GR00T_PYTHON:-}"
GR00T_SITE_PACKAGES="${GR00T_SITE_PACKAGES:-${GR00T_ROOT}/.venv/lib/python3.10/site-packages}"
ISAAC5_PYTHON="${ISAAC5_PYTHON:-/pfs/pfs-oHNwH0/mnt/pfs/humanoid/wjs/miniconda3/envs/unitree_sim_env/bin/python}"
ISAAC45_PYTHON="${ISAAC45_PYTHON:-${PSI0_ROOT}/third_party/SIMPLE/.venv/bin/python}"
ISAAC45_DEPS_DIR="${ISAAC45_DEPS_DIR:-/pfs/pfs-oHNwH0/mnt/pfs/humanoid/wzl/Psi0/.cache/humanoidarena-isaac45-compat}"
ISAACLAB_EXTERNAL_ROOT="${ISAACLAB_EXTERNAL_ROOT:-/pfs/pfs-oHNwH0/mnt/pfs/humanoid/wjs/tts/references/HumanoidArena/external/IsaacLab}"
ISAAC_SIM_RUNTIME="${ISAAC_SIM_RUNTIME:-auto}"
KIMODO_PYTHON="${KIMODO_PYTHON:-/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/miniconda3/envs/kimodo/bin/python}"

GR00T_BACKBONE_PATH="${GR00T_BACKBONE_PATH:-${PSI0_ROOT}/huggingface/hub/models--nvidia--Cosmos-Reason2-2B/snapshots/9ce19a195e423419c349abfc86fd07178b230561}"
GR00T_MODALITY_CONFIG_PATH="${GR00T_MODALITY_CONFIG_PATH:-${GR00T_ROOT}/examples/unitree_g1_rot6d59_config.py}"
GR00T_PORT="${GR00T_PORT:-22196}"
KIMODO_PORT="${KIMODO_PORT:-22185}"
GR00T_EXECUTION_HORIZON="${GR00T_EXECUTION_HORIZON:-30}"
OPEN_DOOR_LEAF_UNLOCK_STIFFNESS="${OPEN_DOOR_LEAF_UNLOCK_STIFFNESS:-}"
OPEN_DOOR_LEAF_UNLOCK_DAMPING="${OPEN_DOOR_LEAF_UNLOCK_DAMPING:-}"

SERVE_GPU="${SERVE_GPU:-0}"
EVAL_GPU="${EVAL_GPU:-1}"
KIMODO_GPU="${KIMODO_GPU:-2}"
EVAL_SEEDS="${EVAL_SEEDS:-0}"
REPEATS_PER_SEED="${REPEATS_PER_SEED:-1}"
VIDEO_FPS="${VIDEO_FPS:-50}"

INSTALLED_NVIDIA_DRIVER=""
if command -v nvidia-smi >/dev/null 2>&1; then
  INSTALLED_NVIDIA_DRIVER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | sed -n '1p' | tr -d '[:space:]')"
fi

# Isaac Sim 5.0 sees the installed 535.309.01 driver as 535.53 through
# Vulkan. Disabling only the version check still crashes in RTX postprocessing,
# so use the locally available Isaac Sim 4.5 runtime on this exact driver.
if [[ "$ISAAC_SIM_RUNTIME" == "auto" ]]; then
  if [[ "$INSTALLED_NVIDIA_DRIVER" == "535.309.01" ]]; then
    ISAAC_SIM_RUNTIME="4.5"
  else
    ISAAC_SIM_RUNTIME="5.0"
  fi
fi

case "$ISAAC_SIM_RUNTIME" in
  4.5)
    EVAL_PYTHON="${EVAL_PYTHON:-$ISAAC45_PYTHON}"
    ;;
  5.0)
    EVAL_PYTHON="${EVAL_PYTHON:-$ISAAC5_PYTHON}"
    if [[ "$INSTALLED_NVIDIA_DRIVER" == "535.309.01" && "${ISAAC_ALLOW_UNSUPPORTED_SIM5:-0}" != "1" ]]; then
      echo "Isaac Sim 5.0 crashes with the installed NVIDIA driver 535.309.01." >&2
      echo "Use ISAAC_SIM_RUNTIME=auto/4.5, or set ISAAC_ALLOW_UNSUPPORTED_SIM5=1 only for debugging." >&2
      exit 2
    fi
    ;;
  *)
    echo "ISAAC_SIM_RUNTIME must be auto, 4.5, or 5.0; got $ISAAC_SIM_RUNTIME" >&2
    exit 2
    ;;
esac

ISAAC_DISABLE_RTX_DRIVER_CHECK="${ISAAC_DISABLE_RTX_DRIVER_CHECK:-auto}"
ISAAC_KIT_ARGS="${ISAAC_KIT_ARGS:-}"

case "$TASK_KIND" in
  pp_box)
    DEFAULT_MAX_STEPS=1300
    DEFAULT_MODEL_PATH="${GR00T_ROOT}/outputs/arena-pp-box-twist2-gr00t-n17-rot6d59/checkpoint-160000"
    TASK_NAME="Isaac-Move-PickPlace-Box-G129-Dex3-Wholedoby"
    TASK_INSTRUCTION="Pick up the box and place it on the shelf."
    DEFAULT_GROUND_TRUTH_PARQUET="${PSI0_ROOT}/data/output/arena_pp_box_twist2_rot6d59/data/chunk-000/episode_000000.parquet"
    DEFAULT_GROUND_TRUTH_MAX_STEPS=421
    case "$ARENA_EVAL_PROFILE" in
      recording0|twist2_recording0) DEFAULT_ENV_CONFIG="${ISAACLAB_ROOT}/tasks/common_test_config/base_test/pp_box_sonic78_recording0.yaml" ;;
      native_recording0) DEFAULT_ENV_CONFIG="${ISAACLAB_ROOT}/tasks/common_test_config/base_test/pp_box_sonic_native_recording0.yaml" ;;
      random) DEFAULT_ENV_CONFIG="${ISAACLAB_ROOT}/tasks/common_test_config/base_test/pp_box_sonic78_train_range.yaml" ;;
      benchmark_random) DEFAULT_ENV_CONFIG="${ISAACLAB_ROOT}/tasks/common_test_config/base_test/pp_box_sonic_test.yaml" ;;
      *)
        echo "Unknown ARENA_EVAL_PROFILE: $ARENA_EVAL_PROFILE (expected recording0/twist2_recording0, native_recording0, random, or benchmark_random)" >&2
        exit 2
        ;;
    esac
    ;;
  football)
    DEFAULT_MAX_STEPS=1300
    DEFAULT_MODEL_PATH="${GR00T_ROOT}/outputs/arena-football-gr00t-n17-rot6d59/checkpoint-160000"
    TASK_NAME="Isaac-Move-Football-Single-G129-Dex3-Wholebody"
    TASK_INSTRUCTION="Move toward the football and kick it."
    DEFAULT_GROUND_TRUTH_PARQUET="${PSI0_ROOT}/data/output/arena_football_rot6d59/data/chunk-000/episode_000000.parquet"
    DEFAULT_GROUND_TRUTH_MAX_STEPS=387
    case "$ARENA_EVAL_PROFILE" in
      recording0) DEFAULT_ENV_CONFIG="${ISAACLAB_ROOT}/tasks/common_test_config/base_test/football_single_sonic78_recording0.yaml" ;;
      random) DEFAULT_ENV_CONFIG="${ISAACLAB_ROOT}/tasks/common_test_config/base_test/football_single_sonic78_train_range.yaml" ;;
      goalframe_only) DEFAULT_ENV_CONFIG="${ISAACLAB_ROOT}/tasks/common_test_config/base_test/football_single_sonic78_train_range_goalframe_only.yaml" ;;
      benchmark_random) DEFAULT_ENV_CONFIG="${ISAACLAB_ROOT}/tasks/common_test_config/base_test/football_single_sonic_test.yaml" ;;
      *)
        echo "Unknown ARENA_EVAL_PROFILE: $ARENA_EVAL_PROFILE (expected recording0, random, goalframe_only, or benchmark_random)" >&2
        exit 2
        ;;
    esac
    ;;
  open_door)
    DEFAULT_MAX_STEPS=1800
    DEFAULT_MODEL_PATH="${GR00T_ROOT}/outputs/arena-open-door-sonic-gr00t-n17-rot6d59-v1-prefixrtc-delay0to12-4gpu-bs256-step20000/checkpoint-10000"
    TASK_NAME="Isaac-Move-Open-Door-G129-Dex3-Wholebody"
    TASK_INSTRUCTION="Open the door."
    DEFAULT_GROUND_TRUTH_PARQUET="${PSI0_ROOT}/data/output/arena_open_door_sonic_rot6d59_v1/data/chunk-000/episode_000000.parquet"
    DEFAULT_GROUND_TRUTH_MAX_STEPS=1760
    case "$ARENA_EVAL_PROFILE" in
      recording0|native_recording0) DEFAULT_ENV_CONFIG="${ISAACLAB_ROOT}/tasks/common_test_config/base_test/open_door_sonic_recording0.yaml" ;;
      random) DEFAULT_ENV_CONFIG="${ISAACLAB_ROOT}/tasks/common_test_config/base_test/open_door_sonic_train_range.yaml" ;;
      benchmark_random) DEFAULT_ENV_CONFIG="${ISAACLAB_ROOT}/tasks/common_test_config/base_test/open_door_sonic_benchmark_random.yaml" ;;
      *)
        echo "Unknown ARENA_EVAL_PROFILE: $ARENA_EVAL_PROFILE (expected recording0/native_recording0, random, or benchmark_random)" >&2
        exit 2
        ;;
    esac
    ;;
  *)
    echo "Unknown task: $TASK_KIND (expected pp_box, football, or open_door)" >&2
    exit 2
    ;;
esac

MAX_STEPS="${MAX_STEPS:-$DEFAULT_MAX_STEPS}"

GR00T_MODEL_PATH="${GR00T_MODEL_PATH:-$DEFAULT_MODEL_PATH}"
ENV_CONFIG_YAML="${ENV_CONFIG_YAML:-$DEFAULT_ENV_CONFIG}"
EVAL_RUNNER="${ISAACLAB_ROOT}/script/eval_scripts/sonic/run_vla_eval.sh"
SERVER_SCRIPT="${PSI0_ROOT}/scripts/deploy/gr00t_n17_rot6d59_server.py"
GROUND_TRUTH_PARQUET="${GROUND_TRUTH_PARQUET:-$DEFAULT_GROUND_TRUTH_PARQUET}"
GROUND_TRUTH_MAX_STEPS="${GROUND_TRUTH_MAX_STEPS:-$DEFAULT_GROUND_TRUTH_MAX_STEPS}"
CHECKPOINT_TAG="${GR00T_MODEL_PATH##*/}"
RESULTS_DIR="${RESULTS_DIR:-${PSI0_ROOT}/evals/humanoidarena_${TASK_KIND}_gr00t_rot6d59_kimodo_textop_${CHECKPOINT_TAG}_${ARENA_EVAL_PROFILE}}"
GROUND_TRUTH_RESULTS_DIR="${GROUND_TRUTH_RESULTS_DIR:-${PSI0_ROOT}/evals/humanoidarena_${TASK_KIND}_rot6d59_ground_truth_recording0}"

TEXTOP_ROOT="${TEXTOP_ROOT:-/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh}"
TEXTOP_TRACKER_RUN="${TEXTOP_TRACKER_RUN:-${TEXTOP_ROOT}/textop/2026-06-11_11-39-47_rgz_loco_manip_obj_transf_vae_1step_ddp_4gpu_gear_sonic_ads_naug}"
TEXTOP_POLICY_ONNX="${TEXTOP_POLICY_ONNX:-${TEXTOP_TRACKER_RUN}/latest.onnx}"
TEXTOP_VAE_RUN="${TEXTOP_VAE_RUN:-${TEXTOP_ROOT}/textop/2026-05-30_04-21-30_npz_rgz_filtered_ddp_save}"
TEXTOP_VAE_ONNX="${TEXTOP_VAE_ONNX:-${TEXTOP_VAE_RUN}/artifacts/motion_transformer_vae_encoder_z_c.onnx}"
TEXTOP_VAE_STATS="${TEXTOP_VAE_STATS:-${TEXTOP_VAE_RUN}/artifacts/stats.npz}"

require_file() {
  [[ -f "$1" ]] || { echo "Required file does not exist: $1" >&2; exit 1; }
}

require_dir() {
  [[ -d "$1" ]] || { echo "Required directory does not exist: $1" >&2; exit 1; }
}

append_isaac_kit_arg() {
  local kit_arg="$1"
  if [[ " $ISAAC_KIT_ARGS " != *" $kit_arg "* ]]; then
    ISAAC_KIT_ARGS="${ISAAC_KIT_ARGS:+$ISAAC_KIT_ARGS }$kit_arg"
  fi
}

configure_rtx_driver_check() {
  local disable_check="$ISAAC_DISABLE_RTX_DRIVER_CHECK"

  if [[ "$disable_check" == "auto" ]]; then
    if [[ "$ISAAC_SIM_RUNTIME" == "4.5" && "$INSTALLED_NVIDIA_DRIVER" == "535.309.01" ]]; then
      disable_check="1"
    else
      disable_check="0"
    fi
  fi

  case "$disable_check" in
    1|true|TRUE|yes|YES)
      append_isaac_kit_arg "--/rtx/verifyDriverVersion/enabled=false"
      append_isaac_kit_arg "--/rtx/post/aa/op=0"
      append_isaac_kit_arg "--/rtx-defaults/post/aa/op=0"
      append_isaac_kit_arg "--/rtx-transient/post/aa/limitedOps=false"
      append_isaac_kit_arg "--/rtx-transient/dlssg/enabled=false"
      ;;
    0|false|FALSE|no|NO)
      ;;
    *)
      echo "ISAAC_DISABLE_RTX_DRIVER_CHECK must be auto, 0, or 1; got $ISAAC_DISABLE_RTX_DRIVER_CHECK" >&2
      exit 2
      ;;
  esac
}

validate_isaac_runtime() {
  [[ "$ISAAC_SIM_RUNTIME" == "4.5" ]] || return 0

  if [[ ! -d "$ISAAC45_DEPS_DIR/onnxruntime" || ! -f "$ISAAC45_DEPS_DIR/libcuda.so" ]]; then
    echo "Isaac Sim 4.5 compatibility dependencies are incomplete: $ISAAC45_DEPS_DIR" >&2
    echo "Run the native SONIC compatibility setup script or set ISAAC45_DEPS_DIR to a complete cache." >&2
    exit 1
  fi
  require_dir "$ISAAC45_DEPS_DIR"
  require_dir "$ISAACLAB_EXTERNAL_ROOT/source/isaaclab/isaaclab"
  require_dir "$ISAACLAB_EXTERNAL_ROOT/source/isaaclab_tasks/isaaclab_tasks"
  require_dir "$ISAACLAB_EXTERNAL_ROOT/source/isaaclab_assets/isaaclab_assets"
}

configure_isaac45_runtime() {
  [[ "$ISAAC_SIM_RUNTIME" == "4.5" ]] || return 0
  validate_isaac_runtime

  local isaaclab_pythonpath
  isaaclab_pythonpath="$ISAACLAB_EXTERNAL_ROOT/source/isaaclab:$ISAACLAB_EXTERNAL_ROOT/source/isaaclab_tasks:$ISAACLAB_EXTERNAL_ROOT/source/isaaclab_assets"
  export PYTHONPATH="$ISAAC45_DEPS_DIR:$isaaclab_pythonpath${PYTHONPATH:+:$PYTHONPATH}"
  export LD_LIBRARY_PATH="$ISAAC45_DEPS_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"
}

select_gr00t_python() {
  local import_check='import fastapi, torch, tyro, uvicorn, gr00t'
  local requested_python="$GR00T_PYTHON"

  if [[ -n "$GR00T_PYTHON" ]]; then
    if [[ -x "$GR00T_PYTHON" ]] && \
      PYTHONPATH="${GR00T_SITE_PACKAGES}:${GR00T_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
        "$GR00T_PYTHON" -c "$import_check" >/dev/null 2>&1; then
      export GR00T_PYTHON
      return
    fi
    echo "Requested GR00T_PYTHON is unusable on this node: $GR00T_PYTHON" >&2
    echo "Falling back to automatic per-node Python discovery." >&2
    GR00T_PYTHON=""
  fi

  local candidate
  for candidate in \
    "$GR00T_ROOT/.venv/bin/python" \
    "$KIMODO_PYTHON" \
    "$GR00T_ROOT/.venv.broken-20260822-py312/bin/python" \
    "$GR00T_ROOT"/.venv*/bin/python; do
    if [[ -x "$candidate" ]] && \
      PYTHONPATH="${GR00T_SITE_PACKAGES}:${GR00T_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
        "$candidate" -c "$import_check" >/dev/null 2>&1; then
      GR00T_PYTHON="$candidate"
      export GR00T_PYTHON
      if [[ -n "$requested_python" && "$candidate" != "$requested_python" ]]; then
        echo "Selected fallback GR00T Python: $GR00T_PYTHON" >&2
      elif [[ -z "$requested_python" ]]; then
        echo "Auto-selected GR00T Python on $(hostname): $GR00T_PYTHON" >&2
      fi
      return
    fi
  done

  # uv created this Python 3.10 venv with an interpreter under /root. On a
  # different compute node that interpreter path may not exist even though
  # the shared venv site-packages are complete. Use the system Python 3.10
  # with those packages, scoped to the GR00T server process only.
  local fallback_python="/usr/bin/python3.10"
  if [[ -x "$fallback_python" && -d "$GR00T_SITE_PACKAGES" ]] && \
    PYTHONPATH="${GR00T_SITE_PACKAGES}:${GR00T_ROOT}" \
      "$fallback_python" -c "$import_check" >/dev/null 2>&1; then
    GR00T_PYTHON="$fallback_python"
    export GR00T_PYTHON GR00T_SITE_PACKAGES
    echo "Auto-selected GR00T Python on $(hostname): $GR00T_PYTHON" >&2
    return
  fi
  echo "No Python on this node can import fastapi, torch, tyro, uvicorn, and gr00t." >&2
  echo "Set GR00T_PYTHON to a node-local Python 3.10 environment containing those packages." >&2
  exit 1
}

preflight() {
  select_gr00t_python
  validate_isaac_runtime
  configure_rtx_driver_check
  require_file "$GR00T_PYTHON"
  require_file "$EVAL_PYTHON"
  require_file "$KIMODO_PYTHON"
  require_file "$SERVER_SCRIPT"
  require_file "$EVAL_RUNNER"
  require_file "$ENV_CONFIG_YAML"
  require_file "$GR00T_MODALITY_CONFIG_PATH"
  require_file "$TEXTOP_POLICY_ONNX"
  require_file "$TEXTOP_VAE_ONNX"
  require_file "$TEXTOP_VAE_STATS"
  require_dir "$GR00T_MODEL_PATH"
  require_dir "$GR00T_BACKBONE_PATH"
  require_dir "$KIMODO_ROOT"
  if ! PYTHONPATH="${GR00T_SITE_PACKAGES}:${GR00T_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
    "$GR00T_PYTHON" -c 'import fastapi, torch, tyro, uvicorn, gr00t' >/dev/null 2>&1; then
    echo "GR00T Python cannot import the required server packages: $GR00T_PYTHON" >&2
    exit 1
  fi
  if (( GR00T_EXECUTION_HORIZON < 1 || GR00T_EXECUTION_HORIZON >= 40 )); then
    echo "Prefix-RTC requires GR00T_EXECUTION_HORIZON in [1, 39]" >&2
    exit 2
  fi
}

export_common_eval_env() {
  export YZH_PSI0_ROOT="$PSI0_ROOT"
  export GR00T_TASK_INSTRUCTION="$TASK_INSTRUCTION"
  export GR00T_EXECUTION_HORIZON
  export KIMODO_ROOT KIMODO_PYTHON
  export KIMODO_SERVER_URL="http://127.0.0.1:${KIMODO_PORT}"
  export KIMODO_WORK_DIR="${KIMODO_WORK_DIR:-${RESULTS_DIR}/kimodo}"
  export KIMODO_ANCHOR_MODE="${KIMODO_ANCHOR_MODE:-policy_only}"
  export KIMODO_SOURCE_FPS=50 KIMODO_OUTPUT_FPS=50 KIMODO_FPS=30
  export TEXTOP_ROOT TEXTOP_TRACKER_RUN TEXTOP_POLICY_ONNX
  export TEXTOP_VAE_RUN TEXTOP_VAE_ONNX TEXTOP_VAE_STATS
  export TEXTOP_TASK="Tracking-Flat-G1-ProjGravAnchorEEObsOneStep-TransformerVAE-NMMLP-v0"
  export TEXTOP_FUTURE_STEPS=1 TEXTOP_VAE_WINDOW_STEPS=10 TEXTOP_FPS=50
  export TEXTOP_TARGET_RATE_LIMIT="${TEXTOP_TARGET_RATE_LIMIT:-0}"
  export NO_PROXY="${NO_PROXY:-localhost,127.0.0.1,0.0.0.0,::1}"
  export no_proxy="${no_proxy:-${NO_PROXY}}"
  export OMNI_KIT_ACCEPT_EULA=Y ACCEPT_EULA=Y
  export ISAAC_KIT_ARGS
  export OPEN_DOOR_LEAF_UNLOCK_STIFFNESS OPEN_DOOR_LEAF_UNLOCK_DAMPING
}

print_open_door_unlock_drive() {
  if [[ "$TASK_KIND" == "open_door" ]]; then
    echo "  door_unlock_drive=stiffness=${OPEN_DOOR_LEAF_UNLOCK_STIFFNESS:-usd-default} damping=${OPEN_DOOR_LEAF_UNLOCK_DAMPING:-usd-default} max_force=usd-default"
  fi
}

wait_health() {
  local url="$1"
  local name="$2"
  local timeout="${3:-360}"
  local start=$SECONDS
  until curl --noproxy '*' --fail --silent --max-time 2 "$url" >/dev/null 2>&1; do
    if (( SECONDS - start >= timeout )); then
      echo "Timed out waiting for ${name}: ${url}" >&2
      return 1
    fi
    sleep 1
  done
}

serve() {
  preflight
  cd "$GR00T_ROOT"
  export CUDA_VISIBLE_DEVICES="$SERVE_GPU"
  export PYTHONPATH="${GR00T_SITE_PACKAGES}:${GR00T_ROOT}:${PYTHONPATH:-}"
  export HF_HOME="${HF_HOME:-${PSI0_ROOT}/huggingface}"
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 NO_ALBUMENTATIONS_UPDATE=1
  exec "$GR00T_PYTHON" "$SERVER_SCRIPT" \
    --model-path "$GR00T_MODEL_PATH" \
    --backbone-path "$GR00T_BACKBONE_PATH" \
    --modality-config-path "$GR00T_MODALITY_CONFIG_PATH" \
    --host 0.0.0.0 \
    --port "$GR00T_PORT" \
    --device cuda \
    --action-exec-horizon "$GR00T_EXECUTION_HORIZON" \
    --prefix-rtc \
    --enable-rtc \
    --prefix-rtc-timestep-mode groot_clean
}

kimodo_serve() {
  preflight
  cd "$KIMODO_ROOT"
  export KIMODO_ROOT
  export CUDA_VISIBLE_DEVICES="$KIMODO_GPU"
  export KIMODO_SERVER_HOST=127.0.0.1 KIMODO_SERVER_PORT="$KIMODO_PORT"
  export HF_HOME="${KIMODO_ROOT}/huggingface"
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONNOUSERSITE=1
  export TEXT_ENCODER_MODE="${TEXT_ENCODER_MODE:-local}" LOCAL_CACHE="${LOCAL_CACHE:-true}"
  exec "$KIMODO_PYTHON" "$PSI0_ROOT/scripts/deploy/kimodo_generation_server.py"
}

eval_checkpoint() {
  preflight
  configure_isaac45_runtime
  export_common_eval_env
  print_open_door_unlock_drive
  wait_health "http://127.0.0.1:${GR00T_PORT}/health" "GR00T rot6d59 server"
  wait_health "http://127.0.0.1:${KIMODO_PORT}/health" "Kimodo server"
  unset ROT6D59_GROUND_TRUTH_PARQUET
  cd "$ISAACLAB_ROOT"
  EVAL_PYTHON="$EVAL_PYTHON" \
  SERVER_PYTHON="$GR00T_PYTHON" \
  SERVER_SCRIPT="$SERVER_SCRIPT" \
  MODEL_PATH="$GR00T_MODEL_PATH" \
  ENV_CONFIG_YAML="$ENV_CONFIG_YAML" \
  TASK_NAME="$TASK_NAME" \
  SONIC_VLA_ACTION_FORMAT=rot6d59_textop \
  SONIC_ENCODER_PATH="$TEXTOP_VAE_ONNX" \
  SONIC_DECODER_PATH="$TEXTOP_POLICY_ONNX" \
  EXTERNAL_SERVER=1 \
  SERVER_HOST=127.0.0.1 SERVER_PORT="$GR00T_PORT" SERVER_SCHEME=http \
  ISAAC_DEVICE="cuda:${EVAL_GPU}" \
  EVAL_SEEDS="$EVAL_SEEDS" REPEATS_PER_SEED="$REPEATS_PER_SEED" \
  RESULTS_DIR="$RESULTS_DIR" MAX_STEPS="$MAX_STEPS" VIDEO_FPS="$VIDEO_FPS" \
  PERSISTENT_SIM="${PERSISTENT_SIM:-1}" HEADLESS="${HEADLESS:-1}" \
  RECORD_VIDEO_EVERY_N="${RECORD_VIDEO_EVERY_N:-1}" \
  LEROBOT_VLA_RECORD_OUTPUTS=0 \
  bash "$EVAL_RUNNER"
}

eval_ground_truth() {
  preflight
  configure_isaac45_runtime
  require_file "$GROUND_TRUTH_PARQUET"
  export_common_eval_env
  print_open_door_unlock_drive
  wait_health "http://127.0.0.1:${KIMODO_PORT}/health" "Kimodo server"
  export ROT6D59_GROUND_TRUTH_PARQUET="$GROUND_TRUTH_PARQUET"
  local gt_dir="$GROUND_TRUTH_RESULTS_DIR"
  mkdir -p "$gt_dir/episodes" "$gt_dir/videos/success" "$gt_dir/videos/failure"
  cd "$ISAACLAB_ROOT"
  "$EVAL_PYTHON" \
    script/eval_scripts/sonic/sim_eval_vla.py \
    --task "$TASK_NAME" \
    --env_config_yaml "$ENV_CONFIG_YAML" \
    --seed 0 --repeat_idx 0 --episode_seed 0 --episode_index 0 \
    --max_steps "$GROUND_TRUTH_MAX_STEPS" \
    --sonic_encoder_path "$TEXTOP_VAE_ONNX" \
    --sonic_decoder_path "$TEXTOP_POLICY_ONNX" \
    --sonic_vla_action_format rot6d59_textop \
    --lerobot_server_url "http://127.0.0.1:${GR00T_PORT}" \
    --lerobot_server_timeout 360 \
    --robot_type unitree_g1_refpose_v3_1 \
    --result_json "$gt_dir/episodes/episode_0.json" \
    --success_video_dir "$gt_dir/videos/success" \
    --failure_video_dir "$gt_dir/videos/failure" \
    --video_fps "$VIDEO_FPS" --record_video_every_n 1 \
    --model_label rot6d59_ground_truth --eval_model_path "$GROUND_TRUTH_PARQUET" \
    --device "cuda:${EVAL_GPU}" --enable_cameras --headless \
    --kit_args="$ISAAC_KIT_ARGS"
}

run_all() {
  preflight
  mkdir -p "$RESULTS_DIR/logs"
  bash "$0" serve "$TASK_KIND" >"$RESULTS_DIR/logs/gr00t_server.log" 2>&1 &
  local gr00t_pid=$!
  bash "$0" kimodo-serve "$TASK_KIND" >"$RESULTS_DIR/logs/kimodo_server.log" 2>&1 &
  local kimodo_pid=$!
  trap "kill $gr00t_pid $kimodo_pid 2>/dev/null || true" EXIT
  trap 'exit 130' INT TERM
  wait_health "http://127.0.0.1:${GR00T_PORT}/health" "GR00T rot6d59 server"
  wait_health "http://127.0.0.1:${KIMODO_PORT}/health" "Kimodo server"
  eval_checkpoint
}

run_ground_truth_all() {
  preflight
  local gt_dir="$GROUND_TRUTH_RESULTS_DIR"
  mkdir -p "$gt_dir/logs"
  bash "$0" kimodo-serve "$TASK_KIND" >"$gt_dir/logs/kimodo_server.log" 2>&1 &
  local kimodo_pid=$!
  trap "kill $kimodo_pid 2>/dev/null || true" EXIT
  trap 'exit 130' INT TERM
  wait_health "http://127.0.0.1:${KIMODO_PORT}/health" "Kimodo server"
  eval_ground_truth
}

dry_run() {
  preflight
  echo "HumanoidArena native GR00T rot6d59 + Kimodo + TextOp"
  echo "  task=$TASK_KIND isaac_task=$TASK_NAME profile=$ARENA_EVAL_PROFILE"
  echo "  host=$(hostname)"
  echo "  gr00t_python=$GR00T_PYTHON"
  echo "  isaac_sim_runtime=$ISAAC_SIM_RUNTIME eval_python=$EVAL_PYTHON"
  echo "  installed_driver=${INSTALLED_NVIDIA_DRIVER:-not-probed}"
  echo "  isaac_kit_args=${ISAAC_KIT_ARGS:-<none>}"
  echo "  checkpoint=$GR00T_MODEL_PATH"
  echo "  env_config=$ENV_CONFIG_YAML"
  echo "  horizon=predict40/execute${GR00T_EXECUTION_HORIZON}/overlap$((40 - GR00T_EXECUTION_HORIZON))"
  print_open_door_unlock_drive
  echo "  max_steps=$MAX_STEPS"
  echo "  GPUs: gr00t=$SERVE_GPU isaac_textop=$EVAL_GPU kimodo=$KIMODO_GPU"
  echo "  seeds=$EVAL_SEEDS repeats_per_seed=$REPEATS_PER_SEED"
  echo "  results=$RESULTS_DIR"
  echo "  instruction=$TASK_INSTRUCTION"
  echo "  ground_truth=$GROUND_TRUTH_PARQUET steps=$GROUND_TRUTH_MAX_STEPS"
}

case "$COMMAND" in
  serve) serve ;;
  kimodo-serve) kimodo_serve ;;
  eval) eval_checkpoint ;;
  ground-truth) eval_ground_truth ;;
  all) run_all ;;
  all-ground-truth) run_ground_truth_all ;;
  dry-run) dry_run ;;
  *)
    echo "Usage: bash $0 {serve|kimodo-serve|eval|ground-truth|all|all-ground-truth|dry-run} [pp_box|football|open_door]" >&2
    exit 2
    ;;
esac
