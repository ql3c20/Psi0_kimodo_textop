#!/usr/bin/env bash
set -euo pipefail

# Common four-terminal launcher for Task1--Task4:
#   serve | kimodo-serve | isaac | eval
# MuJoCo in SIMPLE is authoritative; the standalone Isaac process only mirrors
# qpos and publishes synchronized HVEGO01 RGB frames.
# LQB Prefix-RTC contract (when configured as 40/30): GR00T returns 40 frames;
# its next request reuses the previous tail[30:40] as the 10-frame VLA prefix.
# Kimodo consumes and regenerates all 40 frames (no separate qpos-prefix
# override), TextOp receives the full 40-frame full-body + root/EE reference,
# and SIMPLE executes frames 0:30 before the next blocking request.

PSI0_ROOT="${PSI0_ROOT:-/home/ubuntu/yzh/Psi0_kimodo_textop}"
SIMPLE_ROOT="${SIMPLE_ROOT:-${PSI0_ROOT}/third_party/SIMPLE}"
GR00T_ROOT="${GR00T_ROOT:-/home/ubuntu/yzh/Isaac-GR00T-rtc}"
HUMANOID_VLA_MJ_ROOT="${HUMANOID_VLA_MJ_ROOT:-/home/ubuntu/yzh/HumanoidVLA_MJ_backup/HumanoidVLA_MJ}"
BASE_SCRIPT="${PSI0_ROOT}/scripts/deploy/simple_psi0_kimodo_eval_commands.sh"
ISAAC_PYTHON="${ISAAC_PYTHON:-/home/ubuntu/isaacsim/python.sh}"
ISAAC_VIEWER="${ISAAC_VIEWER:-${HUMANOID_VLA_MJ_ROOT}/scripts/deploy/task3_isaac_hssd_viewer.py}"

TASK_NUMBER="${TASK_NUMBER:-1}"
seed_var="TASK${TASK_NUMBER}_RECORDING_SEED"
index_var="TASK${TASK_NUMBER}_RECORDING_INDEX"
FULLSTATE_GR00T_TASK="${FULLSTATE_GR00T_TASK:-G1Fullstate20260615Task1-v0}"
TASK_RECORDINGS_DIR="${TASK_RECORDINGS_DIR:-/home/ubuntu/yzh/mujoco_recordings/20260615_task1_new}"
TASK_HSSD_USD="${TASK_HSSD_USD:-${SIMPLE_ROOT}/data/scenes/hssd/102344280/102344280.usd}"
TASK_EGO_EYE="${TASK_EGO_EYE:-0.06 0.06 0.40}"
TASK_EGO_FORWARD="${TASK_EGO_FORWARD:-0.71735609 0.0 -0.69670671}"
TASK_EGO_UP="${TASK_EGO_UP:-0.69670671 0.0 0.71735609}"
TASK_EGO_PITCH_DEG="${TASK_EGO_PITCH_DEG:-15}"
TASK_EGO_FOVY="${TASK_EGO_FOVY:-70}"
TASK_TRANSLATE="${TASK_TRANSLATE:-0.0 0.0 0.0}"
TASK_YAW_DEG="${TASK_YAW_DEG:-0.0}"
TASK_BACKGROUND_TRANSLATE="${TASK_BACKGROUND_TRANSLATE:-0.0 0.0 0.0}"
TASK_HIDE_BACKGROUND_PRIM="${TASK_HIDE_BACKGROUND_PRIM:-}"
ISAAC_TRASH_TRANSLATE="${ISAAC_TRASH_TRANSLATE:-0.0 0.0 0.0}"
GR00T_MODEL_PATH="${GR00T_MODEL_PATH:-/home/ubuntu/yzh/ckpt/gr00tn17/task${TASK_NUMBER}_newbg}"
GR00T_BACKBONE_PATH="${GR00T_BACKBONE_PATH:-${GR00T_ROOT}/huggingface/Cosmos-Reason2-2B}"
GR00T_MODALITY_CONFIG_PATH="${GR00T_MODALITY_CONFIG_PATH:-${GR00T_ROOT}/examples/unitree_g1_rot6d59_config.py}"
ISAAC_RANDOMIZE_LIGHTING="${ISAAC_RANDOMIZE_LIGHTING:-1}"
ISAAC_LIGHTING_SEED="${ISAAC_LIGHTING_SEED:-42}"
ISAAC_LIGHT_RIG="${ISAAC_LIGHT_RIG:-scripted}"
ISAAC_LIVE_EGO_WIDTH="${ISAAC_LIVE_EGO_WIDTH:-640}"
ISAAC_LIVE_EGO_HEIGHT="${ISAAC_LIVE_EGO_HEIGHT:-480}"
ISAAC_LIVE_EGO_JPEG_QUALITY="${ISAAC_LIVE_EGO_JPEG_QUALITY:-95}"
ISAAC_MAIN_WIDTH="${ISAAC_MAIN_WIDTH:-2560}"
ISAAC_MAIN_HEIGHT="${ISAAC_MAIN_HEIGHT:-1440}"

GR00T_PORT="${GR00T_PORT:-22085}"
KIMODO_SERVER_PORT="${KIMODO_SERVER_PORT:-22185}"
ISAAC_UDP_PORT="${ISAAC_UDP_PORT:-23331}"
ISAAC_EGO_FRAME_PATH="${ISAAC_EGO_FRAME_PATH:-/dev/shm/simple_task${TASK_NUMBER}_isaac_ego.frame}"
NUM_EPISODES="${NUM_EPISODES:-10}"
EPISODE_START="${EPISODE_START:-0}"
SAVE_VIDEO="${SAVE_VIDEO:-1}"
TASK_RECORDING_SEED="${TASK_RECORDING_SEED:-${!seed_var:-0}}"
TASK_RECORDING_INDEX="${TASK_RECORDING_INDEX:-${!index_var:-}}"
FULLSTATE_GR00T_EVAL_DIR="${FULLSTATE_GR00T_EVAL_DIR:-${SIMPLE_ROOT}/data/evals_task${TASK_NUMBER}_external_isaac}"
FULLSTATE_GR00T_KIMODO_WORK_DIR="${FULLSTATE_GR00T_KIMODO_WORK_DIR:-${PSI0_ROOT}/outputs/kimodo_task${TASK_NUMBER}_external_isaac}"
PROCESS_CPUSET="${PROCESS_CPUSET:-16-31}"

KIMODO_ROOT="${KIMODO_ROOT:-/home/ubuntu/yzh/kimodo_my}"
KIMODO_PYTHON="${KIMODO_PYTHON:-/home/ubuntu/miniconda3/envs/kimodo/bin/python}"
KIMODO_DISTILL_CONFIG="${KIMODO_DISTILL_CONFIG:-/home/ubuntu/yzh/ckpt/kimodo/scq_final/resolved_config.yaml}"
KIMODO_DISTILL_CKPT="${KIMODO_DISTILL_CKPT:-/home/ubuntu/yzh/ckpt/kimodo/scq_final/ema_final.pt}"
KIMODO_TRT_ENGINE_PATH="${KIMODO_TRT_ENGINE_PATH:-/home/ubuntu/yzh/ckpt/kimodo/scq_final/trt_engines/kimodo_T24.trt}"
KIMODO_TRT_METADATA_PATH="${KIMODO_TRT_METADATA_PATH:-/home/ubuntu/yzh/ckpt/kimodo/scq_final/trt_engines/kimodo_T24.json}"

require_file() {
  [[ -f "$1" ]] || { echo "Missing required file: $1" >&2; exit 1; }
}

require_dir() {
  [[ -d "$1" ]] || { echo "Missing required directory: $1" >&2; exit 1; }
}

preflight_recordings() {
  require_dir "$TASK_RECORDINGS_DIR"
  require_file "${SIMPLE_ROOT}/scripts/validate_fullstate_recordings.py"
  require_file "${SIMPLE_ROOT}/.venv/bin/python"
  (
    cd "$SIMPLE_ROOT"
    PYTHONPATH=src taskset -c "${PREFLIGHT_CPUSET:-$PROCESS_CPUSET}" \
      .venv/bin/python scripts/validate_fullstate_recordings.py \
      --task "$TASK_NUMBER" \
      --recordings-dir "$TASK_RECORDINGS_DIR"
  )
}

export PSI0_ROOT SIMPLE_ROOT GR00T_ROOT HUMANOID_VLA_MJ_ROOT
export GR00T_MODEL_PATH GR00T_BACKBONE_PATH GR00T_MODALITY_CONFIG_PATH GR00T_PORT
export KIMODO_ROOT KIMODO_PYTHON KIMODO_DISTILL_CONFIG KIMODO_DISTILL_CKPT
export KIMODO_SERVER_PORT
export KIMODO_USE_TRT="${KIMODO_USE_TRT:-1}"
export KIMODO_ANCHOR_MODE="${KIMODO_ANCHOR_MODE:-policy_only}"
export KIMODO_KEYFRAME_STEP="${KIMODO_KEYFRAME_STEP:-10}"
if [[ "$KIMODO_USE_TRT" == "1" ]]; then
  export KIMODO_TRT_ENGINE_PATH KIMODO_TRT_METADATA_PATH
else
  unset KIMODO_TRT_ENGINE_PATH KIMODO_TRT_METADATA_PATH
fi
export KIMODO_SERVER_HOST="${KIMODO_SERVER_HOST:-127.0.0.1}"
export PORT="$GR00T_PORT"

RECORDING_PREFIX="TASK${TASK_NUMBER}"
export "${RECORDING_PREFIX}_RECORDINGS_DIR=${TASK_RECORDINGS_DIR}"
export "${RECORDING_PREFIX}_RECORDING_SEED=${TASK_RECORDING_SEED}"
if [[ -n "${TASK_RECORDING_INDEX:-}" ]]; then
  export "${RECORDING_PREFIX}_RECORDING_INDEX=${TASK_RECORDING_INDEX}"
else
  unset "${RECORDING_PREFIX}_RECORDING_INDEX" 2>/dev/null || true
fi

serve_gr00t() {
  require_dir "$GR00T_MODEL_PATH"
  require_dir "$GR00T_BACKBONE_PATH"
  require_file "$GR00T_MODALITY_CONFIG_PATH"
  cd "$GR00T_ROOT"
  export CUDA_VISIBLE_DEVICES="${SERVE_GPU:-1}"
  export PYTHONPATH="${GR00T_ROOT}:${PYTHONPATH:-}"
  export HF_HOME="${HF_HOME:-${GR00T_ROOT}/huggingface}"
  export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
  export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
  export NO_ALBUMENTATIONS_UPDATE=1
  args=(
    --model-path "$GR00T_MODEL_PATH"
    --backbone-path "$GR00T_BACKBONE_PATH"
    --modality-config-path "$GR00T_MODALITY_CONFIG_PATH"
    --host 0.0.0.0
    --port "$GR00T_PORT"
    --device cuda
    --action-exec-horizon "${GR00T_EXECUTION_HORIZON:-34}"
    --expected-action-horizon "${GR00T_ACTION_HORIZON:-40}"
    --rtc-frozen-steps "${GR00T_RTC_FROZEN_STEPS:-2}"
    --rtc-ramp-rate "${GR00T_RTC_RAMP_RATE:-2.0}"
  )
  if [[ "${GR00T_PREFIX_RTC:-1}" == "1" ]]; then
    args+=(--prefix-rtc)
    [[ -z "${GR00T_PREFIX_RTC_TIMESTEP_MODE:-}" ]] || args+=(--prefix-rtc-timestep-mode "$GR00T_PREFIX_RTC_TIMESTEP_MODE")
  fi
  if [[ "${GR00T_USE_RTC:-1}" == "1" || "${GR00T_PREFIX_RTC:-1}" == "1" ]]; then
    args+=(--enable-rtc)
  else
    args+=(--no-enable-rtc)
  fi
  exec taskset -c "${SERVE_CPUSET:-$PROCESS_CPUSET}" .venv/bin/python \
    "${PSI0_ROOT}/scripts/deploy/gr00t_n17_rot6d59_server.py" "${args[@]}"
}

serve_kimodo() {
  require_file "$KIMODO_DISTILL_CONFIG"
  require_file "$KIMODO_DISTILL_CKPT"
  if [[ "$KIMODO_USE_TRT" == "1" ]]; then
    require_file "$KIMODO_TRT_ENGINE_PATH"
    require_file "$KIMODO_TRT_METADATA_PATH"
  fi
  KIMODO_GPU="${KIMODO_GPU:-2}" exec taskset -c "${KIMODO_CPUSET:-$PROCESS_CPUSET}" \
    bash "$BASE_SCRIPT" kimodo-serve
}

serve_isaac() {
  require_file "$ISAAC_PYTHON"
  require_file "$ISAAC_VIEWER"
  require_file "$TASK_HSSD_USD"
  shopt -s nullglob
  scenes=("${TASK_RECORDINGS_DIR}"/*/model_snapshot/mujoco/model/g1/scene_43dof.xml)
  shopt -u nullglob
  [[ ${#scenes[@]} -gt 0 ]] || { echo "No scene snapshots below $TASK_RECORDINGS_DIR" >&2; exit 1; }
  scene="${scenes[0]}"
  read -r eye_x eye_y eye_z <<<"$TASK_EGO_EYE"
  read -r forward_x forward_y forward_z <<<"$TASK_EGO_FORWARD"
  read -r up_x up_y up_z <<<"$TASK_EGO_UP"
  read -r task_x task_y task_z <<<"$TASK_TRANSLATE"
  read -r background_x background_y background_z <<<"$TASK_BACKGROUND_TRANSLATE"
  read -r trash_x trash_y trash_z <<<"$ISAAC_TRASH_TRANSLATE"
  args=(
    --mjcf "$scene"
    --hssd-usd "$TASK_HSSD_USD"
    --background-mode hssd
    --background-translate "$background_x" "$background_y" "$background_z"
    --output-dir "${ISAAC_OUTPUT_DIR:-${HUMANOID_VLA_MJ_ROOT}/output/simple_task${TASK_NUMBER}_isaac_eval}"
    --udp-host 127.0.0.1
    --udp-port "$ISAAC_UDP_PORT"
    --live-ego-frame-path "$ISAAC_EGO_FRAME_PATH"
    --live-ego-rate-hz 50
    --live-ego-throttle-render
    --live-ego-strict-request-sync
    --live-ego-width "$ISAAC_LIVE_EGO_WIDTH"
    --live-ego-height "$ISAAC_LIVE_EGO_HEIGHT"
    --live-ego-jpeg-quality "$ISAAC_LIVE_EGO_JPEG_QUALITY"
    --camera-eye -2.5 -1.65 1.8
    --camera-target 0.55 -0.05 0.72
    --camera-fovy 70
    --width "$ISAAC_MAIN_WIDTH"
    --height "$ISAAC_MAIN_HEIGHT"
    --head-camera-eye "$eye_x" "$eye_y" "$eye_z"
    --head-camera-near-clip "${TASK_EGO_NEAR_CLIP:-0.2}"
    --head-camera-forward "$forward_x" "$forward_y" "$forward_z"
    --head-camera-up "$up_x" "$up_y" "$up_z"
    --ego-camera-fovy "$TASK_EGO_FOVY"
    --task-translate "$task_x" "$task_y" "$task_z"
    --task-yaw-deg "$TASK_YAW_DEG"
    --trash-translate "$trash_x" "$trash_y" "$trash_z"
    --robot-z-offset "${ISAAC_ROBOT_Z_OFFSET:-0.012}"
    --trash-z-offset "${ISAAC_TRASH_Z_OFFSET:-0.006}"
    --free-object-z-offset "${ISAAC_FREE_OBJECT_Z_OFFSET:-0.0}"
    --light-rig "$ISAAC_LIGHT_RIG"
  )
  if [[ -n "$TASK_HIDE_BACKGROUND_PRIM" ]]; then
    args+=(--hide-background-prim "$TASK_HIDE_BACKGROUND_PRIM")
  fi
  if [[ "${ISAAC_TRANSFORM_TASK_ROOT:-0}" == "1" ]]; then
    args+=(--transform-task-root)
  fi
  if [[ "${ISAAC_REMOVE_TASK6_TOP_SHELF_VISUAL:-0}" == "1" ]]; then
    args+=(--remove-task6-top-shelf-visual)
  fi
  if [[ "${ISAAC_ROBOT_COLORS_SRGB_TO_LINEAR:-0}" == "1" ]]; then
    args+=(--robot-colors-srgb-to-linear)
  fi
  if [[ "${ISAAC_SMOOTH_TASK_TABLE_CYLINDER:-0}" == "1" ]]; then
    args+=(--smooth-task-table-cylinder)
  fi
  if [[ "${ISAAC_HEADLESS:-0}" == "1" ]]; then
    args+=(--headless --camera-mode head --head-camera-pitch-deg "$TASK_EGO_PITCH_DEG")
  else
    args+=(
      --camera-mode world
      --no-head-camera
      --ego-inset
      --ego-width "$ISAAC_LIVE_EGO_WIDTH"
      --ego-height "$ISAAC_LIVE_EGO_HEIGHT"
      --ego-window-width "${ISAAC_EGO_WINDOW_WIDTH:-480}"
      --ego-window-height "${ISAAC_EGO_WINDOW_HEIGHT:-270}"
      --ego-camera-pitch-deg "$TASK_EGO_PITCH_DEG"
    )
  fi
  if [[ "${ISAAC_RANDOMIZE_LIGHTING:-0}" == "1" ]]; then
    if [[ -z "${ISAAC_TRAINING_LIGHTING_MANIFEST:-}" ]]; then
      args+=(--randomize-lighting --lighting-seed "${ISAAC_LIGHTING_SEED:-$TASK_RECORDING_SEED}")
    fi
  fi
  if [[ -n "${ISAAC_TRAINING_LIGHTING_MANIFEST:-}" ]]; then
    require_file "$ISAAC_TRAINING_LIGHTING_MANIFEST"
    args+=(--live-lighting-manifest "$ISAAC_TRAINING_LIGHTING_MANIFEST")
  fi
  cd "$HUMANOID_VLA_MJ_ROOT"
  exec taskset -c "${ISAAC_CPUSET:-$PROCESS_CPUSET}" env \
    -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u CONDA_PROMPT_MODIFIER \
    -u PYTHONPATH -u LD_LIBRARY_PATH \
    VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json \
    "$ISAAC_PYTHON" "$ISAAC_VIEWER" "${args[@]}"
}

verify_kimodo_backend() {
  config_json="$(curl -fsS --max-time 3 "http://${KIMODO_SERVER_HOST}:${KIMODO_SERVER_PORT}/config")" || {
    echo "Kimodo is not reachable on ${KIMODO_SERVER_HOST}:${KIMODO_SERVER_PORT}" >&2
    exit 1
  }
  CONFIG_JSON="$config_json" \
  EXPECTED_BACKEND="$([[ "$KIMODO_USE_TRT" == "1" ]] && echo tensorrt || echo pytorch)" \
  EXPECTED_ANCHOR_MODE="$KIMODO_ANCHOR_MODE" \
  EXPECTED_KEYFRAME_STEP="$KIMODO_KEYFRAME_STEP" \
  python - <<'PY'
import json, os
c = json.loads(os.environ["CONFIG_JSON"])
assert c.get("inference_backend") == os.environ["EXPECTED_BACKEND"], c
assert c.get("anchor_mode") == os.environ["EXPECTED_ANCHOR_MODE"], c
assert int(c.get("keyframe_step", -1)) == int(os.environ["EXPECTED_KEYFRAME_STEP"]), c
print("[preflight] Kimodo config:", json.dumps(c, sort_keys=True))
PY
  if command -v fuser >/dev/null 2>&1; then
    pid="$(fuser -n tcp "$KIMODO_SERVER_PORT" 2>/dev/null | awk '{print $1}' | head -n1 || true)"
    if [[ -n "$pid" && -r "/proc/${pid}/environ" ]]; then
      env_dump="$(tr '\0' '\n' <"/proc/${pid}/environ")"
      grep -qx "KIMODO_USE_TRT=${KIMODO_USE_TRT}" <<<"$env_dump"
      if [[ "$KIMODO_USE_TRT" == "1" ]]; then
        grep -qx "KIMODO_TRT_ENGINE_PATH=${KIMODO_TRT_ENGINE_PATH}" <<<"$env_dump"
        echo "[preflight] Kimodo PID $pid has TensorRT enabled."
      else
        echo "[preflight] Kimodo PID $pid has the PyTorch backend enabled."
      fi
    fi
  fi
}

run_eval() {
  require_dir "$GR00T_MODEL_PATH"
  require_dir "$TASK_RECORDINGS_DIR"
  verify_kimodo_backend
  if [[ "${RESUME:-0}" == "1" ]]; then
    results_path="${FULLSTATE_GR00T_EVAL_DIR}/results.jsonl"
    if [[ -f "$results_path" ]]; then
      completed="$(RESULTS_PATH="$results_path" python - <<'PY'
import json, os
episodes = []
for line in open(os.environ["RESULTS_PATH"], encoding="utf-8"):
    if line.strip():
        episodes.append(int(json.loads(line)["episode"]))
print(max(episodes) + 1 if episodes else 0)
PY
)"
      planned="$NUM_EPISODES"
      EPISODE_START="$completed"
      NUM_EPISODES=$((planned - completed))
      if (( NUM_EPISODES <= 0 )); then
        echo "All ${planned} planned episodes are already present in $results_path"
        return 0
      fi
    fi
  fi
  export EXTERNAL_ISAAC_UDP_HOST=127.0.0.1
  export EXTERNAL_ISAAC_UDP_PORT="$ISAAC_UDP_PORT"
  export EXTERNAL_ISAAC_EGO_FRAME_PATH="$ISAAC_EGO_FRAME_PATH"
  export EXTERNAL_ISAAC_EGO_WIDTH="$ISAAC_LIVE_EGO_WIDTH"
  export EXTERNAL_ISAAC_EGO_HEIGHT="$ISAAC_LIVE_EGO_HEIGHT"
  export EXTERNAL_ISAAC_EGO_TIMEOUT_S="${EXTERNAL_ISAAC_EGO_TIMEOUT_S:-30}"
  export EXTERNAL_ISAAC_CAMERA_LOCAL_EYE="$TASK_EGO_EYE"
  export GR00T_MODEL_PATH
  TASK="$FULLSTATE_GR00T_TASK" \
  SIM_MODE=mujoco_external_isaac \
  DATA_FORMAT=fixed DATA_DIR=unused \
  NUM_EPISODES="$NUM_EPISODES" EPISODE_START="$EPISODE_START" SAVE_VIDEO="$SAVE_VIDEO" \
  EVAL_GPU="${EVAL_GPU:-3}" \
  EVAL_CPUSET="${EVAL_CPUSET:-$PROCESS_CPUSET}" \
  TEXTOP_POLICY_ROOT_EE="${TEXTOP_POLICY_ROOT_EE:-1}" \
  VLA_PROPRIO_SOURCE="${VLA_PROPRIO_SOURCE:-actual}" \
  POLICY_EXECUTION_HORIZON="${POLICY_EXECUTION_HORIZON:-${GR00T_EXECUTION_HORIZON:-34}}" \
  POLICY_ACTION_HORIZON="${POLICY_ACTION_HORIZON:-${GR00T_ACTION_HORIZON:-40}}" \
  KIMODO_RTC_PREFIX_FRAMES="${KIMODO_RTC_PREFIX_FRAMES:-0}" \
  SKIP_STABILIZE="${SKIP_STABILIZE:-1}" \
  MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-900}" \
  KIMODO_WORK_DIR="$FULLSTATE_GR00T_KIMODO_WORK_DIR" \
  ROT6D59_ONESTEP_TEXTOP_EVAL_DIR="$FULLSTATE_GR00T_EVAL_DIR" \
  ROT6D59_ONESTEP_KIMODO_WORK_DIR="$FULLSTATE_GR00T_KIMODO_WORK_DIR" \
  bash "$BASE_SCRIPT" eval-rot6d59-textop-onestep
}

case "${1:-}" in
  serve) preflight_recordings; serve_gr00t ;;
  kimodo-serve) preflight_recordings; serve_kimodo ;;
  isaac) preflight_recordings; serve_isaac ;;
  eval) preflight_recordings; run_eval ;;
  *)
    echo "Usage: $0 {serve|kimodo-serve|isaac|eval}" >&2
    echo "  terminal 1: SERVE_GPU=1 bash $0 serve" >&2
    echo "  terminal 2: KIMODO_GPU=2 bash $0 kimodo-serve" >&2
    echo "  terminal 3: ISAAC_HEADLESS=0 bash $0 isaac" >&2
    echo "  terminal 4: EVAL_GPU=3 NUM_EPISODES=10 bash $0 eval" >&2
    echo "  tracker demo: SIMPLE_MUJOCO_GUI=1 SIMPLE_TRACKER_GHOST=1 NUM_EPISODES=1 bash $0 eval" >&2
    exit 2
    ;;
esac
