#!/usr/bin/env bash
set -euo pipefail

# Three-terminal Task2 native baseline:
#   GR00T N1.7 (UNITREE_G1_SONIC) -> SONIC decoder -> SIMPLE/MuJoCo
#   SIMPLE/MuJoCo -> external Isaac Ego -> GR00T observation

PSI0_ROOT="${PSI0_ROOT:-/home/ubuntu/yzh/Psi0_kimodo_textop}"
SIMPLE_ROOT="${SIMPLE_ROOT:-${PSI0_ROOT}/third_party/SIMPLE}"
GR00T_ROOT="${GR00T_ROOT:-/home/ubuntu/yzh/Isaac-GR00T-rtc}"
HUMANOID_VLA_MJ_ROOT="${HUMANOID_VLA_MJ_ROOT:-/home/ubuntu/yzh/HumanoidVLA_MJ_backup/HumanoidVLA_MJ}"
NATIVE_BASE_SCRIPT="${PSI0_ROOT}/scripts/deploy/fullstate_task1_gr00t_n17_sonic_eval.sh"
TASK2_ISAAC_SCRIPT="${PSI0_ROOT}/scripts/deploy/fullstate_task2_gr00t_rot6d59_kimodo_textop_eval.sh"

GR00T_MODEL_PATH="${GR00T_MODEL_PATH:-/home/ubuntu/yzh/ckpt/gr00tn17/task2_sonic}"
GR00T_BACKBONE_PATH="${GR00T_BACKBONE_PATH:-${GR00T_ROOT}/huggingface/Cosmos-Reason2-2B}"
SONIC_DECODER_ONNX="${SONIC_DECODER_ONNX:-/home/ubuntu/yzh/GR00T-WholeBodyControl/gear_sonic_deploy/policy/low_latency/model_decoder.onnx}"
TASK2_RECORDINGS_DIR="${TASK2_RECORDINGS_DIR:-/home/ubuntu/yzh/mujoco_recordings/20260805_task2_new}"

GR00T_PORT="${GR00T_PORT:-22095}"
ISAAC_UDP_PORT="${ISAAC_UDP_PORT:-23331}"
ISAAC_EGO_FRAME_PATH="${ISAAC_EGO_FRAME_PATH:-/dev/shm/simple_task2_isaac_ego.frame}"
GR00T_EXECUTION_HORIZON="${GR00T_EXECUTION_HORIZON:-34}"
NUM_EPISODES="${NUM_EPISODES:-10}"
EPISODE_START="${EPISODE_START:-0}"
SAVE_VIDEO="${SAVE_VIDEO:-1}"
PROCESS_CPUSET="${PROCESS_CPUSET:-16-31}"
GR00T_SONIC_EVAL_DIR="${GR00T_SONIC_EVAL_DIR:-${SIMPLE_ROOT}/data/evals_task2_sonic_external_isaac}"

require_file() {
  [[ -f "$1" ]] || { echo "Missing required file: $1" >&2; exit 1; }
}

require_dir() {
  [[ -d "$1" ]] || { echo "Missing required directory: $1" >&2; exit 1; }
}

preflight_recordings() {
  require_dir "$TASK2_RECORDINGS_DIR"
  require_file "${SIMPLE_ROOT}/scripts/validate_fullstate_recordings.py"
  require_file "${SIMPLE_ROOT}/.venv/bin/python"
  (
    cd "$SIMPLE_ROOT"
    PYTHONPATH=src taskset -c "${PREFLIGHT_CPUSET:-$PROCESS_CPUSET}" \
      .venv/bin/python scripts/validate_fullstate_recordings.py \
      --task 2 \
      --recordings-dir "$TASK2_RECORDINGS_DIR"
  )
}

preflight_checkpoint() {
  require_dir "$GR00T_MODEL_PATH"
  require_dir "$GR00T_BACKBONE_PATH"
  require_file "${GR00T_MODEL_PATH}/model-00001-of-00002.safetensors"
  require_file "${GR00T_MODEL_PATH}/model-00002-of-00002.safetensors"
  require_file "${GR00T_MODEL_PATH}/processor/embodiment_id.json"
  require_file "${GR00T_MODEL_PATH}/processor/processor_config.json"
  GR00T_MODEL_PATH="$GR00T_MODEL_PATH" "${SIMPLE_ROOT}/.venv/bin/python" - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["GR00T_MODEL_PATH"])
embodiments = json.loads((root / "processor/embodiment_id.json").read_text())
if "unitree_g1_sonic" not in embodiments:
    raise SystemExit("checkpoint does not contain unitree_g1_sonic embodiment")
processor = json.loads((root / "processor/processor_config.json").read_text())
kwargs = processor["processor_kwargs"]
modalities = kwargs["modality_configs"]["unitree_g1_sonic"]
expected_state = [
    "left_leg", "right_leg", "waist", "left_arm", "right_arm",
    "left_hand", "right_hand", "projected_gravity",
]
expected_action = ["motion_token", "left_hand_joints", "right_hand_joints"]
assert modalities["video"]["modality_keys"] == ["ego_view"], modalities["video"]
assert modalities["state"]["modality_keys"] == expected_state, modalities["state"]
assert modalities["action"]["modality_keys"] == expected_action, modalities["action"]
assert modalities["action"]["delta_indices"] == list(range(40)), modalities["action"]
assert int(kwargs["max_action_horizon"]) == 40, kwargs["max_action_horizon"]
print(
    "[preflight] checkpoint embodiment=unitree_g1_sonic "
    "state=46 action=78 horizon=40"
)
PY
}

configure_ort_cuda_runtime() {
  # uv can install the ORT CUDA provider below .venv/lib64 while Python imports
  # onnxruntime from .venv/lib.  The provider and pip NVIDIA runtime libraries
  # must be visible before the Python process starts.
  local ort_cuda_provider
  local ort_cuda_dir
  local ort_nvidia_libs=""
  ort_cuda_provider="$(
    find "${SIMPLE_ROOT}/.venv" -type f \
      -path '*/site-packages/onnxruntime/capi/libonnxruntime_providers_cuda.so' \
      -print -quit
  )"
  if [[ -z "$ort_cuda_provider" ]]; then
    echo "Missing libonnxruntime_providers_cuda.so below ${SIMPLE_ROOT}/.venv" >&2
    exit 1
  fi
  ort_cuda_dir="$(dirname "$ort_cuda_provider")"
  while IFS= read -r nvidia_lib; do
    ort_nvidia_libs="${ort_nvidia_libs}:${nvidia_lib}"
  done < <(
    find "${SIMPLE_ROOT}/.venv" -type d \
      -path '*/site-packages/nvidia/*/lib' -print
  )
  export LD_LIBRARY_PATH="${ort_cuda_dir}${ort_nvidia_libs}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
}

preflight_decoder() {
  require_file "$SONIC_DECODER_ONNX"
  SONIC_DECODER_ONNX="$SONIC_DECODER_ONNX" "${SIMPLE_ROOT}/.venv/bin/python" - <<'PY'
import os
import onnxruntime as ort

path = os.environ["SONIC_DECODER_ONNX"]
available = ort.get_available_providers()
if "CUDAExecutionProvider" not in available:
    raise SystemExit(
        "SONIC decoder requires CUDAExecutionProvider; available=" + repr(available)
    )
session = ort.InferenceSession(
    path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)
if session.get_providers()[0] != "CUDAExecutionProvider":
    raise SystemExit("SONIC decoder did not activate CUDAExecutionProvider")
input_shape = session.get_inputs()[0].shape
output_shape = session.get_outputs()[0].shape
assert input_shape == [1, 994], input_shape
assert output_shape == [1, 29], output_shape
print(
    "[preflight] SONIC decoder input=994 output=29 "
    f"providers={session.get_providers()}"
)
PY
}

verify_gr00t_server() {
  local config_json
  config_json="$(curl -fsS --max-time 3 "http://127.0.0.1:${GR00T_PORT}/config")" || {
    echo "GR00T SONIC server is not reachable on 127.0.0.1:${GR00T_PORT}" >&2
    exit 1
  }
  CONFIG_JSON="$config_json" \
  GR00T_MODEL_PATH="$GR00T_MODEL_PATH" \
  GR00T_EXECUTION_HORIZON="$GR00T_EXECUTION_HORIZON" \
    "${SIMPLE_ROOT}/.venv/bin/python" - <<'PY'
import json
import os
from pathlib import Path

config = json.loads(os.environ["CONFIG_JSON"])
expected_model = str(Path(os.environ["GR00T_MODEL_PATH"]).resolve())
expected_execution = int(os.environ["GR00T_EXECUTION_HORIZON"])
assert config["model_path"] == expected_model, config
assert config["prediction_horizon"] == 40, config
assert config["execution_horizon"] == expected_execution == 34, config
assert config["overlap_steps"] == 6, config
assert config["prefix_rtc"] is True, config
print("[preflight] GR00T SONIC config:", json.dumps(config, sort_keys=True))
PY
}

configure_runtime() {
  export PSI0_ROOT SIMPLE_ROOT GR00T_ROOT HUMANOID_VLA_MJ_ROOT
  export GR00T_MODEL_PATH GR00T_BACKBONE_PATH SONIC_DECODER_ONNX
  export GR00T_PORT GR00T_EXECUTION_HORIZON GR00T_SONIC_EVAL_DIR
  export GR00T_PREFIX_RTC="${GR00T_PREFIX_RTC:-1}"
  export GR00T_PREFIX_RTC_TIMESTEP_MODE="${GR00T_PREFIX_RTC_TIMESTEP_MODE:-groot_clean}"
  export GR00T_SONIC_TASK="G1Fullstate20260805Task2-v0"
  export MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-900}"
  export SKIP_STABILIZE="${SKIP_STABILIZE:-1}"
  export GR00T_INITIAL_POSE_STEPS="${GR00T_INITIAL_POSE_STEPS:-0}"
  export SIM_MODE=mujoco_external_isaac
  export TASK2_RECORDINGS_DIR
  export TASK2_RECORDING_SEED="${TASK2_RECORDING_SEED:-0}"
  if [[ -n "${TASK2_RECORDING_INDEX:-}" ]]; then
    export TASK2_RECORDING_INDEX
  else
    unset TASK2_RECORDING_INDEX 2>/dev/null || true
  fi
  export EXTERNAL_ISAAC_UDP_HOST=127.0.0.1
  export EXTERNAL_ISAAC_UDP_PORT="$ISAAC_UDP_PORT"
  export EXTERNAL_ISAAC_EGO_FRAME_PATH="$ISAAC_EGO_FRAME_PATH"
  export EXTERNAL_ISAAC_EGO_WIDTH=640
  export EXTERNAL_ISAAC_EGO_HEIGHT=480
  export EXTERNAL_ISAAC_CAMERA_LOCAL_EYE="0.06 0.06 0.45"
}

apply_resume() {
  [[ "${RESUME:-0}" == "1" ]] || return 0
  local results_path="${GR00T_SONIC_EVAL_DIR}/results.jsonl"
  [[ -f "$results_path" ]] || return 0
  local resume_values
  resume_values="$(
    RESULTS_PATH="$results_path" \
    EPISODE_START="$EPISODE_START" \
    NUM_EPISODES="$NUM_EPISODES" \
      "${SIMPLE_ROOT}/.venv/bin/python" - <<'PY'
import json
import os

start = int(os.environ["EPISODE_START"])
count = int(os.environ["NUM_EPISODES"])
end = start + count
episodes = []
with open(os.environ["RESULTS_PATH"], encoding="utf-8") as stream:
    for line in stream:
        if line.strip():
            episode = int(json.loads(line)["episode"])
            if start <= episode < end:
                episodes.append(episode)
next_episode = max(episodes) + 1 if episodes else start
print(next_episode, max(0, end - next_episode))
PY
  )"
  read -r EPISODE_START NUM_EPISODES <<<"$resume_values"
  if (( NUM_EPISODES == 0 )); then
    echo "All requested episodes are already present in $results_path"
    exit 0
  fi
  echo "[resume] episode_start=${EPISODE_START} remaining=${NUM_EPISODES}"
}

run_serve() {
  preflight_recordings
  preflight_checkpoint
  configure_runtime
  SERVE_GPU="${SERVE_GPU:-0}" exec bash "$NATIVE_BASE_SCRIPT" serve
}

run_isaac() {
  preflight_recordings
  configure_runtime
  ISAAC_UDP_PORT="$ISAAC_UDP_PORT" \
  ISAAC_EGO_FRAME_PATH="$ISAAC_EGO_FRAME_PATH" \
    exec bash "$TASK2_ISAAC_SCRIPT" isaac
}

run_eval() {
  preflight_recordings
  preflight_checkpoint
  configure_runtime
  configure_ort_cuda_runtime
  preflight_decoder
  verify_gr00t_server
  apply_resume
  NUM_EPISODES="$NUM_EPISODES" \
  EPISODE_START="$EPISODE_START" \
  SAVE_VIDEO="$SAVE_VIDEO" \
  EVAL_GPU="${EVAL_GPU:-0}" \
    exec bash "$NATIVE_BASE_SCRIPT" eval
}

case "${1:-}" in
  serve) run_serve ;;
  isaac) run_isaac ;;
  eval) run_eval ;;
  *)
    echo "Usage: $0 {serve|isaac|eval}" >&2
    echo "  terminal 1: SERVE_GPU=0 bash $0 serve" >&2
    echo "  terminal 2: ISAAC_HEADLESS=1 bash $0 isaac" >&2
    echo "  terminal 3: EVAL_GPU=0 NUM_EPISODES=10 bash $0 eval" >&2
    exit 2
    ;;
esac
