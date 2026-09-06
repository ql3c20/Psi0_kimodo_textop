#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/ubuntu/yzh/Psi0_kimodo_textop
SIMPLE_ROOT="${ROOT}/third_party/SIMPLE"
CONDITION="${1:-}"

if [[ "${CONDITION}" == "both" ]]; then
  "$0" raw-yaw
  "$0" face-can
  exit 0
fi

case "${CONDITION}" in
  raw-yaw)
    FACE_CAN=0
    RUN_TAG=task6_sonic_heading_ab_raw_yaw_can_front_edge_center_left_h40_eps20
    ;;
  face-can)
    FACE_CAN=1
    RUN_TAG=task6_sonic_heading_ab_face_can_can_front_edge_center_left_h40_eps20
    ;;
  *)
    echo "Usage: $0 {raw-yaw|face-can|both}" >&2
    exit 2
    ;;
esac

EVAL_DIR="${SIMPLE_ROOT}/data/evals_task6_scq_isaac_${RUN_TAG}"
if [[ -s "${EVAL_DIR}/results.jsonl" ]]; then
  echo "Refusing to append to an existing A/B result: ${EVAL_DIR}/results.jsonl" >&2
  exit 1
fi

if pgrep -f 'eval_decoupled_wbc.py.*G1Fullstate20260828Task6' >/dev/null; then
  echo "Another Task6 evaluator is running; wait for it to finish before starting the A/B run." >&2
  exit 1
fi

curl --noproxy '*' -fsS --max-time 3 http://127.0.0.1:22095/config >/dev/null || {
  echo "GR00T SONIC is not ready on TCP 22095." >&2
  exit 1
}
ss -H -lun | awk '{print $4}' | grep -Eq '(^|:)23331$' || {
  echo "Isaac Ego is not ready on UDP 23331." >&2
  exit 1
}

cd "${ROOT}"

PSI0_ROOT="${ROOT}" \
SIMPLE_ROOT="${SIMPLE_ROOT}" \
GR00T_ROOT=/home/ubuntu/yzh/Isaac-GR00T-rtc \
GR00T_MODEL_PATH=/home/ubuntu/yzh/ckpt/gr00tn17/task6_sonic_newdata \
GR00T_BACKBONE_PATH=/home/ubuntu/yzh/Isaac-GR00T-rtc/huggingface/Cosmos-Reason2-2B \
SONIC_DECODER_ONNX=/home/ubuntu/yzh/GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/model_decoder.onnx \
GR00T_PREFIX_RTC=0 \
GR00T_EXECUTION_HORIZON=40 \
GR00T_PORT=22095 \
GR00T_SONIC_TASK=G1Fullstate20260828Task6-v0 \
GR00T_SONIC_EVAL_DIR="${EVAL_DIR}" \
HUMANOID_VLA_MJ_ROOT=/home/ubuntu/yzh/HumanoidVLA_MJ_backup/HumanoidVLA_MJ \
TASK6_RECORDINGS_DIR=/home/ubuntu/yzh/mujoco_recordings/copy/20260828_task6_scq/20260828 \
TASK6_RECORDING_INDEX= \
TASK6_RECORDING_SEED=42 \
TASK6_CAN_RANDOMIZE_LEFT=1 \
TASK6_CAN_RANDOM_SEED=42 \
TASK6_CAN_FIXED_X=1.220 \
TASK6_CAN_FIXED_Y=0.020 \
TASK6_CAN_LEFT_JITTER_X=0.0 \
TASK6_CAN_LEFT_JITTER_Y=0.0 \
TASK6_ROBOT_TRANSLATE="0.0 0.0 0.0" \
TASK6_ROBOT_FACE_CAN="${FACE_CAN}" \
SIM_MODE=mujoco_external_isaac \
EXTERNAL_ISAAC_UDP_PORT=23331 \
EXTERNAL_ISAAC_EGO_FRAME_PATH=/dev/shm/simple_task6_isaac_ego.frame \
EXTERNAL_ISAAC_EGO_WIDTH=1280 \
EXTERNAL_ISAAC_EGO_HEIGHT=720 \
EXTERNAL_ISAAC_EGO_TIMEOUT_S=120 \
EXTERNAL_ISAAC_CAMERA_LOCAL_EYE="0.10 0.06 0.50" \
SKIP_STABILIZE=1 \
GR00T_INITIAL_POSE_STEPS=0 \
EVAL_GPU=0 \
TASK6_INSTRUCTION="Pick up the object from the shelf and place it on the shelf below." \
NUM_EPISODES=20 \
EPISODE_START=0 \
MAX_EPISODE_STEPS=700 \
SAVE_VIDEO=1 \
bash scripts/deploy/fullstate_task1_gr00t_n17_sonic_eval.sh eval
