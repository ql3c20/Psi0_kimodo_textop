# ----------------------- Terminal 1: GR00T SONIC -----------------------

```bash
cd /home/ubuntu/yzh/Psi0_kimodo_textop

PSI0_ROOT=/home/ubuntu/yzh/Psi0_kimodo_textop \
SIMPLE_ROOT=/home/ubuntu/yzh/Psi0_kimodo_textop/third_party/SIMPLE \
GR00T_ROOT=/home/ubuntu/yzh/Isaac-GR00T-rtc \
GR00T_MODEL_PATH=/home/ubuntu/yzh/ckpt/gr00tn17/task1_isaac_sonic_lqb/checkpoint-60000 \
GR00T_BACKBONE_PATH=/home/ubuntu/yzh/Isaac-GR00T-rtc/huggingface/Cosmos-Reason2-2B \
GR00T_PREFIX_RTC=0 \
GR00T_EXECUTION_HORIZON=40 \
GR00T_PORT=22095 \
SERVE_GPU=0 \
bash scripts/deploy/fullstate_task1_gr00t_n17_sonic_eval.sh serve
```

# Readiness (must show prediction=40 and prefix_rtc=false):
#   curl -fsS http://127.0.0.1:22095/config


# ------------------------- Terminal 2: Isaac Ego -------------------------
```bash
cd /home/ubuntu/yzh/Psi0_kimodo_textop

TASK_RECORDINGS_DIR=/home/ubuntu/yzh/mujoco_recordings/20260615_task1_lqb \
ISAAC_HEADLESS=1 \
ISAAC_UDP_PORT=23331 \
ISAAC_EGO_FRAME_PATH=/dev/shm/simple_task1_isaac_ego.frame \
ISAAC_RANDOMIZE_LIGHTING=0 \
ISAAC_TRAINING_LIGHTING_MANIFEST=/home/ubuntu/yzh/HumanoidVLA_MJ_backup/HumanoidVLA_MJ/output/task1_lqb_isaac_lerobot/scene3/task1_lqb_all_groot_sonic_release_train/meta/lighting.jsonl \
TASK_EGO_EYE="0.10 0.06 0.70" \
TASK_EGO_UP="0.69670649 0.00079633 0.71735586" \
ISAAC_LIVE_EGO_WIDTH=1280 \
ISAAC_LIVE_EGO_HEIGHT=720 \
ISAAC_ROBOT_COLORS_SRGB_TO_LINEAR=1 \
ISAAC_SMOOTH_TASK_TABLE_CYLINDER=1 \
ISAAC_MAIN_WIDTH=1280 \
ISAAC_MAIN_HEIGHT=720 \
ISAAC_OUTPUT_DIR=/home/ubuntu/yzh/HumanoidVLA_MJ_backup/HumanoidVLA_MJ/output/task1_isaac_hssd_scene3 \
bash scripts/deploy/fullstate_task1_gr00t_rot6d59_kimodo_textop_eval.sh isaac
```

# 如需世界相机和 Ego 小窗，将 ISAAC_HEADLESS 改成 0。


# ----------------------- Terminal 3: SONIC eval -----------------------
```bash
cd /home/ubuntu/yzh/Psi0_kimodo_textop

RUN_TAG=lqb_sonic_run002_ckpt60k_h40

PSI0_ROOT=/home/ubuntu/yzh/Psi0_kimodo_textop \
SIMPLE_ROOT=/home/ubuntu/yzh/Psi0_kimodo_textop/third_party/SIMPLE \
GR00T_ROOT=/home/ubuntu/yzh/Isaac-GR00T-rtc \
GR00T_MODEL_PATH=/home/ubuntu/yzh/ckpt/gr00tn17/task1_isaac_sonic_lqb/checkpoint-60000 \
GR00T_BACKBONE_PATH=/home/ubuntu/yzh/Isaac-GR00T-rtc/huggingface/Cosmos-Reason2-2B \
SONIC_DECODER_ONNX=/home/ubuntu/yzh/GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/model_decoder.onnx \
GR00T_PREFIX_RTC=0 \
GR00T_EXECUTION_HORIZON=40 \
GR00T_PORT=22095 \
GR00T_SONIC_TASK=G1Fullstate20260615Task1-v0 \
GR00T_SONIC_EVAL_DIR="/home/ubuntu/yzh/Psi0_kimodo_textop/third_party/SIMPLE/data/evals_task1_lqb_isaac_${RUN_TAG}" \
TASK1_RECORDINGS_DIR=/home/ubuntu/yzh/mujoco_recordings/20260615_task1_lqb \
TASK1_RECORDING_SEED=42 \
SIM_MODE=mujoco_external_isaac \
EXTERNAL_ISAAC_UDP_PORT=23331 \
EXTERNAL_ISAAC_EGO_FRAME_PATH=/dev/shm/simple_task1_isaac_ego.frame \
EXTERNAL_ISAAC_EGO_WIDTH=1280 \
EXTERNAL_ISAAC_EGO_HEIGHT=720 \
EXTERNAL_ISAAC_CAMERA_LOCAL_EYE="0.10 0.06 0.70" \
SKIP_STABILIZE=1 \
GR00T_INITIAL_POSE_STEPS=0 \
EVAL_GPU=0 \
NUM_EPISODES=10 \
EPISODE_START=0 \
MAX_EPISODE_STEPS=500 \
SAVE_VIDEO=1 \
bash scripts/deploy/fullstate_task1_gr00t_n17_sonic_eval.sh eval
```

# 固定 recording：增加 TASK1_RECORDING_INDEX=0。
# 手动续跑：保持 RUN_TAG 不变，并同时修改 EPISODE_START 与 NUM_EPISODES。
# 原生 SONIC 链路只有三个终端，不启动 Kimodo/TextOp。
