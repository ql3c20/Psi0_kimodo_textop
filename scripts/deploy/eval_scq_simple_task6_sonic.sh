# ----------------------- Terminal 1: GR00T SONIC -----------------------

```bash
cd /home/ubuntu/yzh/Psi0_kimodo_textop

PSI0_ROOT=/home/ubuntu/yzh/Psi0_kimodo_textop \
SIMPLE_ROOT=/home/ubuntu/yzh/Psi0_kimodo_textop/third_party/SIMPLE \
GR00T_ROOT=/home/ubuntu/yzh/Isaac-GR00T-rtc \
GR00T_MODEL_PATH=/home/ubuntu/yzh/ckpt/gr00tn17/task6_sonic_newdata \
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

TASK_NUMBER=6 \
FULLSTATE_GR00T_TASK=G1Fullstate20260828Task6-v0 \
TASK_RECORDINGS_DIR=/home/ubuntu/yzh/mujoco_recordings/copy/20260828_task6_scq/20260828 \
TASK_HSSD_USD=/home/ubuntu/yzh/Psi0_kimodo_textop/third_party/SIMPLE/data/scenes/hssd/107734119_175999932/107734119_175999932.usd \
ISAAC_HEADLESS=1 \
ISAAC_UDP_PORT=23331 \
ISAAC_EGO_FRAME_PATH=/dev/shm/simple_task6_isaac_ego.frame \
ISAAC_LIGHT_RIG=scripted \
ISAAC_RANDOMIZE_LIGHTING=0 \
TASK_EGO_EYE="0.10 0.06 0.50" \
TASK_EGO_FORWARD="0.71735609 0.0 -0.69670671" \
TASK_EGO_UP="0.69670649 0.00079633 0.71735586" \
TASK_EGO_PITCH_DEG=0 \
TASK_EGO_FOVY=70 \
TASK_EGO_NEAR_CLIP=0.0324 \
TASK_BACKGROUND_TRANSLATE="1.3 0.0 0.0" \
TASK_HIDE_BACKGROUND_PRIM=furniture/d1bb1e76ecd549767fd650aa211e3ce29be75ad6 \
TASK_TRANSLATE="-2.964 0.785 0.0" \
TASK_YAW_DEG=180 \
ISAAC_TRANSFORM_TASK_ROOT=1 \
ISAAC_REMOVE_TASK6_TOP_SHELF_VISUAL=1 \
ISAAC_TRASH_TRANSLATE="0.0 0.0 0.0" \
ISAAC_LIVE_EGO_WIDTH=1280 \
ISAAC_LIVE_EGO_HEIGHT=720 \
ISAAC_ROBOT_COLORS_SRGB_TO_LINEAR=1 \
ISAAC_SMOOTH_TASK_TABLE_CYLINDER=0 \
ISAAC_MAIN_WIDTH=1280 \
ISAAC_MAIN_HEIGHT=720 \
ISAAC_ROBOT_Z_OFFSET=0.012 \
ISAAC_TRASH_Z_OFFSET=0.0 \
ISAAC_FREE_OBJECT_Z_OFFSET=0.0 \
ISAAC_OUTPUT_DIR=/home/ubuntu/yzh/HumanoidVLA_MJ_backup/HumanoidVLA_MJ/output/task6_scq_isaac_hssd_scene0 \
bash scripts/deploy/fullstate_task1_gr00t_rot6d59_kimodo_textop_eval.sh isaac
```

# 如需世界相机和 Ego 小窗，将 ISAAC_HEADLESS 改成 0。


# ----------------------- Terminal 3: SONIC eval -----------------------
```bash
cd /home/ubuntu/yzh/Psi0_kimodo_textop

RUN_TAG=task6_sonic_newdata_can_front_edge_center_left_h40_eps100

PSI0_ROOT=/home/ubuntu/yzh/Psi0_kimodo_textop \
SIMPLE_ROOT=/home/ubuntu/yzh/Psi0_kimodo_textop/third_party/SIMPLE \
GR00T_ROOT=/home/ubuntu/yzh/Isaac-GR00T-rtc \
GR00T_MODEL_PATH=/home/ubuntu/yzh/ckpt/gr00tn17/task6_sonic_newdata \
GR00T_BACKBONE_PATH=/home/ubuntu/yzh/Isaac-GR00T-rtc/huggingface/Cosmos-Reason2-2B \
SONIC_DECODER_ONNX=/home/ubuntu/yzh/GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/model_decoder.onnx \
GR00T_PREFIX_RTC=0 \
GR00T_EXECUTION_HORIZON=40 \
GR00T_PORT=22095 \
GR00T_SONIC_TASK=G1Fullstate20260828Task6-v0 \
GR00T_SONIC_EVAL_DIR="/home/ubuntu/yzh/Psi0_kimodo_textop/third_party/SIMPLE/data/evals_task6_scq_isaac_${RUN_TAG}" \
HUMANOID_VLA_MJ_ROOT=/home/ubuntu/yzh/HumanoidVLA_MJ_backup/HumanoidVLA_MJ \
TASK6_RECORDINGS_DIR=/home/ubuntu/yzh/mujoco_recordings/copy/20260828_task6_scq/20260828 \
TASK6_RECORDING_INDEX= \
TASK6_RECORDING_SEED=42 \
TASK6_CAN_RANDOMIZE_LEFT=1 \
TASK6_CAN_RANDOM_SEED=42 \
TASK6_CAN_FIXED_X=1.220 \
TASK6_CAN_FIXED_Y=0.020 \
TASK6_CAN_LEFT_MIN_Y=0.05 \
TASK6_CAN_LEFT_JITTER_X=0.0 \
TASK6_CAN_LEFT_JITTER_Y=0.0 \
TASK6_CAN_LEFT_WEIGHT_POWER=2.0 \
TASK6_ROBOT_TRANSLATE="0.0 0.0 0.0" \
SIM_MODE=mujoco_external_isaac \
EXTERNAL_ISAAC_UDP_PORT=23331 \
EXTERNAL_ISAAC_EGO_FRAME_PATH=/dev/shm/simple_task6_isaac_ego.frame \
EXTERNAL_ISAAC_EGO_WIDTH=1280 \
EXTERNAL_ISAAC_EGO_HEIGHT=720 \
EXTERNAL_ISAAC_CAMERA_LOCAL_EYE="0.10 0.06 0.50" \
SKIP_STABILIZE=1 \
GR00T_INITIAL_POSE_STEPS=0 \
EVAL_GPU=0 \
TASK6_INSTRUCTION="Pick up the object from the shelf and place it on the shelf below." \
NUM_EPISODES=100 \
EPISODE_START=0 \
MAX_EPISODE_STEPS=700 \
SAVE_VIDEO=1 \
bash scripts/deploy/fullstate_task1_gr00t_n17_sonic_eval.sh eval
```

# 固定第 0 条 recording：将 TASK6_RECORDING_INDEX= 改成 TASK6_RECORDING_INDEX=0。
# Task6 提示词：Pick up the object from the shelf and place it on the shelf below.
# 机器人使用 recording 中的原始初始位置；货架不动。
# SONIC 专用：易拉罐固定在靠机器人一侧的安全前沿 X=1.220 m，
# 并位于中心左侧 2 cm（Y=+0.020 m）；XY 不加抖动。
# MuJoCo 是物理与成功判定权威；Isaac 只读同步 qpos 并渲染客厅背景。
# 训练使用原生 SONIC 40 帧动作，因此 GR00T_PREFIX_RTC=0。
# 手动续跑：保持 RUN_TAG 不变，并同时修改 EPISODE_START 与 NUM_EPISODES。
# 原生 SONIC 链路只有三个终端，不启动 Kimodo/TextOp。
