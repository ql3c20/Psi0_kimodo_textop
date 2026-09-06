# ----------------------- Terminal 1: GR00T SONIC -----------------------

```bash
cd /home/ubuntu/yzh/Psi0_kimodo_textop

PSI0_ROOT=/home/ubuntu/yzh/Psi0_kimodo_textop \
SIMPLE_ROOT=/home/ubuntu/yzh/Psi0_kimodo_textop/third_party/SIMPLE \
GR00T_ROOT=/home/ubuntu/yzh/Isaac-GR00T-rtc \
GR00T_MODEL_PATH=/home/ubuntu/yzh/ckpt/gr00tn17/task3_isaac_sonic_lqb/checkpoint-60000 \
GR00T_BACKBONE_PATH=/home/ubuntu/yzh/Isaac-GR00T-rtc/huggingface/Cosmos-Reason2-2B \
GR00T_PREFIX_RTC=0 \
GR00T_EXECUTION_HORIZON=40 \
GR00T_PORT=22095 \
SERVE_GPU=0 \
bash scripts/deploy/fullstate_task1_gr00t_n17_sonic_eval.sh serve
```

# Readiness (must show prediction_horizon=40, execution_horizon=34,
# prefix_rtc=true, and prefix_rtc_timestep_mode=groot_clean):
#   curl -fsS http://127.0.0.1:22095/config


# ------------------------- Terminal 2: Isaac Ego -------------------------
```bash
cd /home/ubuntu/yzh/Psi0_kimodo_textop

TASK_RECORDINGS_DIR=/home/ubuntu/yzh/mujoco_recordings/20260804_task3_lqb \
TASK_HSSD_USD=/home/ubuntu/yzh/Psi0_kimodo_textop/third_party/SIMPLE/data/scenes/hssd/102344250/102344250_local.usd \
ISAAC_HEADLESS=1 \
ISAAC_UDP_PORT=23331 \
ISAAC_EGO_FRAME_PATH=/dev/shm/simple_task3_isaac_ego.frame \
ISAAC_LIGHT_RIG=scripted \
ISAAC_RANDOMIZE_LIGHTING=0 \
ISAAC_TRAINING_LIGHTING_MANIFEST=/home/ubuntu/yzh/HumanoidVLA_MJ_backup/HumanoidVLA_MJ/output/task3_lqb_isaac_lerobot/scene3/task3_lqb_all_groot_sonic_release_train/meta/lighting.jsonl \
TASK_EGO_EYE="0.10 0.06 0.70" \
TASK_EGO_FORWARD="0.71735609 0.0 -0.69670671" \
TASK_EGO_UP="0.69670649 0.00079633 0.71735586" \
TASK_TRANSLATE="0.6 -0.5 0.0" \
ISAAC_LIVE_EGO_WIDTH=1280 \
ISAAC_LIVE_EGO_HEIGHT=720 \
ISAAC_ROBOT_COLORS_SRGB_TO_LINEAR=1 \
ISAAC_SMOOTH_TASK_TABLE_CYLINDER=1 \
ISAAC_MAIN_WIDTH=1280 \
ISAAC_MAIN_HEIGHT=720 \
ISAAC_ROBOT_Z_OFFSET=0.012 \
ISAAC_TRASH_Z_OFFSET=0.006 \
ISAAC_FREE_OBJECT_Z_OFFSET=0.0 \
ISAAC_OUTPUT_DIR=/home/ubuntu/yzh/HumanoidVLA_MJ_backup/HumanoidVLA_MJ/output/task3_lqb_isaac_hssd_scene3 \
bash scripts/deploy/fullstate_task3_gr00t_rot6d59_kimodo_textop_eval.sh isaac
```

# 如需世界相机和右上角 Ego 小窗，将 ISAAC_HEADLESS 改成 0。


# ------------------------- Terminal 3: SONIC eval -------------------------
```bash
cd /home/ubuntu/yzh/Psi0_kimodo_textop

RUN_TAG=lqb_sonic_run001_ckpt60k_eps100

PSI0_ROOT=/home/ubuntu/yzh/Psi0_kimodo_textop \
SIMPLE_ROOT=/home/ubuntu/yzh/Psi0_kimodo_textop/third_party/SIMPLE \
GR00T_ROOT=/home/ubuntu/yzh/Isaac-GR00T-rtc \
HUMANOID_VLA_MJ_ROOT=/home/ubuntu/yzh/HumanoidVLA_MJ_backup/HumanoidVLA_MJ \
GR00T_MODEL_PATH=/home/ubuntu/yzh/ckpt/gr00tn17/task3_isaac_sonic_lqb/checkpoint-60000 \
GR00T_BACKBONE_PATH=/home/ubuntu/yzh/Isaac-GR00T-rtc/huggingface/Cosmos-Reason2-2B \
SONIC_DECODER_ONNX=/home/ubuntu/yzh/GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/model_decoder.onnx \
GR00T_PREFIX_RTC=0 \
GR00T_EXECUTION_HORIZON=40 \
GR00T_PORT=22095 \
GR00T_SONIC_TASK=G1Fullstate20260804Task3-v0 \
GR00T_SONIC_EVAL_DIR="/home/ubuntu/yzh/Psi0_kimodo_textop/third_party/SIMPLE/data/evals_task3_lqb_isaac_${RUN_TAG}" \
TASK3_RECORDINGS_DIR=/home/ubuntu/yzh/mujoco_recordings/20260804_task3_lqb \
TASK3_RECORDING_INDEX= \
TASK3_RECORDING_SEED=42 \
SIM_MODE=mujoco_external_isaac \
EXTERNAL_ISAAC_UDP_PORT=23331 \
EXTERNAL_ISAAC_EGO_FRAME_PATH=/dev/shm/simple_task3_isaac_ego.frame \
EXTERNAL_ISAAC_EGO_WIDTH=1280 \
EXTERNAL_ISAAC_EGO_HEIGHT=720 \
EXTERNAL_ISAAC_EGO_TIMEOUT_S=30 \
EXTERNAL_ISAAC_CAMERA_LOCAL_EYE="0.10 0.06 0.70" \
SKIP_STABILIZE=1 \
GR00T_INITIAL_POSE_STEPS=0 \
EVAL_GPU=0 \
NUM_EPISODES=100 \
EPISODE_START=0 \
MAX_EPISODE_STEPS=500 \
SAVE_VIDEO=1 \
bash scripts/deploy/fullstate_task1_gr00t_n17_sonic_eval.sh eval
```

# 固定第 0 条 recording：将 TASK3_RECORDING_INDEX= 改成 TASK3_RECORDING_INDEX=0。
# 随机顺序由 TASK3_RECORDING_SEED=42 固定；同一 seed 会复现相同 episode 顺序。
# 手动续跑：保持 RUN_TAG 不变，并同时修改 EPISODE_START 与 NUM_EPISODES。
# 原生 SONIC 链路只有三个终端，不启动 Kimodo/TextOp。
