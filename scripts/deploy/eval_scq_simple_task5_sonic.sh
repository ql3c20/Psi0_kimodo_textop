# ----------------------- Terminal 1: GR00T SONIC -----------------------

```bash
cd /home/ubuntu/yzh/Psi0_kimodo_textop

PSI0_ROOT=/home/ubuntu/yzh/Psi0_kimodo_textop \
SIMPLE_ROOT=/home/ubuntu/yzh/Psi0_kimodo_textop/third_party/SIMPLE \
GR00T_ROOT=/home/ubuntu/yzh/Isaac-GR00T-rtc \
GR00T_MODEL_PATH=/home/ubuntu/yzh/ckpt/gr00tn17/task5_newlight_sonic \
GR00T_BACKBONE_PATH=/home/ubuntu/yzh/Isaac-GR00T-rtc/huggingface/Cosmos-Reason2-2B \
GR00T_PREFIX_RTC=0 \
GR00T_EXECUTION_HORIZON=40 \
GR00T_PORT=22095 \
SERVE_GPU=0 \
bash scripts/deploy/fullstate_task5_gr00t_n17_sonic_eval.sh serve
```

# Readiness (must show prediction=40 and prefix_rtc=false):
#   curl -fsS http://127.0.0.1:22095/config


# ------------------------- Terminal 2: Isaac Ego -------------------------
```bash
cd /home/ubuntu/yzh/Psi0_kimodo_textop

TASK_NUMBER=5 \
TASK_RECORDINGS_DIR=/home/ubuntu/yzh/mujoco_recordings/copy/20260825_task5_scq \
TASK_HSSD_USD=/home/ubuntu/yzh/Psi0_kimodo_textop/third_party/SIMPLE/data/scenes/hssd/103997919_171031233/103997919_171031233_local.usd \
ISAAC_HEADLESS=1 \
ISAAC_UDP_PORT=23331 \
ISAAC_EGO_FRAME_PATH=/dev/shm/simple_task5_isaac_ego.frame \
ISAAC_LIGHT_RIG=scripted \
ISAAC_RANDOMIZE_LIGHTING=0 \
ISAAC_TRAINING_LIGHTING_MANIFEST=/home/ubuntu/yzh/HumanoidVLA_MJ_backup/HumanoidVLA_MJ/output/task5_scq_isaac_hssd_bedroom_scene28/lerobot/task5_scq_all/meta/lighting.jsonl \
TASK_EGO_EYE="0.10 0.06 0.70" \
TASK_EGO_FORWARD="0.71735609 0.0 -0.69670671" \
TASK_EGO_UP="0.69670649 0.00079633 0.71735586" \
TASK_TRANSLATE="1.4 -1.3 0.0" \
TASK_YAW_DEG=-90 \
ISAAC_TRASH_TRANSLATE="-0.10 0.10 0.0" \
ISAAC_LIVE_EGO_WIDTH=1280 \
ISAAC_LIVE_EGO_HEIGHT=720 \
ISAAC_ROBOT_COLORS_SRGB_TO_LINEAR=1 \
ISAAC_SMOOTH_TASK_TABLE_CYLINDER=0 \
ISAAC_MAIN_WIDTH=1280 \
ISAAC_MAIN_HEIGHT=720 \
ISAAC_ROBOT_Z_OFFSET=0.012 \
ISAAC_TRASH_Z_OFFSET=0.0 \
ISAAC_FREE_OBJECT_Z_OFFSET=0.0 \
ISAAC_OUTPUT_DIR=/home/ubuntu/yzh/HumanoidVLA_MJ_backup/HumanoidVLA_MJ/output/task5_scq_isaac_hssd_bedroom_scene28 \
bash scripts/deploy/fullstate_task1_gr00t_rot6d59_kimodo_textop_eval.sh isaac
```

# 如需世界相机和 Ego 小窗，将 ISAAC_HEADLESS 改成 0。


# ----------------------- Terminal 3: SONIC eval -----------------------
```bash
cd /home/ubuntu/yzh/Psi0_kimodo_textop

RUN_TAG=scq_sonic_newlight_ckpt80k_h40_drawer_right10cm_closer10cm_eps100_final

PSI0_ROOT=/home/ubuntu/yzh/Psi0_kimodo_textop \
SIMPLE_ROOT=/home/ubuntu/yzh/Psi0_kimodo_textop/third_party/SIMPLE \
GR00T_ROOT=/home/ubuntu/yzh/Isaac-GR00T-rtc \
GR00T_MODEL_PATH=/home/ubuntu/yzh/ckpt/gr00tn17/task5_newlight_sonic \
GR00T_BACKBONE_PATH=/home/ubuntu/yzh/Isaac-GR00T-rtc/huggingface/Cosmos-Reason2-2B \
SONIC_DECODER_ONNX=/home/ubuntu/yzh/GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/model_decoder.onnx \
GR00T_PREFIX_RTC=0 \
GR00T_EXECUTION_HORIZON=40 \
GR00T_PORT=22095 \
GR00T_SONIC_TASK=G1Fullstate20260825Task5-v0 \
GR00T_SONIC_EVAL_DIR="/home/ubuntu/yzh/Psi0_kimodo_textop/third_party/SIMPLE/data/evals_task5_scq_isaac_${RUN_TAG}" \
TASK5_RECORDINGS_DIR=/home/ubuntu/yzh/mujoco_recordings/copy/20260825_task5_scq \
TASK5_RECORDING_INDEX= \
TASK5_RECORDING_SEED=42 \
TASK5_DRAWER_TRANSLATE="-0.10 -0.10 0.0" \
SIM_MODE=mujoco_external_isaac \
EXTERNAL_ISAAC_UDP_PORT=23331 \
EXTERNAL_ISAAC_EGO_FRAME_PATH=/dev/shm/simple_task5_isaac_ego.frame \
EXTERNAL_ISAAC_EGO_WIDTH=1280 \
EXTERNAL_ISAAC_EGO_HEIGHT=720 \
EXTERNAL_ISAAC_CAMERA_LOCAL_EYE="0.10 0.06 0.70" \
SKIP_STABILIZE=1 \
GR00T_INITIAL_POSE_STEPS=0 \
EVAL_GPU=0 \
TASK5_INSTRUCTION="Walk forward and pull open the upper drawer." \
NUM_EPISODES=100 \
EPISODE_START=0 \
MAX_EPISODE_STEPS=500 \
SAVE_VIDEO=1 \
bash scripts/deploy/fullstate_task5_gr00t_n17_sonic_eval.sh eval
```

# 固定第 0 条 recording：将 TASK5_RECORDING_INDEX= 改成 TASK5_RECORDING_INDEX=0。
# 柜子相对机器人右移 10 cm、靠近 10 cm；MuJoCo 为 (-X,-Y)，Isaac 经 -90 度旋转后为 (-X,+Y)。
# 训练使用原生 SONIC 40 帧动作，因此 GR00T_PREFIX_RTC=0。
# 手动续跑：保持 RUN_TAG 不变，并同时修改 EPISODE_START 与 NUM_EPISODES。
# 原生 SONIC 链路只有三个终端，不启动 Kimodo/TextOp。
