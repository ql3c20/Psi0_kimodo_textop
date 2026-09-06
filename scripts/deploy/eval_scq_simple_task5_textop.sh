
# --------------------------- Terminal 1: GR00T ---------------------------
```bash
cd /home/ubuntu/yzh/Psi0_kimodo_textop

TASK_NUMBER=5 \
FULLSTATE_GR00T_TASK=G1Fullstate20260825Task5-v0 \
TASK_RECORDINGS_DIR=/home/ubuntu/yzh/mujoco_recordings/copy/20260825_task5_scq \
SERVE_GPU=0 \
GR00T_PORT=22085 \
GR00T_PREFIX_RTC=1 \
GR00T_USE_RTC=1 \
GR00T_PREFIX_RTC_TIMESTEP_MODE=groot_clean \
GR00T_ACTION_HORIZON=40 \
GR00T_EXECUTION_HORIZON=30 \
GR00T_MODEL_PATH=/home/ubuntu/yzh/ckpt/gr00tn17/task5_newlight_textop \
GR00T_BACKBONE_PATH=/home/ubuntu/yzh/Isaac-GR00T-rtc/huggingface/Cosmos-Reason2-2B \
bash scripts/deploy/fullstate_task1_gr00t_rot6d59_kimodo_textop_eval.sh serve
```

# Readiness:
#   curl --noproxy '*' -fsS http://127.0.0.1:22085/health


# ---------------------- Terminal 2: Kimodo PyTorch -----------------------
```bash
cd /home/ubuntu/yzh/Psi0_kimodo_textop

TASK_NUMBER=5 \
FULLSTATE_GR00T_TASK=G1Fullstate20260825Task5-v0 \
TASK_RECORDINGS_DIR=/home/ubuntu/yzh/mujoco_recordings/copy/20260825_task5_scq \
KIMODO_GPU=0 \
KIMODO_SERVER_PORT=22185 \
KIMODO_USE_TRT=0 \
KIMODO_TEXT_ENCODER_MODE=original \
KIMODO_ANCHOR_MODE=policy_only \
KIMODO_DIFFUSION_STEPS=20 \
KIMODO_KEYFRAME_STEP=10 \
KIMODO_DISTILL_CONFIG=/home/ubuntu/yzh/ckpt/kimodo/scq_final/resolved_config.yaml \
KIMODO_DISTILL_CKPT=/home/ubuntu/yzh/ckpt/kimodo/scq_final/ema_final.pt \
bash scripts/deploy/fullstate_task1_gr00t_rot6d59_kimodo_textop_eval.sh kimodo-serve
```

# Readiness (must show pytorch / original / policy_only / keyframe_step=10):
#   curl --noproxy '*' -fsS http://127.0.0.1:22185/config


# ------------------------- Terminal 3: Isaac Ego -------------------------
```bash
cd /home/ubuntu/yzh/Psi0_kimodo_textop

TASK_NUMBER=5 \
FULLSTATE_GR00T_TASK=G1Fullstate20260825Task5-v0 \
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
ISAAC_TRASH_TRANSLATE="-0.05 0.05 0.0" \
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

# 如需世界相机和右上角 Ego 小窗，将 ISAAC_HEADLESS 改成 0。
# 启动完成后应能看到 task3_isaac_hssd_viewer.py 进程，并监听 UDP 23331。


# ----------------------- Terminal 4: TextOp eval -------------------------
```bash
cd /home/ubuntu/yzh/Psi0_kimodo_textop

RUN_TAG=scq_textop_newlight_ckpt80k_drawer_right5cm_closer5cm_eps50_auto
KIMODO_RUN=/home/ubuntu/yzh/ckpt/kimodo/scq_final

TASK_NUMBER=5 \
FULLSTATE_GR00T_TASK=G1Fullstate20260825Task5-v0 \
TASK_RECORDINGS_DIR=/home/ubuntu/yzh/mujoco_recordings/copy/20260825_task5_scq \
GR00T_MODEL_PATH=/home/ubuntu/yzh/ckpt/gr00tn17/task5_newlight_textop \
TASK5_INSTRUCTION="Walk forward and pull open the upper drawer." \
POLICY_ACTION_HORIZON=40 \
POLICY_EXECUTION_HORIZON=30 \
TASK5_RECORDING_INDEX= \
TASK5_RECORDING_SEED=42 \
TASK5_DRAWER_TRANSLATE="-0.05 -0.05 0.0" \
KIMODO_POLICY_ONLY_INITIAL_QPOS=1 \
KIMODO_RTC_PREFIX_FRAMES=0 \
KIMODO_USE_TRT=0 \
KIMODO_DISTILL_CONFIG="$KIMODO_RUN/resolved_config.yaml" \
KIMODO_DISTILL_CKPT="$KIMODO_RUN/ema_final.pt" \
TASK_EGO_EYE="0.10 0.06 0.70" \
ISAAC_UDP_PORT=23331 \
ISAAC_EGO_FRAME_PATH=/dev/shm/simple_task5_isaac_ego.frame \
ISAAC_LIVE_EGO_WIDTH=1280 \
ISAAC_LIVE_EGO_HEIGHT=720 \
EVAL_GPU=0 \
NUM_EPISODES=50 \
EPISODE_START=0 \
MAX_EPISODE_STEPS=500 \
SAVE_VIDEO=1 \
FULLSTATE_GR00T_EVAL_DIR="/home/ubuntu/yzh/Psi0_kimodo_textop/third_party/SIMPLE/data/evals_task5_scq_isaac_${RUN_TAG}" \
FULLSTATE_GR00T_KIMODO_WORK_DIR="/home/ubuntu/yzh/Psi0_kimodo_textop/outputs/kimodo_task5_scq_isaac_${RUN_TAG}" \
bash scripts/deploy/fullstate_task1_gr00t_rot6d59_kimodo_textop_eval.sh eval
```

# 固定第 0 条 recording：将 TASK5_RECORDING_INDEX= 改成 TASK5_RECORDING_INDEX=0。
# 柜子相对机器人右移 5 cm、靠近 5 cm；MuJoCo 为 (-X,-Y)，Isaac 经 -90 度旋转后为 (-X,+Y)。
# 随机顺序由 TASK5_RECORDING_SEED=42 固定；同一 seed 会复现相同 episode 顺序。
# 断点续跑：保持 RUN_TAG 不变，并在终端 4 命令前增加 RESUME=1。
