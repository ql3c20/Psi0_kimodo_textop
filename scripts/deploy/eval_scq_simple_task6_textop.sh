
# --------------------------- Terminal 1: GR00T ---------------------------
```bash
cd /home/ubuntu/yzh/Psi0_kimodo_textop

TASK_NUMBER=6 \
FULLSTATE_GR00T_TASK=G1Fullstate20260828Task6-v0 \
TASK_RECORDINGS_DIR=/home/ubuntu/yzh/mujoco_recordings/copy/20260828_task6_scq/20260828 \
SERVE_GPU=0 \
GR00T_PORT=22086 \
GR00T_PREFIX_RTC=1 \
GR00T_USE_RTC=1 \
GR00T_PREFIX_RTC_TIMESTEP_MODE=groot_clean \
GR00T_ACTION_HORIZON=40 \
GR00T_EXECUTION_HORIZON=30 \
GR00T_MODEL_PATH=/home/ubuntu/yzh/ckpt/gr00tn17/task6_textop_newdata \
GR00T_BACKBONE_PATH=/home/ubuntu/yzh/Isaac-GR00T-rtc/huggingface/Cosmos-Reason2-2B \
bash scripts/deploy/fullstate_task1_gr00t_rot6d59_kimodo_textop_eval.sh serve
```

# Readiness:
#   curl --noproxy '*' -fsS http://127.0.0.1:22086/health


# ---------------------- Terminal 2: Kimodo PyTorch -----------------------
```bash
cd /home/ubuntu/yzh/Psi0_kimodo_textop

TASK_NUMBER=6 \
FULLSTATE_GR00T_TASK=G1Fullstate20260828Task6-v0 \
TASK_RECORDINGS_DIR=/home/ubuntu/yzh/mujoco_recordings/copy/20260828_task6_scq/20260828 \
KIMODO_GPU=0 \
GR00T_PORT=22086 \
KIMODO_SERVER_PORT=22186 \
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
#   curl --noproxy '*' -fsS http://127.0.0.1:22186/config


# ------------------------- Terminal 3: Isaac Ego -------------------------
```bash
cd /home/ubuntu/yzh/Psi0_kimodo_textop

TASK_NUMBER=6 \
FULLSTATE_GR00T_TASK=G1Fullstate20260828Task6-v0 \
TASK_RECORDINGS_DIR=/home/ubuntu/yzh/mujoco_recordings/copy/20260828_task6_scq/20260828 \
TASK_HSSD_USD=/home/ubuntu/yzh/Psi0_kimodo_textop/third_party/SIMPLE/data/scenes/hssd/107734119_175999932/107734119_175999932.usd \
ISAAC_HEADLESS=1 \
ISAAC_UDP_PORT=23332 \
ISAAC_EGO_FRAME_PATH=/dev/shm/simple_task6_textop_isaac_ego.frame \
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
ISAAC_OUTPUT_DIR=/home/ubuntu/yzh/HumanoidVLA_MJ_backup/HumanoidVLA_MJ/output/task6_scq_isaac_hssd_scene0_textop_parallel \
bash scripts/deploy/fullstate_task1_gr00t_rot6d59_kimodo_textop_eval.sh isaac
```

# 如需世界相机和右上角 Ego 小窗，将 ISAAC_HEADLESS 改成 0。
# 启动完成后应能看到 task3_isaac_hssd_viewer.py 进程，并监听 UDP 23331。


# ----------------------- Terminal 4: TextOp eval -------------------------
```bash
cd /home/ubuntu/yzh/Psi0_kimodo_textop

RUN_TAG=task6_textop_newdata_can_left_data_aug_h40e30_eps50
KIMODO_RUN=/home/ubuntu/yzh/ckpt/kimodo/scq_final

TASK_NUMBER=6 \
FULLSTATE_GR00T_TASK=G1Fullstate20260828Task6-v0 \
TASK_RECORDINGS_DIR=/home/ubuntu/yzh/mujoco_recordings/copy/20260828_task6_scq/20260828 \
GR00T_MODEL_PATH=/home/ubuntu/yzh/ckpt/gr00tn17/task6_textop_newdata \
GR00T_PORT=22086 \
KIMODO_SERVER_PORT=22186 \
TASK6_INSTRUCTION="Pick up the object from the shelf and place it on the shelf below." \
POLICY_ACTION_HORIZON=40 \
POLICY_EXECUTION_HORIZON=30 \
TASK6_RECORDING_INDEX= \
TASK6_RECORDING_SEED=42 \
TASK6_CAN_RANDOMIZE_LEFT=1 \
TASK6_CAN_RANDOM_SEED=42 \
TASK6_CAN_LEFT_MIN_Y=0.05 \
TASK6_CAN_LEFT_JITTER_X=0.005 \
TASK6_CAN_LEFT_JITTER_Y=0.005 \
TASK6_CAN_LEFT_WEIGHT_POWER=2.0 \
KIMODO_POLICY_ONLY_INITIAL_QPOS=1 \
KIMODO_RTC_PREFIX_FRAMES=0 \
KIMODO_USE_TRT=0 \
KIMODO_DISTILL_CONFIG="$KIMODO_RUN/resolved_config.yaml" \
KIMODO_DISTILL_CKPT="$KIMODO_RUN/ema_final.pt" \
TASK_EGO_EYE="0.10 0.06 0.50" \
ISAAC_UDP_PORT=23332 \
ISAAC_EGO_FRAME_PATH=/dev/shm/simple_task6_textop_isaac_ego.frame \
ISAAC_LIVE_EGO_WIDTH=1280 \
ISAAC_LIVE_EGO_HEIGHT=720 \
EVAL_GPU=0 \
NUM_EPISODES=100 \
EPISODE_START=31 \
MAX_EPISODE_STEPS=700 \
SAVE_VIDEO=1 \
FULLSTATE_GR00T_EVAL_DIR="/home/ubuntu/yzh/Psi0_kimodo_textop/third_party/SIMPLE/data/evals_task6_scq_isaac_${RUN_TAG}" \
FULLSTATE_GR00T_KIMODO_WORK_DIR="/home/ubuntu/yzh/Psi0_kimodo_textop/outputs/kimodo_task6_scq_isaac_${RUN_TAG}" \
bash scripts/deploy/fullstate_task1_gr00t_rot6d59_kimodo_textop_eval.sh eval
```

# 固定第 0 条 recording：将 TASK6_RECORDING_INDEX= 改成 TASK6_RECORDING_INDEX=0。
# Task6 提示词：Pick up the object from the shelf and place it on the shelf below.
# 易拉罐从原始数据 Y>=0.05 的左侧出生点加权抽样，并加 +/-5 mm 抖动扩充，seed=42。
# MuJoCo 是物理与成功判定权威；Isaac 只读同步 qpos 并渲染客厅背景。
# 随机顺序由 TASK6_RECORDING_SEED=42 固定；同一 seed 会复现相同 episode 顺序。
# 断点续跑：保持 RUN_TAG 不变，并在终端 4 命令前增加 RESUME=1。
