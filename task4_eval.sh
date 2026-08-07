# Task4 评测命令备忘录（仅用于复制命令，不要整体执行此文件）
#
# 场景初始化：每次 eval 都从 /dev/urandom 生成新的 TASK4_RUN_SEED，
# 再从 203 条录像初始状态中无放回随机选取，因此每次 10 条的顺序不同。
# 如需复现实验，可将 TASK4_RUN_SEED="$(...)" 改成固定数字，例如：
# TASK4_RUN_SEED=12345
#
# 视频开关：
# SAVE_VIDEO=1  保存视频
# SAVE_VIDEO=0  不保存视频


# ============================================================================
# GR00T -> Kimodo -> TextOp 管线（三个终端）
# ============================================================================

# 终端1：启动 rot6d59 GR00T server
```bash
cd /pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Psi0

SERVE_GPU=1 \
GR00T_PORT=22096 \
GR00T_PREFIX_RTC=1 \
GR00T_USE_RTC=1 \
GR00T_PREFIX_RTC_TIMESTEP_MODE=groot_clean \
GR00T_EXECUTION_HORIZON=34 \
GR00T_MODEL_PATH=/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Isaac-GR00T/outputs/task4-gr00t-n17-rot6d59-kimodo-textop-prefix-rtc-groot-clean-A800/checkpoint-80000 \
bash scripts/deploy/fullstate_task4_gr00t_rot6d59_kimodo_textop_eval.sh serve
```


# 终端2：启动 Kimodo/TextOp server
```bash
cd /pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Psi0

KIMODO_GPU=2 \
KIMODO_KEYFRAME_STEP=1 \
FULLSTATE_TASK4_GR00T_KIMODO_WORK_DIR=/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Psi0/outputs/kimodo_task4_gr00t_rot6d59_prefixrtc_grootclean \
bash scripts/deploy/fullstate_task4_gr00t_rot6d59_kimodo_textop_eval.sh kimodo-serve
```


# 终端3：运行 Kimodo/TextOp Task4 评测
# 修改 SAVE_VIDEO=1/0 选择是否保存视频。
# 默认随机 seed；固定 TASK4_RUN_SEED 可复现相同的 10 条场景顺序。
```bash
cd /pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Psi0

unset TASK4_RECORDING_INDEX
TASK4_RUN_SEED="$(od -An -N4 -tu4 /dev/urandom | tr -d '[:space:]')"

EVAL_GPU=3 \
GR00T_PORT=22096 \
NUM_EPISODES=10 \
EPISODE_START=0 \
MAX_EPISODE_STEPS=700 \
SAVE_VIDEO=1 \
GR00T_EXECUTION_HORIZON=34 \
SKIP_STABILIZE=1 \
TASK4_INIT_FROM_RECORDINGS=1 \
TASK4_RECORDINGS_DIR=/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/HumanoidVLA_MJ_1/output/20260729_task4 \
TASK4_RECORDING_SEED="$TASK4_RUN_SEED" \
FULLSTATE_TASK4_GR00T_KIMODO_WORK_DIR=/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Psi0/outputs/kimodo_task4_gr00t_rot6d59_prefixrtc_grootclean \
FULLSTATE_TASK4_GR00T_EVAL_DIR="data/evals_task4_gr00t_n17_rot6d59_kimodo_textop_prefix_rtc_groot_clean_seed${TASK4_RUN_SEED}_10eps" \
bash scripts/deploy/fullstate_task4_gr00t_rot6d59_kimodo_textop_eval.sh eval
```


# ============================================================================
# GR00T -> Sonic 管线（两个终端）
# ============================================================================

# 终端1：启动 Sonic 格式 GR00T server
# 注意必须使用 UNITREE_G1_SONIC checkpoint-80000，不能使用 rot6d59 checkpoint。
```bash
cd /pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Psi0

SERVE_GPU=1 \
GR00T_PORT=22095 \
GR00T_PREFIX_RTC=1 \
GR00T_PREFIX_RTC_TIMESTEP_MODE=groot_clean \
GR00T_EXECUTION_HORIZON=34 \
GR00T_MODEL_PATH=/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Isaac-GR00T/outputs/task4-gr00t-n17-sonic-prefix-rtc-groot-clean/checkpoint-80000 \
bash scripts/deploy/fullstate_task4_gr00t_n17_sonic_eval.sh serve
```


# 终端2：运行 Sonic Task4 评测
# 修改 SAVE_VIDEO=1/0 选择是否保存视频。
# 默认随机 seed；固定 TASK4_RUN_SEED 可复现相同的 10 条场景顺序。
```bash
cd /pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Psi0

unset TASK4_RECORDING_INDEX
TASK4_RUN_SEED="$(od -An -N4 -tu4 /dev/urandom | tr -d '[:space:]')"

EVAL_GPU=4 \
GR00T_PORT=22095 \
NUM_EPISODES=100 \
EPISODE_START=0 \
MAX_EPISODE_STEPS=500 \
SAVE_VIDEO=1 \
GR00T_EXECUTION_HORIZON=34 \
SKIP_STABILIZE=1 \
TASK4_INIT_FROM_RECORDINGS=1 \
TASK4_RECORDINGS_DIR=/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/HumanoidVLA_MJ_1/output/20260729_task4 \
TASK4_RECORDING_SEED="$TASK4_RUN_SEED" \
GR00T_SONIC_EVAL_DIR="data/evals_task4_gr00t_n17_sonic_prefix_rtc_groot_clean_ckpt80000_seed${TASK4_RUN_SEED}_10eps" \
bash scripts/deploy/fullstate_task4_gr00t_n17_sonic_eval.sh eval
```
