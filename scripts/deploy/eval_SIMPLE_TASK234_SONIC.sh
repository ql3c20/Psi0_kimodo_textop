#!/usr/bin/env bash
set -euo pipefail

# Task2--Task4 native GR00T N1.7 + SONIC + SIMPLE + Isaac-Ego
# 三终端评测命令备忘录（2026-08-19）

: <<'TASK234_SONIC_EVAL_COMMANDS'

Task2 / Task3 / Task4 原生 GR00T N1.7 -> SONIC -> SIMPLE 评测备忘录
=======================================================================

用途
----

这份文件只记录可复制到三个终端的命令，不要直接执行整个文件。

链路：

  GR00T task<N>_sonic -> SONIC decoder -> SIMPLE/MuJoCo
                                      ^
  SIMPLE qpos -> Isaac Sim -> 640x480 Ego

它不经过 Kimodo/TextOp，不能与四终端 Kimodo/TextOp 链路混接。
一次只评测一个任务；Task2/3/4 共用 GR00T 端口 22095 和 Isaac UDP 23331。
切换任务时必须先 Ctrl-C 关闭当前 GR00T 与 Isaac，再启动下一任务。

共同配置
--------

  GR00T root : /home/ubuntu/yzh/Isaac-GR00T-rtc
  Backbone   : /home/ubuntu/yzh/Isaac-GR00T-rtc/huggingface/Cosmos-Reason2-2B
  Decoder    : /home/ubuntu/yzh/GR00T-WholeBodyControl/gear_sonic_deploy/policy/low_latency/model_decoder.onnx
  Prefix-RTC : prediction=40, execute=34, overlap=6
  Ego        : 640x480 @ 50 Hz, VFOV 70 deg, pitch-down 15 deg
  GR00T port : 22095
  Isaac UDP  : 23331

服务启动后可检查：

  curl -fsS http://127.0.0.1:22095/config

应看到 prediction_horizon=40、execution_horizon=34、overlap_steps=6、
prefix_rtc=true，并且 model_path 与当前任务权重一致。


=======================================================================
Task2：抓取桌上瓶子并放入垃圾桶
=======================================================================

Checkpoint:
  /home/ubuntu/yzh/ckpt/gr00tn17/task2_sonic

Prompt:
  Pick up the bottle on the table in front of me and throw it into the trash can.

场景/相机：Scene3 102344280，Ego local eye=(0.06,0.06,0.45)

------------------------ Task2 终端 1：GR00T ----------------------------

cd /home/ubuntu/yzh/Psi0_kimodo_textop

SERVE_GPU=0 \
bash scripts/deploy/fullstate_task2_gr00t_n17_sonic_eval.sh serve

------------------------ Task2 终端 2：Isaac Ego ------------------------

cd /home/ubuntu/yzh/Psi0_kimodo_textop

ISAAC_HEADLESS=1 \
bash scripts/deploy/fullstate_task2_gr00t_n17_sonic_eval.sh isaac

# 看世界相机和右上角 Ego 小窗时将 ISAAC_HEADLESS 改成 0。

------------------------ Task2 终端 3：SIMPLE Eval ----------------------

cd /home/ubuntu/yzh/Psi0_kimodo_textop

RUN_TAG=task2_sonic_seed0

EVAL_GPU=0 \
NUM_EPISODES=10 \
EPISODE_START=0 \
TASK2_RECORDING_SEED=0 \
SAVE_VIDEO=1 \
GR00T_SONIC_EVAL_DIR="/home/ubuntu/yzh/Psi0_kimodo_textop/third_party/SIMPLE/data/evals_task2_sonic_external_isaac_${RUN_TAG}" \
bash scripts/deploy/fullstate_task2_gr00t_n17_sonic_eval.sh eval

# 固定 recording：增加 TASK2_RECORDING_INDEX=0。
# 断点续跑：保持 RUN_TAG 不变并增加 RESUME=1。


=======================================================================
Task3：踩踏板打开垃圾桶
=======================================================================

Checkpoint:
  /home/ubuntu/yzh/ckpt/gr00tn17/task3_sonic

Prompt:
  Step on the pedal to open the trash can in front of you.

场景/相机：Scene13 102344250_local，Ego local eye=(0.06,0.06,0.45)，
task translate=(0.6,-0.5,0)

------------------------ Task3 终端 1：GR00T ----------------------------

cd /home/ubuntu/yzh/Psi0_kimodo_textop

PSI0_ROOT=/home/ubuntu/yzh/Psi0_kimodo_textop \
SIMPLE_ROOT=/home/ubuntu/yzh/Psi0_kimodo_textop/third_party/SIMPLE \
GR00T_ROOT=/home/ubuntu/yzh/Isaac-GR00T-rtc \
GR00T_MODEL_PATH=/home/ubuntu/yzh/ckpt/gr00tn17/task3_sonic \
GR00T_BACKBONE_PATH=/home/ubuntu/yzh/Isaac-GR00T-rtc/huggingface/Cosmos-Reason2-2B \
GR00T_PREFIX_RTC=1 \
GR00T_PREFIX_RTC_TIMESTEP_MODE=groot_clean \
GR00T_EXECUTION_HORIZON=34 \
GR00T_PORT=22095 \
SERVE_GPU=0 \
bash scripts/deploy/fullstate_task1_gr00t_n17_sonic_eval.sh serve

------------------------ Task3 终端 2：Isaac Ego ------------------------

cd /home/ubuntu/yzh/Psi0_kimodo_textop

ISAAC_HEADLESS=1 \
bash scripts/deploy/fullstate_task3_gr00t_rot6d59_kimodo_textop_eval.sh isaac

------------------------ Task3 终端 3：SIMPLE Eval ----------------------

cd /home/ubuntu/yzh/Psi0_kimodo_textop

RUN_TAG=task3_sonic_seed0

PSI0_ROOT=/home/ubuntu/yzh/Psi0_kimodo_textop \
SIMPLE_ROOT=/home/ubuntu/yzh/Psi0_kimodo_textop/third_party/SIMPLE \
GR00T_ROOT=/home/ubuntu/yzh/Isaac-GR00T-rtc \
GR00T_MODEL_PATH=/home/ubuntu/yzh/ckpt/gr00tn17/task3_sonic \
GR00T_BACKBONE_PATH=/home/ubuntu/yzh/Isaac-GR00T-rtc/huggingface/Cosmos-Reason2-2B \
SONIC_DECODER_ONNX=/home/ubuntu/yzh/GR00T-WholeBodyControl/gear_sonic_deploy/policy/low_latency/model_decoder.onnx \
GR00T_PREFIX_RTC=1 \
GR00T_EXECUTION_HORIZON=34 \
GR00T_PORT=22095 \
GR00T_SONIC_TASK=G1Fullstate20260804Task3-v0 \
GR00T_SONIC_EVAL_DIR="/home/ubuntu/yzh/Psi0_kimodo_textop/third_party/SIMPLE/data/evals_task3_sonic_external_isaac_${RUN_TAG}" \
HUMANOID_VLA_MJ_ROOT=/home/ubuntu/yzh/HumanoidVLA_MJ_backup/HumanoidVLA_MJ \
TASK3_RECORDINGS_DIR=/home/ubuntu/yzh/mujoco_recordings/20260804_task3_new \
TASK3_RECORDING_SEED=0 \
SIM_MODE=mujoco_external_isaac \
EXTERNAL_ISAAC_UDP_PORT=23331 \
EXTERNAL_ISAAC_EGO_FRAME_PATH=/dev/shm/simple_task3_isaac_ego.frame \
EXTERNAL_ISAAC_EGO_WIDTH=640 \
EXTERNAL_ISAAC_EGO_HEIGHT=480 \
EXTERNAL_ISAAC_CAMERA_LOCAL_EYE="0.06 0.06 0.45" \
MAX_EPISODE_STEPS=800 \
SKIP_STABILIZE=1 \
GR00T_INITIAL_POSE_STEPS=0 \
EVAL_GPU=0 \
NUM_EPISODES=10 \
EPISODE_START=0 \
SAVE_VIDEO=1 \
bash scripts/deploy/fullstate_task1_gr00t_n17_sonic_eval.sh eval

# 固定 recording：增加 TASK3_RECORDING_INDEX=0。
# 手动续跑：保持 RUN_TAG 不变，同时设置 EPISODE_START=<已完成数>、
# NUM_EPISODES=<剩余数>。


=======================================================================
Task4：把瓶子放入绿色箱子
=======================================================================

Checkpoint:
  /home/ubuntu/yzh/ckpt/gr00tn17/task4_sonic

Prompt:
  Walk forward and put the bottle into the box.

场景/相机：Scene3 102344280，Ego local eye=(0.06,0.06,0.45)

------------------------ Task4 终端 1：GR00T ----------------------------

cd /home/ubuntu/yzh/Psi0_kimodo_textop

PSI0_ROOT=/home/ubuntu/yzh/Psi0_kimodo_textop \
SIMPLE_ROOT=/home/ubuntu/yzh/Psi0_kimodo_textop/third_party/SIMPLE \
GR00T_ROOT=/home/ubuntu/yzh/Isaac-GR00T-rtc \
GR00T_MODEL_PATH=/home/ubuntu/yzh/ckpt/gr00tn17/task4_sonic \
GR00T_BACKBONE_PATH=/home/ubuntu/yzh/Isaac-GR00T-rtc/huggingface/Cosmos-Reason2-2B \
GR00T_PREFIX_RTC=1 \
GR00T_PREFIX_RTC_TIMESTEP_MODE=groot_clean \
GR00T_EXECUTION_HORIZON=34 \
GR00T_PORT=22095 \
SERVE_GPU=0 \
bash scripts/deploy/fullstate_task1_gr00t_n17_sonic_eval.sh serve

------------------------ Task4 终端 2：Isaac Ego ------------------------

cd /home/ubuntu/yzh/Psi0_kimodo_textop

ISAAC_HEADLESS=1 \
bash scripts/deploy/fullstate_task4_gr00t_rot6d59_kimodo_textop_eval.sh isaac

------------------------ Task4 终端 3：SIMPLE Eval ----------------------

cd /home/ubuntu/yzh/Psi0_kimodo_textop

RUN_TAG=task4_sonic_seed0

PSI0_ROOT=/home/ubuntu/yzh/Psi0_kimodo_textop \
SIMPLE_ROOT=/home/ubuntu/yzh/Psi0_kimodo_textop/third_party/SIMPLE \
GR00T_ROOT=/home/ubuntu/yzh/Isaac-GR00T-rtc \
GR00T_MODEL_PATH=/home/ubuntu/yzh/ckpt/gr00tn17/task4_sonic \
GR00T_BACKBONE_PATH=/home/ubuntu/yzh/Isaac-GR00T-rtc/huggingface/Cosmos-Reason2-2B \
SONIC_DECODER_ONNX=/home/ubuntu/yzh/GR00T-WholeBodyControl/gear_sonic_deploy/policy/low_latency/model_decoder.onnx \
GR00T_PREFIX_RTC=1 \
GR00T_PREFIX_RTC_TIMESTEP_MODE=groot_clean \
GR00T_EXECUTION_HORIZON=34 \
GR00T_PORT=22095 \
GR00T_SONIC_TASK=G1Fullstate20260729Task4-v0 \
GR00T_SONIC_EVAL_DIR="/home/ubuntu/yzh/Psi0_kimodo_textop/third_party/SIMPLE/data/evals_task4_sonic_external_isaac_${RUN_TAG}" \
HUMANOID_VLA_MJ_ROOT=/home/ubuntu/yzh/HumanoidVLA_MJ_backup/HumanoidVLA_MJ \
TASK4_RECORDINGS_DIR=/home/ubuntu/yzh/mujoco_recordings/20260729_task4 \
TASK4_RECORDING_SEED=0 \
SIM_MODE=mujoco_external_isaac \
EXTERNAL_ISAAC_UDP_PORT=23331 \
EXTERNAL_ISAAC_EGO_FRAME_PATH=/dev/shm/simple_task4_isaac_ego.frame \
EXTERNAL_ISAAC_EGO_WIDTH=640 \
EXTERNAL_ISAAC_EGO_HEIGHT=480 \
EXTERNAL_ISAAC_CAMERA_LOCAL_EYE="0.06 0.06 0.45" \
MAX_EPISODE_STEPS=900 \
SKIP_STABILIZE=1 \
GR00T_INITIAL_POSE_STEPS=0 \
EVAL_GPU=0 \
NUM_EPISODES=10 \
EPISODE_START=0 \
SAVE_VIDEO=1 \
bash scripts/deploy/fullstate_task1_gr00t_n17_sonic_eval.sh eval

# 固定 recording：增加 TASK4_RECORDING_INDEX=0。
# 手动续跑：保持 RUN_TAG 不变，同时设置 EPISODE_START=<已完成数>、
# NUM_EPISODES=<剩余数>。


=======================================================================
输出与注意事项
=======================================================================

1. 每次评测输出 results.jsonl、summary.json、eval_stats.txt 和视频。
2. 视频文件名带 success 或 failed，可直接判断结果。
3. Task2/3/4 的 checkpoint 都是 unitree_g1_sonic、state46、action78、
   action horizon 40；客户端每 34 步重新规划，最后 6 步只作为 overlap。
4. 每条 episode 的第一次 GR00T 请求会发送 history={"reset": true}，
   不继承上一条 episode 的 Prefix-RTC tail。
5. Isaac 只负责渲染 Ego；MuJoCo 始终负责动作执行、碰撞与成功判定。
6. Task2 可用 RESUME=1 自动续跑；Task3/4 当前按上面的 EPISODE_START 和
   NUM_EPISODES 手动续跑。

TASK234_SONIC_EVAL_COMMANDS

printf '%s\n' \
  '这是 Task2--Task4 原生 GR00T -> SONIC 三终端评测命令备忘录。' \
  '请打开本文件，按对应任务的终端 1 -> 2 -> 3 顺序复制执行。'
