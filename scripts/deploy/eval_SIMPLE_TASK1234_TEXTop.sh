#!/usr/bin/env bash
set -euo pipefail

# Task1--Task4 SIMPLE + GR00T + Kimodo TRT + TextOp + Isaac-Ego
# 四终端评测命令备忘录（2026-08-19）
#
# 这是命令备忘录，不是一键拉起脚本。请分别复制对应任务的四段命令，
# 按终端 1 -> 2 -> 3 -> 4 的顺序启动。
#
# 链路边界：
#   GR00T -> Kimodo(scq_final TensorRT) -> TextOp -> SIMPLE/MuJoCo
#                                                |
#                                                +-> UDP 23331 -> Isaac Sim
#                                                             -> 640x480 Ego
#                                                             -> SIMPLE observation
#   SIMPLE/MuJoCo 是动作执行、碰撞和成功判定的唯一权威；Isaac 只镜像状态和渲染。
#
# 本机当前只有 GPU 0 可用，因此下面四个终端都写 GPU 0。
# 多 GPU 机器可自行改为 SERVE_GPU=1、KIMODO_GPU=2、EVAL_GPU=3。
# Task1--Task4 共用 22085/22185/23331，不能同时运行；切换任务前先在旧任务四个终端 Ctrl+C。

: <<'TASK1_TASK4_SIMPLE_EVAL_COMMANDS'

================================================================================
Task1：向前走并抓起绿色瓶子
================================================================================

Prompt:
  walk forward and then pick up the green cylinder

配置：
  SIMPLE env : simple/G1Fullstate20260615Task1-v0
  Recording  : /home/ubuntu/yzh/mujoco_recordings/20260615_task1_new
  GR00T ckpt : /home/ubuntu/yzh/ckpt/gr00tn17/task1_newbg
  HSSD       : Scene3 / 102344280.usd
  Ego        : local eye=(0.06,0.06,0.40), pitch=15deg, VFOV=70deg,
               640x480 @ 50Hz
  Lighting   : 与训练导出一致，seed=42+原始 recording index
  Scene move : task-translate=(0,0,0)
  成功规则   : 绿色圆柱相对初始高度抬升 >= 0.09m，连续保持 10 个控制步
  最大步数   : 800

------------------------------ 终端 1：GR00T ------------------------------

cd /home/ubuntu/yzh/Psi0_kimodo_textop

SERVE_GPU=0 \
bash scripts/deploy/fullstate_task1_gr00t_rot6d59_kimodo_textop_eval.sh serve

# 等待 22085 服务启动；可检查：
# curl -fsS http://127.0.0.1:22085/health

-------------------------- 终端 2：Kimodo TRT ----------------------------

cd /home/ubuntu/yzh/Psi0_kimodo_textop

KIMODO_GPU=0 \
bash scripts/deploy/fullstate_task1_gr00t_rot6d59_kimodo_textop_eval.sh kimodo-serve

# 必须确认 inference_backend=tensorrt、anchor_mode=policy_only、keyframe_step=10：
# curl -fsS http://127.0.0.1:22185/config

-------------------------- 终端 3：Isaac Ego -----------------------------

cd /home/ubuntu/yzh/Psi0_kimodo_textop

ISAAC_HEADLESS=1 \
bash scripts/deploy/fullstate_task1_gr00t_rot6d59_kimodo_textop_eval.sh isaac

# 如需世界相机和右上角 Ego 预览，将 ISAAC_HEADLESS 改为 0。
# Isaac 只启动一次，连续服务本次评测的全部 episode。

----------------------------- 终端 4：评测 -------------------------------

cd /home/ubuntu/yzh/Psi0_kimodo_textop

RUN_TAG=run001

EVAL_GPU=0 \
NUM_EPISODES=10 \
EPISODE_START=0 \
TASK1_RECORDING_SEED=0 \
SAVE_VIDEO=1 \
FULLSTATE_GR00T_EVAL_DIR="/home/ubuntu/yzh/Psi0_kimodo_textop/third_party/SIMPLE/data/evals_task1_external_isaac_${RUN_TAG}" \
FULLSTATE_GR00T_KIMODO_WORK_DIR="/home/ubuntu/yzh/Psi0_kimodo_textop/outputs/kimodo_task1_external_isaac_${RUN_TAG}" \
bash scripts/deploy/fullstate_task1_gr00t_rot6d59_kimodo_textop_eval.sh eval

# 固定第 0 条 recording：增加 TASK1_RECORDING_INDEX=0。
# 断点续跑：保持 RUN_TAG 不变，并在命令前增加 RESUME=1。


================================================================================
Task2：把桌上的瓶子投入垃圾桶
================================================================================

Prompt:
  Pick up the bottle on the table in front of me and throw it into the trash can.

配置：
  SIMPLE env : simple/G1Fullstate20260805Task2-v0
  Recording  : /home/ubuntu/yzh/mujoco_recordings/20260805_task2_new
  GR00T ckpt : /home/ubuntu/yzh/ckpt/gr00tn17/task2_newbg
  HSSD       : Scene3 / 102344280.usd
  Ego        : local eye=(0.06,0.06,0.45), pitch=15deg, VFOV=70deg,
               640x480 @ 50Hz
  Lighting   : 与训练导出一致，seed=42+原始 recording index
  Scene move : task-translate=(0,0,0)
  成功规则   : 瓶子完全位于垃圾桶局部内腔，线速度 <= 0.10m/s、
               角速度 <= 2.0rad/s，连续保持 10 个控制步
  最大步数   : 900

------------------------------ 终端 1：GR00T ------------------------------

cd /home/ubuntu/yzh/Psi0_kimodo_textop

SERVE_GPU=0 \
bash scripts/deploy/fullstate_task2_gr00t_rot6d59_kimodo_textop_eval.sh serve

# curl -fsS http://127.0.0.1:22085/health

-------------------------- 终端 2：Kimodo TRT ----------------------------

cd /home/ubuntu/yzh/Psi0_kimodo_textop

KIMODO_GPU=0 \
bash scripts/deploy/fullstate_task2_gr00t_rot6d59_kimodo_textop_eval.sh kimodo-serve

# curl -fsS http://127.0.0.1:22185/config

-------------------------- 终端 3：Isaac Ego -----------------------------

cd /home/ubuntu/yzh/Psi0_kimodo_textop

ISAAC_HEADLESS=1 \
bash scripts/deploy/fullstate_task2_gr00t_rot6d59_kimodo_textop_eval.sh isaac

# 如需世界相机和右上角 Ego 预览，将 ISAAC_HEADLESS 改为 0。

----------------------------- 终端 4：评测 -------------------------------

cd /home/ubuntu/yzh/Psi0_kimodo_textop

RUN_TAG=run001

EVAL_GPU=0 \
NUM_EPISODES=10 \
EPISODE_START=0 \
TASK2_RECORDING_SEED=0 \
SAVE_VIDEO=1 \
FULLSTATE_GR00T_EVAL_DIR="/home/ubuntu/yzh/Psi0_kimodo_textop/third_party/SIMPLE/data/evals_task2_external_isaac_${RUN_TAG}" \
FULLSTATE_GR00T_KIMODO_WORK_DIR="/home/ubuntu/yzh/Psi0_kimodo_textop/outputs/kimodo_task2_external_isaac_${RUN_TAG}" \
bash scripts/deploy/fullstate_task2_gr00t_rot6d59_kimodo_textop_eval.sh eval

# 固定第 0 条 recording：增加 TASK2_RECORDING_INDEX=0。
# 断点续跑：保持 RUN_TAG 不变，并在命令前增加 RESUME=1。

================================================================================
Task2 原生对照：GR00T -> SONIC（不经过 Kimodo/TextOp）
================================================================================

用途：
  使用与上面 Task2 完全一致的 SIMPLE/MuJoCo、recording、Isaac Scene3、
  640x480 Ego、Prompt 和成功规则，仅将动作执行链换成训练时的原生 SONIC。
  该链路用于判断失败是否来自 Kimodo/TextOp 的 root/reference 对齐。

配置：
  GR00T ckpt : /home/ubuntu/yzh/ckpt/gr00tn17/task2_sonic
  SONIC ONNX : /home/ubuntu/yzh/GR00T-WholeBodyControl/gear_sonic_deploy/policy/low_latency/model_decoder.onnx
  Prefix-RTC : prediction=40, execute=34, overlap=6
  GR00T port : 22095
  Isaac UDP  : 23331

------------------------ SONIC 终端 1：GR00T -------------------------------

cd /home/ubuntu/yzh/Psi0_kimodo_textop

SERVE_GPU=0 \
bash scripts/deploy/fullstate_task2_gr00t_n17_sonic_eval.sh serve

# 等待服务完成模型加载，并检查 40/34/6：
# curl -fsS http://127.0.0.1:22095/config

------------------------ SONIC 终端 2：Isaac Ego ---------------------------

cd /home/ubuntu/yzh/Psi0_kimodo_textop

ISAAC_HEADLESS=1 \
bash scripts/deploy/fullstate_task2_gr00t_n17_sonic_eval.sh isaac

# 该子命令复用上面四终端 Task2 的同一个 Isaac 启动实现。
# 如需世界相机和右上角 Ego 预览，将 ISAAC_HEADLESS 改为 0。

------------------------ SONIC 终端 3：SIMPLE Eval -------------------------

cd /home/ubuntu/yzh/Psi0_kimodo_textop

RUN_TAG=run001_sonic

EVAL_GPU=0 \
NUM_EPISODES=10 \
EPISODE_START=0 \
TASK2_RECORDING_SEED=0 \
SAVE_VIDEO=1 \
GR00T_SONIC_EVAL_DIR="/home/ubuntu/yzh/Psi0_kimodo_textop/third_party/SIMPLE/data/evals_task2_sonic_external_isaac_${RUN_TAG}" \
bash scripts/deploy/fullstate_task2_gr00t_n17_sonic_eval.sh eval

# 固定 recording：增加 TASK2_RECORDING_INDEX=0。
# 断点续跑：保持 RUN_TAG 不变并增加 RESUME=1。
# 原生链路只有三个终端，不能启动 Kimodo；也不能把 task2_newbg 权重放到这里。

================================================================================
Task3：踩踏板打开垃圾桶
================================================================================

Prompt:
  Step on the pedal to open the trash can in front of you.

配置：
  SIMPLE env : simple/G1Fullstate20260804Task3-v0
  Recording  : /home/ubuntu/yzh/mujoco_recordings/20260804_task3_new
  GR00T ckpt : /home/ubuntu/yzh/ckpt/gr00tn17/task3_newbg
  HSSD       : Scene13 / 102344250_local.usd
  Ego        : local eye=(0.06,0.06,0.45), pitch=15deg, VFOV=70deg,
               640x480 @ 50Hz
  Lighting   : 与训练导出一致，seed=42+原始 recording index
  Scene move : task-translate=(0.6,-0.5,0)
  成功规则   : 垃圾桶盖角度 >= 0.8 rad，连续保持 10 个控制步

------------------------------ 终端 1：GR00T ------------------------------

cd /home/ubuntu/yzh/Psi0_kimodo_textop

SERVE_GPU=0 \
bash scripts/deploy/fullstate_task3_gr00t_rot6d59_kimodo_textop_eval.sh serve

# 等待 22085 服务启动；可在另一个终端检查：
# curl -fsS http://127.0.0.1:22085/health

-------------------------- 终端 2：Kimodo TRT ----------------------------

cd /home/ubuntu/yzh/Psi0_kimodo_textop

KIMODO_GPU=0 \
bash scripts/deploy/fullstate_task3_gr00t_rot6d59_kimodo_textop_eval.sh kimodo-serve

# 必须确认配置返回：inference_backend=tensorrt、anchor_mode=policy_only、
# keyframe_step=10。
# curl -fsS http://127.0.0.1:22185/config

-------------------------- 终端 3：Isaac Ego -----------------------------

cd /home/ubuntu/yzh/Psi0_kimodo_textop

ISAAC_HEADLESS=1 \
bash scripts/deploy/fullstate_task3_gr00t_rot6d59_kimodo_textop_eval.sh isaac

# ISAAC_HEADLESS=0：显示世界相机，并在右上角显示 480x360 Ego 预览。
# ISAAC_HEADLESS=1：无 GUI，只输出训练/推理使用的 640x480 Ego 帧。
# 该终端会跨全部 episode 持续运行，不需要每条 recording 重启一次。

----------------------------- 终端 4：评测 -------------------------------

cd /home/ubuntu/yzh/Psi0_kimodo_textop

# 每次新评测修改 RUN_TAG，避免与旧结果混在一起。
RUN_TAG=run001_normal

EVAL_GPU=0 \
NUM_EPISODES=100 \
EPISODE_START=0 \
TASK3_RECORDING_SEED=0 \
SAVE_VIDEO=1 \
FULLSTATE_GR00T_EVAL_DIR="/home/ubuntu/yzh/Psi0_kimodo_textop/third_party/SIMPLE/data/evals_task3_external_isaac_${RUN_TAG}" \
FULLSTATE_GR00T_KIMODO_WORK_DIR="/home/ubuntu/yzh/Psi0_kimodo_textop/outputs/kimodo_task3_external_isaac_${RUN_TAG}" \
bash scripts/deploy/fullstate_task3_gr00t_rot6d59_kimodo_textop_eval.sh eval

# 固定调试第 0 条 recording：在上面增加 TASK3_RECORDING_INDEX=0。
# 从同一输出目录断点续跑：保持 RUN_TAG 不变，并在命令前增加 RESUME=1。
# 输出包括 results.jsonl、summary.json、eval_stats.txt 和视频。


================================================================================
Task4：把瓶子放进绿色箱子
================================================================================

Prompt:
  Walk forward and put the bottle into the box.

配置：
  SIMPLE env : simple/G1Fullstate20260729Task4-v0
  Recording  : /home/ubuntu/yzh/mujoco_recordings/20260729_task4
  GR00T ckpt : /home/ubuntu/yzh/ckpt/gr00tn17/task4_newbg
  HSSD       : Scene3 / 102344280.usd
  Ego        : local eye=(0.06,0.06,0.45), pitch=15deg, VFOV=70deg,
               640x480 @ 50Hz
  Lighting   : 与训练导出一致，seed=42+原始 recording index
  Scene move : task-translate=(0,0,0)
  成功规则   : 瓶子位于绿色箱子局部内腔且速度稳定，连续保持 8 个控制步

------------------------------ 终端 1：GR00T ------------------------------

cd /home/ubuntu/yzh/Psi0_kimodo_textop

SERVE_GPU=0 \
bash scripts/deploy/fullstate_task4_gr00t_rot6d59_kimodo_textop_eval.sh serve

# curl -fsS http://127.0.0.1:22085/health

-------------------------- 终端 2：Kimodo TRT ----------------------------

cd /home/ubuntu/yzh/Psi0_kimodo_textop

KIMODO_GPU=0 \
bash scripts/deploy/fullstate_task4_gr00t_rot6d59_kimodo_textop_eval.sh kimodo-serve

# curl -fsS http://127.0.0.1:22185/config

-------------------------- 终端 3：Isaac Ego -----------------------------

cd /home/ubuntu/yzh/Psi0_kimodo_textop

ISAAC_HEADLESS=1 \
bash scripts/deploy/fullstate_task4_gr00t_rot6d59_kimodo_textop_eval.sh isaac

# GUI 下应只有绿色箱子是绿色，桌子保持自己的原始材质。
# 启动日志应显示派生视觉 MJCF 使用了纹理重命名/remap；若仍看到整张绿桌，
# 先 Ctrl+C 关闭并重新启动本终端，避免继续使用旧 Isaac 派生场景缓存。

----------------------------- 终端 4：评测 -------------------------------

cd /home/ubuntu/yzh/Psi0_kimodo_textop

RUN_TAG=run001_test

EVAL_GPU=0 \
NUM_EPISODES=10 \
EPISODE_START=0 \
TASK4_RECORDING_SEED=0 \
SAVE_VIDEO=1 \
FULLSTATE_GR00T_EVAL_DIR="/home/ubuntu/yzh/Psi0_kimodo_textop/third_party/SIMPLE/data/evals_task4_external_isaac_${RUN_TAG}" \
FULLSTATE_GR00T_KIMODO_WORK_DIR="/home/ubuntu/yzh/Psi0_kimodo_textop/outputs/kimodo_task4_external_isaac_${RUN_TAG}" \
bash scripts/deploy/fullstate_task4_gr00t_rot6d59_kimodo_textop_eval.sh eval

# 固定调试第 0 条 recording：在上面增加 TASK4_RECORDING_INDEX=0。
# 从同一输出目录断点续跑：保持 RUN_TAG 不变，并在命令前增加 RESUME=1。


================================================================================
通用注意事项
================================================================================

1. 四个终端必须使用同一个任务 wrapper，不能把 Task1--Task4 混用。
2. 启动顺序固定：GR00T -> Kimodo -> Isaac -> eval。
3. Isaac 终端只启动一次，连续服务本次评测的所有 recording/episode。
4. eval 会先检查 recording、Kimodo TRT 配置和 Isaac Ego 帧；缺帧、旧帧、
   图像尺寸错误或策略服务退出都会 fail-fast，不会回退到 MuJoCo RGB。
5. 随机顺序由 TASK1_RECORDING_SEED ... TASK4_RECORDING_SEED 控制，为无放回排列。
6. EPISODE_START 用于从同一种子排列的指定位置开始；RESUME=1 会读取同一
   FULLSTATE_GR00T_EVAL_DIR/results.jsonl 并自动续跑。
7. 正常停止请在四个对应终端分别按 Ctrl+C；不要在一个任务服务未退出时直接
   启动另一任务，因为四个任务默认共用端口：GR00T 22085、Kimodo 22185、Isaac 23331。
8. Task2 原生 SONIC 对照只有 GR00T、Isaac、SIMPLE 三个终端，GR00T 使用 22095；
   它仍与四终端链路共用 Isaac UDP 23331 和共享帧文件，因此两套 eval 不可同时运行。

TASK1_TASK4_SIMPLE_EVAL_COMMANDS

printf '%s\n' \
  '这是 Task1--Task4 四终端评测命令备忘录。' \
  '请打开本文件，按对应任务的终端 1 -> 2 -> 3 -> 4 顺序复制执行。'
