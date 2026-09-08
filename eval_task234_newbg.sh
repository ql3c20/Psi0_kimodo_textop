# Task2/3/4 新背景 SIMPLE Eval 终端备忘录
#
# 本文件只用于复制各终端命令，不要把整个文件作为脚本一次性执行。
# 每个任务都要先启动对应的 GR00T 服务和 Kimodo/TextOp 服务，再运行
# SIMPLE eval。MuJoCo 是唯一物理源；Isaac 只渲染严格复现场景和
# 640x480 第一视角 RGB。

# ============================================================================
# 终端0：首次补全 SIMPLE 依赖并预检 Task2/3/4
# ============================================================================
```bash
cd /pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Psi0/third_party/SIMPLE

source .venv/bin/activate

# XRoboToolkit 只用于真机 XR 遥操作，本评测不需要，因此跳过该包。
uv sync --active \
  --group dev \
  --group sonic \
  --no-install-package xrobotoolkit-sdk

export SIMPLE_TASK_ASSETS_ROOT=/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/assets
PYTHONPATH=src uv run --no-sync \
  python scripts/task234_reproduction_preflight.py \
  --task all \
  --manifest-dir outputs/task234_manifests
```


# ############################################################################
# Task2：把瓶子放入垃圾桶，Scene3 / 102344280
# ############################################################################

# ============================================================================
# Task2 终端1：启动 GR00T 服务（GPU 0，端口 22096）
# ============================================================================
```bash
cd /pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Psi0

GR00T_PYTHON=/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Isaac-GR00T/.venv/bin/python \
SERVE_GPU=0 \
GR00T_PORT=22096 \
GR00T_PREFIX_RTC=1 \
GR00T_USE_RTC=1 \
GR00T_PREFIX_RTC_TIMESTEP_MODE=groot_clean \
GR00T_EXECUTION_HORIZON=34 \
GR00T_MODEL_PATH=/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Isaac-GR00T/outputs/task2-gr00t-n17-rot6d59-kimodo-textop-prefix-rtc-groot-clean-A800-newbg/checkpoint-80000 \
bash scripts/deploy/fullstate_task2_gr00t_rot6d59_kimodo_textop_eval.sh serve
```

# ============================================================================
# Task2 终端2：启动 Kimodo/TextOp 服务（GPU 1）
# ============================================================================
```bash
cd /pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Psi0

KIMODO_GPU=1 \
KIMODO_KEYFRAME_STEP=1 \
FULLSTATE_TASK2_GR00T_KIMODO_WORK_DIR=/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Psi0/outputs/kimodo_task2_newbg_ckpt80000 \
bash scripts/deploy/fullstate_task2_gr00t_rot6d59_kimodo_textop_eval.sh kimodo-serve
```

# ============================================================================
# Task2 终端3：运行严格新背景 SIMPLE eval（GPU 2）
# ============================================================================
```bash
cd /pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Psi0

EVAL_GPU=2 \
GR00T_PORT=22096 \
TASK234_STRICT_ISAAC_EVAL=1 \
SIMPLE_TASK_ASSETS_ROOT=/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/assets \
NUM_EPISODES=10 \
EPISODE_START=0 \
MAX_EPISODE_STEPS=700 \
SAVE_VIDEO=1 \
GR00T_EXECUTION_HORIZON=34 \
SKIP_STABILIZE=1 \
FULLSTATE_TASK2_GR00T_KIMODO_WORK_DIR=/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Psi0/outputs/kimodo_task2_newbg_ckpt80000 \
FULLSTATE_TASK2_GR00T_EVAL_DIR=data/evals_task2_20260805_newbg_ckpt80000_10eps \
bash scripts/deploy/fullstate_task2_gr00t_rot6d59_kimodo_textop_eval.sh eval
```


# ############################################################################
# Task3：踩下垃圾桶踏板，Scene13 / 102344250
# ############################################################################

# ============================================================================
# Task3 终端1：启动 GR00T 服务（GPU 0，端口 22096）
# ============================================================================
```bash
cd /pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Psi0

GR00T_PYTHON=/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Isaac-GR00T/.venv/bin/python \
SERVE_GPU=0 \
GR00T_PORT=22096 \
GR00T_PREFIX_RTC=1 \
GR00T_USE_RTC=1 \
GR00T_PREFIX_RTC_TIMESTEP_MODE=groot_clean \
GR00T_EXECUTION_HORIZON=34 \
GR00T_MODEL_PATH=/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Isaac-GR00T/outputs/task3-gr00t-n17-rot6d59-kimodo-textop-prefix-rtc-groot-clean-A800-newbg/checkpoint-80000 \
bash scripts/deploy/fullstate_task3_gr00t_rot6d59_kimodo_textop_eval.sh serve
```

# ============================================================================
# Task3 终端2：启动 Kimodo/TextOp 服务（GPU 1）
# ============================================================================
```bash
cd /pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Psi0

KIMODO_GPU=1 \
KIMODO_KEYFRAME_STEP=1 \
FULLSTATE_TASK3_GR00T_KIMODO_WORK_DIR=/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Psi0/outputs/kimodo_task3_newbg_ckpt80000 \
bash scripts/deploy/fullstate_task3_gr00t_rot6d59_kimodo_textop_eval.sh kimodo-serve
```

# ============================================================================
# Task3 终端3：运行严格新背景 SIMPLE eval（GPU 2）
# ============================================================================
```bash
cd /pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Psi0

EVAL_GPU=2 \
GR00T_PORT=22096 \
TASK234_STRICT_ISAAC_EVAL=1 \
SIMPLE_TASK_ASSETS_ROOT=/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/assets \
NUM_EPISODES=10 \
EPISODE_START=0 \
MAX_EPISODE_STEPS=500 \
SAVE_VIDEO=1 \
GR00T_EXECUTION_HORIZON=34 \
SKIP_STABILIZE=1 \
FULLSTATE_TASK3_GR00T_KIMODO_WORK_DIR=/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Psi0/outputs/kimodo_task3_newbg_ckpt80000 \
FULLSTATE_TASK3_GR00T_EVAL_DIR=data/evals_task3_20260804_newbg_ckpt80000_10eps \
bash scripts/deploy/fullstate_task3_gr00t_rot6d59_kimodo_textop_eval.sh eval
```


# ############################################################################
# Task4：把瓶子放入绿色箱子，Scene3 / 102344280
# ############################################################################

# ============================================================================
# Task4 终端1：启动 GR00T 服务（GPU 0，端口 22096）
# ============================================================================
```bash
cd /pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Psi0

GR00T_PYTHON=/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Isaac-GR00T/.venv/bin/python \
SERVE_GPU=0 \
GR00T_PORT=22096 \
GR00T_PREFIX_RTC=1 \
GR00T_USE_RTC=1 \
GR00T_PREFIX_RTC_TIMESTEP_MODE=groot_clean \
GR00T_EXECUTION_HORIZON=34 \
GR00T_USE_TRT=0 \
GR00T_MODEL_PATH=/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Isaac-GR00T/outputs/task4-gr00t-n17-rot6d59-kimodo-textop-prefix-rtc-groot-clean-A800-newbg/checkpoint-80000 \
bash scripts/deploy/fullstate_task4_gr00t_rot6d59_kimodo_textop_eval.sh serve
```

# ============================================================================
# Task4 终端2：启动 Kimodo/TextOp 服务（GPU 1）
# ============================================================================
```bash
cd /pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Psi0

KIMODO_GPU=1 \
KIMODO_KEYFRAME_STEP=1 \
FULLSTATE_TASK4_GR00T_KIMODO_WORK_DIR=/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Psi0/outputs/kimodo_task4_newbg_ckpt80000 \
bash scripts/deploy/fullstate_task4_gr00t_rot6d59_kimodo_textop_eval.sh kimodo-serve
```

# ============================================================================
# Task4 终端3：运行严格新背景 SIMPLE eval（GPU 2）
# ============================================================================
```bash
cd /pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Psi0

EVAL_GPU=2 \
GR00T_PORT=22096 \
TASK234_STRICT_ISAAC_EVAL=1 \
SIMPLE_TASK_ASSETS_ROOT=/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/assets \
NUM_EPISODES=10 \
EPISODE_START=0 \
MAX_EPISODE_STEPS=700 \
SAVE_VIDEO=1 \
GR00T_EXECUTION_HORIZON=34 \
SKIP_STABILIZE=1 \
GR00T_USE_TRT=0 \
FULLSTATE_TASK4_GR00T_KIMODO_WORK_DIR=/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Psi0/outputs/kimodo_task4_newbg_ckpt80000 \
FULLSTATE_TASK4_GR00T_EVAL_DIR=data/evals_task4_20260729_newbg_ckpt80000_10eps \
bash scripts/deploy/fullstate_task4_gr00t_rot6d59_kimodo_textop_eval.sh eval
```

注意事项：
- 三个任务的 GR00T 服务都使用端口 22096，因此同一时间只运行一个任务链路。
- GR00T 使用 GPU 0，Kimodo/TextOp 使用 GPU 1，SIMPLE/Isaac 使用 GPU 2。
- 将 SAVE_VIDEO=1 改为 SAVE_VIDEO=0 可关闭视频保存。
- 严格模式会选择已注册的 Isaac eval Gym ID，并使用任务自身的固定初始化；
  不要在这里添加旧版 TASK*_RECORDINGS_DIR 环境变量。







============================================================================
环境备份与旧成功链路依赖安装（手工执行，不属于 eval 链路）
============================================================================
先备份当前 SIMPLE 环境：

cd /pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Psi0/third_party/SIMPLE
BACKUP_DIR="/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/SIMPLE_env_backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"
cp -a .venv "$BACKUP_DIR/venv"
cp -a pyproject.toml uv.lock flake.nix flake.lock "$BACKUP_DIR/" 2>/dev/null || true
.venv/bin/python -m pip freeze > "$BACKUP_DIR/pip-freeze.txt"
env | sort > "$BACKUP_DIR/environment.txt"
ldconfig -p 2>/dev/null | grep -E 'libcuda|libnvidia-ml' > "$BACKUP_DIR/nvidia-libs.txt"
tar -czf "${BACKUP_DIR}.tar.gz" -C "$(dirname "$BACKUP_DIR")" "$(basename "$BACKUP_DIR")"

在新机器上安装旧成功链路的 SIMPLE 开发依赖。不要安装 sonic 组：

cd /pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Psi0/third_party/SIMPLE
unset LD_PRELOAD PYTHONPATH PYTHONHOME
uv sync --python 3.10 --group dev --no-install-package xrobotoolkit-sdk

补充 CuRobo source、Pinocchio 和 PhysX 的 libcuda soname：

source .venv/bin/activate
export PYTHONPATH="$PWD/third_party/curobo/src:${PYTHONPATH:-}"
python -m pip install "numpy==1.26.4" "pin==2.7.0"
mkdir -p .runtime/host-libcuda
ln -sfn /lib/x86_64-linux-gnu/libcuda.so.1 .runtime/host-libcuda/libcuda.so
export LD_LIBRARY_PATH="$PWD/.runtime/host-libcuda:/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"

验证基础导入：

python - <<'PY'
import ctypes
import numpy
import pinocchio
import simple.envs
ctypes.CDLL("libcuda.so")
print("numpy:", numpy.__version__)
print("pinocchio:", pinocchio.__version__)
print("SIMPLE import: OK")
print("libcuda: OK")
PY

验证 Task2/3/4 资产：

export SIMPLE_TASK_ASSETS_ROOT=/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/assets
PYTHONPATH=src uv run --no-sync \
  python scripts/task234_reproduction_preflight.py \
  --task all --manifest-dir outputs/task234_manifests

不要执行：
uv sync --active --group dev --group sonic
uv pip install --no-build-isolation -e third_party/curobo
