# Task2/3/4 新背景 SIMPLE Eval 复现说明

本文档用于在另一台机器上复现 Psi0 的 Task2、Task3、Task4 新背景评测链路：

```
GR00T N1.7 server -> Kimodo generation server -> SIMPLE decoupled WBC eval
                                              -> MuJoCo + Isaac Sim 渲染
```

## 1. 必需仓库和目录

| 用途 | 当前目录 | 是否必需 |
|---|---|---|
| Psi0 主工程、部署脚本 | `/mnt/pfs/humanoid/yzh/Psi0` | 必需 |
| SIMPLE 环境、Isaac Sim 4.5、CuRobo source | `Psi0/third_party/SIMPLE` | 必需 |
| GR00T N1.7 server 和 checkpoint | `/mnt/pfs/humanoid/yzh/Isaac-GR00T` | 必需 |
| Kimodo policy/server 和 checkpoint | `/mnt/pfs/humanoid/yzh/kimodo_my` | 必需 |
| TextOp policy、ONNX、VAE 统计文件 | `/mnt/pfs/humanoid/yzh/textop` 及其上级目录 | 必需 |
| SIMPLE Task2/3/4 资产 | `/mnt/pfs/humanoid/yzh/assets` | 必需 |
| NVIDIA 驱动用户态库 | 宿主机 `/lib/x86_64-linux-gnu/libcuda.so.1` | 必需 |

严格新背景评测默认不需要 `HumanoidVLA_MJ` 录制目录。`third_party/SIMPLE/third_party/curobo` 已随 SIMPLE 提供；XRoboToolkit 不是严格 eval 的必需依赖。

## 2. 版本和运行时

- Python 3.10
- Isaac Sim `4.5.0`
- PyTorch `2.7.0+cu128`
- Pinocchio `2.7.0`
- 宿主 NVIDIA 驱动用户态库必须能被容器加载
- 不要设置全局 `LD_PRELOAD` 注入 Isaac/cmeel 的 URDF 库

推荐使用 SIMPLE 官方 Nix runtime：

```bash
cd /mnt/pfs/humanoid/yzh/Psi0/third_party/SIMPLE
env -u LD_LIBRARY_PATH nix --extra-experimental-features "nix-command flakes" develop
```

无 Nix 时：

```bash
cd /mnt/pfs/humanoid/yzh/Psi0/third_party/SIMPLE
source .venv/bin/activate
export PYTHONPATH="$PWD/third_party/curobo/src:${PYTHONPATH:-}"
mkdir -p .runtime/host-libcuda
ln -sfn /lib/x86_64-linux-gnu/libcuda.so.1 .runtime/host-libcuda/libcuda.so
export LD_LIBRARY_PATH="$PWD/.runtime/host-libcuda:/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
python -c 'import simple.envs; import ctypes; ctypes.CDLL("libcuda.so"); print("SIMPLE runtime OK")'
```

## 3. SIMPLE 任务环境

```
Task2: simple/G1Fullstate20260805Task2IsaacEval-v0
Task3: simple/G1Fullstate20260804Task3IsaacEval-v0
Task4: simple/G1Fullstate20260729Task4IsaacEval-v0
```

预检：

```bash
cd /mnt/pfs/humanoid/yzh/Psi0/third_party/SIMPLE
export SIMPLE_TASK_ASSETS_ROOT=/mnt/pfs/humanoid/yzh/assets
PYTHONPATH=src uv run --no-sync \
  python scripts/task234_reproduction_preflight.py \
  --task all --manifest-dir outputs/task234_manifests
```

三个任务都应输出 `OK checkpoint-only inference assets`。

## 4. New background checkpoint

```
Task2: /mnt/pfs/humanoid/yzh/Isaac-GR00T/outputs/task2-gr00t-n17-rot6d59-kimodo-textop-prefix-rtc-groot-clean-A800-newbg/checkpoint-80000
Task3: /mnt/pfs/humanoid/yzh/Isaac-GR00T/outputs/task3-gr00t-n17-rot6d59-kimodo-textop-prefix-rtc-groot-clean-A800-newbg/checkpoint-80000
Task4: /mnt/pfs/humanoid/yzh/Isaac-GR00T/outputs/task4-gr00t-n17-rot6d59-kimodo-textop-prefix-rtc-groot-clean-A800-newbg/checkpoint-80000
```

## 5. 三终端启动顺序（Task2 示例）

### 终端 A：GR00T

```bash
cd /mnt/pfs/humanoid/yzh/Psi0
source .venv/bin/activate
SERVE_GPU=0 GR00T_PORT=22096 \
GR00T_PREFIX_RTC=1 GR00T_USE_RTC=1 \
GR00T_PREFIX_RTC_TIMESTEP_MODE=groot_clean \
GR00T_EXECUTION_HORIZON=34 \
GR00T_MODEL_PATH=/mnt/pfs/humanoid/yzh/Isaac-GR00T/outputs/task2-gr00t-n17-rot6d59-kimodo-textop-prefix-rtc-groot-clean-A800-newbg/checkpoint-80000 \
bash scripts/deploy/fullstate_task2_gr00t_rot6d59_kimodo_textop_eval.sh serve
```

### 终端 B：Kimodo/TextOp

```bash
cd /mnt/pfs/humanoid/yzh/Psi0
source .venv/bin/activate
KIMODO_GPU=1 KIMODO_KEYFRAME_STEP=1 \
FULLSTATE_TASK2_GR00T_KIMODO_WORK_DIR=/mnt/pfs/humanoid/yzh/Psi0/outputs/kimodo_task2_newbg_ckpt80000 \
bash scripts/deploy/fullstate_task2_gr00t_rot6d59_kimodo_textop_eval.sh kimodo-serve
```

### 终端 C：SIMPLE strict Isaac eval

```bash
cd /mnt/pfs/humanoid/yzh/Psi0
EVAL_GPU=2 GR00T_PORT=22096 \
TASK234_STRICT_ISAAC_EVAL=1 \
SIMPLE_TASK_ASSETS_ROOT=/mnt/pfs/humanoid/yzh/assets \
NUM_EPISODES=10 EPISODE_START=0 MAX_EPISODE_STEPS=700 \
SAVE_VIDEO=1 GR00T_EXECUTION_HORIZON=34 SKIP_STABILIZE=1 \
FULLSTATE_TASK2_GR00T_KIMODO_WORK_DIR=/mnt/pfs/humanoid/yzh/Psi0/outputs/kimodo_task2_newbg_ckpt80000 \
FULLSTATE_TASK2_GR00T_EVAL_DIR=data/evals_task2_20260805_newbg_ckpt80000_10eps \
bash scripts/deploy/fullstate_task2_gr00t_rot6d59_kimodo_textop_eval.sh eval
```

`NUM_EPISODES=10` 表示评测 10 条 episode，每条最多 700 steps。

## 6. Task3/Task4 替换

Task3 使用环境 `G1Fullstate20260804Task3IsaacEval-v0`、脚本 `fullstate_task3_gr00t_rot6d59_kimodo_textop_eval.sh`、Task3 checkpoint、`kimodo_task3_newbg_ckpt80000` 和 `data/evals_task3_20260804_newbg_ckpt80000_10eps`。

Task4 使用环境 `G1Fullstate20260729Task4IsaacEval-v0`、脚本 `fullstate_task4_gr00t_rot6d59_kimodo_textop_eval.sh`、Task4 checkpoint、`kimodo_task4_newbg_ckpt80000` 和 `data/evals_task4_20260729_newbg_ckpt80000_10eps`。

## 7. 故障定位

- `ModuleNotFoundError: curobo.types`：确认 `PYTHONPATH` 包含 `third_party/curobo/src`。
- `Could not load libcuda.so`：创建 `.runtime/host-libcuda/libcuda.so -> /lib/x86_64-linux-gnu/libcuda.so.1`。
- `liblula_kinematics.so: undefined symbol`：清除 `LD_PRELOAD`，不要全局 preload URDF 库。
- SIMPLE 显示 `0/10` 且 step 为 `-`：查看 `third_party/SIMPLE/data/<eval-dir>/eval_latest.log`；若停在 `rtx driver verification failed`，是 Isaac 渲染 runtime 问题，不是 GR00T 策略问题。

## 8. 现有 memo

```
/mnt/pfs/humanoid/yzh/Psi0/eval_task234_newbg.sh
```

迁移时优先修改该文件中的 `PSI0_ROOT`、`GR00T_ROOT`、`SIMPLE_TASK_ASSETS_ROOT`、Kimodo/TextOp 路径和 checkpoint 路径。
