# simple_psi0_kimodo_eval_commands.sh 说明

这个脚本用于 `Psi0 44D -> Kimodo -> SIMPLE eval`，并包含 TextOp tracker 分支。

脚本位置：

```bash
/pfs/pfs-ilWc5D/yzh/Psi0/scripts/deploy/simple_psi0_kimodo_eval_commands.sh
```

## 子命令后缀

| 子命令 | 作用 | 默认 agent | 默认 eval 输出 |
|---|---|---|---|
| `serve` | 启动 Psi0 action server | - | - |
| `health` | 检查 Psi0 server | - | - |
| `kimodo-serve` | 启动 Kimodo generation server | - | - |
| `download-data` | 下载 SIMPLE eval 数据 | - | `third_party/SIMPLE/data/evals/simple-eval` |
| `eval` | Psi0 + Kimodo + WBC 正常多 episode 测试 | `psi0_kimodo_decoupled_wbc` | `data/evals_decoupled_wbc_ep0_v2_44d` |
| `eval-one` | Psi0 + Kimodo + WBC 单 episode 测试 | `psi0_kimodo_decoupled_wbc` | 同 `eval` |
| `eval-textop` | Psi0 + Kimodo + TextOp tracker 多 episode 测试 | `psi0_kimodo_textop_tracker` | `data/evals_kimodo_textop_ep0_v2_44d` |
| `eval-textop-one` | Psi0 + Kimodo + TextOp tracker 单 episode 测试 | `psi0_kimodo_textop_tracker` | 同 `eval-textop` |
| `eval-textop-direct` | TextOp 分支，跳过 SIMPLE stabilize | `psi0_kimodo_textop_tracker` | `data/evals_kimodo_textop_direct_ep0_v2_44d` |
| `eval-textop-direct-one` | direct 单 episode | `psi0_kimodo_textop_tracker` | 同 `eval-textop-direct` |
| `eval-textop-initref` | TextOp 分支，跳过 stabilize，并把初态设到 Kimodo 第一帧 | `psi0_kimodo_textop_tracker` | `data/evals_kimodo_textop_initref_ep0_v2_44d` |
| `eval-textop-initref-one` | initref 单 episode | `psi0_kimodo_textop_tracker` | 同 `eval-textop-initref` |
| `stitch` | 拼接保存的 Kimodo chunk 为 50Hz qpos36 | - | `${KIMODO_WORK_DIR}/stitched_executed_qpos50.csv` |
| `stitch30` | 拼接保存的 Kimodo chunk 为 30Hz qpos36 | - | `${KIMODO_WORK_DIR}/stitched_executed_qpos30.csv` |

## 常用启动流程

终端 1，Psi0 server：

```bash
cd /pfs/pfs-ilWc5D/yzh/Psi0
SERVE_GPU=1 PORT=22085 bash scripts/deploy/simple_psi0_kimodo_eval_commands.sh serve
```

终端 2，Kimodo server：

```bash
cd /pfs/pfs-ilWc5D/yzh/Psi0
KIMODO_GPU=2 bash scripts/deploy/simple_psi0_kimodo_eval_commands.sh kimodo-serve
```

终端 3，TextOp 单 episode eval：

```bash
cd /pfs/pfs-ilWc5D/yzh/Psi0
EVAL_GPU=3 KIMODO_KEEP_WORK=1 bash scripts/deploy/simple_psi0_kimodo_eval_commands.sh eval-textop-one
```

## 全局默认项

| 变量 | 默认值 | 说明 |
|---|---|---|
| `PSI0_ROOT` | `/pfs/pfs-ilWc5D/yzh/Psi0` | Psi0 根目录 |
| `SIMPLE_ROOT` | `${PSI0_ROOT}/third_party/SIMPLE` | SIMPLE 根目录 |
| `KIMODO_ROOT` | `/pfs/pfs-ilWc5D/yzh/kimodo_my` | Kimodo 根目录 |
| `TASK` | `G1WholebodyXMovePickTeleop-v0` | SIMPLE task |
| `DR` | `level-0` | eval split/难度 |
| `ENTRY` | `eval_decoupled_wbc.py` | SIMPLE eval 入口 |
| `AGENT` | `psi0_kimodo_decoupled_wbc` | 默认 WBC agent |
| `RUN_DIR` | `.runs/finetune/movepick-kimodo-v2.simple.flow1000.cosine.lr1.0e-04.b64.gpus4.2606091729` | Psi0 checkpoint run dir |
| `CKPT_STEP` | `40000` | Psi0 checkpoint step |
| `SERVE_GPU` | `1` | Psi0 server 物理卡 |
| `KIMODO_GPU` | `2` | Kimodo server 物理卡 |
| `EVAL_GPU` | `3` | SIMPLE eval 物理卡 |
| `PORT` | `22085` | Psi0 server 端口 |
| `ACTION_EXEC_HORIZON` | `24` | 每次执行 policy chunk 的前 24 帧 |
| `USE_RTC` | `1` | Psi0 server 使用 RTC |
| `NUM_EPISODES` | `20` | 默认 eval episode 数 |
| `EPISODE_START` | `0` | 从第几个 episode 开始 |
| `MAX_EPISODE_STEPS` | `800` | 每个 episode 最大步数 |

## 离线 HuggingFace 默认项

| 变量 | 默认值 |
|---|---|
| `HF_HOME` | `${PSI0_ROOT}/huggingface` |
| `HF_HUB_CACHE` | `${HF_HOME}/hub` |
| `HUGGINGFACE_HUB_CACHE` | `${HF_HOME}/hub` |
| `HUGGINGFACE_CACHE_DIR` | `${HF_HOME}/hub` |
| `TRANSFORMERS_OFFLINE` | `1` |
| `HF_HUB_OFFLINE` | `1` |

`kimodo-serve` 会把这些 cache 路径切到 `${KIMODO_ROOT}/huggingface`。

## Kimodo 默认项

| 变量 | 默认值 | 说明 |
|---|---|---|
| `KIMODO_PYTHON` | `/pfs/pfs-ilWc5D/yuhao/miniconda3_4090/envs/kimodo/bin/python` | Kimodo 环境 Python |
| `KIMODO_DISTILL_CONFIG` | `${KIMODO_ROOT}/outputs/.../resolved_config.yaml` | Kimodo config |
| `KIMODO_DISTILL_CKPT` | `${KIMODO_ROOT}/outputs/.../ema_final.pt` | Kimodo checkpoint |
| `KIMODO_DIFFUSION_STEPS` | `20` | Kimodo 采样步数 |
| `KIMODO_KEYFRAME_STEP` | `10` | 50Hz policy 约束每 10 帧取关键帧 |
| `KIMODO_BLEND_FRAMES` | `4` | chunk 起始拼接 blend 帧数 |
| `KIMODO_ANCHOR_MODE` | `policy_only` | Kimodo root 平移锚点策略 |
| `KIMODO_WORK_DIR` | `${PSI0_ROOT}/outputs/kimodo_eval_ep0_v2_44d` | Kimodo chunk 保存目录 |
| `KIMODO_KEEP_WORK` | `0` | 是否保留 `chunk_*/g1_generated.*` |
| `KIMODO_EPISODE_SUBDIR` | `1` | 是否按 episode 分开保存为 `ep0000/chunk_*` |
| `KIMODO_SERVER_HOST` | `127.0.0.1` | Kimodo server host |
| `KIMODO_SERVER_PORT` | `22185` | Kimodo server port |
| `KIMODO_SERVER_URL` | `http://${KIMODO_SERVER_HOST}:${KIMODO_SERVER_PORT}` | Kimodo server URL |
| `KIMODO_SOURCE_FPS` | `50` | policy/Kimodo adapter 输入频率 |
| `KIMODO_FPS` | `30` | Kimodo 生成频率 |
| `KIMODO_OUTPUT_FPS` | `50` | 输出给 SIMPLE 的重采样频率 |
| `KIMODO_CFG_TYPE` | `separated` | Kimodo CFG 类型 |
| `KIMODO_CFG_WEIGHT_TEXT` | `2.0` | 文本 CFG 权重 |
| `KIMODO_CFG_WEIGHT_CONSTRAINT` | `2.0` | 约束 CFG 权重 |
| `KIMODO_PROMPT` | `a humanoid bends and picks up an object` | fallback prompt |
| `KIMODO_SEED` | 未设置 | 设置后固定 Kimodo seed |

base command 平滑/限幅默认：

| 变量 | 默认值 |
|---|---|
| `KIMODO_BASE_CMD_SMOOTH_WINDOW` | `5` |
| `KIMODO_MAX_ABS_VX` | `0.5` |
| `KIMODO_MAX_ABS_VY` | `0.08` |
| `KIMODO_MAX_ABS_VYAW` | `0.25` |

## TextOp 默认项

| 变量 | 默认值 | 说明 |
|---|---|---|
| `TEXTOP_ROOT` | `/pfs/pfs-ilWc5D/yzh` | textop 根路径前缀 |
| `TEXTOP_TRACKER_RUN` | `${TEXTOP_ROOT}/textop/2026-05-18_19-55-31_transformer_vae_eeobs_g1_before_2023` | tracker run |
| `TEXTOP_VAE_RUN` | `${TEXTOP_ROOT}/textop/2026-05-12_16-34-08_optitrack_npz_soma_before_2023` | VAE run |
| `TEXTOP_POLICY_ONNX` | `${TEXTOP_TRACKER_RUN}/exported/policy.onnx` | TextOp tracker ONNX |
| `TEXTOP_VAE_ONNX` | `${TEXTOP_VAE_RUN}/artifacts/motion_transformer_vae_encoder_z_c.onnx` | transformer VAE encoder ONNX |
| `TEXTOP_VAE_STATS` | `${TEXTOP_VAE_RUN}/artifacts/stats.npz` | VAE normalization stats |
| `TEXTOP_FUTURE_STEPS` | `10` | TextOp tracker 看未来 10 帧 ref |
| `TEXTOP_FPS` | `50` | TextOp adapter 计算速度用的 FPS |
| `TEXTOP_DEBUG` | `0` | 是否保存 TextOp debug npz |
| `TEXTOP_DEBUG_DIR` | `${PSI0_ROOT}/outputs/textop_debug` | debug 输出目录 |
| `TEXTOP_TARGET_RATE_LIMIT` | `0` | 是否限制 target_q 变化速度 |
| `TEXTOP_LEG_MAX_DELTA` | `0.03` | rate limit 腿部每步最大变化 |
| `TEXTOP_TORSO_MAX_DELTA` | `0.04` | rate limit torso 每步最大变化 |
| `TEXTOP_ARM_MAX_DELTA` | `0.08` | rate limit 手臂每步最大变化 |
| `TEXTOP_INIT_TO_REF` | `0` | 是否把初始仿真 qpos 直接设为 Kimodo 第一帧 |
| `TEXTOP_INIT_HOLD_ACTION` | `1` | initref 后是否先插入一帧 hold action |
| `TEXTOP_SKIP_STABILIZE` | `0` | 对 `eval-textop/eval-textop-one` 可选跳过 stabilize |

TextOp 分支默认输出目录：

| 子命令 | `EVAL_DIR` 默认 | `KIMODO_WORK_DIR` 默认 |
|---|---|---|
| `eval-textop`, `eval-textop-one` | `data/evals_kimodo_textop_ep0_v2_44d` | `${PSI0_ROOT}/outputs/kimodo_textop_ep0_v2_44d` |
| `eval-textop-direct`, `eval-textop-direct-one` | `data/evals_kimodo_textop_direct_ep0_v2_44d` | `${PSI0_ROOT}/outputs/kimodo_textop_direct_ep0_v2_44d` |
| `eval-textop-initref`, `eval-textop-initref-one` | `data/evals_kimodo_textop_initref_ep0_v2_44d` | `${PSI0_ROOT}/outputs/kimodo_textop_initref_ep0_v2_44d` |

## Fall-stop 自动停止

`eval-textop*` 分支默认打开：

```bash
FALL_STOP=1
```

检测条件：

| 变量 | 默认值 | 说明 |
|---|---|---|
| `FALL_STOP` | TextOp 分支默认 `1`，普通 eval 默认 `0` | 是否倒地提前结束 episode |
| `FALL_MIN_ROOT_Z` | `0.45` | root 高度低于该值则停止 |
| `FALL_MAX_ABS_RP` | `0.9` | root roll/pitch 绝对值超过该值则停止 |

关闭：

```bash
FALL_STOP=0 bash scripts/deploy/simple_psi0_kimodo_eval_commands.sh eval-textop-one
```

## Stitch 默认项

`stitch` 用于把保留下来的 `chunk_*/g1_generated.csv` 拼接成完整 qpos36。

如果 `KIMODO_EPISODE_SUBDIR=1`，chunk 会保存为：

```text
${KIMODO_WORK_DIR}/ep0000/chunk_000000/g1_generated.csv
${KIMODO_WORK_DIR}/ep0001/chunk_000000/g1_generated.csv
...
```

此时可以用 `STITCH_EPISODE=0` 拼接第 0 个 episode。

| 变量 | 默认值 | 说明 |
|---|---|---|
| `STITCH_EPISODE` | 空 | 设置为 `0` 或 `ep0000` 时拼接对应 episode 子目录 |
| `STITCH_EXEC_FRAMES` | `${ACTION_EXEC_HORIZON}`，默认 `24` | 每个 50Hz chunk 取多少帧 |
| `STITCH_BLEND_FRAMES` | `${KIMODO_BLEND_FRAMES}`，默认 `4` | chunk 边界 blend 帧 |
| `STITCH_OUTPUT` | `${KIMODO_WORK_DIR}/stitched_executed_qpos50.csv` | 50Hz 拼接 csv |
| `STITCH_PLOT` | `${KIMODO_WORK_DIR}/stitched_root.png` | root 轨迹图 |
| `STITCH_START_CHUNK` | 空 | 只拼接起始 chunk |
| `STITCH_END_CHUNK` | 空 | 只拼接结束 chunk |
| `STITCH_NUM_CHUNKS` | 空 | 最多拼接几个 chunk |

`stitch30` 默认：

| 变量 | 默认值 |
|---|---|
| `STITCH30_FRAMES_PER_CHUNK` | `15` |
| `STITCH30_BLEND_FRAMES` | `2` |
| `STITCH30_OUTPUT` | `${KIMODO_WORK_DIR}/stitched_executed_qpos30.csv` |
| `STITCH30_PLOT` | `${KIMODO_WORK_DIR}/stitched_root_30hz.png` |

## 例子

TextOp 单次测试，保留 Kimodo chunk：

```bash
cd /pfs/pfs-ilWc5D/yzh/Psi0

EVAL_GPU=3 \
KIMODO_KEEP_WORK=1 \
bash scripts/deploy/simple_psi0_kimodo_eval_commands.sh eval-textop-one
```

TextOp rot6d 修复后建议的单次测试，单独保存目录：

```bash
cd /pfs/pfs-ilWc5D/yzh/Psi0

TEXTOP_DEBUG=1 \
TEXTOP_DEBUG_DIR=/pfs/pfs-ilWc5D/yzh/Psi0/outputs/textop_debug_rot6d_fixed \
TEXTOP_EVAL_DIR=data/evals_kimodo_textop_rot6d_fixed \
TEXTOP_KIMODO_WORK_DIR=/pfs/pfs-ilWc5D/yzh/Psi0/outputs/kimodo_textop_rot6d_fixed \
FALL_STOP=1 \
EVAL_GPU=3 \
KIMODO_KEEP_WORK=1 \
bash scripts/deploy/simple_psi0_kimodo_eval_commands.sh eval-textop-initref-one
```

拼接该次 Kimodo chunk：

```bash
cd /pfs/pfs-ilWc5D/yzh/Psi0

KIMODO_WORK_DIR=/pfs/pfs-ilWc5D/yzh/Psi0/outputs/kimodo_textop_rot6d_fixed \
STITCH_EPISODE=0 \
bash scripts/deploy/simple_psi0_kimodo_eval_commands.sh stitch
```
