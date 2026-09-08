# Project Describe

## 项目范围

当前主链路是 Humanoid loco-manipulation 的 `GR00T VLA + Kimodo distill + TextOp tracker`：

```text
RGB + state49 + instruction
  -> GR00T N1.7 policy server
  -> 40 x 59D VLA plan
  -> Kimodo 蒸馏动作生成器
  -> 40 帧 qpos36 全身参考
  -> TextOp tracker
  -> 29D body target + hand14 target
  -> SIMPLE / MuJoCo 控制
```

结论：本仓库没有在 `src/psi` 下重新定义 GR00T VLA。VLA 主体来自相邻仓库 `../Isaac-GR00T`；本仓库负责 rot6d59 数据/schema、GR00T HTTP bridge、Prefix-RTC runtime adapter、Kimodo server wrapper，以及 SIMPLE 侧 TextOp tracker 接入。

## 环境与源码边界（2026-08-22）

不要使用终端里的 `(base)` 启动训练、TensorRT 转换或评测；当前 `(base)` 是 Python 3.14，不符合两套 GR00T 源码要求。两套 GR00T 虽然包名相同，但源码和环境不能混用：

| 链路 | 源码 | Python | 用途 |
| --- | --- | --- | --- |
| 本项目 rot6d59 | `/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Isaac-GR00T` | `yzh/Isaac-GR00T/.venv/bin/python`（3.10.20） | 微调、Prefix-RTC、TensorRT 导出/构建、GR00T server |
| 原生 GR00T-Sonic | `wzl/Psi0/third_party/Isaac-GR00T-N1.7-General-Release` | `wzl/Psi0/.venv-gr00t-n17-sonic-4gpu/bin/python`（3.12.13） | 原生 Sonic 微调、TensorRT、GR00T server；操作说明见 `wzl/Psi0/finetune-note.md` |
| Kimodo | `/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/kimodo_my` | `yzh/miniconda3/envs/kimodo/bin/python`（3.10.20） | Kimodo generation server |
| SIMPLE / TextOp | `third_party/SIMPLE` | `third_party/SIMPLE/.venv/bin/python`（3.10.20） | MuJoCo、IsaacSim、SONIC decoder、TextOp ONNX tracker |

当前指定代码树中共有 11 个不同的环境目录；`yzh/Isaac-GR00T/.venv.bak-py310-20260731 -> .venv` 只是兼容软链，不另算一个环境：

- 当前主链路可用：上表四个环境，以及不被当前链路直接调用的 `yzh/miniconda3/envs/text_tracker`。
- 不完整或失效：`yzh/Psi0/.venv`、`yzh/Psi0/.venv-psi`、`yzh/Psi0/.venv-dp`、`yzh/Psi0/src/gr00t/.venv`、`yzh/miniconda3/envs/ardy`、`yzh/miniconda3/envs/humi_tracker`。
- 原生 Sonic 的 SIMPLE eval wrapper 默认使用 `/usr/bin/python3.10` 并注入 yzh SIMPLE 的 site-packages；它是系统解释器，不计入虚拟环境数量。
- 当前 TextOp 直接由 SIMPLE 进程通过 ONNX Runtime 执行，不需要单独启动 `text_tracker` 环境。

## VLA 定义位置

GR00T VLA 的核心定义在 `../Isaac-GR00T`：

- `../Isaac-GR00T/gr00t/policy/gr00t_policy.py`：`Gr00tPolicy`，负责 observation 校验、processor、model inference、action decode。
- `../Isaac-GR00T/gr00t/model/gr00t_n1d7/gr00t_n1d7.py`：`Gr00tN1d7` 和 `Gr00tN1d7ActionHead`，action head 是 flow-matching DiT / AlternateVLDiT。
- `../Isaac-GR00T/examples/unitree_g1_rot6d59_config.py`：本项目 `NEW_EMBODIMENT` 的 rot6d59 modality 配置。

本仓库的 GR00T 服务入口是 `scripts/deploy/gr00t_n17_rot6d59_server.py`。它加载 `Gr00tPolicy`，导入 rot6d59 modality，把 SIMPLE 侧 `state49` 中的 root RPY 转成 rot6d `state52`，最后返回完整 `(T, 59)` action chunk。

## State / Action Layout

SIMPLE 侧 policy state 是：

```text
state49 = hand14 + body29 + root6(xyz,rpy)
```

GR00T bridge 会转换成：

```text
state52 = hand14 + body29 + root9(xyz,rot6d)
```

`state52` 切片：

| Range | 含义 |
| --- | --- |
| `0:7` | left hand |
| `7:14` | right hand |
| `14:20` | left leg |
| `20:26` | right leg |
| `26:29` | waist |
| `29:36` | left arm |
| `36:43` | right arm |
| `43:52` | root xyz + rot6d |

GR00T 输出 action 是：

```text
action59 = hand14 + root9 + left_hand_pose9 + right_hand_pose9 + left_foot_pose9 + right_foot_pose9
```

`action59` 切片：

| Range | 含义 |
| --- | --- |
| `0:14` | hand14 |
| `14:23` | root xyz + rot6d |
| `23:32` | left hand / wrist pose xyz + rot6d |
| `32:41` | right hand / wrist pose xyz + rot6d |
| `41:50` | left foot pose xyz + rot6d |
| `50:59` | right foot pose xyz + rot6d |

## hand14 具体维度

rot6d59 VLA schema 的权威来源是 `scripts/data/build_movepick_rot6d59_dataset.py` 里的 `HAND14_ORDER`：

| Dim | Joint |
| --- | --- |
| `0` | `left_hand_thumb_0_joint` |
| `1` | `left_hand_thumb_1_joint` |
| `2` | `left_hand_thumb_2_joint` |
| `3` | `left_hand_middle_0_joint` |
| `4` | `left_hand_middle_1_joint` |
| `5` | `left_hand_index_0_joint` |
| `6` | `left_hand_index_1_joint` |
| `7` | `right_hand_thumb_0_joint` |
| `8` | `right_hand_thumb_1_joint` |
| `9` | `right_hand_thumb_2_joint` |
| `10` | `right_hand_index_0_joint` |
| `11` | `right_hand_index_1_joint` |
| `12` | `right_hand_middle_0_joint` |
| `13` | `right_hand_middle_1_joint` |

注意：SIMPLE 里部分 robot/controller 文件把手部“自然顺序”写成 thumb/index/middle，而 MuJoCo state extraction 注释里又提到 MJCF order。不要只改一个地方的 hand 顺序。若要调整 hand14 schema，必须同时检查并更新 dataset builder、modality metadata、server bridge、tracker hand slicing 和 robot control interpretation。

## Prefix RTC 核心 Adapter

最关键文件是 `scripts/deploy/gr00t_n17_prefix_rtc.py`。

Psi0 RTC 的核心思想是 action inpainting：训练时随机给模型一段 clean prefix，让模型只学习预测 suffix；推理时只执行当前 chunk 的前 `Ta` 帧，把未执行的尾部 `Tp-Ta` 帧作为下一次生成的 clean prefix。

本项目的实现是对已加载的 GR00T policy 做 runtime monkeypatch，不新增参数，原 GR00T 权重可以继续加载。它 patch 了：

- action encoder：支持逐 action `(B,T)` timestep。
- timestep encoder：支持逐 token timestep。
- AdaLayerNorm：支持 token-level `(B,T,D)` modulation。
- DiT / AlternateVLDiT forward：把逐 token timestep 传入 action token。
- action-head training loss：clean prefix 不参与 loss。
- action-head inference decode：denoise 过程中 hard rewrite prefix。

训练侧入口是 `_prefix_rtc_compute_loss`：

```text
随机 clean prefix 长度
  -> noisy trajectory 的 prefix 覆盖为 clean action
  -> prefix action timestep = clean bucket
  -> suffix action timestep = 当前 flow timestep
  -> DiT 接收逐 token timestep
  -> prefix loss mask = 0
```

`groot_clean` 模式把 prefix timestep 设为 `num_timestep_buckets - 1`；如果是 1000 个 bucket，就是 bucket `999`。这对应 GR00T flow 约定 `x_t = (1-t) * noise + t * action`，clean data 在 `t=1` 端点。

推理侧入口是 `_prefix_rtc_get_action_with_features`：

```text
读取上一轮 normalized chunk
  -> 提取上一 chunk 末尾 [Ta:Tp]
  -> 初始化当前 chunk 前 overlap 帧
  -> 每个 denoise step 前后都 hard rewrite prefix
  -> prefix 使用 clean bucket，suffix 使用当前 denoise timestep
```

`scripts/deploy/gr00t_n17_rot6d59_server.py` 负责缓存 `previous_normalized_action`，下一轮通过 `options["rtc_prev_action"]` 传回 action head。部署时用 `--prefix-rtc` 或 `GR00T_PREFIX_RTC=1` 启用。

## 部署链路

task1 主入口是 `scripts/deploy/fullstate_task1_gr00t_rot6d59_kimodo_textop_eval.sh`：

```bash
bash scripts/deploy/fullstate_task1_gr00t_rot6d59_kimodo_textop_eval.sh serve
bash scripts/deploy/fullstate_task1_gr00t_rot6d59_kimodo_textop_eval.sh kimodo-serve
bash scripts/deploy/fullstate_task1_gr00t_rot6d59_kimodo_textop_eval.sh eval
```

关键 RTC 配置：

- `GR00T_PREFIX_RTC=1`：启用 hard-prefix RTC adapter。
- `GR00T_PREFIX_RTC_TIMESTEP_MODE=groot_clean`：使用 GR00T clean endpoint bucket。
- `GR00T_EXECUTION_HORIZON=34`：下游实际执行 34 帧。
- GR00T modality 预测 40 帧，所以 overlap 是 `40 - 34 = 6` 帧。
- `POLICY_EXECUTION_HORIZON` 必须等于 `GR00T_EXECUTION_HORIZON`，否则 server 认为“已执行”的帧数会和 client 实际执行不一致。

`scripts/deploy/fullstate_task4_gr00t_rot6d59_kimodo_textop_eval.sh` 是 task4 wrapper，默认打开 prefix RTC、`groot_clean`、horizon 34，并指向 task4 checkpoint。

## Kimodo 与 TextOp

Kimodo 自己没有 RTC 修改。本仓库的 `scripts/deploy/kimodo_generation_server.py` 只负责加载 Kimodo distill student checkpoint，并暴露 `/generate`。

SIMPLE 侧主 agent 是 `third_party/SIMPLE/src/simple/baselines/psi0_kimodo_textop_tracker.py`。它先把 GR00T `59D` action 转成 legacy `44D`：

```text
59D: hand14 + root9 + 4 * pose9
44D: hand14 + root6 + 4 * pose6
```

转换方式是把 root/EE 的 rot6d 还原成 RPY。然后 `third_party/SIMPLE/src/simple/baselines/kimodo_adapter.py` 把 root/EE plan 转成 Kimodo constraints，调用 Kimodo 生成 `qpos36`。TextOp 路径中，hand14 直接来自 VLA 输出；TextOp 根据 Kimodo full-body reference 和当前机器人状态预测 29D body target。

## 架构注意点

这条链路更准确地说是“Psi0-style prefix RTC + receding-horizon execution”，不是完整异步版 Psi0 runtime。模型侧连续性已经实现：上一 chunk 的未执行尾部会条件化下一 chunk。部署 agent 仍然是在本地 action queue 为空时同步 query policy，因此 wall-clock asynchronous inference/execution 在这条链路里还不是完整实现。

调连续性时要分清三个 horizon：

- VLA prediction horizon `Tp = 40`。
- execution horizon `Ta = 34`。
- RTC overlap `Tp - Ta = 6`。

Kimodo 和 TextOp 始终接收完整 40 帧 chunk；只有最终仿真执行阶段会裁成前 34 个 tracker step，然后重新规划。


## GR00T N1.7 Prefix-RTC Clean Full TRT 全流程测试

当前推荐使用 clean full-pipeline TRT，不再使用 merged engine 目录。这个 TRT engine 不绑定 Task4 场景本身；它绑定的是 GR00T N1.7 rot6d59 checkpoint、Prefix-RTC 输入/输出形状、execution horizon 和 TensorRT 构建 profile。Task4 eval 脚本默认使用该 checkpoint，因此可以直接复用这个 engine。全链路 TRT 目录是：

```bash
/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Psi0/outputs/gr00t_trt/gr00t_n17_rot6d59_prefixrtc_ckpt160k_full_pipeline/engines
```

该目录来自一次完整 `prefix_rtc_full_pipeline` 导出和构建，包含 7 个 engine：

```text
vit.engine
llm_bf16.engine
vl_self_attention.engine
state_encoder.engine
action_encoder.engine
dit_bf16.engine
action_decoder.engine
```

数据验证结果：

```text
final action cosine = 0.999998
prefix cosine      = 1.000000
prefix Linf        = 0.000000
suffix cosine      = 0.999996
PASS -- Prefix-RTC TRT matches PyTorch
```

历史构建记录：2026-08-09 的 `prefix_rtc_full_pipeline` 和 2026-08-10 的 fused sampler 都使用当时名为 `.venv.bak-py310-20260731` 的 Python 3.10 环境。该物理环境现已迁移为正式的 `yzh/Isaac-GR00T/.venv`，旧名称只是兼容软链；今后的训练、TRT 转换和 server 命令统一写 `.venv` 或使用 `uv run --no-sync`。当前环境包含 TensorRT `10.15.1.29` 和 ONNX `1.20.1`。它没有安装 `onnxruntime`，因此日志中的 ORT 验证被跳过，但 TensorRT engine 构建及后续 PyTorch/TRT 数值验证均已通过。

`scripts/deploy/fullstate_task4_gr00t_rot6d59_kimodo_textop_eval.sh` 默认已经设置：

```bash
GR00T_USE_TRT=1
GR00T_TRT_MODE=prefix_rtc_full_pipeline
GR00T_TRT_ENGINE_DIR=$PSI0_ROOT/outputs/gr00t_trt/gr00t_n17_rot6d59_prefixrtc_ckpt160k_full_pipeline/engines
```

正常运行时不需要再手动指向 merged 目录。如果要临时关闭 TRT，用：

```bash
export GR00T_USE_TRT=0
```

### 终端 1：GR00T full TRT server

```bash
cd /pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Psi0

export PSI0_ROOT=/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Psi0
export GR00T_ROOT=/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Isaac-GR00T
export GR00T_PYTHON=$GR00T_ROOT/.venv/bin/python

export SERVE_GPU=6
export GR00T_PORT=22096
export GR00T_USE_TRT=1  # 设为0可使用原生的无TRT server
export GR00T_TRT_ENGINE_DIR=$PSI0_ROOT/outputs/gr00t_trt/gr00t_n17_rot6d59_prefixrtc_ckpt160k_full_pipeline/engines
export GR00T_TRT_MODE=prefix_rtc_full_pipeline

export TMPDIR=$PSI0_ROOT/outputs/tmp_trt_server
export CUDA_CACHE_PATH=$PSI0_ROOT/outputs/cuda_cache
mkdir -p "$TMPDIR" "$CUDA_CACHE_PATH"

bash scripts/deploy/fullstate_task4_gr00t_rot6d59_kimodo_textop_eval.sh serve
```

server 启动日志应出现：

```text
Loading ViT engine
Loading LLM engine
Loading VL Self-Attention engine
Deleted PyTorch vl_self_attention (replaced by TRT engine)
Prefix-RTC action head TRT engines loaded
TensorRT enabled: mode=prefix_rtc_full_pipeline
RTC enabled ... overlap=6
```

server 每次 `/act` 会打印 `policy.get_action` 推理耗时，例如：

```text
[gr00t-rot6d59-server] policy.get_action latency: request=..., last=... ms, mean20=... ms, req_hz=..., exec_fps=... (Ta=34), chunk_fps=... (Tp=40), mode=prefix_rtc_full_pipeline
```

如需看模型内部耗时，启动 GR00T server 前加：

```bash
export GR00T_LOG_TIMING_BREAKDOWN=1
export GR00T_TIMING_SYNC_CUDA=1  # 严格计时时打开；日常低开销观察可不设
```

日志会额外打印 `processor/collate/backbone/vl_sa/state/act_enc/dit/act_dec/sampler/post_decode/kv_entries/trt_launches`。

### 实验：Prefix-RTC fused sampler TRT

`prefix_rtc_full_pipeline` 当前使用 7 个 engine，action head 在 4 个 denoise step 中重复调用 `action_encoder.engine -> dit_bf16.engine -> action_decoder.engine`。实验模式 `prefix_rtc_full_pipeline_sampler` 会重新导出一个 `prefix_rtc_action_sampler.engine`，把 4-step sampler 融合成单个 TRT engine，并在图内复用 DiT cross-attention 的 encoder K/V。默认 eval 不会自动使用这个模式。

已构建并通过一次数据验证的目录：

```bash
/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Psi0/outputs/gr00t_trt/gr00t_n17_rot6d59_prefixrtc_ckpt160k_full_pipeline_sampler/engines
```

包含 5 个 engine：`vit.engine`、`llm_bf16.engine`、`vl_self_attention.engine`、`state_encoder.engine`、`prefix_rtc_action_sampler.engine`。验证结果：

```text
final action cosine = 0.999999
prefix cosine      = 1.000000
suffix cosine      = 0.999996
PASS -- Prefix-RTC TRT matches PyTorch
```

小型同输入 microbenchmark，GPU7，8 次计时、2 次 warmup、`GR00T_TIMING_SYNC_CUDA=1`：

```text
old prefix_rtc_full_pipeline:      mean_total=45.5 ms, sampler=19.3 ms, trt_launches=12
new prefix_rtc_full_pipeline_sampler: mean_total=41.7 ms, sampler=15.3 ms, trt_launches=1, kv_entries=16
```

导出/构建/验证：

```bash
cd /pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Isaac-GR00T
export CUDA_VISIBLE_DEVICES=6

uv run --no-sync python scripts/deployment/build_trt_pipeline.py \
  --model-path /pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Isaac-GR00T/outputs/task4-gr00t-n17-rot6d59-kimodo-textop-prefix-rtc-groot-clean/checkpoint-160000 \
  --dataset-path /pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Psi0/data/output/fullstate_20260729_task4_rot6d59 \
  --modality-config-path examples/unitree_g1_rot6d59_config.py \
  --embodiment-tag new_embodiment \
  --output-dir /pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Psi0/outputs/gr00t_trt/gr00t_n17_rot6d59_prefixrtc_ckpt160k_full_pipeline_sampler \
  --export-mode prefix_rtc_full_pipeline_sampler \
  --prefix-rtc-timestep-mode groot_clean \
  --rtc-overlap-steps 6
```

启动 fused sampler TRT server 时，把 engine 目录和 mode 换成：

```bash
export GR00T_TRT_ENGINE_DIR=$PSI0_ROOT/outputs/gr00t_trt/gr00t_n17_rot6d59_prefixrtc_ckpt160k_full_pipeline_sampler/engines
export GR00T_TRT_MODE=prefix_rtc_full_pipeline_sampler
```

### 终端 2：Kimodo server

```bash
cd /pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Psi0

export PSI0_ROOT=/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Psi0
export KIMODO_GPU=5

bash scripts/deploy/fullstate_task4_gr00t_rot6d59_kimodo_textop_eval.sh kimodo-serve
```

### 终端 3：SIMPLE / MuJoCo eval

先跑单条 debug：

```bash
cd /pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Psi0

export PSI0_ROOT=/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Psi0
export EVAL_GPU=4
export NUM_EPISODES=1
export SAVE_VIDEO=1
export TASK4_RECORDING_INDEX=0
export KIMODO_KEEP_WORK=1
export GR00T_PORT=22096

export TMPDIR=$PSI0_ROOT/outputs/tmp_simple_eval
mkdir -p "$TMPDIR"

bash scripts/deploy/fullstate_task4_gr00t_rot6d59_kimodo_textop_eval.sh eval
```

确认单条能正常跑完后，跑 20 episode：

```bash
cd /pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Psi0

export PSI0_ROOT=/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Psi0
unset TASK4_RECORDING_INDEX
export EVAL_GPU=4
export NUM_EPISODES=20
export SAVE_VIDEO=1
export TASK4_RECORDING_SEED=0
export KIMODO_KEEP_WORK=1
export GR00T_PORT=22096

export TMPDIR=$PSI0_ROOT/outputs/tmp_simple_eval
mkdir -p "$TMPDIR"

bash scripts/deploy/fullstate_task4_gr00t_rot6d59_kimodo_textop_eval.sh eval
```

eval 日志和视频目录在 SIMPLE 子仓库下，因为 eval 脚本会 `cd third_party/SIMPLE`：

```bash
/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Psi0/third_party/SIMPLE/data/evals_fullstate_20260729_task4_gr00t_rot6d59_kimodo_textop_prefixrtc_grootclean_checkpoint-160000_recording0_xoff0_yoff0
/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Psi0/third_party/SIMPLE/data/evals_fullstate_20260729_task4_gr00t_rot6d59_kimodo_textop_prefixrtc_grootclean_checkpoint-160000_recordingseed0_xoff0_yoff0
```

Kimodo 中间结果在 Psi0 根目录下：

```bash
/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Psi0/outputs/kimodo_fullstate_20260729_task4_gr00t_rot6d59_prefixrtc_grootclean_checkpoint-160000_recording0_xoff0_yoff0
/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Psi0/outputs/kimodo_fullstate_20260729_task4_gr00t_rot6d59_prefixrtc_grootclean_checkpoint-160000_recordingseed0_xoff0_yoff0
```

已记录的一次 full TRT 20 episode 结果：`40%`

```bash
/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Psi0/third_party/SIMPLE/data/evals_fullstate_20260729_task4_gr00t_rot6d59_kimodo_textop_prefixrtc_grootclean_checkpoint-160000_recordingseed0_xoff0_yoff0
```

## HumanoidArena 足球与搬箱子

### 数据位置

- 足球原始数据：`/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/dataset/HumanoidArena_football/HOI_football_v2`
- 足球 rot6d59：`/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Psi0/data/output/arena_football_rot6d59`（100 episodes，66,669 帧）
- 搬箱子原始数据：`/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/dataset/HumanoidArena_pp_box`
- 搬箱子 rot6d59：`/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Psi0/data/output/arena_pp_box_rot6d59`（200 episodes，115,053 帧）
- 搬箱子 Sonic-only rot6d59：`/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Psi0/data/output/arena_pp_box_sonic_rot6d59_v3`（100 episodes，69,176 帧）
- 搬箱子 Sonic-only native realized：`/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Psi0/data/output/arena_pp_box_sonic_native_realized_v1`（100 episodes，69,176 帧；由 `robot_qpos_before_decimation` 重新编码）

搬箱子和足球使用相同的 `state52/action59` modality 与 schema，可以共用下面的训练配置。任务文本分别是 `Move toward the football and kick it.` 和 `Pick up the box and place it on the shelf.`。

### Sonic-only 搬箱子 8 卡串行微调

统一入口是 `scripts/train/gr00t/train_arena_pp_box_sonic_8gpu_sequential.sh`。它固定使用 8 卡、global batch 256、20,000 steps，先训练原生 Sonic，再训练 rot6d59 Prefix-RTC；第一条失败时不会启动第二条。rot6d59 使用 `groot_clean`，RTC delay 为 `[0, 12]`。两条链路都启用 W&B，并自动从输出目录中最新的 `checkpoint-*` 续训；达到 20,000 steps 的链路会自动跳过。首次训练前先运行 `scripts/data/convert_arena_pp_box_sonic_v2_all.sh` 生成 realized native 和 rot6d59 v3 数据集。

```bash
cd /pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Psi0

# 只检查环境、数据与配置，不启动训练
PREFLIGHT_ONLY=1 \
  bash scripts/train/gr00t/train_arena_pp_box_sonic_8gpu_sequential.sh

# 依次训练 native SONIC -> rot6d59 Prefix-RTC
bash scripts/train/gr00t/train_arena_pp_box_sonic_8gpu_sequential.sh
```

原生 Sonic 以 wzl preset `finetune_arena_pp_box_sonic_native_v2_8gpu_bs256_step20000.yaml` 为基础；统一入口会显式覆盖 realized 数据集、experiment name、batch、steps 和保存间隔。rot6d59 的对应设置和 `TRAIN_RTC_MIN_DELAY=0`、`TRAIN_RTC_MAX_DELAY=12` 也在统一入口脚本中。global batch 256 在 8 卡下等于每卡 forward batch 32，gradient accumulation 为 1。

默认输出：

```text
/pfs/pfs-oHNwH0/mnt/pfs/humanoid/wzl/Psi0/checkpoints/gr00t-n17-sonic-arena-pp-box-sonic-native-realized-v1-8gpu-bs256-step20000
/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Isaac-GR00T/outputs/arena-pp-box-sonic-gr00t-n17-rot6d59-v3-prefixrtc-delay0to12-8gpu-bs256-step20000
```

### GR00T N1.7 微调

先在同一终端选择一组路径。

足球：

```bash
DATASET_PATH=/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Psi0/data/output/arena_football_rot6d59
OUTPUT_DIR=/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Isaac-GR00T/outputs/arena-football-gr00t-n17-rot6d59
```

搬箱子：

```bash
DATASET_PATH=/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Psi0/data/output/arena_pp_box_rot6d59
OUTPUT_DIR=/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Isaac-GR00T/outputs/arena-pp-box-gr00t-n17-rot6d59
```

启动训练：

```bash
cd /pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Isaac-GR00T
CUDA_VISIBLE_DEVICES=4,5,6,7 \
NUM_GPUS=4 MASTER_PORT=29531 \
MAX_STEPS=160000 SAVE_STEPS=10000 \
GLOBAL_BATCH_SIZE=64 DATALOADER_NUM_WORKERS=4 \
TRAIN_PREFIX_RTC=1 GR00T_PREFIX_RTC_TIMESTEP_MODE=groot_clean \
GR00T_BACKBONE_PATH=/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Psi0/huggingface/hub/models--nvidia--Cosmos-Reason2-2B/snapshots/9ce19a195e423419c349abfc86fd07178b230561 \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 USE_WANDB=1 \
uv run --no-sync bash examples/finetune.sh \
  --base-model-path /pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Psi0/checkpoints/GR00T-N1.7-3B \
  --dataset-path "$DATASET_PATH" \
  --modality-config-path /pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Isaac-GR00T/examples/unitree_g1_rot6d59_config.py \
  --embodiment-tag NEW_EMBODIMENT \
  --output-dir "$OUTPUT_DIR" \
  --wandb-project gr00t-n1.7
```


只用 twist2 搬箱子数据（100 条带完整视频的原始记录）：
```bash
cd /pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Isaac-GR00T
CUDA_VISIBLE_DEVICES=4,5,6,7 \
NUM_GPUS=4 MASTER_PORT=29531 \
MAX_STEPS=160000 SAVE_STEPS=10000 \
GLOBAL_BATCH_SIZE=32 DATALOADER_NUM_WORKERS=4 \
TRAIN_PREFIX_RTC=1 \
GR00T_PREFIX_RTC_TIMESTEP_MODE=groot_clean \
GR00T_BACKBONE_PATH=/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Psi0/huggingface/hub/models--nvidia--Cosmos-Reason2-2B/snapshots/9ce19a195e423419c349abfc86fd07178b230561 \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
USE_WANDB=1 \
uv run --no-sync bash examples/finetune.sh \
  --base-model-path /pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Psi0/checkpoints/GR00T-N1.7-3B \
  --dataset-path /pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Psi0/data/output/arena_pp_box_twist2_rot6d59 \
  --modality-config-path /pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Isaac-GR00T/examples/unitree_g1_rot6d59_config.py \
  --embodiment-tag NEW_EMBODIMENT \
  --output-dir /pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Isaac-GR00T/outputs/arena-pp-box-twist2-gr00t-n17-rot6d59 \
  --wandb-project gr00t-n1.7
```
续训示例（保持与原训练相同的数据、batch size 和 Prefix-RTC 配置）：

```bash
cd /pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Isaac-GR00T
CUDA_VISIBLE_DEVICES=4,5,6,7 \
NUM_GPUS=4 \
MASTER_PORT=29531 \
MAX_STEPS=160000 \
SAVE_STEPS=10000 \
GLOBAL_BATCH_SIZE=32 \
DATALOADER_NUM_WORKERS=4 \
TRAIN_PREFIX_RTC=1 \
GR00T_PREFIX_RTC_TIMESTEP_MODE=groot_clean \
GR00T_BACKBONE_PATH=/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Psi0/huggingface/hub/models--nvidia--Cosmos-Reason2-2B/snapshots/9ce19a195e423419c349abfc86fd07178b230561 \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
USE_WANDB=1 \
WANDB_RUN_ID=73jgz047 \
WANDB_RESUME=allow \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
uv run --no-sync bash examples/finetune.sh \
  --base-model-path /pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Psi0/checkpoints/GR00T-N1.7-3B \
  --dataset-path /pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Psi0/data/output/arena_pp_box_twist2_rot6d59 \
  --modality-config-path /pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Isaac-GR00T/examples/unitree_g1_rot6d59_config.py \
  --embodiment-tag NEW_EMBODIMENT \
  --output-dir /pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Isaac-GR00T/outputs/arena-pp-box-twist2-gr00t-n17-rot6d59 \
  --wandb-project gr00t-n1.7 \
  --resume-from-checkpoint
```

若端口或 GPU 已被占用，修改 `MASTER_PORT` 或 `CUDA_VISIBLE_DEVICES`；续训时在命令末尾加 `--resume-from-checkpoint`。

### 足球闭环测试

在 Psi0 根目录分别启动三个终端。`GR00T_PORT` 必须在 serve 和 eval 中一致；这里用 `22196` 避免与其他任务的 `22096` 冲突。

```bash
# 终端 1：足球 GR00T
GR00T_PORT=22196 SERVE_GPU=1 \
  bash scripts/deploy/fullstate_arena_football_gr00t_rot6d59_kimodo_textop_eval.sh serve

# 终端 2：Kimodo
KIMODO_GPU=2 \
  bash scripts/deploy/fullstate_arena_football_gr00t_rot6d59_kimodo_textop_eval.sh kimodo-serve

# 终端 3：SIMPLE MuJoCo + Isaac 闭环；固定使用0作为种子初始化场景
GR00T_PORT=22196 EVAL_GPU=3 NUM_EPISODES=1  ARENA_FOOTBALL_RECORDING_SEED=0 \
  bash scripts/deploy/fullstate_arena_football_gr00t_rot6d59_kimodo_textop_eval.sh eval
```

默认 checkpoint 是 `arena-football-gr00t-n17-rot6d59/checkpoint-160000`。结果保存在：

```text
third_party/SIMPLE/data/evals_arena_football_gr00t_rot6d59_kimodo_textop_<checkpoint>_<init_tag>/
```

可用 `ARENA_FOOTBALL_GR00T_EVAL_DIR=/absolute/path` 覆盖结果目录。

### twist2 搬箱子闭环测试

该环境只从原始 `twist2/yb` 中加载 NPZ 与 `vision_rgb_video_path` 均完整的 100 条记录；17 条缺视频记录和 117 条重录分支都不会参与初始化。场景并非在连续空间任意随机摆放：每个 episode 从这些训练记录读取第 0 帧的机器人、箱子和货架位姿，默认用 `ARENA_PP_BOX_RECORDING_SEED=0` 做无重复确定性排列，所有 box/shelf offset 默认为 0。因此当前成功率属于训练分布初始状态上的闭环评测，不是 held-out 泛化结果。MuJoCo 负责机器人、箱子、桌子和货架的物理碰撞，Isaac 仅渲染并同步 MuJoCo 位姿。成功条件是箱子完整落在货架中层、底面高度对齐且连续 8 帧稳定。

在 Psi0 根目录启动三个终端：

```bash
# 可先检查 checkpoint、seed、horizon 和结果目录
bash scripts/deploy/fullstate_arena_pp_box_gr00t_rot6d59_kimodo_textop_eval.sh dry-run

# 终端 1：搬箱子 GR00T；默认使用 checkpoint-160000
GR00T_PORT=22196 SERVE_GPU=1 \
  bash scripts/deploy/fullstate_arena_pp_box_gr00t_rot6d59_kimodo_textop_eval.sh serve

# 终端 2：Kimodo
KIMODO_GPU=2 \
  bash scripts/deploy/fullstate_arena_pp_box_gr00t_rot6d59_kimodo_textop_eval.sh kimodo-serve

# 终端 3：SIMPLE MuJoCo + Isaac；固定使用完整记录中的第 0 条
GR00T_PORT=22196 EVAL_GPU=3 NUM_EPISODES=1 ARENA_PP_BOX_RECORDING_INDEX=0 \
  bash scripts/deploy/fullstate_arena_pp_box_gr00t_rot6d59_kimodo_textop_eval.sh eval
```

训练尚未到 160000 step 时，可以在 serve 和 eval 两端同时覆盖 checkpoint，例如：

```bash
GR00T_MODEL_PATH=/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Isaac-GR00T/outputs/arena-pp-box-twist2-gr00t-n17-rot6d59/checkpoint-10000
```

不设置 `ARENA_PP_BOX_RECORDING_INDEX` 时，100 条完整记录按固定 seed 无重复轮换。rot6d59 GR00T 每次预测 40 帧、执行 34 帧，并把剩余 6 帧作为 Prefix-RTC 条件带入下一次预测。结果和逐 episode/总成功率默认保存在：

```text
third_party/SIMPLE/data/evals_arena_pp_box_gr00t_rot6d59_kimodo_textop_<checkpoint>_<init_tag>/
third_party/SIMPLE/data/evals_arena_pp_box_gr00t_rot6d59_kimodo_textop_<checkpoint>_<init_tag>/eval_stats.txt
```

可用 `ARENA_PP_BOX_GR00T_EVAL_DIR=/absolute/path` 覆盖结果目录。

与原生 Sonic 做公平的单场景比较时，两边设置相同的 `ARENA_PP_BOX_RECORDING_INDEX`；做完整 100 条比较时，两边都不设置 index，并统一使用 `ARENA_PP_BOX_RECORDING_SEED=0 NUM_EPISODES=100 EPISODE_START=0`。原生 Sonic 使用相同场景顺序，但当前执行策略是预测 40、执行 30、直接丢弃后 10 帧，不使用 Prefix-RTC。

### HumanoidArena 原生环境中的 rot6d59 + Kimodo + TextOp

原生 PP-box/football adapter 位于 HumanoidArena 的 `action_provider/action_provider_rot6d59_textop.py`。它从 Isaac Lab 直接构造训练时的 `state49`，调用本项目 rot6d59 server 得到 `40 x 59D` 动作，再沿用原来的 Kimodo 和 TextOp，最后按 joint name 把 29D body 与 14D hand target 写回 Isaac Lab。该路径不经过 Sonic decoder。wrapper 接口是 `{serve|kimodo-serve|eval|ground-truth|all|all-ground-truth|dry-run} [pp_box|football]`；省略任务参数时仍默认 `pp_box`。

先用训练集 episode 0 做 ground-truth gate；它只启动 Kimodo，不启动 GR00T：

```bash
cd /pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Psi0
EVAL_GPU=1 KIMODO_GPU=2 GROUND_TRUTH_MAX_STEPS=421 \
  bash scripts/deploy/humanoidarena_gr00t_n17_rot6d59_kimodo_textop_eval.sh all-ground-truth
```

测试 checkpoint（默认 checkpoint-160000、seed 0、1 episode）：

```bash
cd /pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Psi0
SERVE_GPU=0 EVAL_GPU=1 KIMODO_GPU=2 \
  bash scripts/deploy/humanoidarena_gr00t_n17_rot6d59_kimodo_textop_eval.sh all
```

50 episodes 示例：

```bash
SERVE_GPU=0 EVAL_GPU=1 KIMODO_GPU=2 \
EVAL_SEEDS="0 1 2 3 4" REPEATS_PER_SEED=10 MAX_STEPS=1300 \
RESULTS_DIR=$PWD/evals/humanoidarena_pp_box_rot6d59_checkpoint160000_50ep \
  bash scripts/deploy/humanoidarena_gr00t_n17_rot6d59_kimodo_textop_eval.sh all
```

足球先做训练集 recording 0 的 ground-truth gate 和 checkpoint 冒烟：

```bash
cd /pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Psi0

EVAL_GPU=1 KIMODO_GPU=2 GROUND_TRUTH_MAX_STEPS=387 \
GROUND_TRUTH_RESULTS_DIR=$PWD/evals/humanoidarena_football_rot6d59_ground_truth_recording0 \
  bash scripts/deploy/humanoidarena_gr00t_n17_rot6d59_kimodo_textop_eval.sh all-ground-truth football

ARENA_EVAL_PROFILE=recording0 SERVE_GPU=0 EVAL_GPU=1 KIMODO_GPU=2 \
EVAL_SEEDS=0 REPEATS_PER_SEED=1 MAX_STEPS=1300 \
RESULTS_DIR=$PWD/evals/humanoidarena_football_rot6d59_checkpoint160000_recording0_smoke1 \
  bash scripts/deploy/humanoidarena_gr00t_n17_rot6d59_kimodo_textop_eval.sh all football
```

冒烟通过后，用训练数据位姿范围内的确定性场景测 50 episodes：

```bash
ARENA_EVAL_PROFILE=random SERVE_GPU=0 EVAL_GPU=1 KIMODO_GPU=2 \
EVAL_SEEDS="0 1 2 3 4" REPEATS_PER_SEED=10 MAX_STEPS=1300 \
RESULTS_DIR=$PWD/evals/humanoidarena_football_rot6d59_checkpoint160000_train_range_50ep \
  bash scripts/deploy/humanoidarena_gr00t_n17_rot6d59_kimodo_textop_eval.sh all football
```

`recording0`、`random`、`benchmark_random` 分别对应固定的训练记录 0、训练数据位姿范围、HumanoidArena benchmark 位姿范围。足球默认 checkpoint 是 `arena-football-gr00t-n17-rot6d59/checkpoint-160000`，任务文本是 `Move toward the football and kick it.`。

rot6d59 的 Prefix-RTC 保持 `predict 40 / execute 34 / carry 6`。逐 episode JSON 会额外记录 `initial_box_z`、`max_box_z` 和 `max_box_lift_m`；结果汇总和视频沿用 HumanoidArena 原生 runner 的 `episodes/`、`videos/`、`summary.json` 与 `final.csv` 目录结构。

### 原生 GR00T-Sonic

原生 Sonic 的训练、TensorRT 和闭环评测统一记录在 `/pfs/pfs-oHNwH0/mnt/pfs/humanoid/wzl/Psi0/finetune-note.md`。不要在本文件复制手工 `PYTHONPATH` 命令；使用 wzl wrapper，避免把原生 Python 3.12 server 与 rot6d59 Python 3.10 环境混用。

```bash
#测试自研链路开门
cd /pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Psi0
GR00T_PYTHON=/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Isaac-GR00T/.venv/bin/python \
GR00T_MODEL_PATH=/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Isaac-GR00T/outputs/arena-open-door-sonic-gr00t-n17-rot6d59-v1-prefixrtc-delay0to12-4gpu-bs256-step20000/checkpoint-10000 \
ARENA_EVAL_PROFILE=random \
SERVE_GPU=0 \
EVAL_GPU=1 \
KIMODO_GPU=2 \
GR00T_EXECUTION_HORIZON=30 \
EVAL_SEEDS="0 1 2 3 4" \
REPEATS_PER_SEED=10 \
MAX_STEPS=1800 \
RECORD_VIDEO_EVERY_N=5 \
RESULTS_DIR=$PWD/evals/humanoidarena_open_door_rot6d59_ckpt10000_random_eval50-video \
bash scripts/deploy/humanoidarena_gr00t_n17_rot6d59_kimodo_textop_eval.sh all open_door


#测试sonic开门
cd /pfs/pfs-oHNwH0/mnt/pfs/humanoid/wzl/Psi0

checkpoints/gr00t-n17-sonic-arena-open-door-sonic-native-realized-v1-8gpu-bs512-step10000/checkpoint-10000

GR00T_MODEL_PATH=$PWD/checkpoints/gr00t-n17-sonic-arena-open-door-sonic-native-v2-4gpu-bs256-step20000/checkpoint-20000 \
ARENA_EVAL_PROFILE=random \
SERVE_GPU=0 \
EVAL_GPU=1 \
GR00T_PORT=18443 \
SONIC_VLA_EXECUTION_HORIZON=30 \
EVAL_SEEDS="0 1 2 3 4" \
REPEATS_PER_SEED=10 \
MAX_STEPS=1800 \
RECORD_VIDEO_EVERY_N=5 \
RESULTS_DIR=$PWD/evals/humanoidarena_open_door_sonic_checkpoint20000_random_eval50 \
bash scripts/deploy/humanoidarena_gr00t_n17_sonic_native_eval.sh open_door


#重新测试新转换的数据的sonic开门
cd /pfs/pfs-oHNwH0/mnt/pfs/humanoid/wzl/Psi0
GR00T_MODEL_PATH=$PWD/checkpoints/gr00t-n17-sonic-arena-open-door-sonic-native-realized-v1-8gpu-bs512-step10000/checkpoint-10000 \
ARENA_EVAL_PROFILE=random \
SERVE_GPU=0 \
EVAL_GPU=1 \
SONIC_VLA_EXECUTION_HORIZON=30 \
EVAL_SEEDS="0 1 2 3 4" \
REPEATS_PER_SEED=10 \
MAX_STEPS=1800 \
RECORD_VIDEO_EVERY_N=1 \
RESULTS_DIR=$PWD/evals/humanoidarena_open_door_sonic_native_realized_ckpt10000_random_exec30_eval50 \
bash scripts/deploy/humanoidarena_gr00t_n17_sonic_native_eval.sh open_door


#sonic开门 修改门阻尼后重测
OPEN_DOOR_LEAF_UNLOCK_STIFFNESS=1.5 \
OPEN_DOOR_LEAF_UNLOCK_DAMPING=5 \
GR00T_MODEL_PATH=$PWD/checkpoints/gr00t-n17-sonic-arena-open-door-sonic-native-realized-v1-8gpu-bs512-step10000/checkpoint-10000 \
ARENA_EVAL_PROFILE=random \
SERVE_GPU=5 \
EVAL_GPU=6 \
GR00T_PORT=22095 \
SONIC_VLA_EXECUTION_HORIZON=40 \
EVAL_SEEDS="0 1 2 3 4" \
REPEATS_PER_SEED=10 \
MAX_STEPS=1800 \
RECORD_VIDEO_EVERY_N=1 \
RESULTS_DIR=$PWD/evals/humanoidarena_open_door_sonic_realized_ckpt10000_random_exec40_k1p5_d5_eval50 \
bash scripts/deploy/humanoidarena_gr00t_n17_sonic_native_eval.sh open_door



OPEN_DOOR_SUCCESS_LEAF_ANGLE_DEG=20 \
OPEN_DOOR_LEAF_UNLOCK_STIFFNESS=1.5 \
OPEN_DOOR_LEAF_UNLOCK_DAMPING=3 \
GR00T_MODEL_PATH=$PWD/checkpoints/gr00t-n17-sonic-arena-open-door-twist2-sonic-native-realized-4gpu-bs512-step10000/checkpoint-10000 \
ARENA_EVAL_PROFILE=random \
SERVE_GPU=5 \
EVAL_GPU=6 \
GR00T_PORT=22095 \
SONIC_VLA_EXECUTION_HORIZON=40 \
EVAL_SEEDS="0 1 2 3 4" \
REPEATS_PER_SEED=10 \
MAX_STEPS=1700 \
RECORD_VIDEO_EVERY_N=1 \
RESULTS_DIR=$PWD/evals/humanoidarena_open_door_twist2_sonic_realized_ckpt10000_random_exec40_angle20_k1p5_d3_eval50 \
bash scripts/deploy/humanoidarena_gr00t_n17_sonic_native_eval.sh open_door


#自研链路开门 修改门阻尼
cd /pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Psi0
OPEN_DOOR_LEAF_UNLOCK_STIFFNESS=2.5 \
OPEN_DOOR_LEAF_UNLOCK_DAMPING=10 \
GR00T_MODEL_PATH=/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Isaac-GR00T/outputs/arena-open-door-sonic-gr00t-n17-rot6d59-v1-prefixrtc-delay0to12-4gpu-bs256-step20000/checkpoint-10000 \
ARENA_EVAL_PROFILE=random \
SERVE_GPU=0 \
EVAL_GPU=1 \
KIMODO_GPU=2 \
GR00T_EXECUTION_HORIZON=30 \
EVAL_SEEDS="0 1 2 3 4" \
REPEATS_PER_SEED=10 \
MAX_STEPS=1800 \
RECORD_VIDEO_EVERY_N=1 \
RESULTS_DIR=$PWD/evals/humanoidarena_open_door_rot6d59_ckpt10000_random_exec30_k2p5_d10_eval50 \
bash scripts/deploy/humanoidarena_gr00t_n17_rot6d59_kimodo_textop_eval.sh all open_door



GR00T_MODEL_PATH=/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Isaac-GR00T/outputs/arena-open-door-twist2-gr00t-n17-rot6d59-prefixrtc-delay0to12-4gpu-bs512-step10000 \
ARENA_EVAL_PROFILE=random \
SERVE_GPU=1 \
EVAL_GPU=2 \
KIMODO_GPU=3 \
GR00T_EXECUTION_HORIZON=30 \
EVAL_SEEDS="0 1 2 3 4" \
REPEATS_PER_SEED=10 \
MAX_STEPS=1800 \
RECORD_VIDEO_EVERY_N=1 \
RESULTS_DIR=$PWD/evals/humanoidarena_open_door_twist2_rot6d59_ckpt10000_random_exec30_eval50 \
bash scripts/deploy/humanoidarena_gr00t_n17_rot6d59_kimodo_textop_eval.sh all open_door



#测试自己链路踢足球，sonic数据训练
cd /pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Psi0

GR00T_MODEL_PATH=/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Isaac-GR00T/outputs/arena-football-sonic-gr00t-n17-rot6d59-v2-prefixrtc-delay0to12-8gpu-bs512-step10000/checkpoint-10000 \
ARENA_EVAL_PROFILE=random \
SERVE_GPU=0 \
EVAL_GPU=1 \
KIMODO_GPU=2 \
GR00T_EXECUTION_HORIZON=30 \
EVAL_SEEDS="0 1 2 3 4" \
REPEATS_PER_SEED=10 \
MAX_STEPS=1300 \
RECORD_VIDEO_EVERY_N=5 \
RESULTS_DIR=$PWD/evals/humanoidarena_football_sonic_rot6d59_v2_ckpt10000_random_eval50 \
bash scripts/deploy/humanoidarena_gr00t_n17_rot6d59_kimodo_textop_eval.sh all football
```

native SONIC 当前可以在调用 .sh 时覆盖 CUDA_VISIBLE_DEVICES、NUM_GPUS、MASTER_PORT、PRESET、DATASET_PATH 和 OUTPUT_DIR。但 MAX_STEPS、GLOBAL_BATCH_SIZE 等训练超参不能用同名环境变量直接覆盖，它们由 YAML preset 决定。需要动态更换时，可以通过 NATIVE_PRESET=/path/to/another.yaml 选择另一份配置


用正确的sonic遥操数据latent训练sonic踢足球，
```bash
cd /pfs/pfs-oHNwH0/mnt/pfs/humanoid/wzl/Psi0

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
RUN_ROT6D=0 \
RUN_NATIVE=1 \
bash baselines/gr00t-n1.7/train_arena_football_sonic_v2_8gpu_sequential.sh
```

输出目录：
  /pfs/pfs-oHNwH0/mnt/pfs/humanoid/wzl/Psi0/checkpoints/gr00t-n17-sonic-arena-football-sonic-native-realized-v1-8gpu-bs512-step10000


测试 正确的latent训练的sonic踢球
```bash
GR00T_MODEL_PATH=/pfs/pfs-oHNwH0/mnt/pfs/humanoid/wzl/Psi0/checkpoints/gr00t-n17-sonic-arena-football-sonic-native-realized-v1-4gpu-bs512-step10000/checkpoint-10000 \
RESULTS_DIR=/pfs/pfs-oHNwH0/mnt/pfs/humanoid/wzl/Psi0/evals/humanoidarena_football_sonic_native_realized_v1_4gpu_bs512_ckpt10000_goalframe_only_eval50-allvideo \
SERVER_GPU=1 \
EVAL_GPU=2 \
SERVER_PORT=22207 \
EVAL_SEEDS='0 1 2 3 4' \
REPEATS_PER_SEED=10 \
MAX_STEPS=1300 \
RECORD_VIDEO_EVERY_N=1 \
VIDEO_FPS=50 \
SONIC_VLA_EXECUTION_HORIZON=30 \
bash scripts/deploy/humanoidarena_football_gr00t_n17_sonic_eval.sh all
```


微调原生sonic，twist2数据开门任务
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NUM_GPUS=4 \
MASTER_PORT=29548 \
PRESET=$PWD/baselines/gr00t-n1.7/presets/train/finetune_arena_open_door_sonic_native_realized_v1_4gpu_bs256_step20000.yaml \
DATASET_PATH=/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Psi0/data/output/arena_open_door_twist2_sonic_native_realized \
OUTPUT_DIR=$PWD/checkpoints \
bash baselines/gr00t-n1.7/train_gr00t_n17_sonic_gr00t_new_4gpu.sh \
  --experiment-name gr00t-n17-sonic-arena-open-door-twist2-sonic-native-realized-4gpu-bs512-step10000 \
  --max-steps 10000 \
  --save-steps 5000 \
  --global-batch-size 512 \
  --use-wandb
```


```bash
#移植到SIMPLE中测试自研踢球sonic数据训练，修复场地纹理和相机视角
cd /pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Psi0

GR00T_MODEL_PATH=/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Isaac-GR00T/outputs/arena-football-sonic-gr00t-n17-rot6d59-v2-prefixrtc-delay0to12-8gpu-bs512-step10000/checkpoint-10000 \
ARENA_FOOTBALL_GR00T_EVAL_DIR=/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Psi0/third_party/SIMPLE/data/evals_arena_football_sonic_rot6d59_v2_ckpt10000_recordingseed0 \
SERVE_GPU=4 \
KIMODO_GPU=5 \
EVAL_GPU=6 \
NUM_EPISODES=50 \
bash scripts/deploy/fullstate_arena_football_gr00t_rot6d59_kimodo_textop_eval.sh all

#移植到SIMPLE中测试自研踢球twist2数据训练
GR00T_MODEL_PATH=/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Isaac-GR00T/outputs/arena-football-gr00t-n17-rot6d59/checkpoint-160000 \
ARENA_FOOTBALL_GR00T_EVAL_DIR=/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Psi0/third_party/SIMPLE/data/evals_arena_football_twist2_rot6d59_ckpt160000_recordingseed0 \
SERVE_GPU=2 \
KIMODO_GPU=3 \
EVAL_GPU=4 \
NUM_EPISODES=50 \
GR00T_PORT=22098 \
KIMODO_SERVER_PORT=22186 \
bash scripts/deploy/fullstate_arena_football_gr00t_rot6d59_kimodo_textop_eval.sh all


#SIMPLE移植测试正确encode sonic踢球，sonic数据训练
cd /pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Psi0
GR00T_MODEL_PATH=/pfs/pfs-oHNwH0/mnt/pfs/humanoid/wzl/Psi0/checkpoints/gr00t-n17-sonic-arena-football-sonic-native-realized-v1-4gpu-bs512-step10000/checkpoint-10000 \
ARENA_FOOTBALL_SONIC_EVAL_DIR=/pfs/pfs-oHNwH0/mnt/pfs/humanoid/wzl/Psi0/evals/simple_arena_football_sonic_native_realized_v1_ckpt10000_recordingseed0_cameraaligned \
ARENA_FOOTBALL_RECORDING_SEED=0 \
SERVE_GPU=2 \
EVAL_GPU=1 \
NUM_EPISODES=50 \
MAX_EPISODE_STEPS=1000 \
SAVE_VIDEO=1 \
bash scripts/deploy/fullstate_arena_football_gr00t_n17_sonic_eval.sh all

#twist数据训练的sonic踢球，SIMPLE移植测试
cd /pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Psi0
GR00T_MODEL_PATH=/pfs/pfs-oHNwH0/mnt/pfs/humanoid/wzl/Psi0/checkpoints/gr00t-n17-sonic-arena-football-native-4gpu/checkpoint-160000 \
ARENA_FOOTBALL_SONIC_EVAL_DIR=/pfs/pfs-oHNwH0/mnt/pfs/humanoid/wzl/Psi0/evals/simple_arena_football_twist2_sonic_ckpt160000_recordingseed0_cameraaligned \
ARENA_FOOTBALL_RECORDING_SEED=0 \
SERVE_GPU=2 \
EVAL_GPU=1 \
NUM_EPISODES=50 \
MAX_EPISODE_STEPS=1000 \
SAVE_VIDEO=1 \
GR00T_PORT=22098 \
bash scripts/deploy/fullstate_arena_football_gr00t_n17_sonic_eval.sh all


#限制足球初始位置随机范围，SIMPLE测试twist数据训练的踢足球
cd /pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Psi0

GR00T_MODEL_PATH=/pfs/pfs-oHNwH0/mnt/pfs/humanoid/wzl/Psi0/checkpoints/gr00t-n17-sonic-arena-football-native-4gpu/checkpoint-160000 \
ARENA_FOOTBALL_SONIC_EVAL_DIR=/pfs/pfs-oHNwH0/mnt/pfs/humanoid/wzl/Psi0/evals/simple_arena_football_twist2_sonic_ckpt160000_ballrange_x0p4_y0p8to0p2_seed0 \
ARENA_FOOTBALL_INIT_FROM_DATA=0 \
ARENA_FOOTBALL_RANDOMIZE_BALL=1 \
ARENA_FOOTBALL_OBJECT_SEED=0 \
ARENA_FOOTBALL_BALL_X_RANGE="-0.4,0.4" \
ARENA_FOOTBALL_BALL_Y_RANGE="-0.8,0.2" \
SERVE_GPU=2 \
EVAL_GPU=3 \
GR00T_PORT=22123 \
GR00T_EXECUTION_HORIZON=40 \
NUM_EPISODES=50 \
MAX_EPISODE_STEPS=1000 \
SAVE_VIDEO=1 \
bash scripts/deploy/fullstate_arena_football_gr00t_n17_sonic_eval.sh all
```
