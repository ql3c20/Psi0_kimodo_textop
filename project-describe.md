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
export GR00T_PYTHON=$GR00T_ROOT/.venv.bak-py310-20260731/bin/python

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

python scripts/deployment/build_trt_pipeline.py \
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
