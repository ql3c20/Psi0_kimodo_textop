# GR00T N1.7 RTC 权重 TensorRT 转换与使用指南

本文针对以下 checkpoint：

```text
/home/ubuntu/yzh/ckpt/gr00tn17/task4_real_20260828_lqb
```

该 checkpoint 已经训练了 RTC 和 Prefix-RTC：

```text
action_horizon=40
train_rtc=true
train_prefix_rtc=true
prefix_rtc_timestep_mode=groot_clean
```

TensorRT转换不会修改checkpoint，而是在独立派生目录生成ONNX和engine。当前目标链路为
hard Prefix-RTC：`Tp=40`、`Ta=30`、`overlap=10`，运行mode为
`prefix_rtc_full_pipeline`。ViT、LLM、VL self-attention、State Encoder、
Prefix Action Encoder、Prefix DiT和Action Decoder均使用TensorRT；processor、
张量编排和HTTP仍由Python执行。此前的soft-RTC action-head目录保留为回滚分支。

## 1. 代码在哪里

GR00T TensorRT 代码不在 Kimodo TRT 脚本中，涉及三个 checkout：

| 路径 | 职责 |
| --- | --- |
| `/home/ubuntu/yzh/Isaac-GR00T-rtc/scripts/deployment/build_trt_pipeline.py` | 总构建入口：导出 ONNX、构建 engine |
| `/home/ubuntu/yzh/Isaac-GR00T-rtc/scripts/deployment/export_onnx_n1d7.py` | GR00T N1.7 ONNX 导出 |
| `/home/ubuntu/yzh/Isaac-GR00T-rtc/scripts/deployment/build_tensorrt_engine.py` | ONNX 转 TensorRT engine |
| `/home/ubuntu/yzh/Isaac-GR00T-rtc/scripts/deployment/trt_model_forward.py` | TRT runtime 适配与 mode 选择 |
| `/home/ubuntu/yzh/Isaac-GR00T-rtc/scripts/deployment/verify_n1d7_trt.py` | 标准 action head 的 PyTorch/TRT 单样本对拍 |
| `/home/ubuntu/yzh/Isaac-GR00T-rtc/scripts/deployment/rtc_prefix_action_head_parity_check.py` | PyTorch/TRT Prefix-RTC 离线对拍 |
| `/home/ubuntu/yzh/Psi0_kimodo_textop/scripts/deploy/gr00t_n17_prefix_rtc.py` | `groot_clean` Prefix-RTC adapter |
| `/home/ubuntu/yzh/HumanoidVLA_MJ_backup/HumanoidVLA_MJ/scripts/deploy/mujoco_psi0_kimodo_textop_commands.sh` | 旧 checkpoint 布局的 `build-gr00t-trt` 和 `serve-gr00t` 包装入口；本 checkpoint 不直接使用 |
| `/home/ubuntu/yzh/HumanoidVLA_MJ_backup/HumanoidVLA_MJ/scripts/deploy/gr00t_n17_rot6d59_server.py` | 支持 PyTorch/TRT 切换的 GR00T HTTP server |

`/home/ubuntu/yzh/Psi0_kimodo_textop/scripts/deploy/build_kimodo_trt_engine.py` 只用于 Kimodo denoiser，不能转换 GR00T。

本 checkpoint 的 `processor_config.json`、`statistics.json` 和 `embodiment_id.json` 位于权重根目录。旧包装脚本的 `check_gr00t_assets` 强制查找 `checkpoint/processor/` 子目录，因此会在真正构建前误报缺失。下面使用底层构建器和 server，避免该布局检查；不要为了通过检查而复制或移动 checkpoint 文件。

## 2. 本机已核对的输入

```bash
export GR00T_ROOT=/home/ubuntu/yzh/Isaac-GR00T-rtc
export GR00T_MODEL_PATH=/home/ubuntu/yzh/ckpt/gr00tn17/task4_real_20260828_lqb
export GR00T_BACKBONE_PATH=/home/ubuntu/yzh/Isaac-GR00T-rtc/huggingface/Cosmos-Reason2-2B
export GR00T_TRT_DATASET_PATH=/home/ubuntu/yzh/GR00T-WholeBodyControl/outputs/g1_optitrack_soma_sonic_real_20260825_112832_to_20260826_000403_merged_lerobot_heading0_is_pos_y_local_right_rot6d59_ep_local_xy_ep_yaw_pos_y_1280x720
export GR00T_TRT_OUTPUT_DIR=/home/ubuntu/yzh/ckpt/gr00tn17/task4_real_20260828_lqb_trt_soft_rtc_action_head_4090
export GR00T_TRT_ENGINE_PATH="${GR00T_TRT_OUTPUT_DIR}/engines"
```

当前数据集包含构建需要的 schema：

```text
observation.images.ego_view    [720,1280,3]
observation.full_state_rot6d   [52]
action.policy_action_rot6d59   [59]
```

checkpoint 的两个 `safetensors` 分片完整。本机环境为 Python 3.10.20、PyTorch 2.7.1+cu128、ONNX 1.20.1、TensorRT 10.15.1.29、RTX 4090 SM 8.9。

## 3. 构建前检查

先停止或避开正在使用同一 GPU 的 GR00T 服务。构建时不要同时做正式测速。

```bash
cd /home/ubuntu/yzh/Isaac-GR00T-rtc

.venv/bin/python - <<'PY'
import onnx
import tensorrt as trt
import torch

print("python/torch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("ONNX:", onnx.__version__)
print("TensorRT:", trt.__version__)
print("GPU:", torch.cuda.get_device_name(0))
print("capability:", torch.cuda.get_device_capability(0))
print("builder:", trt.Builder(trt.Logger()) is not None)
PY

uv pip check --python .venv/bin/python
```

检查所有路径，并拒绝覆盖已有输出：

```bash
for input_path in \
  "${GR00T_MODEL_PATH}/config.json" \
  "${GR00T_MODEL_PATH}/processor_config.json" \
  "${GR00T_MODEL_PATH}/statistics.json" \
  "${GR00T_MODEL_PATH}/embodiment_id.json" \
  "${GR00T_MODEL_PATH}/model-00001-of-00002.safetensors" \
  "${GR00T_MODEL_PATH}/model-00002-of-00002.safetensors" \
  "${GR00T_BACKBONE_PATH}" \
  "${GR00T_TRT_DATASET_PATH}"; do
  if [[ ! -e "${input_path}" ]]; then
    echo "missing: ${input_path}" >&2
    exit 1
  fi
done

if [[ -e "${GR00T_TRT_OUTPUT_DIR}" ]]; then
  echo "refusing existing output: ${GR00T_TRT_OUTPUT_DIR}" >&2
  exit 1
fi
```

## 4. 执行转换

直接使用 GR00T 底层构建入口：

```bash
cd /home/ubuntu/yzh/Isaac-GR00T-rtc

CUDA_VISIBLE_DEVICES=0 \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
NO_ALBUMENTATIONS_UPDATE=1 \
PYTHONPATH=/home/ubuntu/yzh/Isaac-GR00T-rtc \
.venv/bin/python scripts/deployment/build_trt_pipeline.py \
  --model-path /home/ubuntu/yzh/ckpt/gr00tn17/task4_real_20260828_lqb \
  --backbone-path /home/ubuntu/yzh/Isaac-GR00T-rtc/huggingface/Cosmos-Reason2-2B \
  --dataset-path /home/ubuntu/yzh/GR00T-WholeBodyControl/outputs/g1_optitrack_soma_sonic_real_20260825_112832_to_20260826_000403_merged_lerobot_heading0_is_pos_y_local_right_rot6d59_ep_local_xy_ep_yaw_pos_y_1280x720 \
  --embodiment-tag new_embodiment \
  --output-dir /home/ubuntu/yzh/ckpt/gr00tn17/task4_real_20260828_lqb_trt_soft_rtc_action_head_4090 \
  --batch-size 1 \
  --precision bf16 \
  --export-mode action_head \
  --workspace 8192 \
  --steps export,build
```

该入口实际调用：

```text
build_trt_pipeline.py
  --export-mode action_head
  --steps export,build
  --batch-size 1（默认）
  --precision bf16（当前唯一完整支持精度）
```

产物位于：

```text
${GR00T_TRT_OUTPUT_DIR}/onnx/
${GR00T_TRT_OUTPUT_DIR}/engines/
${GR00T_TRT_OUTPUT_DIR}/pipeline.log
```

## 5. 默认运行模式需要的 engine

`action_head` runtime 只加载下面四个标准 action-head engine：

```text
state_encoder.engine
action_encoder.engine
dit_bf16.engine
action_decoder.engine
```

检查：

```bash
for engine in \
  state_encoder.engine \
  action_encoder.engine \
  dit_bf16.engine \
  action_decoder.engine; do
  test -s "${GR00T_TRT_ENGINE_PATH}/${engine}" || exit 1
  ls -lh "${GR00T_TRT_ENGINE_PATH}/${engine}"
done

sha256sum "${GR00T_TRT_ENGINE_PATH}"/*.engine
```

在 `--export-mode action_head` 下，ViT、LLM 和 VL self-attention 不会导出；builder 会跳过这些不存在的 ONNX，只生成上面的四个 engine。第一轮不要改成 `full_pipeline`/`n17_full_pipeline`：旧 checkpoint 曾出现 TRT LLM 在 RTC 续接时明显发散；新 checkpoint 必须重新做分层对拍后才能扩大 TRT 范围。

当前构建器已增加独立 `action_head` build mode，pipeline会只要求这四个ONNX/engine。
不要再把 `action_head` 映射为 `full_pipeline`，否则四个目标engine虽然能够生成，最终汇总仍会因
缺少ViT、LLM和VL self-attention ONNX而错误退出1。

## 6. 必做：标准 action head PyTorch/TRT 对拍

先使用同一 checkpoint、同一真实数据样本和同一随机种子比较 PyTorch/TRT 输出：

```bash
cd /home/ubuntu/yzh/Isaac-GR00T-rtc

CUDA_VISIBLE_DEVICES=0 \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
NO_ALBUMENTATIONS_UPDATE=1 \
.venv/bin/python scripts/deployment/verify_n1d7_trt.py \
  --model-path /home/ubuntu/yzh/ckpt/gr00tn17/task4_real_20260828_lqb \
  --backbone-path /home/ubuntu/yzh/Isaac-GR00T-rtc/huggingface/Cosmos-Reason2-2B \
  --dataset-path /home/ubuntu/yzh/GR00T-WholeBodyControl/outputs/g1_optitrack_soma_sonic_real_20260825_112832_to_20260826_000403_merged_lerobot_heading0_is_pos_y_local_right_rot6d59_ep_local_xy_ep_yaw_pos_y_1280x720 \
  --engine-dir /home/ubuntu/yzh/ckpt/gr00tn17/task4_real_20260828_lqb_trt_soft_rtc_action_head_4090/engines \
  --mode action_head \
  --embodiment-tag NEW_EMBODIMENT \
  --batch-size 1
```

该脚本固定 `torch.manual_seed(42)`，输出 final action 的 cosine、L1 和 L-infinity 误差。它把 cosine `>0.999` 标为 PASS，`0.99~0.999` 标为 WARN，低于 `0.99` 标为 FAIL；WARN/FAIL 都不要直接进入在线链路。

这个单样本检查只证明标准 action head 的一次输出接近，不覆盖 soft RTC 的多 chunk 续接。启动服务后还要用同一请求序列分别跑 PyTorch 和 TRT，检查第二个及后续 chunk，确认 `Ta=25/overlap=15`、shape、有限值和实际动作轨迹一致。

只有下面条件都满足才能启动 TRT 服务：

- PyTorch/TRT 都没有 NaN/Inf；
- 单样本 final action cosine 大于 0.999，且误差量级可接受；
- 多 chunk soft RTC 续接没有 NaN/Inf 或明显发散；
- action shape、dtype、40 帧合约一致。

## 7. 启动 TRT 服务

当前 `22095` 已有 PyTorch 服务时，先在原终端正常 `Ctrl+C` 停止它，再启动 TRT；不要同时保留两个服务。

保持当前真实链路的 `Ta=25/overlap=15`。这里也直接启动 TRT-capable server，绕过旧包装脚本的 `processor/` 布局检查：

```bash
cd /home/ubuntu/yzh/Isaac-GR00T-rtc

CUDA_VISIBLE_DEVICES=0 \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
NO_ALBUMENTATIONS_UPDATE=1 \
PYTHONPATH=/home/ubuntu/yzh/Isaac-GR00T-rtc \
.venv/bin/python \
  /home/ubuntu/yzh/HumanoidVLA_MJ_backup/HumanoidVLA_MJ/scripts/deploy/gr00t_n17_rot6d59_server.py \
  --model-path /home/ubuntu/yzh/ckpt/gr00tn17/task4_real_20260828_lqb \
  --backbone-path /home/ubuntu/yzh/Isaac-GR00T-rtc/huggingface/Cosmos-Reason2-2B \
  --host 0.0.0.0 \
  --port 22095 \
  --device cuda \
  --action-horizon 40 \
  --action-exec-horizon 25 \
  --inference-seed 42 \
  --inference-backend tensorrt \
  --trt-mode action_head \
  --trt-engine-path /home/ubuntu/yzh/ckpt/gr00tn17/task4_real_20260828_lqb_trt_soft_rtc_action_head_4090/engines \
  --rtc \
  --no-prefix-rtc \
  --rtc-frozen-steps 15 \
  --rtc-ramp-rate 6.0
```

必须看到类似日志：

```text
Action head TRT engines loaded and forward method patched.
[gr00t-server] TensorRT enabled: mode=action_head ...
[gr00t-server] ready ... horizon=40 exec=25 ... prefix_rtc=False overlap=15 backend=tensorrt trt_mode=action_head
```

## 8. 激活检查

```bash
curl --noproxy '*' -fsS http://127.0.0.1:22095/health
```

预期关键字段：

```json
{
  "inference_backend": "tensorrt",
  "trt_mode": "action_head",
  "action_horizon": 40,
  "action_exec_horizon": 25,
  "prefix_rtc": false,
  "rtc_overlap_steps": 15
}
```

再检查实际进程参数：

```bash
pgrep -af 'gr00t_n17_rot6d59_server.py'
```

不能仅凭 engine 文件存在或启动命令文本宣称 TRT 已启用。

## 9. T40/E30 hard Prefix-RTC full-pipeline

```text
构建 export mode: prefix_rtc_full_pipeline
运行 TRT mode:   prefix_rtc_full_pipeline
运行参数:         --rtc --prefix-rtc
engine:           vit.engine
                  llm_bf16.engine
                  vl_self_attention.engine
                  state_encoder.engine
                  action_encoder_prefix_rtc.engine
                  dit_prefix_rtc_bf16.engine
                  action_decoder.engine
```

派生目录：

```text
/home/ubuntu/yzh/ckpt/gr00tn17/task4_real_20260828_lqb_trt_prefix_rtc_full_pipeline_t40_4090
```

此分支必须额外运行：

```bash
cd /home/ubuntu/yzh/Isaac-GR00T-rtc

CUDA_VISIBLE_DEVICES=0 \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
NO_ALBUMENTATIONS_UPDATE=1 \
.venv/bin/python scripts/deployment/verify_prefix_rtc_backbone_trt.py \
  --model-path /home/ubuntu/yzh/ckpt/gr00tn17/task4_real_20260828_lqb \
  --backbone-path /home/ubuntu/yzh/Isaac-GR00T-rtc/huggingface/Cosmos-Reason2-2B \
  --dataset-path /home/ubuntu/yzh/GR00T-WholeBodyControl/outputs/g1_optitrack_soma_sonic_real_20260825_112832_to_20260826_000403_merged_lerobot_heading0_is_pos_y_local_right_rot6d59_ep_local_xy_ep_yaw_pos_y_1280x720 \
  --engine-dir /home/ubuntu/yzh/ckpt/gr00tn17/task4_real_20260828_lqb_trt_prefix_rtc_full_pipeline_t40_4090/engines \
  --mode prefix_rtc_full_pipeline \
  --prefix-rtc-module-path /home/ubuntu/yzh/Psi0_kimodo_textop/scripts/deploy/gr00t_n17_prefix_rtc.py \
  --overlap 10 \
  --calls 5 \
  --seed 42 \
  --min-cosine 0.99 \
  --max-mean-abs-error 0.02 \
  --max-abs-error 0.2
```

`groot_clean`来自checkpoint的`config.json`。TRT Prefix forward必须把前缀token的
timestep设为bucket 999；旧实现硬编码bucket 0时，第二块cosine仅`0.97939`、
最大误差`1.7578125`。修复后连续5块cosine均不低于`0.999967`，没有逐块发散。

## 10. 回滚

停止 TRT 服务，恢复：

```bash
GR00T_INFERENCE_BACKEND=pytorch
```

并移除 `GR00T_TRT_MODE`、`GR00T_TRT_ENGINE_PATH` 即可。checkpoint 和派生 engine 都不需要删除。

## 11. 状态说明

截至2026-08-29，本机已完成：

- 在独立目录生成4个标准action-head ONNX和engine，原checkpoint未修改；
- 构建环境为PyTorch 2.7.1+cu128、ONNX 1.20.1、TensorRT 10.15.1.29、RTX 4090 SM 8.9；
- 同一真实样本、seed 42的final action对拍：cosine `1.000000`、L1 `0.000274`、L∞ `0.005515`，结果PASS；
- TRT服务实际激活为 `action_head`、40帧预测、25帧执行、soft RTC overlap 15；
- episode 0真实chunk 0和1均返回有限 `(40,59)`，第二块确认复用上一块 `[25:40]`；
- 现场首次请求约0.425秒，续接请求约0.074秒；这些是共享GPU上的功能烟测，不是隔离后的正式性能基准。

MuJoCo GUI中的绿影、手指、机器人动作和pause行为仍需完整四终端肉眼验收，不能由上述组件和HTTP验证替代。

同日新增并验证了T40 Prefix-RTC full-pipeline分支：

- 独立目录生成7个ONNX、7个TensorRT engine和`deployment_metadata.json`；
- 运行时为`prefix_rtc_full_pipeline`，ViT、LLM、VL self-attention和Prefix action head均为TRT；
- 修复TRT Prefix forward硬编码bucket 0的问题，按本checkpoint使用`groot_clean` bucket 999；
- 同一真实样本连续5次Prefix-RTC对拍，最低cosine `0.999967098`、最大MAE `0.0051658`、最大绝对误差`0.1210938`，没有逐块发散；
- 在线episode 0连续5块均返回有限`(40,59)`，chunk 1到4的10帧硬overlap最大误差均为`0.0`；
- 当前功能烟测首块GR00T约`0.206s`，后续约`0.054-0.059s`。该数字不是隔离GPU后的正式性能基准。

完整MuJoCo GUI中的绿影、机器人稳定性和pause行为仍需肉眼验收。
