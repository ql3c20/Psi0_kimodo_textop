# Kimodo 新权重 TensorRT 转换与使用指南

本文说明如何在不破坏原 PyTorch 链路的前提下，为一个 Kimodo distill checkpoint 增加 TensorRT denoiser 分支。目标读者是第一次接触本项目 TensorRT 接入、但已经能正常运行 Kimodo 四终端链路的开发者。

> 核心原则：PyTorch 永远是默认后端；只有显式设置 `KIMODO_USE_TRT=1` 才启用 TensorRT。构建成功、engine 文件存在，都不代表在线服务已经使用 TensorRT。

## 先看这里：之前的 TRT 转换代码在哪里

代码所在仓库是：

```text
/home/ubuntu/yzh/Psi0_kimodo_textop
```

真正执行“checkpoint -> ONNX -> TensorRT engine”的主脚本是：

```text
/home/ubuntu/yzh/Psi0_kimodo_textop/scripts/deploy/build_kimodo_trt_engine.py
```

相关文件及职责如下：

| 绝对路径 | 职责 | 是否负责转换 |
| --- | --- | --- |
| `/home/ubuntu/yzh/Psi0_kimodo_textop/scripts/deploy/build_kimodo_trt_engine.py` | 加载 Kimodo distill config/checkpoint，导出静态 ONNX，构建 TRT engine，做一次合成数值检查，生成 v2 metadata JSON | **是，这是转换入口** |
| `/home/ubuntu/yzh/Psi0_kimodo_textop/scripts/deploy/kimodo_trt_backend.py` | 加载和严格校验 engine/metadata；离线比较 PyTorch 与 TRT 的误差和速度；在线提供 PyTorch denoiser 兼容接口 | 否，负责验证和执行 |
| `/home/ubuntu/yzh/Psi0_kimodo_textop/scripts/deploy/kimodo_generation_server.py` | Kimodo HTTP 服务；仅当 `KIMODO_USE_TRT=1` 时，用 TRT backend 替换 `self.model.denoiser` | 否，负责运行时接入 |
| `/home/ubuntu/yzh/Psi0_kimodo_textop/scripts/deploy/simple_psi0_kimodo_eval_commands.sh` | 通用任务启动脚本，`kimodo-serve` 子命令最终启动上面的 server | 否，负责启动 |
| `/home/ubuntu/yzh/Psi0_kimodo_textop/scripts/deploy/KIMODO_TENSORRT_GUIDE.md` | 本使用文档 | 否 |

`/home/ubuntu/yzh/kimodo_my` 不是 TRT 转换代码目录。它是 Kimodo 模型源码和模型加载依赖，构建时通过下面的参数传给转换脚本：

```bash
--kimodo-root /home/ubuntu/yzh/kimodo_my
```

### 代码调用关系

```text
离线转换：
build_kimodo_trt_engine.py
  -> 从 kimodo_my 导入 _build_model_from_distill()
  -> 读取 resolved_config.yaml + ema_final.pt
  -> 导出 kimodo_T24.onnx
  -> 构建并验证 kimodo_T24.trt
  -> 写入 kimodo_T24.json

在线运行：
simple_psi0_kimodo_eval_commands.sh kimodo-serve
  -> kimodo_generation_server.py
  -> KIMODO_USE_TRT=0：保留原 PyTorch denoiser
  -> KIMODO_USE_TRT=1：加载 kimodo_trt_backend.py
  -> 用 KimodoTensorRTDenoiser 替换 self.model.denoiser
```

### 之前生成的 TRT 产物在哪里

转换代码在仓库中，生成物不在仓库中，而是在对应权重目录下。目前现场存在以下几组：

```text
/home/ubuntu/yzh/ckpt/kimodo/notext-sft/trt_engines/
/home/ubuntu/yzh/ckpt/kimodo/scq_final/trt_engines/
/home/ubuntu/yzh/ckpt/kimodo/scq_final/trt_engines_no_tf32/
/home/ubuntu/yzh/ckpt/kimodo/g1_distill_16to8_100to20_dagger_teacher_gt03_100k_bs4x4_k3_cosine_selfacc50_rootacc10_headingacc10_gtbranch/trt_engines/
/home/ubuntu/yzh/ckpt/kimodo/g1_distill_16to8_100to20_dagger_teacher_gt03_100k_bs4x4_k3_cosine_selfacc50_rootacc10_headingacc10_gtbranch/trt_engines_no_tf32/
```

每个有效目录内应成套包含：

```text
kimodo_T24.onnx
kimodo_T24.trt
kimodo_T24.json
```

下面这个目录是旧 metadata v1 的备份，只用于保留历史，不要用于当前 runtime：

```text
/home/ubuntu/yzh/ckpt/kimodo/notext-sft/trt_engines_v1_backup_20260812_145403/
```

当前 backend 要求 `format_version=2`。转换新权重时应重新运行 `build_kimodo_trt_engine.py`，让 ONNX、TRT 和 JSON 一起生成；不要复制旧 engine，也不要手改 v1 JSON。

## 0. 适用范围

这套脚本转换的是 **Kimodo distill student denoiser**，不是任意 `.pt` 权重的通用 TensorRT 转换器。目标权重至少要满足：

- 有与 checkpoint 配套的 Hydra `resolved_config.yaml`；
- config 中存在 `model.student_denoiser` 和 `model.num_base_steps`；
- checkpoint 是脚本支持的 `student`、`ema.shadow` 或直接 state dict 格式；
- 能被 Kimodo 的 `_build_model_from_distill()` 正常加载；
- denoiser 使用当前的 8 输入、4096 维 text feature、`cfg_type=separated` 调用合约。

GR00T、SONIC、Psi0 action head 或完全不同结构的 Kimodo 权重不能使用这份脚本。`--check-only` 只做依赖、config 字段和 checkpoint 外层格式检查；它通过后，仍要以实际模型加载、ONNX 导出和数值验证结果为准。

## 1. 先理解加速范围

这个方案只替换 Kimodo 的 student denoiser：

```text
默认路径
HTTP request -> Kimodo sampler -> PyTorch denoiser -> motion 后处理 -> qpos/CSV

TensorRT 路径
HTTP request -> Kimodo sampler -> TensorRT denoiser -> motion 后处理 -> qpos/CSV
```

以下部分保持原样：

- 文本编码和约束预处理；
- diffusion sampler、时间步和 hard projection；
- motion representation、FK、CSV/qpos 导出；
- HTTP 协议、Bridge、TextOp、MuJoCo 和 pause 逻辑；
- GR00T、VLA、VAE 和机器人控制链路。

因此，纯 denoiser 的加速比不等于完整 Kimodo 请求或四终端端到端加速比。

本项目当前验证过的静态合约是：

| 项目 | 值 |
| --- | --- |
| batch | 1 |
| 帧数 | 24 |
| 帧率语义 | 30 FPS 下 0.8 秒 |
| motion dimension | 417 |
| CFG type | `separated` |
| CFG weight | `[2.0, 2.0]` |
| diffusion calls | 20 次；这不是 20 帧 |

TensorRT engine 与 checkpoint、配置、GPU 架构、TensorRT 版本和静态 shape 绑定。任何一项改变都应重新构建和验证。

## 2. 文件结构

TensorRT 分支由三个文件组成：

```text
scripts/deploy/
├── build_kimodo_trt_engine.py   # 离线导出 ONNX、构建 engine、写 metadata
├── kimodo_trt_backend.py        # 运行时适配器和离线数值验证入口
└── kimodo_generation_server.py  # 默认关闭的在线选择分支
```

构建产物放在 checkpoint 目录的独立子目录，不覆盖原权重：

```text
CHECKPOINT_DIR/
├── resolved_config.yaml
├── ema_final.pt
└── trt_engines_no_tf32/
    ├── kimodo_T24.onnx
    ├── kimodo_T24.trt
    └── kimodo_T24.json
```

三个派生产物必须成套使用。不要把另一个 checkpoint 的 `.trt`、`.json` 或 `.onnx` 混进来。

## 3. 转换其他权重：最短流程

每次换权重，只修改下面的 `TARGET_CONFIG`、`TARGET_CKPT` 和 `TARGET_TRT_DIR`。config 和 checkpoint 不要求固定文件名，但必须来自同一次训练；输出目录应是新的独立目录：

```bash
export PSI0_ROOT=/home/ubuntu/yzh/Psi0_kimodo_textop
export KIMODO_ROOT=/home/ubuntu/yzh/kimodo_my
export TARGET_CONFIG=/absolute/path/to/new_weight/resolved_config.yaml
export TARGET_CKPT=/absolute/path/to/new_weight/ema_final.pt
export TARGET_TRT_DIR=/absolute/path/to/new_weight/trt_engines_t24_no_tf32
```

确认文件：

```bash
for input_path in "${TARGET_CONFIG}" "${TARGET_CKPT}"; do
  if [[ ! -f "${input_path}" ]]; then
    echo "missing input: ${input_path}" >&2
    exit 1
  fi
done
if [[ -e "${TARGET_TRT_DIR}" ]]; then
  echo "refusing existing output directory: ${TARGET_TRT_DIR}" >&2
  exit 1
fi
echo "input pair and new output directory OK"
```

### 3.1 激活环境

```bash
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate kimodo
cd "${PSI0_ROOT}"
```

不要在看到 `Run 'conda init' before 'conda activate'` 后反复运行 `conda init`。当前终端应先 source 上面的 `conda.sh`。

### 3.2 检查 Python、CUDA、ONNX 和 TensorRT

```bash
python - <<'PY'
import sys
import torch
import onnx
import tensorrt as trt

print("python:", sys.version.split()[0])
print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("ONNX:", onnx.__version__)
print("TensorRT:", trt.__version__)
print("GPU:", torch.cuda.get_device_name(0))
print("compute capability:", torch.cuda.get_device_capability(0))
print("TRT builder:", bool(trt.Builder(trt.Logger(trt.Logger.WARNING))))
PY

python -m pip check
```

本机已验证组合：Python 3.10、PyTorch 2.6.0+cu124、ONNX 1.22.0、TensorRT 10.15.1.29、RTX 4090 SM 8.9。其他组合不是一定不可用，但必须重新构建，不能直接复用这里的 engine。

### 3.3 只读检查 checkpoint

```bash
taskset -c 16-31 python scripts/deploy/build_kimodo_trt_engine.py \
  --distill-config "${TARGET_CONFIG}" \
  --distill-ckpt "${TARGET_CKPT}" \
  --kimodo-root "${KIMODO_ROOT}" \
  --check-only
```

预期结尾：

```text
check-only passed; no files written
```

### 3.4 构建 no-TF32 engine

新 checkpoint 建议先构建严格的 no-TF32 版本：

```bash
taskset -c 16-31 python scripts/deploy/build_kimodo_trt_engine.py \
  --distill-config "${TARGET_CONFIG}" \
  --distill-ckpt "${TARGET_CKPT}" \
  --kimodo-root "${KIMODO_ROOT}" \
  --frames 24 \
  --workspace-gib 2 \
  --disable-tf32 \
  --output-dir "${TARGET_TRT_DIR}"
```

为什么优先 no-TF32：TensorRT 默认可能选择 TF32 tactic。合成输入可能通过，但真实五点约束输入的局部最大误差仍可能超阈值。no-TF32 通常略慢，但更接近禁用 TF32 的 PyTorch FP32 输出。

已有同名产物时脚本会拒绝覆盖。这是保护机制。确认路径和 checkpoint 无误后才使用 `--force`；更稳妥的做法是改用一个新输出目录。

如果在线请求不是 24 帧，应按真实请求帧数构建。例如同时生成 T15 和 T24：

```bash
taskset -c 16-31 python scripts/deploy/build_kimodo_trt_engine.py \
  --distill-config "${TARGET_CONFIG}" \
  --distill-ckpt "${TARGET_CKPT}" \
  --kimodo-root "${KIMODO_ROOT}" \
  --frames 15 24 \
  --disable-tf32 \
  --output-dir "${TARGET_TRT_DIR}"
```

每个 `T` 都会生成独立的 `.onnx/.trt/.json`。运行时必须选择与实际请求帧数一致的一组文件。

### 3.5 检查产物

```bash
ls -lh "${TARGET_TRT_DIR}"/kimodo_T24.{onnx,trt,json}
sha256sum "${TARGET_TRT_DIR}"/kimodo_T24.{onnx,trt,json}

python - "${TARGET_TRT_DIR}/kimodo_T24.json" <<'PY'
import json
import sys

meta = json.load(open(sys.argv[1]))
print("checkpoint:", meta["distill_checkpoint"])
print("config:", meta["distill_config"])
print("frames:", meta["frames"])
print("motion_dim:", meta["motion_dim"])
print("cfg:", meta["cfg_type"], meta["cfg_weight"])
print("build_parameters:", meta["build_parameters"])
print("versions:", meta["versions"])
print("sha256:", meta["sha256"])
PY
```

至少确认：

- checkpoint/config 绝对路径正确；
- `frames` 与目标请求一致；当前常用 T24；
- `motion_dim` 与目标模型一致；当前已验证模型通常为 417；
- `cfg_type=separated`、权重为 `[2.0, 2.0]`；
- `tf32_enabled=false`；
- `.onnx/.trt/.json` 哈希均已记录。

至此只能说“构建完成”，还不能在线启用。

## 4. 必做：真实输入离线验证

### 4.1 为什么合成检查不够

构建脚本会用合成输入检查 shape、NaN/Inf、MAE 和 max error。但真实链路中的文本特征、五点约束、heading 和 diffusion 中间状态具有不同分布，因此必须再用一个正常 PyTorch chunk 的真实输入验证。

从原 PyTorch 链路选一个包含以下文件的正常 chunk：

```text
chunk_xxxxxx/
├── constraints.json
└── heading_source.npz
```

设置路径：

```bash
export REAL_CHUNK=/absolute/path/to/a/known-good/chunk_xxxxxx
```

执行 20 次真实 denoiser 调用对比：

```bash
taskset -c 16-31 python scripts/deploy/kimodo_trt_backend.py \
  --engine "${TARGET_TRT_DIR}/kimodo_T24.trt" \
  --metadata "${TARGET_TRT_DIR}/kimodo_T24.json" \
  --distill-config "${TARGET_CONFIG}" \
  --distill-ckpt "${TARGET_CKPT}" \
  --kimodo-root "${KIMODO_ROOT}" \
  --cases 0 \
  --real-constraints "${REAL_CHUNK}/constraints.json" \
  --real-heading-source "${REAL_CHUNK}/heading_source.npz" \
  --prompt "perform the manipulation task in simulation" \
  --diffusion-steps 20 \
  --max-mean-error 0.001 \
  --max-abs-error 0.01
```

必须检查：

- 20 次调用全部 shape 相同；
- PyTorch/TRT 都没有 NaN/Inf；
- worst MAE 和 worst max 都在当前项目约定阈值内；
- 错误帧数、CFG type 和 CFG weight 会被 backend 拒绝；
- 计时包含 CUDA synchronize、预热和多次正式采样。

不要在不同 checkpoint 间使用不同阈值后直接比较“通过/失败”。例如一个 engine 的 `max=0.017` 在阈值 `0.03` 下会通过，但在阈值 `0.01` 下应失败。

### 4.2 TF32 版本失败怎么办

典型现象：合成输入通过，但真实输入的 worst max 超过 `0.01`。

处理方式：

1. 不启用在线 TRT；
2. 保留失败 engine 供对照；
3. 使用新的输出目录和 `--disable-tf32` 重建；
4. 用同一个真实 chunk、prompt、seed 和阈值重新验证；
5. 只有新 engine 通过后才进入在线接入。

本机 gtbranch no-TF32 样例的真实 20-call 结果：

```text
worst MAE       0.000362
worst max       0.007283
PyTorch         6.893 ms/step
TensorRT        2.823 ms/step
speedup         2.44x
nonfinite       0
```

这些数字只用于 sanity check，不是其他 checkpoint 的自动验收标准。

## 5. 接入在线服务

如果当前分支还没有 TensorRT 选择逻辑，原则上只需要一个默认关闭的门控点：先完整加载原 PyTorch model，再按环境变量替换 denoiser。

```python
use_trt = os.environ.get("KIMODO_USE_TRT", "0").strip()
if use_trt not in {"0", "1"}:
    raise ValueError("KIMODO_USE_TRT must be 0 or 1")

# 原 PyTorch model 加载逻辑保持不变
self.inference_backend = "pytorch"

if use_trt == "1":
    from kimodo_trt_backend import KimodoTensorRTDenoiser

    self.model.denoiser = KimodoTensorRTDenoiser(
        os.environ["KIMODO_TRT_ENGINE_PATH"],
        metadata_path=os.environ["KIMODO_TRT_METADATA_PATH"],
        expected_config=distill_config,
        expected_checkpoint=distill_ckpt,
        device=self.device,
    )
    self.inference_backend = "tensorrt"
    print(f"[kimodo-server] TensorRT enabled: engine={os.environ['KIMODO_TRT_ENGINE_PATH']}")
else:
    print("[kimodo-server] PyTorch backend enabled")
```

适配器负责校验 metadata/hash、TensorRT 版本、GPU compute capability、输入名称、shape、dtype、帧数和 CFG 合约。校验失败必须显式退出，不能静默回退到 PyTorch，否则测速和排错都会失真。

不要在服务启动时隐式导出或重建 engine。

## 6. 启动命令

### 6.1 默认 PyTorch 回归

接入代码后，先不设置任何 TRT 变量，重新运行原四终端。只有原 PyTorch 仍正常，才能继续。

```bash
KIMODO_ROOT="${KIMODO_ROOT}" \
KIMODO_DISTILL_CONFIG="${TARGET_CONFIG}" \
KIMODO_DISTILL_CKPT="${TARGET_CKPT}" \
KIMODO_DIFFUSION_STEPS=20 \
KIMODO_KEYFRAME_STEP=10 \
KIMODO_ANCHOR_MODE=policy_only \
KIMODO_USE_TRT=0 \
KIMODO_SERVER_PORT=22185 \
taskset -c 16-31 python scripts/deploy/kimodo_generation_server.py
```

预期日志：

```text
[kimodo-server] PyTorch backend enabled
```

### 6.2 显式启用 TensorRT

只替换四终端中的 Kimodo 终端，GR00T、MuJoCo 和 Bridge 命令保持原样：

```bash
KIMODO_ROOT="${KIMODO_ROOT}" \
KIMODO_DISTILL_CONFIG="${TARGET_CONFIG}" \
KIMODO_DISTILL_CKPT="${TARGET_CKPT}" \
KIMODO_DIFFUSION_STEPS=20 \
KIMODO_KEYFRAME_STEP=10 \
KIMODO_ANCHOR_MODE=policy_only \
KIMODO_USE_TRT=1 \
KIMODO_TRT_ENGINE_PATH="${TARGET_TRT_DIR}/kimodo_T24.trt" \
KIMODO_TRT_METADATA_PATH="${TARGET_TRT_DIR}/kimodo_T24.json" \
KIMODO_SERVER_PORT=22185 \
taskset -c 16-31 python scripts/deploy/kimodo_generation_server.py
```

上面使用当前仓库真实存在的 server 入口，适合先做单服务验证。如果你已经有任务专用 launcher，不要改写原命令，只把同一组 `KIMODO_DISTILL_*`、`KIMODO_USE_TRT` 和 `KIMODO_TRT_*` 变量加到原 Kimodo 终端。例如本仓库通用 launcher 的子命令是：

```bash
bash scripts/deploy/simple_psi0_kimodo_eval_commands.sh kimodo-serve
```

预期日志必须包含：

```text
[kimodo-server] TensorRT enabled: engine=/absolute/path/.../kimodo_T24.trt
```

## 7. 证明在线 TRT 真的启用

不能只看启动命令。先找到真实 PID：

```bash
pgrep -af '(^|/)python(3(\.10)?)? .*/kimodo_generation_server\.py$'
```

检查进程环境：

```bash
PID=$(pgrep -f '(^|/)python(3(\.10)?)? .*/kimodo_generation_server\.py$' | head -1)
test -n "${PID}" || { echo "Kimodo PID not found" >&2; exit 1; }
tr '\0' '\n' < "/proc/${PID}/environ" | \
  rg '^KIMODO_(USE_TRT|TRT_ENGINE_PATH|TRT_METADATA_PATH|DISTILL_CONFIG|DISTILL_CKPT)='
```

必须同时满足：

```text
KIMODO_USE_TRT=1
KIMODO_TRT_ENGINE_PATH=期望的绝对路径
KIMODO_TRT_METADATA_PATH=期望的绝对路径
KIMODO_DISTILL_CONFIG=构建 engine 时的配置
KIMODO_DISTILL_CKPT=构建 engine 时的权重
```

若服务提供 `/config`，还可以检查：

```bash
curl --noproxy '*' -fsS http://127.0.0.1:22185/config
```

只有命令、进程环境和服务日志三者一致，才能说“运行时激活成功”。

## 8. 四终端端到端验收

完成单服务验证后再进入四终端。至少观察：

- Kimodo 持续收到真实 24 帧请求并输出新 motion；
- Bridge 消费的是新 chunk，不是 stale 或重复 ready chunk；
- 绿影连续，不消失、不闪烁；
- 五点约束与 PyTorch 基线一致；
- 机器人不乱动、不倒地；
- pause 次数和原因与 PyTorch 基线相当；
- 端口 `22185` 没有旧服务或 SSH 转发抢占；
- 完整请求耗时和纯 denoiser 耗时分开记录。

推荐验收表：

```text
PyTorch 基线：通过/失败
ONNX/TRT/JSON：通过/失败
合成数值：MAE=... max=... nonfinite=...
真实 T24：worst MAE=... worst max=... nonfinite=...
纯 denoiser：PyTorch=... ms TRT=... ms speedup=...x
运行时激活：PID=... 日志=... 环境变量=...
四终端：绿影=... 五点=... 动作=... pause=...
回滚：通过/失败
```

## 9. 公平测速

PyTorch 和 TensorRT 必须使用：

- 同一 checkpoint/config；
- 同一真实 constraints 和 heading；
- 同一 prompt、seed、24 帧和 20 diffusion steps；
- 同一 CFG type/weight；
- 相同预热次数和正式样本数；
- CUDA synchronize 或 CUDA events；
- median、P90/P95，而不是单次最快值。

计算方式：

```text
speedup = PyTorch latency / TensorRT latency
reduction = (PyTorch latency - TensorRT latency) / PyTorch latency * 100%
```

完整 Kimodo 日志中的 `inference time` 还包含文本编码、采样循环和其他 GPU 工作。`total time` 还包含约束解析、motion inverse、qpos/CSV 和文件 IO。不要拿它们与单步 denoiser 时间直接相除。

## 10. 一键回滚

停止当前 Kimodo 服务，去掉以下变量后重新运行原命令：

```text
KIMODO_USE_TRT
KIMODO_TRT_ENGINE_PATH
KIMODO_TRT_METADATA_PATH
```

或者显式设置：

```bash
export KIMODO_USE_TRT=0
unset KIMODO_TRT_ENGINE_PATH KIMODO_TRT_METADATA_PATH
```

确认日志恢复为：

```text
[kimodo-server] PyTorch backend enabled
```

无需删除 `.onnx/.trt/.json`，也无需修改 checkpoint。保留产物便于离线调查。

## 11. 常见故障

| 现象 | 优先检查 | 正确处理 |
| --- | --- | --- |
| 日志仍是 PyTorch | `/proc/<pid>/environ`、旧 PID、shell 续行 | 停止旧服务，修正变量后重启 |
| config/checkpoint mismatch | metadata 路径和 SHA-256 | 用目标权重重新构建，不要手改 JSON |
| TensorRT version mismatch | 构建/运行版本 | 在当前运行环境重建 engine |
| GPU capability mismatch | GPU 型号和 SM | 在目标 GPU 上重建并重新验证 |
| frame shape mismatch | 请求是否真为 T24 | 构建匹配帧数的独立 engine |
| 合成通过、真实失败 | 真实 20-call worst max | no-TF32 重建；不要在线启用失败 engine |
| 绿影消失、一直 pause | 端口、旧服务、Bridge 数据流 | 先回退 PyTorch，定位链路层级 |
| 绿影闪烁或机器人乱动 | 同一真实输入逐步数值差异 | 停止在线试错，离线捕获和比较 |
| PyTorch 也异常 | 原链路、输入、端口、主机稳定性 | 不继续归因于 TensorRT |
| 偶发解释器/数值异常 | CPU affinity | 本机可先用 `taskset -c 16-31` 排除 CPU 10/11 问题 |

### Shell 续行陷阱

反斜杠必须是该行最后一个字符，后面不能有空格。不要把注释插进续行命令：

```bash
# 错误示例：注释会切断同一条环境变量命令
KIMODO_DISTILL_CONFIG=/path/config.yaml \
# KIMODO_DISTILL_CKPT=/another/path.pt \
KIMODO_USE_TRT=1 \
bash launcher.sh
```

备选命令应写成两个完整代码块。

## 12. 新 checkpoint 迁移检查单

复制以下清单逐项确认：

```text
[ ] 原 PyTorch 四终端基线正常
[ ] config 与 checkpoint 是明确的一对
[ ] kimodo 环境中的 ONNX/TRT/CUDA/GPU 已记录
[ ] check-only 通过
[ ] 使用独立目录生成 ONNX/TRT/JSON
[ ] metadata 中路径、hash、T24、CFG 和 GPU 合约正确
[ ] 合成输入无 NaN/Inf，误差合格
[ ] 同一真实 chunk 的 20-call 对比通过统一阈值
[ ] 接入后 KIMODO_USE_TRT=0 的 PyTorch 回归正常
[ ] KIMODO_USE_TRT=1 的单服务日志和 /proc 环境一致
[ ] 四终端绿影、五点、动作、pause 正常
[ ] 去掉 TRT 变量即可恢复 PyTorch
```

在最后一项之前，报告应使用“构建通过”“离线验证通过”或“运行时激活通过”，不要提前写“完整 TensorRT 链路通过”。
