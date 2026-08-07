# Fullstate 新任务接入 GR00T → Kimodo → TextOp 评测指南

本文用于指导 Codex 将一批新的 HumanoidVLA MuJoCo 录制数据接入 Psi0 的
GR00T → Kimodo → TextOp 三段式评测链路。目标不是简单复制 task1，而是先证明录制场景能够被 SIMPLE/MuJoCo 正确重建，再定义与训练数据一致的任务和可验证的成功条件，最后提供可重复运行的三终端脚本。

## 1. 每次任务必须提供的输入

开始前确认以下信息；能够从仓库或数据中推导的内容应由 Codex 自行检查，不要要求用户重复提供。

- 任务编号和日期，例如 `20260729_task4`。
- 原始录制目录，例如：
  `/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/HumanoidVLA_MJ_1/output/20260729_task4`。
- 对应 LeRobot 数据目录及语言 prompt。优先读取：
  `Psi0/data/output/<dataset>/meta/tasks.jsonl`。
- GR00T checkpoint 路径、推理模式和 execution horizon。
- Kimodo checkpoint/服务配置；如果仍由公共脚本提供默认值，也要确认当前默认值正确。
- TextOp tracker、Transformer-VAE、ONNX 和 stats 路径。
- episode 数、最大步数、GPU、端口和输出目录命名。

必须区分两个根目录：录制数据可能位于 `HumanoidVLA_MJ_1`，可复用的任务
MJCF 资产可能位于 `HumanoidVLA_MJ`。不得仅因已有任务使用后者就擅自替换前者。

## 2. 总体流程与阶段门槛

按以下顺序工作。上一阶段没有通过时，不要启动完整三服务评测。

1. 审计录制数据和场景快照。
2. 在 MuJoCo 中重建并验证初始化。
3. 新建 SIMPLE task Python 文件并注册 Gym 环境。
4. 新建任务专用部署 wrapper，复用公共 task1 主链路。
5. 做静态检查、单环境 reset/step smoke test。
6. 启动三个终端做单 episode 集成测试。
7. 人工检查视频和状态量，再做多 episode 正式评测。

完成标准不是“脚本能启动”，而是：场景布局、相机、机器人姿态、物体姿态、语言
prompt、动作时序和 success 判定都与该任务的数据语义一致。

## 3. 阶段一：审计录制数据是否足以初始化 MuJoCo

### 3.1 必查文件

对 `recordings_dir/*/data.csv` 做批量检查：

- 至少存在一条有效录制，每个 CSV 至少有 header 和一行数据。
- `scene_path` 指向的 `model_snapshot/mujoco/model/g1/scene_43dof.xml` 实际存在。
- 快照递归依赖的 include、mesh 和 texture 均存在。
- 第一帧 `qpos0..qpos49` 完整，对应 G1 floating root 7 维和 43 个关节。
- 任务物体的 free-joint qpos 必须根据 CSV header 名称识别，不能假设每个任务都使用相同下标。
- 检查 NaN/Inf、四元数范数、机器人 root 高度以及不同 episode 第一帧的分布。
- 比较多份 scene snapshot，确认资产拓扑、物体名称和静态布局是否一致；若不一致，要决定按录制逐 episode 重建，还是只支持同构子集。

第一帧通常是“录制开始时已经 settle 后的状态”，而 XML 中是 authored pose。机器人和
自由物体初始化优先采用 CSV 第一帧；静态桌子、篮筐等布局优先采用对应 snapshot。

### 3.2 MuJoCo 编译与动力学检查

在写任务逻辑前，至少验证一个代表性 snapshot：

- `mujoco.MjModel.from_xml_path(scene_path)` 能成功编译。
- `nq/nv` 与 CSV qpos/qvel 命名能够解释。
- 写入第一帧 qpos、调用 `mj_forward` 后没有异常或明显穿模。
- 空动作短暂仿真时机器人不会因错误 qpos 顺序爆飞，物体不会因错误姿态或碰撞立即弹飞。
- 渲染头相机确认桌子、目标物和容器的位置、尺度、朝向与录制视频一致。

若 snapshot 能编译但 SIMPLE 的额外 MJCF 合并失败，应检查资源相对路径和名字前缀，不能通过删除碰撞体来掩盖问题。

### 3.3 输出一份初始化审计摘要

Codex 在实现前应报告：

- 有效/无效 episode 数。
- 选用的代表性 snapshot 和 fallback recording。
- robot qpos、每个自由物体 qpos 的 header 名称及下标范围。
- 静态资产的世界位姿和动态资产的初始位姿范围。
- 相机分辨率、FOV、pose 是否可以沿用 task1。
- 已发现的数据缺失、拓扑差异或初始化风险。

## 4. 阶段二：定义新的 SIMPLE 任务

新文件命名为：

```text
Psi0/third_party/SIMPLE/src/simple/tasks/g1_fullstate_<date>_task<N>.py
```

参考现有实现：

- `g1_fullstate_20260615_task1.py`：相机和基础 fullstate task 模板。
- `g1_fullstate_20260625_task2.py`：动态瓶子、录制初始化、容器局部坐标 success。
- `g1_fullstate_20260612_task3.py`：按 episode 轮换录制初始化、关节状态 success。

### 4.1 必须定义的内容

- 唯一的 `TaskRegistry.register(...)`、`uid`、class 名、label 和 description。
- 合理的 `metadata["max_episode_steps"]`。
- `mujoco_extra_mjcf`：每个资产的路径、唯一 prefix、世界位姿；自由物体还要指定 body/freejoint。
- `mujoco_object_body_names`：至少正确暴露 `target`，需要容器时增加 `container`。
- `mujoco_initial_robot_qpos` fallback。
- `reset()`：布局、相机、指令、录制选择、物体位姿、episode 状态机清零和 robot reset。
- `compute_reward()` 与 `check_success()`：使用 MuJoCo 状态，不使用图像猜测结果。

任务语言必须与对应训练集 `meta/tasks.jsonl` 完全一致，包括大小写和标点。只有在明确做语言消融时才允许环境变量覆盖。

### 4.2 录制初始化建议

沿用 task2/task3 的可复现轮换模式：

- `TASK<N>_INIT_FROM_RECORDINGS=1` 开启。
- `TASK<N>_RECORDINGS_DIR` 指定目录。
- `TASK<N>_RECORDING_SEED` 控制每轮 seeded permutation。
- `TASK<N>_RECORDING_INDEX` 固定单条录制，便于 debug。
- 默认模式按 seed 打乱后无放回遍历一轮，不要每次独立随机导致小样本重复。

解析 qpos 时应按 header 中的 `[qposN]` 和 joint/body 名双重校验。不要只切固定列，也不要把录制中额外机构的 qpos 拼进 50 维 robot qpos。

### 4.3 success/reward 设计原则

- success 必须对应 prompt 的最终语义，而不是某个容易达到的中间动作。
- 多阶段任务可记录 `was_lifted`、`was_pedal_pressed`、`was_lid_opened` 等历史状态用于 shaped reward。
- 最终 success 尽量用几何/关节状态直接判定，并在必要时要求连续满足若干 simulation steps。
- 容器判断应在容器局部坐标系中进行，不能直接写死世界坐标；容器旋转或随机化后仍应正确。
- 阈值应从 MJCF 内腔、碰撞几何或成功 demo 末帧统计得到，并留出瓶子半径/高度的安全 margin。
- success 应锁存，避免物体轻微抖动导致 episode 结果反复变化。
- 必须做正例和反例单元检查：容器内应成功；容器外、上方掠过、桌面上、地面上不应成功。

### 4.4 不要遗漏注册

新增任务文件后还必须修改：

1. `third_party/SIMPLE/src/simple/tasks/__init__.py`：导入新 task class。
2. `third_party/SIMPLE/src/simple/envs/__init__.py`：注册新的 Gym ID，entry point 通常沿用
   `simple.envs.sonic_loco_manip:SonicLocoManipEnv`。

推荐命名：

```text
TaskRegistry key: g1_fullstate_<date>_task<N>
Gym ID:          simple/G1Fullstate<Date>Task<N>-v0
shell TASK 值:   G1Fullstate<Date>Task<N>-v0
```

## 5. 阶段三：编写任务专用部署脚本

不要为每个任务复制 100 多行主链路。保留
`fullstate_task1_gr00t_rot6d59_kimodo_textop_eval.sh` 作为公共实现，新任务采用 task2/task3 的薄 wrapper 形式：

```text
Psi0/scripts/deploy/fullstate_task<N>_gr00t_rot6d59_kimodo_textop_eval.sh
```

wrapper 至少覆盖：

- `GR00T_MODEL_PATH`
- `GR00T_PREFIX_RTC`、`GR00T_USE_RTC` 和 `GR00T_PREFIX_RTC_TIMESTEP_MODE`
- `GR00T_EXECUTION_HORIZON` 和 `GR00T_PORT`
- `FULLSTATE_GR00T_TASK`
- 唯一的 `FULLSTATE_GR00T_EVAL_DIR`
- 唯一的 `FULLSTATE_GR00T_KIMODO_WORK_DIR`
- `MAX_EPISODE_STEPS`
- `HUMANOID_VLA_MJ_ROOT`
- 该任务的录制初始化、随机化和 instruction 环境变量

最后使用：

```bash
exec bash "$BASE_SCRIPT" "${1:-}"
```

公共链路中必须保持：GR00T 预测 horizon、client 实际 execution horizon 和 RTC carry-over 的定义一致。例如预测 40、执行 34，则下一 chunk 的 continuity prefix 是 6 帧。Kimodo 和 TextOp 接收完整预测 chunk，仅 MuJoCo 执行前 34 帧。

输出目录必须包含任务、checkpoint/RTC 变体和关键随机化标识，禁止多个实验静默写入同一路径。

## 6. 验证顺序

### 6.1 静态和单环境验证

- Python 文件可编译，shell 通过 `bash -n`。
- import `simple.tasks` 后能从 Gym registry 找到新 ID。
- `gym.make()`、`reset()` 成功，observation shape 与 GR00T modality 配置一致。
- 检查 `mjModel` 中目标 body、collision geom、容器 body/site/joint 名均存在。
- 固定 `TASK<N>_RECORDING_INDEX`，确认 reset 得到的 robot/物体 qpos 与对应 CSV 第一帧一致。
- 连续 reset 一轮，确认 seeded permutation 可复现且没有越界。
- 保存 reset 帧或短视频，与录制开头并排核对相机和场景。

### 6.2 三终端单 episode

先用独立 GPU 启动：

1. `serve`
2. `kimodo-serve`
3. `eval`，设置 `NUM_EPISODES=1`、`SAVE_VIDEO=1`，并固定 recording index

确认服务端口健康、GR00T action shape、rot6d59 schema、Kimodo 输出长度、TextOp reference window 和 execution horizon 一致。单 episode 通过后，才解除 recording index 并运行多 episode。

### 6.3 正式评测验收

- 每个 episode 保存 recording 名、随机化参数、success、步数和视频。
- 输出目录没有复用旧结果。
- 视频中没有初始化穿模、相机错位、机器人瞬移或错误容器。
- success 时对应物体确实完成最终任务；抽查失败 episode，排除 success 判定漏检。

## 7. Task4 的具体建议

当前可用信息：

- 数据：`HumanoidVLA_MJ_1/output/20260729_task4`。
- prompt：`Walk forward and put the bottle into the box.`
- snapshot 包含圆桌、`green_box_2real_*` 和自由瓶子 `task_obj_0_001_bottle_13_*`。
- 一份 snapshot 中桌子 mount 为 `[0.95, 0.0, 0.0]`，绿色盒子 mount 为
  `[0.98, 0.1, 0.75]`，瓶子的 authored pose 为 `[0.78, -0.08, 0.755]`；实际 reset 应以所选 CSV 第一帧 free-joint pose 为准。
- 当前 CSV schema 中 robot 是 `qpos0..49`，瓶子是 `qpos50..56`；实现仍需对所有录制逐个验证 header。

Task4 的最终成功条件建议定义为“瓶子碰撞几何质心落入绿色盒子内腔并稳定”，而不是只判断瓶子靠近盒子中心：

1. 读取瓶子 collision geom 的世界坐标。
2. 用绿色盒子 body/mount 的 `xpos/xmat` 转换成盒子局部坐标。
3. 根据盒子内壁碰撞几何推导内腔 `x/y` 范围及底部/口沿 `z` 范围。
4. 收缩 `x/y` 边界至少一个瓶子半径，避免质心在墙外但数值恰好过线。
5. 要求瓶子质心低于盒口且高于盒底；可再要求速度低于阈值并连续满足 5～10 个控制步。
6. 可将 `was_lifted` 和到盒子中心的距离用于 shaped reward，但最终 success 只由稳定 inside 判定。

在没有从实际 MJCF/成功 demo 统计出内腔边界之前，不要猜一个固定阈值并宣称 Task4 已完成。

## 8. 可直接交给 Codex 的任务模板

```text
请为 <date_taskN> 接入 GR00T(rot6d59) → Kimodo → TextOp 的 SIMPLE MuJoCo 评测。

录制数据：<recordings_dir>
LeRobot 数据/prompt：<lerobot_dataset_dir>
GR00T checkpoint：<checkpoint>
Kimodo 配置/checkpoint：<kimodo_path_or_current_default>
TextOp tracker/VAE：<tracker_and_vae_paths>
期望输出目录：<eval_dir>

请严格按照 scripts/deploy/fullstate_new_task_codex_guide.md 执行：
1. 先审计全部 data.csv、scene snapshot、qpos schema 和资产依赖，并验证 MuJoCo 初始化；
2. 新建 g1_fullstate_<date>_task<N>.py，定义录制初始化、准确 prompt、reward/success；
3. 同步更新 tasks/__init__.py 和 envs/__init__.py 注册；
4. 新建 task<N> 的薄 wrapper，复用 task1 公共主脚本；
5. 做 Python/shell 静态检查、Gym reset/step 和 success 正反例检查；
6. 给出三个终端的最终启动命令。不要直接启动长时间多 episode 评测。

不要假设 qpos 下标、MJCF 名称、资产根目录或容器阈值与旧任务相同；所有数值必须来自本任务 CSV、snapshot/MJCF 或成功 demo 统计。
```

## 9. Codex 最终交付清单

- 初始化审计结论及未解决风险。
- 新 task Python 文件。
- `tasks/__init__.py` 和 `envs/__init__.py` 注册修改。
- 新任务部署 wrapper。
- 实际执行过的静态/smoke test 及结果。
- 三终端命令和所有关键环境变量说明。
- success 判定的坐标系、阈值来源和正反例验证结果。

