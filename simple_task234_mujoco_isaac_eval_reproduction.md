# Task2/3/4：MuJoCo 执行动作、Isaac Sim 渲染的 SIMPLE Eval 复现规格

更新时间：2026-08-17

本文只冻结仿真与渲染边界、场景放置和第一视角参数，供服务器端在
SIMPLE eval 中复现。策略服务、Kimodo、训练命令、数据转换和成功率判定不在本文范围内。

## 1. 必须保持的架构

```text
SIMPLE policy / controller
          |
          | action / joint command
          v
MuJoCo（唯一物理与动作执行）
          |
          | mujoco_sim_state DDS: qpos/qvel/sim_time/running
          v
DDS -> localhost UDP relay（60 Hz，BestEffort，KeepLast(1)）
          |
          | UDP JSON, 127.0.0.1:23331
          v
Isaac Sim（只读状态镜像 + HSSD 场景 + Ego RGB）
          |
          | 640x480 RGB
          v
SIMPLE policy image input
```

硬性边界：

- MuJoCo 是唯一动力学、接触、碰撞、任务物体状态和仿真时间来源。
- Isaac Sim 不执行动作、不推进第二套物理、不发布控制、不回写 MuJoCo。
- Isaac 只按 MuJoCo `qpos` 设置机器人、垃圾桶关节和自由物体姿态，然后渲染。
- HSSD 房间、Isaac 地面和视觉 Z offset 只改变画面，不改变 MuJoCo 碰撞。
- Isaac 卡顿不得阻塞 MuJoCo；relay 每次只传最新状态。
- 训练/推理使用的是 Isaac Ego render product，不是世界相机截图，也不是右上角小窗的窗口尺寸。

本机参考实现：

- MuJoCo/DDS 启动入口：`scripts/deploy/mujoco_psi0_kimodo_textop_commands.sh`
- DDS relay：`scripts/deploy/task3_mujoco_dds_udp_relay.py`
- Isaac viewer：`scripts/deploy/task3_isaac_hssd_viewer.py`
- Task2 参数备忘：`scripts/deploy/replay_get_Isaccsim_task2.sh`
- Task3 参数备忘：`scripts/deploy/replay_get_Isaccsim_task3.sh`
- Task4 参数备忘：`scripts/deploy/replay_get_Isaccsim_task4.sh`

## 2. MuJoCo -> Isaac 状态契约

DDS topic：`mujoco_sim_state`，类型：`MujocoSimStateDDS`。

relay 默认使用：

```text
host       = 127.0.0.1
UDP port   = 23331
rate       = 60 Hz
QoS        = BestEffort + KeepLast(1)
```

UDP JSON 至少包含：

```json
{
  "nq": 52,
  "nv": 51,
  "sim_time": 12.34,
  "running": 1,
  "qpos": [],
  "qvel": []
}
```

每个 episode 必须使用该 recording 自己的
`model_snapshot/mujoco/model/g1/scene_43dof.xml` 和初始状态，不能只换 `data.csv`
而复用不同内容的 MJCF。Isaac 导入缓存至少应按 MJCF/XML bundle hash 隔离。

qpos 映射不能依赖 Isaac DOF 顺序，必须按 MJCF joint name -> `jnt_qposadr` ->
Isaac DOF name 显式建立：

```text
qpos[0:3] = floating-base xyz
qpos[3:7] = floating-base quaternion，顺序 wxyz
qpos[7:50] = G1 43 个 body/hand scalar joints
qpos[50:] = 任务物体，按下文各任务处理
```

首帧必须检查 `nq/nv`；不匹配就停止，不能截断或补零继续渲染。四元数传给
Isaac 时保持 `wxyz` 并归一化。

## 3. 三个任务共用的第一视角冻结参数

下面这一组是训练渲染和服务器推理都必须显式使用的参数。不要依赖 viewer
默认值，以免不同版本产生轻微偏差。

| 参数 | 冻结值 |
|---|---:|
| 相机父节点 | `/World/Task3/Geometry/pelvis/waist_yaw_link/waist_roll_link/torso_link` |
| local eye XYZ | `(0.06, 0.06, 0.45)` m |
| local forward XYZ | `(0.71735609, 0.0, -0.69670671)` |
| local up XYZ | `(0.69670649, 0.00079633, 0.71735586)` |
| 额外向下 pitch | `15°` |
| vertical FOV | `70°` |
| near clip | `0.2` m |
| 策略 RGB 分辨率 | `640x480`，4:3 |
| 图像类型 | RGB，禁止桶形畸变 |
| MuJoCo/control 频率 | `50 Hz` |

等价 Isaac 参数：

```bash
--head-camera-parent /World/Task3/Geometry/pelvis/waist_yaw_link/waist_roll_link/torso_link \
--head-camera-eye 0.06 0.06 0.45 \
--head-camera-forward 0.71735609 0.0 -0.69670671 \
--head-camera-up 0.69670649 0.00079633 0.71735586 \
--head-camera-pitch-deg 15 \
--camera-fovy 70 \
--head-camera-near-clip 0.2 \
--camera-mode head \
--width 640 \
--height 480
```

若 SIMPLE 同时保留世界主窗口，策略 RGB 应建立独立 render product。可以用
`--ego-inset` 相机作为它的 camera source，但 render product 固定为 `640x480`：

```bash
--camera-mode world \
--ego-inset \
--ego-width 480 --ego-height 360 \
--ego-camera-fovy 70 \
--ego-camera-pitch-deg 15 \
--live-ego-width 640 --live-ego-height 480
```

其中 `480x360` 只是 GUI 右上角预览窗口，不能把它送给策略。在线发布频率可为
`15 Hz` 以降低 Isaac 开销，但每次策略请求必须读取最新帧，并校验图像的
`sim_time` 与当前 MuJoCo 状态之差不超过 `0.25 s`；若服务器侧策略使用连续视频
序列而不是单帧，则必须改为与训练采样一致的 `50 Hz`。

## 4. HSSD 房间的统一放置算法

HSSD 只作为 visual background。服务器端应按以下顺序放置，不要手改源 USD：

1. 将 HSSD 根 Xform 绕 X 轴旋转 `+90°`，从 Y-up 转成 Z-up。
2. 应用各场景的 scale。
3. 计算下表 insertion surface 的世界包围盒中心 `surface_center`。
4. 设置 `room_offset = (0.0, 0.25, 0.0) - surface_center`。
5. 重新计算整个房间 bounds，再做 `room_offset.z -= room_bounds.min.z`，使可见地面到 `z=0`。
6. 隐藏 insertion surface、ceiling 和下表列出的冲突家具。
7. 隐藏 MJCF 导入得到的 Isaac 可见 floor，只显示 HSSD floor；碰撞仍由 MuJoCo floor 负责。

| SIMPLE/HSSD 场景 | USD | scale | insertion surface |
|---|---|---:|---|
| Scene13 / `102344250` | `/home/ubuntu/yzh/Psi0_kimodo_textop/third_party/SIMPLE/data/scenes/hssd/102344250/102344250_local.usd` | `1.0` | `furniture/node_e7c674ea7231612862a5bb66960f0cfef5d8f0e` |
| Scene3 / `102344280` | `/home/ubuntu/yzh/Psi0_kimodo_textop/third_party/SIMPLE/data/scenes/hssd/102344280/102344280.usd` | `0.01` | `furniture/d68aaf2484eec4c754d3b6e07adc09293d8b36de` |

Scene13 还要隐藏以下原生家具，避免与任务垃圾桶重叠：

```text
furniture/node_a9c2d2f765a2399442f845b7da0ddaa61e82071
furniture/ccf3f0ee76dd2263f77ad90a7cec59da84e047c8
furniture/f71c22e2956fcc6e97ace5d6bc9334ddcd1842c1
```

`room_offset` 是由目标 USD 的实际 bounds 动态计算的，不应从另一台机器硬抄一个
固定 XYZ。真正需要冻结的是 USD、scale、rotation、surface prim 和上述算法。

## 5. Task2

任务语义：拿起前方圆桌上的瓶子并投入垃圾桶。

| 项目 | 冻结值 |
|---|---|
| recording root | `/home/ubuntu/yzh/mujoco_recordings/20260805_task2_new` |
| 参考 instance | `20260805_152442_g1_sim` |
| MJCF | `<instance>/model_snapshot/mujoco/model/g1/scene_43dof.xml` |
| HSSD | Scene3 / `102344280.usd` |
| expected model | `nq=59, nv=57, nu=43` |
| task translate XYZ | `(0.0, 0.0, 0.0)` m |
| robot visual Z offset | `+0.012` m |
| trash visual Z offset | `+0.006` m |
| free bottle visual Z offset | `0.0` m |
| Ego | 使用第 3 节共用冻结参数 |

任务物体 qpos：

```text
qpos[50], qpos[51] = 垃圾桶的两个 scalar hinge；必须按当前 MJCF joint name 解析顺序
qpos[52:59]         = 瓶子 free joint：xyz + wxyz
```

世界相机仅用于人工观察，不进入策略：

```text
eye    = (-1.35, -1.05, 1.45)
target = ( 0.50, -0.18, 0.78)
vfov   = 69.24°
```

## 6. Task3

任务语义：踩下踏板，打开前方垃圾桶。

| 项目 | 冻结值 |
|---|---|
| recording root | `/home/ubuntu/yzh/mujoco_recordings/20260804_task3_new` |
| 参考 instance | `20260804_171003_g1_sim` |
| MJCF | `<instance>/model_snapshot/mujoco/model/g1/scene_43dof.xml` |
| HSSD | Scene13 / `102344250_local.usd` |
| expected model | `nq=52, nv=51, nu=43` |
| task translate XYZ | `(0.6, -0.5, 0.0)` m |
| robot visual Z offset | `+0.012` m |
| trash visual Z offset | `+0.006` m |
| free object visual Z offset | `0.0` m |
| Ego | 使用第 3 节共用冻结参数 |

这里的 `task translate` 必须同时作用于 Isaac 中导入的机器人和任务物体，不能作用于
HSSD 房间，也不能回写 MuJoCo root pose。

本批数据使用 `custom_step_trash_can`，qpos 顺序为：

```text
qpos[50] = custom_step_trash_can_lid_hinge_joint
qpos[51] = custom_step_trash_can_pedal_hinge_joint
```

不要套用旧 Task3 `step_trash_can` 的 pedal/lid 顺序；服务器必须按 joint name 检测。

世界相机仅用于人工观察，不进入策略：

```text
eye    = (-2.50, -1.65, 1.80)
target = ( 0.55, -0.05, 0.72)
vfov   = 69.24°
```

## 7. Task4

任务语义：向前走，把瓶子放入绿色箱子。

| 项目 | 冻结值 |
|---|---|
| recording root | `/home/ubuntu/yzh/mujoco_recordings/20260729_task4` |
| 参考 instance | `20260729_151117_g1_sim` |
| MJCF | `<instance>/model_snapshot/mujoco/model/g1/scene_43dof.xml` |
| HSSD | Scene3 / `102344280.usd` |
| expected model | `nq=57, nv=55, nu=43` |
| task translate XYZ | `(0.0, 0.0, 0.0)` m |
| robot visual Z offset | `+0.012` m |
| trash visual Z offset | 不适用 |
| free bottle visual Z offset | `0.0` m |
| Ego | 使用第 3 节共用冻结参数 |

任务物体 qpos：

```text
qpos[50:57] = 瓶子 free joint：xyz + wxyz
```

绿色箱子是 MJCF 中的任务资产，必须随任务 USD 导入；不能用 HSSD 桌子/柜子的
材质替代，也不要给包含桌子的父 prim 整体设置绿色材质。

世界相机仅用于人工观察，不进入策略：

```text
eye    = (-1.35, -1.05, 1.45)
target = ( 0.50, -0.18, 0.78)
vfov   = 69.24°
```

## 8. SIMPLE 服务器端最小复现流程

1. SIMPLE 保持原策略/动作链路，只把动作送给 MuJoCo。
2. MuJoCo 加载当前 episode 自己的 MJCF 和初始状态，并发布完整 `qpos/qvel`。
3. 在 MuJoCo/DDS 的 Python 环境启动 relay；Isaac Python 不直接加载 DDS 扩展。
4. Isaac 从当前 MJCF 生成派生 USD；源 MJCF、mesh 和 HSSD USD 均只读。
5. 按第 4 节加载并对齐对应 HSSD 房间。
6. Isaac 使用 `torch/cuda:0` 的 `SingleArticulation`，按 joint name 应用最新 qpos。
7. 每次姿态更新后只调用 render 提交画面，禁止让 Isaac 物理推进正常 dt。
8. 用第 3 节参数创建 torso-following Ego render product，将 `640x480` RGB 送给策略。
9. 策略每次取图检查分辨率、帧序号、Isaac 图像 `sim_time` 和 MuJoCo 当前时间；过期直接报错，不能静默切到其他相机。
10. 切换 episode 时重置 MuJoCo 初始状态，并清空旧 UDP/旧 RGB 帧；若 MJCF hash 变化则重建 Isaac task USD。

## 9. 训练/推理一致性验收清单

服务器 Codex 完成后，至少保存以下证据：

- [ ] Task2/3/4 分别打印正确的 `nq/nv`，错误模型会 fail-fast。
- [ ] Ego manifest/log 显示 eye、forward、up、pitch、VFOV、near clip 和 `640x480` 均等于第 3 节。
- [ ] 世界相机和 GUI inset 未被当作策略图像源。
- [ ] 同一 MuJoCo qpos 在离线训练 renderer 与在线 SIMPLE renderer 各截一帧，分辨率相同。
- [ ] 对两张图做像素比较；若非逐像素一致，逐项核对 HSSD asset/hash、task translate、灯光、相机矩阵和材质。
- [ ] 合成改变 root XY、一个手臂关节和一个任务物体状态，Isaac 画面出现对应变化。
- [ ] 停止 Isaac 后 MuJoCo 仍可继续运行，证明渲染是只读旁路。
- [ ] Isaac 日志持续显示增长的 MuJoCo `sim_time`，没有自行推进的第二套物理。

严格复现时，训练与推理还应固定相同的 Isaac/RTX renderer 设置、分辨率缩放方式、
灯光参数和材质资产 hash。若训练使用随机灯光，推理场景可以使用训练分布内固定灯光，
但不能改变相机内外参、HSSD 场景、任务变换或任务资产。
