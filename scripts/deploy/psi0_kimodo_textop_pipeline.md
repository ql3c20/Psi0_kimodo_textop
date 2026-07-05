# PSI0 → Kimodo → TextOp Tracker 链路

## 整体数据流

```text
PSI0
  ├─ root + 双手 + 双脚五点位姿 ──────────────┐
  └─ Kimodo 生成全身未来参考                 │
                  ↓                          │
       TextOp Transformer VAE Encoder        │
                  ↓ 128D latent z/c          │
                                             ↓
TextOp Tracker Policy
  ↓
29D body joint target
  ↓
机器人关节控制
```

实际链路为：

```text
PSI0 → Kimodo → TextOp Tracker → 机器人关节控制
```

TextOp Tracker 根据 Kimodo 生成的未来参考运动和机器人当前状态，直接生成
29D body joint target。该输出不再经过 Decoupled WBC 策略。

本链路默认启用：

```bash
export TEXTOP_POLICY_ROOT_EE=1
```

启用后，Tracker 使用 PSI0/VLA 直接输出的 root、左手、右手、左脚、右脚
五点位姿；Kimodo 负责生成全身未来参考运动，该参考由 Transformer VAE
压缩为 128D latent `z/c`。五点位姿不从 Kimodo 全身动作做 FK 重建。

部分文件名中仍包含 `decoupled_wbc`，这是代码继承关系和历史命名，不代表
TextOp Tracker 后面还会执行 Decoupled WBC policy。

## 统一启动入口

```text
Psi0/scripts/deploy/simple_psi0_kimodo_eval_commands.sh
```

rot6d59 + RGZ one-step + initial-reference 的评测子命令：

```bash
bash scripts/deploy/simple_psi0_kimodo_eval_commands.sh \
  eval-rot6d59-textop-onestep-initref-rgz
```

## PSI0 服务

服务实现：

```text
Psi0/src/psi/deploy/psi0_serve_simple.py
```

PSI0 输出策略动作，rot6d59 模式下动作空间为：

```text
hand14 + root9 + four EE pose9 = 59D
```

## Kimodo 服务

服务实现：

```text
Psi0/scripts/deploy/kimodo_generation_server.py
```

Kimodo adapter：

```text
Psi0/third_party/SIMPLE/src/simple/baselines/kimodo_adapter.py
```

默认蒸馏模型配置：

```text
kimodo_my/outputs/g1_distill_16to8_100to20_dagger_teacher_gt03_100k_bs4x4_k3_cosine_selfacc50_rootacc10_headingacc10_gtbranch/resolved_config.yaml
```

默认 Kimodo 权重：

```text
kimodo_my/outputs/g1_distill_16to8_100to20_dagger_teacher_gt03_100k_bs4x4_k3_cosine_selfacc50_rootacc10_headingacc10_gtbranch/ema_final.pt
```

该模型是 `100 → 20` diffusion steps 的蒸馏 student，部署脚本默认设置：

```bash
KIMODO_DIFFUSION_STEPS=20
```

## PSI0 → Kimodo → TextOp Agent

主 agent：

```text
Psi0/third_party/SIMPLE/src/simple/baselines/psi0_kimodo_textop_tracker.py
```

Kimodo 转换父类：

```text
Psi0/third_party/SIMPLE/src/simple/baselines/psi0_kimodo_decoupled_wbc.py
```

Kimodo adapter：

```text
Psi0/third_party/SIMPLE/src/simple/baselines/kimodo_adapter.py
```

评测时使用的 SIMPLE agent 名称：

```text
psi0_kimodo_textop_tracker
```

## TextOp Tracker

Tracker adapter：

```text
Psi0/third_party/SIMPLE/src/simple/baselines/textop_tracker_adapter.py
```

RGZ one-step Tracker 权重：

```text
textop/2026-06-11_11-39-47_rgz_loco_manip_obj_transf_vae_1step_ddp_4gpu_gear_sonic_ads_naug/latest.onnx
```

Transformer VAE encoder：

```text
textop/2026-05-30_04-21-30_npz_rgz_filtered_ddp_save/artifacts/motion_transformer_vae_encoder_z_c.onnx
```

VAE 归一化统计：

```text
textop/2026-05-30_04-21-30_npz_rgz_filtered_ddp_save/artifacts/stats.npz
```

TextOp 不是只加载一个 Tracker ONNX。完整推理过程为：

```text
机器人当前状态 + Kimodo 未来参考
                 ↓
       Transformer VAE Encoder
                 ↓
             latent z/c
                 ↓
         TextOp Tracker Policy
                 ↓
         29D body joint target
```

## SIMPLE 评测入口

评测 CLI：

```text
Psi0/third_party/SIMPLE/src/simple/cli/eval_decoupled_wbc.py
```

虽然 CLI 文件名包含 `decoupled_wbc`，在本链路中实际选择的 agent 是
`psi0_kimodo_textop_tracker`，最终执行的是 TextOp Tracker 生成的关节目标。

## 默认 RGZ 环境变量

部署脚本中的关键默认值为：

```bash
TEXTOP_RGZ_TRACKER_RUN=/pfs/pfs-ilWc5D/yzh/textop/2026-06-11_11-39-47_rgz_loco_manip_obj_transf_vae_1step_ddp_4gpu_gear_sonic_ads_naug
TEXTOP_RGZ_POLICY_ONNX=${TEXTOP_RGZ_TRACKER_RUN}/latest.onnx

TEXTOP_RGZ_VAE_RUN=/pfs/pfs-ilWc5D/yzh/textop/2026-05-30_04-21-30_npz_rgz_filtered_ddp_save
TEXTOP_RGZ_VAE_ONNX=${TEXTOP_RGZ_VAE_RUN}/artifacts/motion_transformer_vae_encoder_z_c.onnx
TEXTOP_RGZ_VAE_STATS=${TEXTOP_RGZ_VAE_RUN}/artifacts/stats.npz

TEXTOP_FUTURE_STEPS=1
export TEXTOP_POLICY_ROOT_EE=1
```
