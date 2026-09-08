# Lerobo数据构建（fullstate->rot6d59; gr00t->sonic链路的gr00t_new需要用原数据,在采集电脑中转换）
# 放在mnt/pfs/humanoid/yzh/Psi0/data/output


cd /pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Isaac-GR00T

mv .venv .venv.broken-20260822-py312

uv venv .venv \
  --python /root/.local/share/uv/python/cpython-3.10-linux-x86_64-gnu/bin/python3.10

uv sync --frozen

  然后验证：

.venv/bin/python -c \
  "import sys, torch; print(sys.version); print(torch.__version__); print(torch.cuda.is_available())"

head -1 .venv/bin/torchrun
```


```
cd /pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Psi0
source .venv-psi/bin/activate

python scripts/data/build_movepick_rot6d59_dataset.py \
  --src data/output/task6_data/task6_scq_20260907_all   \
  --out data/output/task6_data/task6_scq_20260907_all_rot6d59
```

``` #重新统计stats
cd /pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Isaac-GR00T

uv run --no-sync python gr00t/data/stats.py \
  --dataset-path /pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Psi0/data/output/scq_fullstate_20260825_task5_rot6d59 \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path /pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Isaac-GR00T/examples/unitree_g1_rot6d59_config.py
```

``` # 真机数据转换
cd /pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Psi0
source .venv-psi/bin/activate

python scripts/data/build_real_rot6d59_dataset.py \
  --src data/output/task6_data/task6_scq_20260907_all \
  --out data/output/task6_data/task6_scq_20260907_all_rot6d59 \
  --reference data/output/lqb_fullstate_20260615_task1_rot6d59 \
  --output-fps 30 \
  --fk-clipping none
```

# gr00t 1.7 训练 （更换任务数据集dataset-path）
# 权重路径：output-dir/checkpoint-80000
```bash
cd /pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Isaac-GR00T

CUDA_VISIBLE_DEVICES=0,1,2,3 \
NUM_GPUS=4 \
MASTER_PORT=29533 \
MAX_STEPS=80000 \
SAVE_STEPS=10000 \
GLOBAL_BATCH_SIZE=64 \
DATALOADER_NUM_WORKERS=4 \
TRAIN_PREFIX_RTC=1 \
GR00T_PREFIX_RTC_TIMESTEP_MODE=groot_clean \
USE_WANDB=1 \
uv run --no-sync bash examples/finetune.sh \
  --base-model-path /pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Psi0/checkpoints/GR00T-N1.7-3B \
  --dataset-path /pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Psi0/data/output/lqb_20260831_to_20260901_realtask2_rot6d59 \
  --modality-config-path /pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Isaac-GR00T/examples/unitree_g1_rot6d59_config.py \
  --embodiment-tag NEW_EMBODIMENT \
  --output-dir /pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Isaac-GR00T/outputs/scq_task5-gr00t-n17-rot6d59-kimodo-textop-prefix-rtc-groot-clean-A800-newlight \
  --wandb-project gr00t-n1.7
```

``` # overlap 12的版本
cd /pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Isaac-GR00T

CUDA_VISIBLE_DEVICES=0,1,2,3 \
NUM_GPUS=4 \
MASTER_PORT=29532 \
MAX_STEPS=80000 \
SAVE_STEPS=10000 \
GLOBAL_BATCH_SIZE=64 \
DATALOADER_NUM_WORKERS=4 \
TRAIN_PREFIX_RTC=1 \
TRAIN_RTC_MIN_DELAY=0 \
TRAIN_RTC_MAX_DELAY=12 \
GR00T_PREFIX_RTC_TIMESTEP_MODE=groot_clean \
USE_WANDB=1 \
uv run --no-sync bash examples/finetune.sh \
  --base-model-path /pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Psi0/checkpoints/GR00T-N1.7-3B \
  --dataset-path /pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Psi0/data/output/task6_data/task6_scq_20260907_all_rot6d59 \
  --modality-config-path /pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Isaac-GR00T/examples/unitree_g1_rot6d59_config.py \
  --embodiment-tag NEW_EMBODIMENT \
  --output-dir /pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Isaac-GR00T/outputs/scq_task6_20260907_new \
  --wandb-project gr00t-n1.7
```
#   uv sync --frozen --python /usr/bin/python3.10 --verbose

# 任务场景初始化数据放在/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/HumanoidVLA_MJ_1/output/20260612_task3


# gr00t->kimodo->texop链路评测（写新的fullstate_task1_gr00t_rot6d59_kimodo_textop_eval.sh和Psi0/third_party/SIMPLE/src/simple/tasks/g1_fullstate_20260615_task1.py文件）
# 终端1 gr00t serve
# /pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Isaac-GR00T/outputs/task1-gr00t-n17-rot6d59-kimodo-textop-prefix-rtc-groot-clean/checkpoint-80000
```
cd /pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Psi0

SERVE_GPU=1 \
GR00T_PREFIX_RTC=1 \
GR00T_PREFIX_RTC_TIMESTEP_MODE=groot_clean \
GR00T_EXECUTION_HORIZON=34 \
GR00T_MODEL_PATH=/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Isaac-GR00T/outputs/task1-gr00t-n17-rot6d59-kimodo-textop-prefix-rtc-groot-clean/checkpoint-80000 \
bash scripts/deploy/fullstate_task1_gr00t_rot6d59_kimodo_textop_eval.sh serve
```

# 终端2 kimodo serve
# /pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/kimodo_my/outputs/g1_distill_16to8_100to20_dagger_teacher_gt03_100k_bs4x4_k3_cosine_selfacc50_rootacc10_headingacc10_gtbranch
```
cd /pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Psi0

KIMODO_KEYFRAME_STEP=1 \
KIMODO_GPU=2 \
bash scripts/deploy/fullstate_task1_gr00t_rot6d59_kimodo_textop_eval.sh kimodo-serve
```

# 终端3 textop eval (修改对应任务路径FULLSTATE_TASK1_GR00T_EVAL_DIR)
# 评测数据路径：third_party/SIMPLE/data/evals_task1_gr00t_kimodo_textop_prefix_rtc_groot_clean_obj5cm_10eps
# /pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/textop/2026-05-30_04-21-30_npz_rgz_filtered_ddp_save
# /pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/textop/2026-06-11_11-39-47_rgz_loco_manip_obj_transf_vae_1step_ddp_4gpu_gear_sonic_ads_naug
```
cd /pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Psi0

EVAL_GPU=3 \
NUM_EPISODES=10 \
EPISODE_START=0 \
MAX_EPISODE_STEPS=500 \
SAVE_VIDEO=1 \
SKIP_STABILIZE=1 \
TASK1_RANDOMIZE_OBJECT=1 \
TASK1_OBJECT_SEED=0 \
TASK1_OBJECT_X_RANGE=1.00,1.10 \
TASK1_OBJECT_Y_RANGE=-0.05,0.05 \
FULLSTATE_TASK1_GR00T_EVAL_DIR=data/evals_task1_gr00t_kimodo_textop_prefix_rtc_groot_clean_obj5cm_10eps \
bash scripts/deploy/fullstate_task1_gr00t_rot6d59_kimodo_textop_eval.sh eval
```

# Task2/3/4 SIMPLE checkpoint 推理：MuJoCo physics + Isaac Ego，走 GR00T -> Kimodo -> TextOp
#
# 对应 wrapper：
#   Task2 scripts/deploy/fullstate_task2_gr00t_rot6d59_kimodo_textop_eval.sh
#   Task3 scripts/deploy/fullstate_task3_gr00t_rot6d59_kimodo_textop_eval.sh
#   Task4 scripts/deploy/fullstate_task4_gr00t_rot6d59_kimodo_textop_eval.sh
#
# TASK234_STRICT_ISAAC_EVAL=1 会把旧 Gym task 切换为：
#   Task2 G1Fullstate20260805Task2IsaacEval-v0
#   Task3 G1Fullstate20260804Task3IsaacEval-v0
#   Task4 G1Fullstate20260729Task4IsaacEval-v0
# 并把 FULLSTATE_GR00T_SIM_MODE 切为 mujoco_isaac。下面以 Task2 为例；
# Task3/4 只需换 wrapper 和 GR00T_MODEL_PATH。训练数据与 recording 不参与推理。

# 资产预检（只检查推理场景，不检查训练数据或 recording）
```bash
cd /pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Psi0/third_party/SIMPLE

PYTHONPATH=src python scripts/task234_reproduction_preflight.py \
  --task 2 \
  --manifest-dir outputs/task234_manifests
```

# 终端1：GR00T serve
```bash
cd /pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Psi0

TASK234_STRICT_ISAAC_EVAL=1 \
SERVE_GPU=1 \
GR00T_PREFIX_RTC=1 \
GR00T_PREFIX_RTC_TIMESTEP_MODE=groot_clean \
GR00T_EXECUTION_HORIZON=34 \
GR00T_MODEL_PATH=/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Isaac-GR00T/outputs/task2-gr00t-n17-rot6d59-kimodo-textop-prefix-rtc-groot-clean-A800-newbg/checkpoint-80000 \
bash scripts/deploy/fullstate_task2_gr00t_rot6d59_kimodo_textop_eval.sh serve
```

# 终端2：Kimodo serve（三个任务共用同一服务）
```bash
cd /pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Psi0

TASK234_STRICT_ISAAC_EVAL=1 \
KIMODO_KEYFRAME_STEP=1 \
KIMODO_GPU=2 \
bash scripts/deploy/fullstate_task2_gr00t_rot6d59_kimodo_textop_eval.sh kimodo-serve
```

# 终端3：TextOp eval + MuJoCo physics + Isaac Ego RGB
```bash
cd /pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Psi0

TASK234_STRICT_ISAAC_EVAL=1 \
SIMPLE_TASK_ASSETS_ROOT=/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/assets \
EVAL_GPU=3 \
NUM_EPISODES=10 \
EPISODE_START=0 \
MAX_EPISODE_STEPS=500 \
SAVE_VIDEO=1 \
SKIP_STABILIZE=1 \
GR00T_EXECUTION_HORIZON=34 \
FULLSTATE_TASK2_GR00T_EVAL_DIR=data/evals_task2_20260805_strict_isaac_gr00t_kimodo_textop_10eps \
bash scripts/deploy/fullstate_task2_gr00t_rot6d59_kimodo_textop_eval.sh eval
```

# Task3/4 替换项：使用各自 wrapper 和 checkpoint；场景资产根目录保持不变。



# gr00t->sonic链路
# 原始训练链路
```bash
cd /pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Isaac-GR00T || exit 1

run_train() {
  local dataset_path="$1"
  local output_dir="$2"

  GR00T_BACKBONE_PATH=/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Psi0/huggingface/hub/models--nvidia--Cosmos-Reason2-2B/snapshots/9ce19a195e423419c349abfc86fd07178b230561 \
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  uv run --no-sync torchrun \
    --nproc_per_node=8 \
    --master_port=29531 \
    gr00t/experiment/launch_finetune.py \
    --base-model-path /pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Psi0/checkpoints/GR00T-N1.7-3B \
    --dataset-path "$dataset_path" \
    --embodiment-tag UNITREE_G1_SONIC \
    --num-gpus 8 \
    --output-dir "$output_dir" \
    --max-steps 80000 \
    --save-steps 10000 \
    --save-total-limit 5 \
    --global-batch-size 64 \
    --dataloader-num-workers 4 \
    --warmup-ratio 0.05 \
    --weight-decay 1e-5 \
    --learning-rate 1e-4 \
    --color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
    --use-wandb \
    --wandb-project gr00t-n1.7
}

run_train \
  /pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Psi0/data/output/fullstate_20260816_task2_isaac_gr00t_new \
  /pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Isaac-GR00T/outputs/task2-gr00t-n17-sonic-native-newbg \
&& run_train \
  /pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Psi0/data/output/fullstate_20260816_task3_isaac_new_gr00t_new \
  /pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Isaac-GR00T/outputs/task3-gr00t-n17-sonic-native-newbg \
&& run_train \
  /pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Psi0/data/output/fullstate_20260816_task4_isaac_new_gr00t_new \
  /pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Isaac-GR00T/outputs/task4-gr00t-n17-sonic-native-newbg
```
# 训练 
```bash

cd /pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Isaac-GR00T

CUDA_VISIBLE_DEVICES=0,1,2,3 \
NUM_GPUS=4 \
MASTER_PORT=29531 \
MAX_STEPS=80000 \
SAVE_STEPS=10000 \
GLOBAL_BATCH_SIZE=64 \
DATALOADER_NUM_WORKERS=8 \
TRAIN_RTC=0 \
TRAIN_PREFIX_RTC=0 \
USE_WANDB=1 \
uv run --no-sync bash examples/finetune.sh \
  --base-model-path /pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Psi0/checkpoints/GR00T-N1.7-3B \
  --dataset-path /pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Psi0/data/output/scq_fullstate_20260828_task6_all_gr00t_new \
  --embodiment-tag UNITREE_G1_SONIC \
  --output-dir /pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Isaac-GR00T/outputs/task6-scq-gr00t-n17-sonic-native-A800-20260828 \
  --wandb-project gr00t-n1.7

```

``` #groot->sonic lqb
cd /pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Isaac-GR00T

CUDA_VISIBLE_DEVICES=4,5,6,7 \
NUM_GPUS=4 \
MASTER_PORT=29533 \
MAX_STEPS=80000 \
SAVE_STEPS=10000 \
GLOBAL_BATCH_SIZE=64 \
DATALOADER_NUM_WORKERS=4 \
TRAIN_RTC=0 \
TRAIN_PREFIX_RTC=0 \
USE_WANDB=1 \
uv run --no-sync bash examples/finetune.sh \
  --base-model-path /pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Psi0/checkpoints/GR00T-N1.7-3B \
  --dataset-path /pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Psi0/data/output/scq_fullstate_20260825_task5_newlight_gr00tnew \
  --embodiment-tag UNITREE_G1_SONIC \
  --output-dir /pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Isaac-GR00T/outputs/task5-scq-gr00t-n17-sonic-native-newlight-A800 \
  --wandb-project gr00t-n1.7

# 终端1
```
cd /pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Psi0

SERVE_GPU=1 \
GR00T_PREFIX_RTC=1 \
GR00T_PREFIX_RTC_TIMESTEP_MODE=groot_clean \
GR00T_EXECUTION_HORIZON=34 \
GR00T_MODEL_PATH=/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Isaac-GR00T/outputs/task4-gr00t-n17-rot6d59-kimodo-textop-prefix-rtc-groot-clean/checkpoint-160000 \
bash scripts/deploy/fullstate_task4_gr00t_n17_sonic_eval.sh serve

```

# 终端2
```bash
cd /pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Psi0

EVAL_GPU=3 \
NUM_EPISODES=10 \
EPISODE_START=0 \
MAX_EPISODE_STEPS=500 \
SAVE_VIDEO=1 \
GR00T_EXECUTION_HORIZON=34 \
SKIP_STABILIZE=1 \
TASK3_INIT_FROM_RECORDINGS=1 \
TASK3_RECORDINGS_DIR=/pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/HumanoidVLA_MJ/output/20260612_task3 \
TASK3_RECORDING_SEED=0 \
GR00T_SONIC_EVAL_DIR=data/evals_task3_gr00t_n17_sonic_prefix_rtc_groot_clean_10eps \
bash scripts/deploy/fullstate_task3_gr00t_n17_sonic_eval.sh eval

```

# Task 1 Prefix RTC 八卡串行训练
```bash
cd /pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Isaac-GR00T

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NUM_GPUS=8 \
MASTER_PORT=29532 \
MAX_STEPS=80000 \
SAVE_STEPS=10000 \
GLOBAL_BATCH_SIZE=64 \
DATALOADER_NUM_WORKERS=16 \
TRAIN_PREFIX_RTC=1 \
GR00T_PREFIX_RTC_TIMESTEP_MODE=groot_clean \
USE_WANDB=1 \
uv run --no-sync bash examples/finetune.sh \
  --base-model-path /pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Psi0/checkpoints/GR00T-N1.7-3B \
  --dataset-path /pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Psi0/data/output/fullstate_2026615_task1_isaac_new_rot6d59 \
  --modality-config-path /pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Isaac-GR00T/examples/unitree_g1_rot6d59_config.py \
  --embodiment-tag NEW_EMBODIMENT \
  --output-dir /pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Isaac-GR00T/outputs/task1-gr00t-n17-rot6d59-kimodo-textop-prefix-rtc-groot-clean-A800-newbg \
  --wandb-project gr00t-n1.7 \
&&
cd /pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Isaac-GR00T

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NUM_GPUS=8 \
MASTER_PORT=29531 \
MAX_STEPS=80000 \
SAVE_STEPS=10000 \
GLOBAL_BATCH_SIZE=64 \
DATALOADER_NUM_WORKERS=4 \
TRAIN_PREFIX_RTC=1 \
GR00T_PREFIX_RTC_TIMESTEP_MODE=groot_clean \
USE_WANDB=1 \
uv run --no-sync bash examples/finetune.sh \
  --base-model-path /pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Psi0/checkpoints/GR00T-N1.7-3B \
  --dataset-path /pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Psi0/data/output/fullstate_20260615_task1_isaac_gr00t_new \
  --embodiment-tag UNITREE_G1_SONIC \
  --output-dir /pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Isaac-GR00T/outputs/task1-gr00t-n17-sonic-prefix-rtc-groot-clean \
  --wandb-project gr00t-n1.7 \
  --resume-from-checkpoint
```

# Task 1 Sonic native 八卡训练（不启用 RTC / Prefix RTC）
```bash
cd /pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Isaac-GR00T

CUDA_VISIBLE_DEVICES=4,5,6,7 \
NUM_GPUS=4 \
MASTER_PORT=29531 \
MAX_STEPS=80000 \
SAVE_STEPS=10000 \
GLOBAL_BATCH_SIZE=64 \
DATALOADER_NUM_WORKERS=4 \
TRAIN_RTC=0 \
TRAIN_PREFIX_RTC=0 \
USE_WANDB=1 \
uv run --no-sync bash examples/finetune.sh \
  --base-model-path /pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Psi0/checkpoints/GR00T-N1.7-3B \
  --dataset-path /pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Psi0/data/output/task6_data/task6_scq_isaac_gr00t_new \
  --embodiment-tag UNITREE_G1_SONIC \
  --output-dir /pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Isaac-GR00T/outputs/task6-gr00t-n17-sonic-native-newbg \
  --wandb-project gr00t-n1.7
```
