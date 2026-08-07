# Lerobo数据构建（fullstate->rot6d59; gr00t->sonic链路的gr00t_new需要用原数据,在采集电脑中转换）
# 放在mnt/pfs/humanoid/yzh/Psi0/data/output
```
cd /pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Psi0
source .venv-psi/bin/activate

python scripts/data/build_movepick_rot6d59_dataset.py \
  --src data/output/fullstate_20260625_task2 \
  --out data/output/fullstate_20260625_task2_rot6d59

```


# gr00t 1.7 训练 （更换任务数据集dataset-path）
# 权重路径：output-dir/checkpoint-80000
```bash
cd /pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Isaac-GR00T

CUDA_VISIBLE_DEVICES=0,1,2,3 \
NUM_GPUS=4 \
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
  --dataset-path /pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Psi0/data/output/fullstate_20260729_task4_rot6d59 \
  --modality-config-path /pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Isaac-GR00T/examples/unitree_g1_rot6d59_config.py \
  --embodiment-tag NEW_EMBODIMENT \
  --output-dir /pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Isaac-GR00T/outputs/task4-gr00t-n17-rot6d59-kimodo-textop-prefix-rtc-groot-clean-A800 \
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



# gr00t->sonic链路
# 训练 
```bash

cd /pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Isaac-GR00T

CUDA_VISIBLE_DEVICES=0,1,2,3 \
NUM_GPUS=4 \
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
  --dataset-path /pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Psi0/data/output/fullstate_20260729_task4_gr00t_new \
  --embodiment-tag UNITREE_G1_SONIC \
  --output-dir /pfs/pfs-oHNwH0/mnt/pfs/humanoid/yzh/Isaac-GR00T/outputs/task4-gr00t-n17-sonic-prefix-rtc-groot-clean \
  --wandb-project gr00t-n1.7 \
```
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
