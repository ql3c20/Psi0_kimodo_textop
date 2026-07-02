#!/bin/bash

export OMP_NUM_THREADS=32
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1,2,3,4}

source .venv-psi/bin/activate

NPROC_PER_NODE=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
ulimit -n 65535
echo "Training with $NPROC_PER_NODE GPUs"

export task=${1:-kimodo_augmented_v2}
export exp=${2:-movepick-kimodo-v2}

echo "Task: $task"
echo "Experiment name: $exp"

args="
finetune_simple_psi0_config \
--seed=292285 \
--exp=$exp \
--train.name=finetune \
--train.data_parallel=ddp \
--train.mixed_precision=bf16 \
--train.train_batch_size=16 \
--train.max_checkpoints_to_keep=5 \
--train.gradient_accumulation_steps=1 \
--train.learning_rate=1e-4 \
--train.max_training_steps=40000 \
--train.warmup_ratio=None \
--train.warmup_steps=1000 \
--train.checkpointing_steps=10000 \
--train.validation_steps=500 \
--train.val_num_batches=20 \
--train.max_grad_norm=1.0 \
--train.lr_scheduler_type=cosine \
--train.lr_scheduler_kwargs.weight_decay=1e-6 \
--train.lr_scheduler_kwargs.betas 0.95 0.999 \
--log.report_to=wandb \
--data.root_dir=/pfs/pfs-ilWc5D/yzh/Psi0/data/simple/G1WholebodyXMovePickTeleop-v0 \
--data.train-repo-ids=$task \
--data.transform.repack.state-key=observation.full_state \
--data.transform.repack.action-key=action.policy_action \
--data.transform.repack.pad-action-dim=44 \
--data.transform.repack.pad-state-dim=49 \
--data.transform.field.stat-path=meta/stats.json \
--data.transform.field.stat-action-key=action.policy_action \
--data.transform.field.stat-state-key=observation.full_state \
--data.transform.field.action_norm_type=bounds \
--data.transform.field.no-use-norm-mask \
--data.transform.field.normalize-state \
--data.transform.field.pad-action-dim=44 \
--data.transform.field.pad-state-dim=49 \
--data.transform.model.img-aug \
--data.transform.model.resize.size 180 320 \
--data.transform.model.center_crop.size 180 320 \
--model.model_name_or_path=/pfs/pfs-ilWc5D/yzh/Psi0/huggingface/psi0/pre.fast.1by1.2601091803.ckpt.ego200k.he30k \
--model.pretrained-action-header-path=/pfs/pfs-ilWc5D/yzh/Psi0/huggingface/psi0/postpre.1by1.pad36.2601131206.ckpt.he30k \
--model.noise-scheduler=flow \
--model.train-diffusion-steps=1000 \
--model.n_conditions=0 \
--model.action-chunk-size=30 \
--model.action-dim=44 \
--model.action-exec-horizon=30 \
--model.observation-horizon=1 \
--model.odim=49 \
--model.view_feature_dim=2048 \
--model.no-tune-vlm \
--model.no-use_film \
--model.no-combined_temb \
--model.rtc \
--model.max-delay=8
"

torchrun --nproc_per_node=$NPROC_PER_NODE --master_port=29500 scripts/train.py \
    ${args}
