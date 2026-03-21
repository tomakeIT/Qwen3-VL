#!/bin/bash

# Distributed training configuration
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NNODES=${WORLD_SIZE:-1}
NPROC_PER_NODE=${NPROC_PER_NODE:-2}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}

# DeepSpeed configuration: load from universal checkpoint
deepspeed=./scripts/zero2-universal-load.json

# Model configuration
model_path=/home/jialeng/Qwen3-VL/models/Qwen-VL-2B-Instruct

# Training hyperparameters
lr=5e-4
batch_size=20
grad_accum_steps=8
model_max_length=2048
num_train_epochs=100

# Training entry point
entry_file=qwenvl/train/train_qwen.py

# Dataset configuration
datasets=/home/jialeng/data_1W_Robocasa_X7s_new::ArrangeVegetables+CheesyBread+CoffeeSetupMug+CloseDishwasher+CloseDrawer+OpenDishwasher

# wandb run
run_name="lerobot_Robocasa_X7s_6tasks_RTX6000_universal_resume"

# Important: output_dir should be the parent directory containing converted checkpoint-* folders.
# Example layout:
#   ${output_dir}/checkpoint-1000/global_step1000
#   ${output_dir}/checkpoint-1000/latest_universal
#   ${output_dir}/checkpoint-1000/trainer_state.json
output_dir=./output/lerobot_Robocasa_X7s_6tasks_RTX6000_resume
save_total_limit=10
save_steps=1000

export WANDB_PROJECT="qwen3vl-rewardmodel"

IFS=',' read -r -a VISIBLE_GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
VISIBLE_GPU_COUNT=${#VISIBLE_GPU_IDS[@]}
if (( NPROC_PER_NODE > VISIBLE_GPU_COUNT )); then
  echo "Invalid config: NPROC_PER_NODE=${NPROC_PER_NODE}, but CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} (${VISIBLE_GPU_COUNT} GPUs visible)."
  echo "Please reduce NPROC_PER_NODE or expose more GPUs."
  exit 1
fi

echo "Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}, NPROC_PER_NODE=${NPROC_PER_NODE}"

args="
    --deepspeed ${deepspeed} \
    --model_name_or_path ${model_path} \
    --dataset_use ${datasets} \
    --data_flatten True \
    --tune_mm_vision False \
    --tune_mm_mlp True \
    --tune_mm_llm True \
    --bf16 \
    --lora_enable True \
    --output_dir ${output_dir} \
    --num_train_epochs ${num_train_epochs} \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size $((batch_size*2)) \
    --gradient_accumulation_steps ${grad_accum_steps} \
    --max_pixels 50176 \
    --min_pixels 784 \
    --eval_strategy no \
    --save_strategy steps \
    --save_steps ${save_steps} \
    --save_total_limit ${save_total_limit} \
    --learning_rate ${lr} \
    --weight_decay 0 \
    --warmup_ratio 0.03 \
    --max_grad_norm 1 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --model_max_length ${model_max_length} \
    --gradient_checkpointing True \
    --dataloader_num_workers 12 \
    --dataloader_prefetch_factor 2 \
    --dataloader_persistent_workers True \
    --run_name ${run_name} \
    --report_to wandb"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
torchrun --nproc_per_node=${NPROC_PER_NODE} \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
         ${entry_file} ${args}
