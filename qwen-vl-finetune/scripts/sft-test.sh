#!/bin/bash

#################### 进入单独会话的方法：
# 方案 3：使用 screen（推荐用于长时间运行）
# 安装screen：sudo apt-get install screen  
# 创建新的 screen 会话screen -S training
# 在 screen 中运行cd /home/lightwheel/erdao.liang/Qwen3-VL/qwen-vl-finetunebash; scripts/sft-test.sh
# 按 Ctrl+A 然后按 D 来 detach（分离），进程继续运行# 重新连接：screen -r training
# 在会话中输入exit结束screen会话
########################33

# Distributed training configuration
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NNODES=${WORLD_SIZE:-1}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

# DeepSpeed configuration
deepspeed=./scripts/zero2.json

# Model configuration
model_path=/home/lightwheel/erdao.liang/Qwen3-VL/models/Qwen-VL-2B-Instruct # Using HuggingFace model ID

# Training hyperparameters
lr=5e-4
batch_size=12
grad_accum_steps=1
model_max_length=4096
num_train_epochs=200

# Training entry point
entry_file=qwenvl/train/train_qwen.py

# Dataset configuration (replace with public dataset names)
datasets=pick_up_cream_cheese_and_put_in_tray,put_white_mug_on_plate,put_both_moka_pots_on_stove,stack_middle_black_bowl_on_back_black_bowl

# wandb run
run_name="1129-4Tasks"


# checkpoint saving
output_dir=./output
save_total_limit=10
save_steps=2000

# Wandb configuration
export WANDB_PROJECT="qwen3vl-rewardmodel"

# Training arguments
args="
    --deepspeed ${deepspeed} \
    --model_name_or_path "${model_path}" \
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
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps ${save_steps} \
    --save_total_limit ${save_total_limit} \
    --learning_rate ${lr} \
    --weight_decay 0 \
    --warmup_ratio 0.03 \
    --max_grad_norm 1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length ${model_max_length} \
    --gradient_checkpointing True \
    --dataloader_num_workers 16 \
    --run_name ${run_name} \
    --report_to wandb"

# Launch training
torchrun --nproc_per_node=${NPROC_PER_NODE} \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
         ${entry_file} ${args}


            #  zero2.json: "gradient_clipping": 1.0, 