#!/bin/bash

# ==================== 基础配置 ====================
BASE_MODEL="models/Qwen-VL-2B-Instruct"
ADAPTER="/home/jialeng/Qwen3-VL/qwen-vl-finetune/output/lerobot_Robocasa_X7s_6tasks_RTX6000/checkpoint-3300"

# 只需要这两个根目录：
# 1) 处理后元数据目录（固定包含 train_metadata.json / eval_metadata.json）
PROCESSED_META_ROOT="/home/jialeng/LightwheelData/data_1W_Robocasa_X7s_new"
# 2) 原始数据集根目录（本机路径，用于拼 episode 路径 raw_dataset_root/<task>/<episode_id>；需含 task_descriptions.json）
RAW_DATASET_ROOT="/home/jialeng/LightwheelData/1W_Robocasa_X7s_More"

# 评估配置
# 任务直接按 list 写在这里
TASKS=(
  "ArrangeVegetables"
  "CheesyBread"
)
DEMOS_PER_TASK=10            # 每个任务最多评估 demo 数量

# 推理参数
CONFIG="dataset/configs/build_config_15tasks.yaml"
STEP_INTERVAL=2           # 为空表示不传 --end-frame
BATCH_SIZE=32
NUM_GPUS=6
GLOBAL_BUILD_WORKERS=16

# 输出目录
OUTPUT_ROOT="outputs/eval_multi_tasks_from_metadata"

# ==================================================

python3 -m inference.cli.eval_tasks \
  --base-model "${BASE_MODEL}" \
  --adapter "${ADAPTER}" \
  --processed-meta-root "${PROCESSED_META_ROOT}" \
  --raw-dataset-root "${RAW_DATASET_ROOT}" \
  --tasks "${TASKS[@]}" \
  --demos-per-task "${DEMOS_PER_TASK}" \
  --config "${CONFIG}" \
  --step-interval "${STEP_INTERVAL}" \
  --batch-size "${BATCH_SIZE}" \
  --num-gpus "${NUM_GPUS}" \
  --global-build-workers "${GLOBAL_BUILD_WORKERS}" \
  --output-root "${OUTPUT_ROOT}"
