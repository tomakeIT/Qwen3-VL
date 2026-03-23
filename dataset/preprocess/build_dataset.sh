#!/bin/bash
# 数据集构建流程脚本
# 依次执行：提取任务描述 -> 拆分视频 -> 提取帧 -> 构建Qwen数据集

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 配置参数（根据需要修改）
DATASET_PATH="1W_Libero_X7s"
NUM_VIEWS=6
FPS=2.0
SCALE=1.0
CONFIG_FILE="${SCRIPT_DIR}/build_config.yaml"


# 步骤1: 提取任务描述
echo "[1/4] 提取任务描述..."
python3 "${SCRIPT_DIR}/extract_task_descriptions.py" \
    --dataset-path "$DATASET_PATH" \
    --output "${DATASET_PATH}/task_descriptions.json"
echo "✓ 任务描述提取完成"

# 步骤2: 拆分视频
echo "[2/4] 拆分视频..."
python3 "${SCRIPT_DIR}/split_videos.py" \
    --dataset-path "$DATASET_PATH" \
    --num-views "$NUM_VIEWS"
echo "✓ 视频拆分完成"

# 步骤3: 提取帧
echo "[3/4] 提取帧..."
python3 "${SCRIPT_DIR}/extract_frames.py" \
    --dataset-path "$DATASET_PATH" \
    --fps "$FPS" \
    --scale "$SCALE"
echo "✓ 帧提取完成"

# 步骤4: 构建Qwen数据集
echo "[4/4] 构建Qwen数据集..."
python3 "${SCRIPT_DIR}/build_qwen_dataset.py" \
    --config "$CONFIG_FILE" \
    --root "$DATASET_PATH"
echo "✓ 数据集构建完成"

