#!/bin/bash

# 评估 ArrangeVegetables 任务在验证集上的效果
# 使用 `python -m inference.cli.eval_curves` 批量推理 progress curves

# ==================== 配置区域（请根据你的实际情况修改）====================

# 基础模型路径
BASE_MODEL="models/Qwen-VL-2B-Instruct"

# LoRA 适配器路径（请修改为你训练好的模型路径）
# 示例: ADAPTER="qwen-vl-finetune/output/checkpoint-10000"
ADAPTER=/home/jialeng/Qwen3-VL/qwen-vl-finetune/output/lerobot_Robocasa_X7s_6tasks_RTX6000/checkpoint-3300

# demo list JSON 文件路径（包含要评估的验证集 demo 列表）
DEMO_LIST="inference/data/demo_list_arrange_vegetables_train_2.json"

# reference demo 路径（从训练集中选一个作为 reference）
REFERENCE_DEMO="/home/jialeng/LightwheelData/1W_Robocasa_X7s_More/ArrangeVegetables/ArrangeVegetables_1762240854119666"

# 任务描述
TASK_DESC="Put the vegetables on the cutting board"

# YAML 配置文件路径（包含视角配置等）
CONFIG="dataset/configs/build_config_15tasks.yaml"

# 采样间隔（每隔多少帧采样一次）
STEP_INTERVAL=2

# 起始帧和结束帧（可选，默认从0到最后一帧）
START_FRAME=0
# END_FRAME=100  # 如果不设置，默认到最后一帧

# Batch大小（大于1时使用batch推理加速）
BATCH_SIZE=32

# 输出结果保存路径
OUTPUT="outputs/eval_arrange_vegetables_results_4.json"

# 曲线图保存路径
PLOT_OUTPUT="outputs/eval_arrange_vegetables_curves_4.png"

# 使用的GPU数量
NUM_GPUS=6

# ============================================================================

echo "=========================================="
echo "开始评估 ArrangeVegetables 任务"
echo "=========================================="
echo "模型: $ADAPTER"
echo "评估demo数量: $(python3 -c "import json; data=json.load(open('$DEMO_LIST')); print(len(data.get('eval', {})))")"
echo ""

python3 -m inference.cli.eval_curves \
    --base-model "$BASE_MODEL" \
    --adapter "$ADAPTER" \
    --demo-list "$DEMO_LIST" \
    --reference-demo "$REFERENCE_DEMO" \
    --task-desc "$TASK_DESC" \
    --config "$CONFIG" \
    --step-interval "$STEP_INTERVAL" \
    --start-frame "$START_FRAME" \
    --batch-size "$BATCH_SIZE" \
    --output "$OUTPUT" \
    --plot-output "$PLOT_OUTPUT" \
    --num-gpus "$NUM_GPUS"


echo ""
echo "=========================================="
echo "评估完成！"
echo "结果保存至: $OUTPUT"
echo "曲线图保存至: $PLOT_OUTPUT"
echo "=========================================="
