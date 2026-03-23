#!/bin/bash

# 多 GPU 并行推理示例脚本
# 自动使用所有可用 GPU 加速推理

# ==================== 配置 ====================

BASE_MODEL="models/Qwen-VL-2B-Instruct"
ADAPTER=/home/erdao.liang/Qwen3-VL/qwen-vl-finetune/output/lerobot_Robocasa_X7s_H800_arange_veg_lora_newdata/checkpoint-900
DATA_ROOT=/home/lightwheel/erdao.liang/LightwheelData/slowdata/

# 检测可用 GPU 数量
# NUM_GPUS=$(nvidia-smi -L | wc -l)
NUM_GPUS=4
echo "检测到 $NUM_GPUS 张 GPU"

# ============================================

echo ""
echo "========== 1. Pairwise 批量评估（多 GPU）=========="
# python3 -m inference.cli.eval_pairwise \
#     --base-model "$BASE_MODEL" \
#     --adapter "$ADAPTER" \
#     --data-samples /home/lightwheel/erdao.liang/LightwheelData/slowdata/data_1W_Robocasa_X7s/eval/ArrangeVegetables_eval.json \
#     --data-root "$DATA_ROOT" \
#     --batch-size 32 \
#     --num-gpus "$NUM_GPUS" \
#     --output outputs/eval_pairwise_arrange_vegetables_multi_gpu.json

echo ""
echo "========== 2. Progress Curve 单 Demo（多 GPU）=========="
# python3 -m inference.cli.curve_demo \
#     --base-model "$BASE_MODEL" \
#     --adapter "$ADAPTER" \
#     --target-demo /home/lightwheel/erdao.liang/LightwheelData/slowdata/1W_Robocasa_X7s_More/ArrangeVegetables/ArrangeVegetables_1761706678200653 \
#     --reference-demo /home/lightwheel/erdao.liang/LightwheelData/slowdata/1W_Robocasa_X7s_More/ArrangeVegetables/ArrangeVegetables_1762240854119666 \
#     --task-desc "Put the vegetables on the cutting board" \
#     --config dataset/build_config_15tasks.yaml \
#     --batch-size 8 \
#     --num-gpus "$NUM_GPUS" \
#     --output-dir outputs/inference_progress_curve_multi_gpu

echo ""
echo "========== 3. Progress Curve 批量评估（多 GPU）=========="
python3 -m inference.cli.eval_curves \
    --base-model "$BASE_MODEL" \
    --adapter "$ADAPTER" \
    --demo-list inference/data/demo_list_arrange_vegetables_eval.json \
    --reference-demo /home/lightwheel/erdao.liang/LightwheelData/slowdata/1W_Robocasa_X7s_More/ArrangeVegetables/ArrangeVegetables_1762240854119666 \
    --task-desc "Put the vegetables on the cutting board" \
    --config dataset/build_config_15tasks.yaml \
    --step-interval 2 \
    --batch-size 8 \
    --num-gpus "$NUM_GPUS" \
    --output outputs/eval_curves_multi_gpu.json \
    --plot-output outputs/eval_curves_multi_gpu.png

echo ""
echo "========== 全部完成 =========="
