#!/bin/bash

# 批量处理demos目录下的所有demo，生成密集采样的progress曲线JSON数据
# 使用方法：从项目根目录运行此脚本
# 采样方式：(0, delta_t), (1, delta_t+1), (2, delta_t+2), ..., (T-1, min(T-1+delta_t, T-1))
# 始终输出T个点

# ============ 配置参数 ============
DATASET_ROOT=/home/lightwheel/erdao.liang/LightwheelData/Example_dataset
DEMOS_DIR=$DATASET_ROOT/GrabTheBlockAndLiftItUp
ADAPTER=qwen-vl-finetune/output/archive/pickup_cube
CONFIG=/home/lightwheel/erdao.liang/Qwen3-VL/dataset/PickUpCube_Build_config.yaml
TASK_DESC="Grab the block and lift it up"
REFERENCE_DEMO=$DATASET_ROOT/GrabTheBlockAndLiftItUp/GrabTheBlockAndLiftItUp_0013
BASE_MODEL=models/Qwen-VL-2B-Instruct
DELTA_T=2
OUTPUT_BASE_DIR=outputs/dense_curves_pickup_cube

# ============ 批量处理 ============
echo "开始批量处理..."
echo "Demos目录: $DEMOS_DIR"
echo "输出目录: $OUTPUT_BASE_DIR"
echo "Delta T: $DELTA_T"
echo ""

# 创建输出目录
mkdir -p "$OUTPUT_BASE_DIR"

# 统计变量
total_demos=0
success_demos=0
failed_demos=0

# 遍历所有demo目录
for demo_dir in "$DEMOS_DIR"/*; do
    [ ! -d "$demo_dir" ] && continue

    total_demos=$((total_demos + 1))
    demo_name=$(basename "$demo_dir")
    
    echo "=========================================="
    echo "[$total_demos] 处理: $demo_name"
    echo "=========================================="
    
    # 运行推理脚本
    if python3 inference/inference_dense_curve_from_demo.py \
        --base-model "$BASE_MODEL" \
        --adapter "$ADAPTER" \
        --target-demo "$demo_dir" \
        --task-desc "$TASK_DESC" \
        --config "$CONFIG" \
        --delta-t "$DELTA_T" \
        --output-dir "$OUTPUT_BASE_DIR" \
        --reference-demo "$REFERENCE_DEMO" 2>&1; then
        success_demos=$((success_demos + 1))
        echo "✓ 成功: $demo_name"
    else
        failed_demos=$((failed_demos + 1))
        echo "✗ 失败: $demo_name"
    fi
    echo ""
done

# 输出统计信息
echo "=========================================="
echo "批量处理完成！"
echo "=========================================="
echo "总demo数: $total_demos"
echo "成功: $success_demos"
echo "失败: $failed_demos"
echo "输出目录: $OUTPUT_BASE_DIR"
echo ""

