#!/bin/bash

# 批量处理demos目录下的所有demo，生成progress曲线

DEMOS_DIR=/home/lightwheel/erdao.liang/LightwheelData/1W_Libero_Piper_Negative/L90L6PutTheWhiteMugOnThePlate
ADAPTER=qwen-vl-finetune/output/checkpoint-30800
CONFIG=/home/lightwheel/erdao.liang/Qwen3-VL/dataset/build_config.yaml
TASK_DESC="Put the white mug on the plate"
REFERENCE_DEMO=/home/lightwheel/erdao.liang/LightwheelData/1W_Libero_Piper/L90L6PutTheWhiteMugOnThePlate/L90L6PutTheWhiteMugOnThePlate_1757927758234053
BASE_MODEL=models/Qwen-VL-2B-Instruct
STEP_INTERVAL=2
OUTPUT_BASE_DIR=outputs/batch_curves_negative
OUTPUT_FPS=5.0

for demo_dir in "$DEMOS_DIR"/*; do
    [ ! -d "$demo_dir" ] && continue
    
    demo_name=$(basename "$demo_dir")
    output_dir="$OUTPUT_BASE_DIR/$demo_name"
    
    echo "处理: $demo_name -> $output_dir"
    
    python3 inference/inference_curve_from_demo.py \
        --adapter "$ADAPTER" \
        --target-demo "$demo_dir" \
        --task-desc "$TASK_DESC" \
        --config "$CONFIG" \
        --step-interval "$STEP_INTERVAL" \
        --output-dir "$output_dir" \
        --output-fps "$OUTPUT_FPS" \
        --reference-demo "$REFERENCE_DEMO" 
    echo ""
done

echo "批量处理完成！输出目录: $OUTPUT_BASE_DIR"

