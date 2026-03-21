
ADAPTER="/home/jialeng/LightwheelData/checkpoint-success-6tasks"
DATASET_ROOT="/home/jialeng/LightwheelDataFast/Robocasa_lerobot_6tasks"
OUTPUT_ROOT="/home/jialeng/LightwheelDataFast/Robocasa_lerobot_6tasks_with_progress2"
REFERENCE_MAP="/home/jialeng/Qwen3-VL/inference/data/6tasks_referece_map.json"

python3 -m inference.backfill \
    --adapter $ADAPTER \
    --dataset-root $DATASET_ROOT \
    --output-root $OUTPUT_ROOT \
    --reference-map $REFERENCE_MAP \
    --config dataset/configs/build_config_15tasks.yaml \
    --num-gpus 6 \
    --batch-size 8 \
    --episode-chunk-size 4 \
    --message-chunk-size 128