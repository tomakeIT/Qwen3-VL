
conda activate qwen

ADAPTER=/

python3 inference/run_backfill_lerobot_progress.py \
    --adapter /path/to/checkpoint \
    --dataset-root /home/jialeng/LightwheelData/Task_Robocasa_X7s_lerobot \
    --output-root /path/to/Task_Robocasa_X7s_lerobot_with_progress \
    --reference-map /path/to/reference_map.json \
    --config dataset/configs/build_config_15tasks.yaml \
    --pair-interval 50 \
    --num-gpus 6 \
    --batch-size 8 \
    --episode-chunk-size 4 \
    --message-chunk-size 128