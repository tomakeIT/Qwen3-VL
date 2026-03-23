
ADAPTER="/home/jialeng/LightwheelData/checkpoint-success-6tasks"
DATASET_ROOT="/home/jialeng/LightwheelDataFast/Robocasa_lerobot_6tasks"
OUTPUT_ROOT="/home/jialeng/LightwheelDataFast/Robocasa_lerobot_6tasks_with_progress_new14"
REFERENCE_MAP="/home/jialeng/Qwen3-VL/inference/data/6tasks_referece_map.json"
RUN_NAME="lerobot_backfill_6tasks"

export WANDB_PROJECT="qwen3vl-rewardmodel"

python3 -m inference.cli.backfill_sharded \
    --adapter $ADAPTER \
    --dataset-root $DATASET_ROOT \
    --output-root $OUTPUT_ROOT \
    --reference-map $REFERENCE_MAP \
    --config dataset/configs/build_config_15tasks.yaml \
    --num-gpus 1 \
    --batch-size 128 \
    --episode-chunk-size 4 \
    --input-mode video_local \
    --wandb-run-name $RUN_NAME \
    --limit-episodes 4 \
    --profile-output "./backfill_profile_updated_new1.prof" \
    --global-build-workers 16