ADAPTER_PATH=/home/lightwheel/erdao.liang/Qwen3-VL/qwen-vl-finetune/output/archive/2tasks_success/checkpoint-2000
DATA_SAMPLES_PATH=inference/data/example_data_samples.json
DATA_ROOT=/home/lightwheel/erdao.liang/LightwheelData/
TARGET_DEMO_PATH="$DATA_ROOT/1W_Libero_Piper_Negative/L90L6PutTheWhiteMugOnThePlate/L90L6PutTheWhiteMugOnThePlate_1763973030758697"
REFERENCE_DEMO_PATH="$DATA_ROOT/1W_Libero_Piper/L90L6PutTheWhiteMugOnThePlate/L90L6PutTheWhiteMugOnThePlate_1757927758234053"
TASK_DECRIPTION="Put the white mug on the plate"
CONFIG=dataset/build_config.yaml


python3 -m inference.cli.eval_pairwise \
    --adapter $ADAPTER_PATH \
    --data-samples $DATA_SAMPLES_PATH \
    --data-root $DATA_ROOT \
    --batch-size 64


python3 -m inference.cli.pairwise_demo \
    --adapter $ADAPTER_PATH \
    --target-demo $TARGET_DEMO_PATH \
    --i 3 --j 20 \
    --reference-demo $REFERENCE_DEMO_PATH \
    --task-desc "$TASK_DECRIPTION" \
    --config $CONFIG

python3 -m inference.cli.curve_demo \
    --adapter $ADAPTER_PATH \
    --target-demo $TARGET_DEMO_PATH \
    --reference-demo $REFERENCE_DEMO_PATH \
    --task-desc "$TASK_DECRIPTION" \
    --config $CONFIG \
    --step-interval 2 \ 
    --output-dir outputs/inference_progress_curve \
    --output-fps 2.0

python3 -m inference.cli.eval_curves \
    --adapter $ADAPTER_PATH \
    --demo-list /home/lightwheel/erdao.liang/Qwen3-VL/utils/train_eval_split_index.json \
    --reference-demo $REFERENCE_DEMO_PATH \
    --task-desc "$TASK_DECRIPTION" \
    --config $CONFIG \
    --step-interval 2 \
    --output outputs/eval_curves.json