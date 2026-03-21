"""
demo_path, i, j, reference_demo_path -> inference result
"""

import argparse
from common.demo_scan import scan_demo_frames
from common.io_utils import load_config_namespace
from common.messages import (
    build_messages_from_demo,
    build_messages_from_inputs,
    sample_reference_demo_pack,
)
from utils.data_formatting import compute_delta_progress_label_int


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=str, default="models/Qwen-VL-2B-Instruct", help="基础模型路径")
    parser.add_argument("--adapter", type=str, required=True, help="LoRA适配器路径")
    parser.add_argument("--target-demo", type=str, required=True, help="target demo路径")
    parser.add_argument("--i", type=int, required=True, help="起始帧索引")
    parser.add_argument("--j", type=int, required=True, help="结束帧索引")
    parser.add_argument("--reference-demo", type=str, help="reference demo路径")
    parser.add_argument("--task-desc", type=str, required=True, help="任务描述")
    parser.add_argument("--config", type=str, required=True, help="YAML配置文件路径")
    
    args = parser.parse_args()
    config = load_config_namespace(args.config)
    from inferencer import DeltaProgressInference
    
    target_views = config.sampling.required_views
    inference = DeltaProgressInference(
        base_model_path=args.base_model,
        adapter_path=args.adapter,
    )
    messages = build_messages_from_demo(
        target_demo_path=args.target_demo,
        i=args.i,
        j=args.j,
        reference_demo_path=args.reference_demo,
        task_desc=args.task_desc,
        target_views=target_views,
        reference_config=config.reference,
    )
    predicted_delta_progress = inference.infer_from_messages(messages)

    _, T = scan_demo_frames(args.target_demo, target_views)
    gt_delta_progress = compute_delta_progress_label_int(args.i, args.j, T)
    print(f"Predicted Delta Progress: {predicted_delta_progress}")
    print(f"Ground Truth Delta Progress: {gt_delta_progress} (only if it is a successful demo)")


if __name__ == "__main__":
    main()

