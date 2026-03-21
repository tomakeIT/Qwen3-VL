"""
单个 demo 上做一次 pairwise delta progress 推理。
"""

from __future__ import annotations

import argparse

from inference.demo_utils import build_messages_from_demo, scan_demo_frames
from inference.io_utils import load_config_namespace
from utils.data_formatting import compute_delta_progress_label_int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="对单个 demo 做一次 pairwise delta progress 推理")
    parser.add_argument("--base-model", type=str, default="models/Qwen-VL-2B-Instruct", help="基础模型路径")
    parser.add_argument("--adapter", type=str, required=True, help="LoRA 适配器路径")
    parser.add_argument("--target-demo", type=str, required=True, help="target demo 路径")
    parser.add_argument("--i", type=int, required=True, help="起始帧索引")
    parser.add_argument("--j", type=int, required=True, help="结束帧索引")
    parser.add_argument("--reference-demo", type=str, help="reference demo 路径")
    parser.add_argument("--task-desc", type=str, required=True, help="任务描述")
    parser.add_argument("--config", type=str, required=True, help="YAML 配置文件路径")
    return parser


def main(args: argparse.Namespace) -> None:
    from inference.inferencer import DeltaProgressInference

    config = load_config_namespace(args.config)
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

    _, total_frames = scan_demo_frames(args.target_demo, target_views)
    gt_delta_progress = compute_delta_progress_label_int(args.i, args.j, total_frames)
    print(f"predicted_delta_progress: {predicted_delta_progress}")
    print(f"ground_truth_delta_progress: {gt_delta_progress}")


if __name__ == "__main__":
    main(build_parser().parse_args())
