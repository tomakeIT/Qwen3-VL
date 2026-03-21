from __future__ import annotations

"""
对单个 demo 做密集滑窗推理，输出 JSON 曲线数据。
"""

import argparse
import json
import os
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from inference.demo_utils import build_messages_from_demo, scan_demo_frames
from inference.io_utils import load_config_namespace


def infer_dense_progress_curve(
    inference,
    target_demo_path: str,
    reference_demo_path: Optional[str],
    task_desc: str,
    target_views: List[str],
    reference_config,
    delta_t: int,
) -> Dict[str, Any]:
    _, total_frames = scan_demo_frames(target_demo_path, target_views)
    if total_frames < 2:
        raise ValueError(f"Target demo has insufficient frames: T={total_frames}")

    result = {
        "demo_name": os.path.basename(target_demo_path),
        "total_frames": total_frames,
        "delta_t": delta_t,
        "target_views": list(target_views),
        "delta_progress": [],
        "cumulative_progress": [],
    }

    current_progress = 0
    delta_progress_list = []
    cumulative_progress_list = []

    for i in tqdm(range(total_frames), desc=f"密集推理 T={total_frames}"):
        j = min(i + delta_t, total_frames - 1)
        if i >= j:
            delta_progress = 0
        else:
            messages = build_messages_from_demo(
                target_demo_path=target_demo_path,
                i=i,
                j=j,
                reference_demo_path=reference_demo_path,
                task_desc=task_desc,
                target_views=target_views,
                reference_config=reference_config,
            )
            delta_progress = inference.infer_from_messages(messages)
            if delta_progress is None:
                delta_progress = 0

        current_progress += delta_progress
        delta_progress_list.append(delta_progress)
        cumulative_progress_list.append(current_progress)

    result["delta_progress"] = delta_progress_list
    result["cumulative_progress"] = cumulative_progress_list
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="对单个 demo 进行密集采样推理，输出 JSON")
    parser.add_argument("--base-model", type=str, default="models/Qwen-VL-2B-Instruct", help="基础模型路径")
    parser.add_argument("--adapter", type=str, required=True, help="LoRA 适配器路径")
    parser.add_argument("--target-demo", type=str, required=True, help="target demo 路径")
    parser.add_argument("--reference-demo", type=str, help="reference demo 路径")
    parser.add_argument("--task-desc", type=str, required=True, help="任务描述")
    parser.add_argument("--config", type=str, required=True, help="YAML 配置文件路径")
    parser.add_argument("--delta-t", type=int, required=True, help="窗口大小")
    parser.add_argument("--output-dir", type=str, default="outputs/dense_curves", help="输出目录")
    return parser


def main(args: argparse.Namespace) -> None:
    from inference.inferencer import DeltaProgressInference

    os.makedirs(args.output_dir, exist_ok=True)
    config = load_config_namespace(args.config)
    inference = DeltaProgressInference(
        base_model_path=args.base_model,
        adapter_path=args.adapter,
    )

    target_views = config.sampling.required_views
    result = infer_dense_progress_curve(
        inference=inference,
        target_demo_path=args.target_demo,
        reference_demo_path=args.reference_demo,
        task_desc=args.task_desc,
        target_views=target_views,
        reference_config=config.reference,
        delta_t=args.delta_t,
    )

    demo_name = os.path.basename(args.target_demo)
    output_json_path = os.path.join(args.output_dir, f"{demo_name}.json")
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"output_json: {output_json_path}")
    print(f"num_points: {len(result['delta_progress'])}")


if __name__ == "__main__":
    main(build_parser().parse_args())
