"""
基于固定 metadata 文件的多任务 curve 评估。
"""

from __future__ import annotations

import argparse
import json
import os
import random
from typing import Any, Dict, List, Optional

from inference.curve_eval import run_eval_curves
from inference.io_utils import load_json


def _resolve_episode_path(raw_dataset_root: str, task: str, episode_id: str) -> str:
    return os.path.join(os.path.normpath(raw_dataset_root), task, episode_id)


def _run_single_task(
    task: str,
    task_desc: str,
    eval_demo_paths: List[str],
    reference_demo_path: str,
    output_root: str,
    base_model: str,
    adapter: str,
    config: str,
    step_interval: int,
    start_frame: int,
    end_frame: Optional[int],
    batch_size: int,
    num_gpus: int,
    global_build_workers: int,
) -> Dict[str, Any]:
    task_out_dir = os.path.join(output_root, task)
    os.makedirs(task_out_dir, exist_ok=True)

    demo_list_path = os.path.join(task_out_dir, "demo_list.json")
    with open(demo_list_path, "w", encoding="utf-8") as f:
        json.dump({"eval": {os.path.basename(p): p for p in eval_demo_paths}}, f, ensure_ascii=False, indent=2)

    metrics_output = os.path.join(task_out_dir, "metrics.json")
    plot_output = os.path.join(task_out_dir, "curves.png")

    run_eval_curves(
        base_model=base_model,
        adapter=adapter,
        config_path=config,
        task_desc=task_desc,
        reference_demo=reference_demo_path,
        demo_list=list(eval_demo_paths),
        step_interval=step_interval,
        start_frame=start_frame,
        end_frame=end_frame,
        batch_size=batch_size,
        num_gpus=num_gpus,
        global_build_workers=global_build_workers,
        output_json=metrics_output,
        plot_output=plot_output,
    )

    return {
        "task": task,
        "task_desc": task_desc,
        "reference_demo": reference_demo_path,
        "num_eval_demos": len(eval_demo_paths),
        "metrics_output": metrics_output,
        "plot_output": plot_output,
        "demo_list_output": demo_list_path,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="多任务 curve 评估（固定 metadata 文件名）")
    parser.add_argument("--base-model", type=str, default="models/Qwen-VL-2B-Instruct")
    parser.add_argument("--adapter", type=str, required=True)
    parser.add_argument("--processed-meta-root", type=str, required=True, help="包含 train/eval_metadata.json 的目录")
    parser.add_argument(
        "--raw-dataset-root",
        type=str,
        required=True,
        help="原始数据集根目录，episode 布局为 raw_dataset_root/<task>/<episode_id>",
    )
    parser.add_argument("--tasks", nargs="+", required=True, help="要评估的任务名列表")
    parser.add_argument("--demos-per-task", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--step-interval", type=int, default=2)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--global-build-workers", type=int, default=1)
    parser.add_argument("--output-root", type=str, required=True)
    return parser


def main(args: argparse.Namespace) -> None:
    train_meta = load_json(os.path.join(args.processed_meta_root, "train_metadata.json"))
    eval_meta = load_json(os.path.join(args.processed_meta_root, "eval_metadata.json"))
    task_desc_map = load_json(os.path.join(args.raw_dataset_root, "task_descriptions.json"))

    rng = random.Random(args.seed)
    os.makedirs(args.output_root, exist_ok=True)
    results: List[Dict[str, Any]] = []

    for task in args.tasks:
        eval_episode_ids = eval_meta["tasks"][task]["episodes"]
        train_episode_ids = train_meta["tasks"][task]["episodes"]

        if args.demos_per_task > 0 and len(eval_episode_ids) > args.demos_per_task:
            selected_eval_ids = sorted(rng.sample(eval_episode_ids, args.demos_per_task))
        else:
            selected_eval_ids = sorted(list(eval_episode_ids))

        reference_episode_id = rng.choice(train_episode_ids)
        reference_demo_path = _resolve_episode_path(args.raw_dataset_root, task, reference_episode_id)
        eval_demo_paths = [
            _resolve_episode_path(args.raw_dataset_root, task, episode_id)
            for episode_id in selected_eval_ids
        ]

        results.append(_run_single_task(
            task=task,
            task_desc=task_desc_map[task],
            eval_demo_paths=eval_demo_paths,
            reference_demo_path=reference_demo_path,
            output_root=args.output_root,
            base_model=args.base_model,
            adapter=args.adapter,
            config=args.config,
            step_interval=args.step_interval,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
            batch_size=args.batch_size,
            num_gpus=args.num_gpus,
            global_build_workers=args.global_build_workers,
        ))

    summary_path = os.path.join(args.output_root, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"results": results}, f, ensure_ascii=False, indent=2)
    print(f"summary_path: {summary_path}")


if __name__ == "__main__":
    main(build_parser().parse_args())
