import argparse
import json
import os
import random
from typing import Any, Dict, List, Optional

from eval_curves_from_batch_demos import run_eval_curves_from_batch_demos


def _load_json(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"文件不存在: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_episode_path(raw_dataset_root: str, task: str, episode_id: str) -> str:
    """Episode 目录始终按「本机 raw_dataset_root / task / episode_id」解析，不依赖 metadata 里的 data_path 或 statistics 里的旧机器绝对路径。"""
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

    print("=" * 80)
    print(f"任务: {task}")
    print(f"reference(train): {reference_demo_path}")
    print(f"eval demos: {len(eval_demo_paths)}")
    print("=" * 80)

    run_eval_curves_from_batch_demos(
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


def main(args: argparse.Namespace) -> None:
    train_meta_path = os.path.join(args.processed_meta_root, "train_metadata.json")
    eval_meta_path = os.path.join(args.processed_meta_root, "eval_metadata.json")
    task_desc_path = os.path.join(args.raw_dataset_root, "task_descriptions.json")

    train_meta = _load_json(train_meta_path)
    eval_meta = _load_json(eval_meta_path)
    task_desc_map = _load_json(task_desc_path)

    rng = random.Random(args.seed)

    os.makedirs(args.output_root, exist_ok=True)
    results: List[Dict[str, Any]] = []

    for task in args.tasks:
        if task not in eval_meta.get("tasks", {}):
            raise ValueError(f"任务 {task} 不在 eval_metadata.json 里")
        if task not in train_meta.get("tasks", {}):
            raise ValueError(f"任务 {task} 不在 train_metadata.json 里")
        if task not in task_desc_map:
            raise ValueError(f"任务 {task} 不在 task_descriptions.json 里")

        eval_episode_ids = eval_meta["tasks"][task]["episodes"]
        train_episode_ids = train_meta["tasks"][task]["episodes"]
        if not eval_episode_ids:
            raise ValueError(f"任务 {task} 的 eval episodes 为空")
        if not train_episode_ids:
            raise ValueError(f"任务 {task} 的 train episodes 为空")

        if args.demos_per_task > 0 and len(eval_episode_ids) > args.demos_per_task:
            selected_eval_ids = rng.sample(eval_episode_ids, args.demos_per_task)
        else:
            selected_eval_ids = list(eval_episode_ids)
        selected_eval_ids = sorted(selected_eval_ids)

        reference_episode_id = rng.choice(train_episode_ids)
        reference_demo_path = _resolve_episode_path(args.raw_dataset_root, task, reference_episode_id)
        eval_demo_paths = [
            _resolve_episode_path(args.raw_dataset_root, task, ep_id) for ep_id in selected_eval_ids
        ]

        result = _run_single_task(
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
        )
        results.append(result)

    summary_path = os.path.join(args.output_root, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"results": results}, f, ensure_ascii=False, indent=2)
    print(f"完成，汇总文件: {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="多任务 curve 评估（固定 metadata 文件名）")
    parser.add_argument("--base-model", type=str, default="models/Qwen-VL-2B-Instruct")
    parser.add_argument("--adapter", type=str, required=True)
    parser.add_argument("--processed-meta-root", type=str, required=True, help="包含 train/eval_metadata.json 的目录")
    parser.add_argument(
        "--raw-dataset-root",
        type=str,
        required=True,
        help="原始数据集根目录（本机路径），用于解析各 episode 目录；episode 布局为 raw_dataset_root/<task>/<episode_id>",
    )
    parser.add_argument("--tasks", nargs="+", required=True, help="要评估的任务名列表（在 Python 主循环中依次处理）")
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
    main(parser.parse_args())
