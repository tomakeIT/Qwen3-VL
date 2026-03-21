from __future__ import annotations

"""
批量推理多个demo的整条progress curve，并计算curve level的evaluation指标
定位：inference/eval whole progress curves from validation demo list
"""

import argparse

from workflows.curve_eval import (
    build_episode_jobs_from_demo_list,
    build_sparse_curve_results,
    evaluate_curves,
    infer_job_predictions,
    load_demo_list_from_json,
    run_eval_curves_from_batch_demos,
    save_progress_curves_plot,
)


def main(args: argparse.Namespace) -> None:
    run_eval_curves_from_batch_demos(
        base_model=args.base_model,
        adapter=args.adapter,
        config_path=args.config,
        task_desc=args.task_desc,
        reference_demo=args.reference_demo,
        demo_list_path=args.demo_list,
        step_interval=args.step_interval,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        batch_size=args.batch_size,
        num_gpus=args.num_gpus,
        global_build_workers=args.global_build_workers,
        message_chunk_size=args.message_chunk_size,
        output_json=args.output,
        plot_output=args.plot_output,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="批量推理多个demo的progress curve并计算评估指标")
    parser.add_argument("--base-model", type=str, default="models/Qwen-VL-2B-Instruct", help="基础模型路径")
    parser.add_argument("--adapter", type=str, required=True, help="LoRA适配器路径")
    parser.add_argument("--demo-list", type=str, required=True, help="validation demo列表文件路径（JSON格式）")
    parser.add_argument("--reference-demo", type=str, default=None, help="全局reference demo路径（如果demo列表中没有指定）")
    parser.add_argument("--task-desc", type=str, required=True, help="任务描述")
    parser.add_argument("--config", type=str, required=True, help="YAML配置文件路径")
    parser.add_argument("--step-interval", type=int, default=1, help="采样间隔")
    parser.add_argument("--start-frame", type=int, default=0, help="起始帧")
    parser.add_argument("--end-frame", type=int, default=None, help="结束帧")
    parser.add_argument("--output", type=str, default=None, help="可选：保存结果到JSON文件")
    parser.add_argument("--plot-output", type=str, default="./curves.png", help="可选：保存curve图路径")
    parser.add_argument("--batch-size", type=int, default=1, help="每个demo内部的推理batch大小，大于1时加速推理")
    parser.add_argument("--num-gpus", type=int, default=1, help="使用的GPU数量（默认1，设为-1使用所有可用GPU）")
    parser.add_argument("--global-build-workers", type=int, default=16, help="构建messages的线程数（global模式生效）")
    parser.add_argument("--message-chunk-size", type=int, default=None, help="每次送入多GPU推理的message数量上限")
    args = parser.parse_args()
    main(args)
