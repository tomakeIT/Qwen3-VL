"""
批量评估多个 demo 的 progress curve。
"""

from __future__ import annotations

import argparse

from inference.eval.curve_eval import run_eval_curves


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="批量推理多个 demo 的 progress curve 并计算评估指标")
    parser.add_argument("--base-model", type=str, default="models/Qwen-VL-2B-Instruct", help="基础模型路径")
    parser.add_argument("--adapter", type=str, required=True, help="LoRA 适配器路径")
    parser.add_argument("--demo-list", type=str, required=True, help="validation demo 列表文件路径（JSON）")
    parser.add_argument("--reference-demo", type=str, default=None, help="全局 reference demo 路径")
    parser.add_argument("--task-desc", type=str, required=True, help="任务描述")
    parser.add_argument("--config", type=str, required=True, help="YAML 配置文件路径")
    parser.add_argument("--step-interval", type=int, default=1, help="采样间隔")
    parser.add_argument("--start-frame", type=int, default=0, help="起始帧")
    parser.add_argument("--end-frame", type=int, default=None, help="结束帧")
    parser.add_argument("--output", type=str, default=None, help="可选：保存指标 JSON")
    parser.add_argument("--plot-output", type=str, default=None, help="可选：保存曲线图路径")
    parser.add_argument("--batch-size", type=int, default=1, help="每个 demo 内部的推理 batch 大小")
    parser.add_argument("--num-gpus", type=int, default=1, help="使用的 GPU 数量")
    parser.add_argument("--global-build-workers", type=int, default=16, help="构建 messages 的线程数")
    parser.add_argument("--message-chunk-size", type=int, default=None, help="每次送入多 GPU 推理的 message 数量上限")
    return parser


def main(args: argparse.Namespace) -> None:
    metrics, _ = run_eval_curves(
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

    print("curve_eval_summary")
    for key, value in metrics.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main(build_parser().parse_args())
