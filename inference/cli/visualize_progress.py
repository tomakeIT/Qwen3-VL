import argparse
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from inference.core.io_utils import load_jsonl_rows
from inference.core.metrics import (
    calc_monotonicity_rate,
    calc_total_variation,
    compute_ground_truth_dense_curve,
    pearson_correlation,
    reconstruct_dense_progress,
    spearman_correlation,
)
from inference.viz.viz_utils import (
    build_summary_payload as build_viz_summary_payload,
    group_items_by,
    resolve_default_output_dir,
    safe_mean,
    safe_median,
    sanitize_filename,
    sample_items,
)


@dataclass
class EpisodeCurve:
    episode_index: int
    task_desc: str
    total_frames: int
    num_pairs: int
    num_missing_pairs: int
    pair_indices: List[Tuple[int, int]]
    frame_indices: List[int]
    delta_progress: List[float]
    cumulative_progress: List[float]
    dense_frames: np.ndarray
    dense_progress: np.ndarray
    dense_progress_gt: np.ndarray


def infer_total_frames(row: Dict[str, Any]) -> int:
    pair_indices = row.get("pair_indices", [])
    frame_indices = row.get("frame_indices", [])
    if pair_indices:
        return max(int(pair[1]) for pair in pair_indices) + 1
    if frame_indices:
        return max(int(frame_idx) for frame_idx in frame_indices) + 1
    raise ValueError(f"无法从记录中推断 total_frames: episode_index={row.get('episode_index')}")


def compute_episode_metrics(curve: EpisodeCurve) -> Dict[str, float]:
    pearson = pearson_correlation(curve.dense_progress, curve.dense_progress_gt)
    spearman = spearman_correlation(curve.dense_progress, curve.dense_progress_gt)
    total_variation, norm_total_variation = calc_total_variation(curve.dense_progress)
    monotonicity_rate = calc_monotonicity_rate(curve.dense_progress)
    final_progress = float(curve.dense_progress[-1]) if curve.dense_progress.size > 0 else 0.0
    final_progress_error = final_progress - 100.0 if curve.dense_progress.size > 0 else -100.0

    return {
        "pearson": pearson,
        "spearman": spearman,
        "total_variation": total_variation,
        "norm_total_variation": norm_total_variation,
        "monotonicity_rate": monotonicity_rate,
        "final_progress": final_progress,
        "final_progress_error": final_progress_error,
        "total_frames": float(curve.total_frames),
        "num_pairs": float(curve.num_pairs),
        "num_missing_pairs": float(curve.num_missing_pairs),
    }


def summarize_task_metrics(curves: Sequence[EpisodeCurve]) -> Dict[str, Any]:
    episode_metrics = [compute_episode_metrics(curve) for curve in curves]
    return {
        "num_episodes": len(curves),
        "pearson": safe_mean(metric["pearson"] for metric in episode_metrics),
        "spearman": safe_mean(metric["spearman"] for metric in episode_metrics),
        "total_variation": safe_mean(metric["total_variation"] for metric in episode_metrics),
        "norm_total_variation": safe_mean(metric["norm_total_variation"] for metric in episode_metrics),
        "monotonicity_rate": safe_mean(metric["monotonicity_rate"] for metric in episode_metrics),
        "final_progress_mean": safe_mean(metric["final_progress"] for metric in episode_metrics),
        "final_progress_median": safe_median(metric["final_progress"] for metric in episode_metrics),
        "final_progress_error_mean": safe_mean(metric["final_progress_error"] for metric in episode_metrics),
        "total_frames_mean": safe_mean(metric["total_frames"] for metric in episode_metrics),
        "total_frames_median": safe_median(metric["total_frames"] for metric in episode_metrics),
        "num_pairs_mean": safe_mean(metric["num_pairs"] for metric in episode_metrics),
        "num_missing_pairs_mean": safe_mean(metric["num_missing_pairs"] for metric in episode_metrics),
    }


def build_episode_curve(row: Dict[str, Any]) -> EpisodeCurve:
    pair_indices = [tuple(int(v) for v in pair) for pair in row.get("pair_indices", [])]
    frame_indices = [int(v) for v in row.get("frame_indices", [])]
    delta_progress = [float(v) for v in row.get("delta_progress", [])]
    cumulative_progress = [float(v) for v in row.get("cumulative_progress", [])]
    missing_pair_indices = [tuple(int(v) for v in pair) for pair in row.get("missing_pair_indices", [])]
    total_frames = infer_total_frames(row)
    dense_progress = reconstruct_dense_progress(pair_indices, cumulative_progress, total_frames)
    dense_progress_gt = compute_ground_truth_dense_curve(total_frames)

    return EpisodeCurve(
        episode_index=int(row["episode_index"]),
        task_desc=str(row["task_desc"]),
        total_frames=total_frames,
        num_pairs=len(pair_indices),
        num_missing_pairs=len(missing_pair_indices),
        pair_indices=pair_indices,
        frame_indices=frame_indices,
        delta_progress=delta_progress,
        cumulative_progress=cumulative_progress,
        dense_frames=np.arange(total_frames, dtype=np.float32),
        dense_progress=dense_progress,
        dense_progress_gt=dense_progress_gt,
    )


def format_metrics_text(metrics: Dict[str, Any], num_plotted: int) -> str:
    return "\n".join([
        f"episodes(all/plotted): {metrics['num_episodes']}/{num_plotted}",
        f"pearson: {metrics['pearson']:.3f}",
        f"spearman: {metrics['spearman']:.3f}",
        f"norm_tv: {metrics['norm_total_variation']:.3f}",
        f"monotonicity: {metrics['monotonicity_rate']:.3f}",
        f"final_progress_mean: {metrics['final_progress_mean']:.2f}",
        f"final_progress_err_mean: {metrics['final_progress_error_mean']:.2f}",
    ])


def plot_task_curves(
    task_desc: str,
    all_curves: Sequence[EpisodeCurve],
    plotted_curves: Sequence[EpisodeCurve],
    metrics: Dict[str, Any],
    output_path: str,
    dpi: int = 150,
) -> None:
    import matplotlib.pyplot as plt

    if not plotted_curves:
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
    ax_raw, ax_norm = axes

    for curve in plotted_curves:
        label = f"ep={curve.episode_index}, T={curve.total_frames}, final={curve.dense_progress[-1]:.1f}"
        ax_raw.plot(curve.dense_frames, curve.dense_progress, alpha=0.75, linewidth=1.6, label=label)
        normalized_time = (
            curve.dense_frames / max(curve.total_frames - 1, 1)
            if curve.total_frames > 1
            else np.zeros_like(curve.dense_frames)
        )
        ax_norm.plot(normalized_time, curve.dense_progress, alpha=0.75, linewidth=1.6, label=label)

    gt_curve = compute_ground_truth_dense_curve(200)
    gt_x = np.linspace(0.0, 1.0, gt_curve.size)
    ax_norm.plot(gt_x, gt_curve, linestyle="--", linewidth=2.0, color="black", alpha=0.7, label="GT 0→100")

    ax_raw.set_title("Dense Curve vs Frame Index")
    ax_raw.set_xlabel("Frame Index")
    ax_raw.set_ylabel("Progress")
    ax_raw.grid(True, alpha=0.3)

    ax_norm.set_title("Dense Curve vs Normalized Time")
    ax_norm.set_xlabel("Normalized Time")
    ax_norm.set_ylabel("Progress")
    ax_norm.grid(True, alpha=0.3)

    legend_ncol = 2 if len(plotted_curves) > 6 else 1
    ax_raw.legend(loc="best", fontsize=8, ncol=legend_ncol)
    ax_norm.legend(loc="best", fontsize=8, ncol=legend_ncol)

    fig.suptitle(task_desc, fontsize=14)
    fig.text(
        0.5,
        0.01,
        format_metrics_text(metrics, len(plotted_curves)),
        ha="center",
        va="bottom",
        fontsize=10,
        family="monospace",
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="可视化 progress_sparse_predictions.jsonl 的 dense 曲线和指标")
    parser.add_argument("--input-jsonl", type=str, required=True, help="progress_sparse_predictions.jsonl 路径")
    parser.add_argument("--output-dir", type=str, default=None, help="输出目录，默认在输入文件旁创建")
    parser.add_argument("--episodes-per-task", type=int, default=12, help="每个 task 采样多少个 episode 画图")
    parser.add_argument("--seed", type=int, default=42, help="采样 seed")
    parser.add_argument("--dpi", type=int, default=150, help="图片 DPI")
    parser.add_argument("--tasks", nargs="*", default=None, help="可选：只可视化指定 task_desc")
    return parser


def main(args: argparse.Namespace) -> None:
    rows = load_jsonl_rows(args.input_jsonl)
    curves = [build_episode_curve(row) for row in rows]
    grouped_curves = group_items_by(curves, key_fn=lambda curve: curve.task_desc)

    if args.tasks:
        task_filter = set(args.tasks)
        grouped_curves = {
            task_desc: task_curves
            for task_desc, task_curves in grouped_curves.items()
            if task_desc in task_filter
        }

    if not grouped_curves:
        raise ValueError("没有可用于可视化的 task")

    output_dir = args.output_dir or resolve_default_output_dir(args.input_jsonl, "_viz")
    os.makedirs(output_dir, exist_ok=True)

    plotted_curves_by_task: Dict[str, List[EpisodeCurve]] = {}
    for task_idx, task_desc in enumerate(sorted(grouped_curves.keys())):
        task_curves = sorted(grouped_curves[task_desc], key=lambda curve: curve.episode_index)
        plotted_curves = sample_items(
            task_curves,
            max_items=args.episodes_per_task,
            seed=args.seed + task_idx,
        )
        plotted_curves_by_task[task_desc] = plotted_curves
        metrics = summarize_task_metrics(task_curves)

        plot_task_curves(
            task_desc=task_desc,
            all_curves=task_curves,
            plotted_curves=plotted_curves,
            metrics=metrics,
            output_path=os.path.join(output_dir, sanitize_filename(task_desc) + ".png"),
            dpi=args.dpi,
        )

    summary_payload = build_viz_summary_payload(
        grouped_curves,
        plotted_curves_by_task,
        summarize_fn=summarize_task_metrics,
    )
    summary_path = os.path.join(output_dir, "metrics_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, ensure_ascii=False, indent=2)

    print(f"input_jsonl: {args.input_jsonl}")
    print(f"output_dir: {output_dir}")
    print(f"num_tasks: {len(grouped_curves)}")
    print(f"summary_path: {summary_path}")


if __name__ == "__main__":
    main(build_parser().parse_args())
