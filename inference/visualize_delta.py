import argparse
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from inference.io_utils import load_jsonl_rows
from inference.metrics import calc_total_variation, compute_ground_truth_delta_curve
from inference.viz_utils import (
    build_summary_payload as build_viz_summary_payload,
    group_items_by,
    resolve_default_output_dir,
    safe_mean,
    safe_median,
    sanitize_filename,
    sample_items,
)


@dataclass
class EpisodeDeltaCurve:
    episode_index: int
    task_desc: str
    total_frames: int
    pair_offset: int
    num_pairs: int
    num_missing_pairs: int
    pair_indices: List[Tuple[int, int]]
    frame_indices: np.ndarray
    delta_progress: np.ndarray
    gt_delta_curve: np.ndarray


def infer_total_frames(row: Dict[str, Any]) -> int:
    total_frames = row.get("total_frames")
    if isinstance(total_frames, int):
        return total_frames

    pair_indices = row.get("pair_indices", [])
    frame_indices = row.get("frame_indices", [])
    if pair_indices:
        return max(int(pair[1]) for pair in pair_indices) + 1
    if frame_indices:
        return max(int(frame_idx) for frame_idx in frame_indices) + 1
    delta_progress = row.get("delta_progress", [])
    if delta_progress:
        return len(delta_progress)
    raise ValueError(f"无法从记录中推断 total_frames: episode_index={row.get('episode_index')}")


def infer_pair_offset(row: Dict[str, Any]) -> int:
    pair_offset = row.get("pair_offset")
    if isinstance(pair_offset, int):
        return pair_offset
    pair_indices = row.get("pair_indices", [])
    if pair_indices:
        first_pair = pair_indices[0]
        return int(first_pair[1]) - int(first_pair[0])
    return 0


def resolve_frame_indices(
    row: Dict[str, Any],
    total_frames: int,
    delta_length: int,
) -> np.ndarray:
    raw_frame_indices = row.get("frame_indices", [])
    if delta_length == total_frames:
        if raw_frame_indices and len(raw_frame_indices) == delta_length:
            return np.array([int(v) for v in raw_frame_indices], dtype=np.float32)
        return np.arange(total_frames, dtype=np.float32)

    pair_indices = row.get("pair_indices", [])
    if pair_indices:
        return np.array([int(pair[0]) for pair in pair_indices], dtype=np.float32)
    if raw_frame_indices:
        return np.array([int(v) for v in raw_frame_indices], dtype=np.float32)
    return np.arange(delta_length, dtype=np.float32)


def compute_episode_metrics(curve: EpisodeDeltaCurve) -> Dict[str, float]:
    pred = curve.delta_progress
    gt = curve.gt_delta_curve
    if pred.size == 0:
        return {
            "mae_to_gt_delta": 0.0,
            "rmse_to_gt_delta": 0.0,
            "total_variation": 0.0,
            "norm_total_variation": 0.0,
            "mean_delta": 0.0,
            "median_delta": 0.0,
            "std_delta": 0.0,
            "min_delta": 0.0,
            "max_delta": 0.0,
            "delta_bias": 0.0,
            "positive_rate": 0.0,
            "negative_rate": 0.0,
            "zero_rate": 0.0,
            "gt_delta": 0.0,
            "total_frames": float(curve.total_frames),
            "num_pairs": float(curve.num_pairs),
            "num_missing_pairs": float(curve.num_missing_pairs),
            "pair_offset": float(curve.pair_offset),
        }

    mae = float(np.mean(np.abs(pred - gt)))
    rmse = float(np.sqrt(np.mean((pred - gt) ** 2)))
    total_variation, norm_total_variation = calc_total_variation(pred, normalization="mean_abs")
    mean_delta = float(np.mean(pred))
    median_delta = float(np.median(pred))
    std_delta = float(np.std(pred))
    min_delta = float(np.min(pred))
    max_delta = float(np.max(pred))
    gt_delta = float(gt[0]) if gt.size > 0 else 0.0
    delta_bias = mean_delta - gt_delta
    positive_rate = float(np.mean(pred > 0))
    negative_rate = float(np.mean(pred < 0))
    zero_rate = float(np.mean(pred == 0))

    return {
        "mae_to_gt_delta": mae,
        "rmse_to_gt_delta": rmse,
        "total_variation": total_variation,
        "norm_total_variation": norm_total_variation,
        "mean_delta": mean_delta,
        "median_delta": median_delta,
        "std_delta": std_delta,
        "min_delta": min_delta,
        "max_delta": max_delta,
        "delta_bias": delta_bias,
        "positive_rate": positive_rate,
        "negative_rate": negative_rate,
        "zero_rate": zero_rate,
        "gt_delta": gt_delta,
        "total_frames": float(curve.total_frames),
        "num_pairs": float(curve.num_pairs),
        "num_missing_pairs": float(curve.num_missing_pairs),
        "pair_offset": float(curve.pair_offset),
    }


def summarize_task_metrics(curves: Sequence[EpisodeDeltaCurve]) -> Dict[str, Any]:
    episode_metrics = [compute_episode_metrics(curve) for curve in curves]
    return {
        "num_episodes": len(curves),
        "mae_to_gt_delta": safe_mean(metric["mae_to_gt_delta"] for metric in episode_metrics),
        "rmse_to_gt_delta": safe_mean(metric["rmse_to_gt_delta"] for metric in episode_metrics),
        "total_variation": safe_mean(metric["total_variation"] for metric in episode_metrics),
        "norm_total_variation": safe_mean(metric["norm_total_variation"] for metric in episode_metrics),
        "mean_delta": safe_mean(metric["mean_delta"] for metric in episode_metrics),
        "median_delta": safe_median(metric["median_delta"] for metric in episode_metrics),
        "std_delta": safe_mean(metric["std_delta"] for metric in episode_metrics),
        "min_delta": safe_mean(metric["min_delta"] for metric in episode_metrics),
        "max_delta": safe_mean(metric["max_delta"] for metric in episode_metrics),
        "delta_bias": safe_mean(metric["delta_bias"] for metric in episode_metrics),
        "positive_rate": safe_mean(metric["positive_rate"] for metric in episode_metrics),
        "negative_rate": safe_mean(metric["negative_rate"] for metric in episode_metrics),
        "zero_rate": safe_mean(metric["zero_rate"] for metric in episode_metrics),
        "gt_delta": safe_mean(metric["gt_delta"] for metric in episode_metrics),
        "total_frames_mean": safe_mean(metric["total_frames"] for metric in episode_metrics),
        "total_frames_median": safe_median(metric["total_frames"] for metric in episode_metrics),
        "num_pairs_mean": safe_mean(metric["num_pairs"] for metric in episode_metrics),
        "num_missing_pairs_mean": safe_mean(metric["num_missing_pairs"] for metric in episode_metrics),
        "pair_offset": safe_mean(metric["pair_offset"] for metric in episode_metrics),
    }


def build_episode_curve(row: Dict[str, Any]) -> EpisodeDeltaCurve:
    pair_indices = [tuple(int(v) for v in pair) for pair in row.get("pair_indices", [])]
    delta_progress = np.array([float(v) for v in row.get("delta_progress", [])], dtype=np.float32)
    missing_pair_indices = [tuple(int(v) for v in pair) for pair in row.get("missing_pair_indices", [])]
    total_frames = infer_total_frames(row)
    pair_offset = infer_pair_offset(row)
    frame_indices = resolve_frame_indices(row=row, total_frames=total_frames, delta_length=delta_progress.size)
    if delta_progress.size == total_frames:
        valid_pair_count = len(pair_indices) if pair_indices else max(total_frames - pair_offset, 0)
    else:
        valid_pair_count = delta_progress.size
    gt_delta_curve = compute_ground_truth_delta_curve(
        total_frames=total_frames,
        pair_offset=pair_offset,
        num_points=delta_progress.size,
        valid_pair_count=valid_pair_count,
    )

    return EpisodeDeltaCurve(
        episode_index=int(row["episode_index"]),
        task_desc=str(row["task_desc"]),
        total_frames=total_frames,
        pair_offset=pair_offset,
        num_pairs=len(pair_indices),
        num_missing_pairs=len(missing_pair_indices),
        pair_indices=pair_indices,
        frame_indices=frame_indices,
        delta_progress=delta_progress,
        gt_delta_curve=gt_delta_curve,
    )


def format_metrics_text(metrics: Dict[str, Any], num_plotted: int) -> str:
    return "\n".join([
        f"episodes(all/plotted): {metrics['num_episodes']}/{num_plotted}",
        f"mae: {metrics['mae_to_gt_delta']:.3f}",
        f"rmse: {metrics['rmse_to_gt_delta']:.3f}",
        f"norm_tv: {metrics['norm_total_variation']:.3f}",
        f"mean_delta: {metrics['mean_delta']:.3f}",
        f"delta_bias: {metrics['delta_bias']:.3f}",
        f"positive_rate: {metrics['positive_rate']:.3f}",
    ])


def plot_task_curves(
    task_desc: str,
    plotted_curves: Sequence[EpisodeDeltaCurve],
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
        label = f"ep={curve.episode_index}, T={curve.total_frames}, mean={float(np.mean(curve.delta_progress)):.2f}"
        ax_raw.plot(curve.frame_indices, curve.delta_progress, alpha=0.75, linewidth=1.4, label=label)
        normalized_time = (
            curve.frame_indices / max(curve.total_frames - 1, 1)
            if curve.total_frames > 1
            else np.zeros_like(curve.frame_indices)
        )
        ax_norm.plot(normalized_time, curve.delta_progress, alpha=0.75, linewidth=1.4, label=label)

        if curve.gt_delta_curve.size > 0:
            ax_raw.plot(curve.frame_indices, curve.gt_delta_curve, linestyle="--", linewidth=1.5, color="black", alpha=0.25)
            ax_norm.plot(normalized_time, curve.gt_delta_curve, linestyle="--", linewidth=1.5, color="black", alpha=0.25)

    ax_raw.set_title("Delta Curve vs Frame Index")
    ax_raw.set_xlabel("Frame Index")
    ax_raw.set_ylabel("Delta Progress")
    ax_raw.grid(True, alpha=0.3)

    ax_norm.set_title("Delta Curve vs Normalized Time")
    ax_norm.set_xlabel("Normalized Time")
    ax_norm.set_ylabel("Delta Progress")
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
    parser = argparse.ArgumentParser(description="可视化 progress_sparse_predictions.jsonl 的 dense delta 曲线和指标")
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

    output_dir = args.output_dir or resolve_default_output_dir(args.input_jsonl, "_delta_viz")
    os.makedirs(output_dir, exist_ok=True)

    plotted_curves_by_task: Dict[str, List[EpisodeDeltaCurve]] = {}
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
