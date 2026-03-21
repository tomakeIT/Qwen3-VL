from __future__ import annotations

import json
import os
from functools import partial
import numpy as np
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Tuple

from tqdm import tqdm

from common.demo_scan import scan_demo_frames
from common.io_utils import load_config_namespace, load_json
from common.messages import build_messages_for_job_chunk, build_messages_from_demo
from common.metrics import calc_monotonicity_rate, calc_total_variation, compute_ground_truth_curve
from common.viz_utils import safe_mean


def calc_correlation(pred: np.ndarray, gt: np.ndarray) -> Tuple[float, float]:
    """Calculate Pearson and Spearman correlation coefficients."""
    from scipy.stats import pearsonr, spearmanr

    if pred.size < 2 or np.allclose(pred, pred[0]):
        return 0.0, 0.0
    pearson, _ = pearsonr(pred, gt)
    spearman, _ = spearmanr(pred, gt)
    return float(np.nan_to_num(pearson)), float(np.nan_to_num(spearman))


def _parse_demo_item(demo_item: Any, reference_demo_path: Optional[str]) -> Optional[Tuple[str, Optional[str]]]:
    """统一解析 demo 项，返回 (target_demo_path, reference_demo_path)。"""
    if isinstance(demo_item, str):
        return demo_item, reference_demo_path
    if isinstance(demo_item, dict):
        target_demo_path = demo_item.get("target_demo", demo_item.get("demo", ""))
        demo_reference_demo_path = demo_item.get("reference_demo", reference_demo_path)
        return target_demo_path, demo_reference_demo_path
    return None


def _build_episode_pairs(
    target_demo_path: str,
    target_views: List[str],
    step_interval: int,
    start_frame: int,
    end_frame: Optional[int],
    include_last_pair: bool = False,
) -> Tuple[int, List[Tuple[int, int]], np.ndarray]:
    """构建单个 episode 的 (i, j) 推理对与对应 frame_indices。"""
    _, total_frames = scan_demo_frames(target_demo_path, target_views)
    if total_frames < 2:
        return total_frames, [], np.array([])

    if end_frame is None:
        valid_end_frame = total_frames - 1
    else:
        valid_end_frame = min(end_frame, total_frames - 1)

    if step_interval <= 0:
        raise ValueError(f"step_interval 必须为正整数，当前为 {step_interval}")

    anchors = list(range(start_frame, valid_end_frame + 1, step_interval))
    if not anchors:
        anchors = [start_frame]
    if include_last_pair and anchors[-1] != valid_end_frame:
        anchors.append(valid_end_frame)

    ij_pairs: List[Tuple[int, int]] = []
    frame_indices: List[int] = []
    for i, j in zip(anchors[:-1], anchors[1:]):
        ij_pairs.append((i, j))
        frame_indices.append(j)

    return total_frames, ij_pairs, np.array(frame_indices)


def _build_message_for_job(
    job: Dict[str, Any],
    task_desc: str,
    target_views: List[str],
    reference_config: SimpleNamespace,
) -> Tuple[int, int, List[Dict[str, Any]]]:
    """为单个 job 构建 messages。"""
    messages = build_messages_from_demo(
        target_demo_path=job["target_demo_path"],
        i=job["i"],
        j=job["j"],
        reference_demo_path=job["reference_demo_path"],
        task_desc=task_desc,
        target_views=target_views,
        reference_config=reference_config,
    )
    return job["episode_id"], job["pair_idx"], messages


def build_episode_jobs_from_demo_list(
    demo_list: List[Dict[str, Any]],
    reference_demo_path: Optional[str],
    target_views: List[str],
    step_interval: int = 1,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
    include_last_pair: bool = False,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """构建 episode 元数据和全局 pair jobs，可供不同推理入口复用。"""
    episode_metas: List[Dict[str, Any]] = []
    global_jobs: List[Dict[str, Any]] = []

    for demo_item in tqdm(demo_list, desc="构建全局jobs"):
        parsed = _parse_demo_item(demo_item, reference_demo_path)
        if parsed is None:
            print(f"警告：跳过无效的demo项: {demo_item}")
            continue

        target_demo_path, demo_reference_demo_path = parsed
        if not target_demo_path or not os.path.exists(target_demo_path):
            print(f"警告：demo路径不存在，跳过: {target_demo_path}")
            continue

        try:
            total_frames, ij_pairs, frame_indices = _build_episode_pairs(
                target_demo_path=target_demo_path,
                target_views=target_views,
                step_interval=step_interval,
                start_frame=start_frame,
                end_frame=end_frame,
                include_last_pair=include_last_pair,
            )
            if total_frames < 2 or len(ij_pairs) == 0:
                continue

            episode_id = len(episode_metas)
            episode_metas.append({
                "episode_id": episode_id,
                "target_demo_path": target_demo_path,
                "reference_demo_path": demo_reference_demo_path,
                "frame_indices": frame_indices,
                "T": total_frames,
                "num_pairs": len(ij_pairs),
                "ij_pairs": ij_pairs,
            })
            for pair_idx, (i, j) in enumerate(ij_pairs):
                global_jobs.append({
                    "episode_id": episode_id,
                    "pair_idx": pair_idx,
                    "i": i,
                    "j": j,
                    "target_demo_path": target_demo_path,
                    "reference_demo_path": demo_reference_demo_path,
                })
        except Exception as exc:
            print(f"警告：处理demo {target_demo_path} 时出错: {exc}")
            continue

    return episode_metas, global_jobs


def infer_job_predictions(
    inference,
    episode_metas: List[Dict[str, Any]],
    global_jobs: List[Any],
    build_message_fn: Callable[[Any], Tuple[int, int, List[Dict[str, Any]]]],
    batch_size: int = 1,
    global_build_workers: int = 1,
    message_chunk_size: Optional[int] = None,
    desc: str = "Global inference",
) -> Dict[int, List[Optional[int]]]:
    """按 chunk 构建 messages 并做多 GPU 推理，返回每个 episode 的 pair 预测结果。"""
    episode_predictions: Dict[int, List[Optional[int]]] = {
        meta["episode_id"]: [None] * meta["num_pairs"] for meta in episode_metas
    }
    if len(global_jobs) == 0:
        return episode_predictions

    if message_chunk_size is None or message_chunk_size <= 0:
        message_chunk_size = len(global_jobs)

    total_chunks = (len(global_jobs) + message_chunk_size - 1) // message_chunk_size
    print(
        f"全局任务数={len(global_jobs)}，build_workers={max(1, global_build_workers)}，"
        f"message_chunk_size={message_chunk_size}"
    )

    for chunk_idx, start_idx in enumerate(range(0, len(global_jobs), message_chunk_size), start=1):
        end_idx = min(start_idx + message_chunk_size, len(global_jobs))
        job_chunk = global_jobs[start_idx:end_idx]
        print(f"处理 inference chunk {chunk_idx}/{total_chunks}，jobs={len(job_chunk)}")

        all_meta, all_messages = build_messages_for_job_chunk(
            jobs=job_chunk,
            build_message_fn=build_message_fn,
            global_build_workers=global_build_workers,
        )
        if len(all_messages) == 0:
            continue

        chunk_desc = desc if total_chunks == 1 else f"{desc} chunk {chunk_idx}/{total_chunks}"
        all_predictions = inference.infer_from_messages_batch(
            all_messages,
            batch_size=batch_size,
            desc=chunk_desc,
        )

        for (episode_id, pair_idx), pred in zip(all_meta, all_predictions):
            episode_predictions[episode_id][pair_idx] = pred

    return episode_predictions


def build_sparse_curve_results(
    episode_metas: List[Dict[str, Any]],
    episode_predictions: Dict[int, List[Optional[int]]],
    fill_missing_with_zero: bool = False,
) -> List[Dict[str, Any]]:
    """把 pair 预测结果整理成每个 episode 的 sparse curve 明细。"""
    sparse_results: List[Dict[str, Any]] = []

    for meta in episode_metas:
        episode_id = meta["episode_id"]
        frame_indices_all = meta["frame_indices"]
        ij_pairs = meta.get("ij_pairs", [])
        preds = episode_predictions[episode_id]

        current_progress = 0
        frame_indices_valid: List[int] = []
        pair_indices_valid: List[Tuple[int, int]] = []
        delta_progress_values: List[int] = []
        cumulative_progress_values: List[int] = []
        missing_pair_indices: List[Tuple[int, int]] = []

        for idx, delta_progress in enumerate(preds):
            if delta_progress is None:
                if idx < len(ij_pairs):
                    missing_pair_indices.append(tuple(int(v) for v in ij_pairs[idx]))
                if not fill_missing_with_zero:
                    continue
                delta_progress = 0

            delta_progress = int(delta_progress)
            current_progress += delta_progress
            frame_indices_valid.append(int(frame_indices_all[idx]))
            delta_progress_values.append(delta_progress)
            cumulative_progress_values.append(current_progress)
            if idx < len(ij_pairs):
                pair_indices_valid.append(tuple(int(v) for v in ij_pairs[idx]))

        if len(frame_indices_valid) == 0:
            continue

        sparse_results.append({
            "episode_id": episode_id,
            "target_demo_path": meta.get("target_demo_path"),
            "reference_demo_path": meta.get("reference_demo_path"),
            "T": meta["T"],
            "frame_indices": frame_indices_valid,
            "pair_indices": pair_indices_valid,
            "delta_progress": delta_progress_values,
            "cumulative_progress": cumulative_progress_values,
            "missing_pair_indices": missing_pair_indices,
        })

    return sparse_results


def evaluate_curves(
    inference,
    demo_list: List[Dict[str, Any]],
    reference_demo_path: Optional[str],
    task_desc: str,
    target_views: List[str],
    reference_config: SimpleNamespace,
    step_interval: int = 1,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
    batch_size: int = 1,
    global_build_workers: int = 1,
    message_chunk_size: Optional[int] = None,
) -> Tuple[Dict[str, float], List[Tuple[np.ndarray, np.ndarray, int]]]:
    """批量推理多个 demo 的 progress curve 并计算评估指标。"""
    print(f"正在批量推理 {len(demo_list)} 个demo的progress curve（global模式）...")

    episode_metas, global_jobs = build_episode_jobs_from_demo_list(
        demo_list=demo_list,
        reference_demo_path=reference_demo_path,
        target_views=target_views,
        step_interval=step_interval,
        start_frame=start_frame,
        end_frame=end_frame,
        include_last_pair=False,
    )

    if len(global_jobs) == 0:
        return {
            "pearson": 0.0,
            "spearman": 0.0,
            "norm_total_variation": 0.0,
            "monotonicity_rate": 0.0,
            "num_valid_demos": 0,
        }, []

    build_message_fn = partial(
        _build_message_for_job,
        task_desc=task_desc,
        target_views=target_views,
        reference_config=reference_config,
    )
    episode_predictions = infer_job_predictions(
        inference=inference,
        episode_metas=episode_metas,
        global_jobs=global_jobs,
        build_message_fn=build_message_fn,
        batch_size=batch_size,
        global_build_workers=global_build_workers,
        message_chunk_size=message_chunk_size,
        desc="Global inference",
    )

    pearsons: List[float] = []
    spearmans: List[float] = []
    norm_total_vars: List[float] = []
    monotonicity_rates: List[float] = []
    all_curves: List[Tuple[np.ndarray, np.ndarray, int]] = []

    sparse_results = build_sparse_curve_results(
        episode_metas=episode_metas,
        episode_predictions=episode_predictions,
        fill_missing_with_zero=False,
    )

    for sparse_result in sparse_results:
        frame_indices_np = np.array(sparse_result["frame_indices"])
        progress_values_np = np.array(sparse_result["cumulative_progress"])
        gt_progress = compute_ground_truth_curve(frame_indices_np)
        if gt_progress.size != progress_values_np.size:
            print(f"警告：episode {sparse_result['target_demo_path']} 结果长度不匹配，跳过")
            continue

        pearson, spearman = calc_correlation(progress_values_np, gt_progress)
        _, norm_tv = calc_total_variation(progress_values_np)
        monotonicity_rate = calc_monotonicity_rate(progress_values_np)

        pearsons.append(pearson)
        spearmans.append(spearman)
        norm_total_vars.append(norm_tv)
        monotonicity_rates.append(monotonicity_rate)
        all_curves.append((frame_indices_np, progress_values_np, sparse_result["T"]))

    return {
        "pearson": safe_mean(pearsons),
        "spearman": safe_mean(spearmans),
        "norm_total_variation": safe_mean(norm_total_vars),
        "monotonicity_rate": safe_mean(monotonicity_rates),
        "num_valid_demos": len(pearsons),
    }, all_curves


def load_demo_list_from_json(path: str) -> List[Any]:
    """读取与 CLI 相同格式的 demo 列表 JSON（顶层含 eval 字典）。"""
    data = load_json(path)
    return list(data["eval"].values())


def save_progress_curves_plot(
    all_curves: List[Tuple[np.ndarray, np.ndarray, int]],
    plot_path: str,
) -> None:
    import matplotlib.pyplot as plt

    if not all_curves:
        return
    plt.figure(figsize=(12, 8))
    for frame_indices, progress_values, total_frames in all_curves:
        normalized_frames = (
            (frame_indices - frame_indices[0]) / (frame_indices[-1] - frame_indices[0])
            if len(frame_indices) > 1 and frame_indices[-1] != frame_indices[0]
            else np.linspace(0, 1, len(frame_indices))
        )
        plt.plot(normalized_frames, progress_values, alpha=0.6, linewidth=1, label=f"T={total_frames}")
    plt.xlabel("Normalized Frame Index")
    plt.ylabel("Progress (%)")
    plt.title(f"All Progress Curves (n={len(all_curves)})")
    plt.legend(loc="best", fontsize=8, ncol=2)
    plt.grid(True, alpha=0.3)
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n所有curve图已保存到: {plot_path}")


def run_eval_curves_from_batch_demos(
    base_model: str,
    adapter: str,
    config_path: str,
    task_desc: str,
    reference_demo: Optional[str],
    *,
    demo_list: Optional[List[Any]] = None,
    demo_list_path: Optional[str] = None,
    step_interval: int = 1,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
    batch_size: int = 1,
    num_gpus: int = 1,
    global_build_workers: int = 16,
    message_chunk_size: Optional[int] = None,
    output_json: Optional[str] = None,
    plot_output: Optional[str] = None,
) -> Tuple[Dict[str, float], List[Tuple[np.ndarray, np.ndarray, int]]]:
    """加载配置、构建推理器、跑 evaluate_curves，并可选保存曲线图与指标 JSON。"""
    if demo_list is None:
        if not demo_list_path:
            raise ValueError("demo_list 与 demo_list_path 必须提供其一")
        print(f"正在加载demo列表: {demo_list_path}")
        demo_list = load_demo_list_from_json(demo_list_path)
    else:
        demo_list = list(demo_list)

    print(f"共 {len(demo_list)} 个demo")

    config = load_config_namespace(config_path)

    from multi_gpu_inferencer import MultiGPUDeltaProgressInference

    inference = MultiGPUDeltaProgressInference(
        base_model_path=base_model,
        adapter_path=adapter,
        num_gpus=num_gpus,
    )

    target_views = config.sampling.required_views
    metrics, all_curves = evaluate_curves(
        inference=inference,
        demo_list=demo_list,
        reference_demo_path=reference_demo,
        task_desc=task_desc,
        target_views=target_views,
        reference_config=config.reference,
        step_interval=step_interval,
        start_frame=start_frame,
        end_frame=end_frame,
        batch_size=batch_size,
        global_build_workers=global_build_workers,
        message_chunk_size=message_chunk_size,
    )

    plot_path = plot_output if plot_output else "all_curves.png"
    save_progress_curves_plot(all_curves, plot_path)

    print("\n" + "=" * 50)
    print("评估结果:")
    print(f"有效demo数量: {metrics['num_valid_demos']}")
    print(f"Pearson Correlation: {metrics['pearson']:.4f}")
    print(f"Spearman Correlation: {metrics['spearman']:.4f}")
    print(f"Normalized Total Variation: {metrics['norm_total_variation']:.4f}")
    print(f"Monotonicity Rate: {metrics['monotonicity_rate']:.4f}")
    print("=" * 50)

    if output_json:
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存到: {output_json}")

    return metrics, all_curves
