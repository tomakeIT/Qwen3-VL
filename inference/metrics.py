from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np


def pearson_correlation(pred: np.ndarray, gt: np.ndarray) -> float:
    if pred.size < 2 or gt.size < 2:
        return 0.0
    if np.allclose(pred, pred[0]) or np.allclose(gt, gt[0]):
        return 0.0
    value = np.corrcoef(pred, gt)[0, 1]
    return float(np.nan_to_num(value))


def _average_rankdata(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return np.array([], dtype=np.float32)

    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.zeros(values.size, dtype=np.float32)

    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        average_rank = (start + end - 1) / 2.0 + 1.0
        ranks[order[start:end]] = average_rank
        start = end
    return ranks


def spearman_correlation(pred: np.ndarray, gt: np.ndarray) -> float:
    if pred.size < 2 or gt.size < 2:
        return 0.0
    pred_rank = _average_rankdata(pred)
    gt_rank = _average_rankdata(gt)
    return pearson_correlation(pred_rank, gt_rank)


def calc_total_variation(
    pred: np.ndarray,
    normalization: str = "endpoint",
) -> Tuple[float, float]:
    if pred.size < 2:
        return 0.0, 0.0

    diffs = np.abs(np.diff(pred))
    total_variation = float(np.sum(diffs))
    if normalization == "endpoint":
        denom = float(abs(pred[-1] - pred[0]))
    elif normalization == "mean_abs":
        mean_abs = float(np.mean(np.abs(pred)))
        denom = mean_abs * max(pred.size - 1, 1)
    else:
        raise ValueError(f"不支持的 total variation 归一化模式: {normalization}")
    norm_tv = total_variation / denom if denom > 1e-8 else 0.0
    return total_variation, norm_tv


def calc_monotonicity_rate(pred: np.ndarray) -> float:
    if pred.size < 2:
        return 0.0
    diffs = np.diff(pred)
    return float(np.mean(diffs > 0))


def compute_ground_truth_curve(frame_indices: np.ndarray) -> np.ndarray:
    if frame_indices.size == 0:
        return np.array([], dtype=np.float32)
    if frame_indices.size == 1:
        return np.array([0.0], dtype=np.float32)

    min_frame = float(frame_indices[0])
    max_frame = float(frame_indices[-1])
    if max_frame - min_frame < 1e-8:
        return np.zeros_like(frame_indices, dtype=np.float32)
    progress = ((frame_indices - min_frame) / (max_frame - min_frame)) * 100.0
    return progress.astype(np.float32)


def _merge_duplicate_anchor_frames(
    anchor_frames: Sequence[int],
    anchor_progress: Sequence[float],
) -> Tuple[np.ndarray, np.ndarray]:
    merged_progress_by_frame = {}
    for frame_idx, progress in zip(anchor_frames, anchor_progress):
        merged_progress_by_frame[int(frame_idx)] = float(progress)

    merged_frames = np.array(sorted(merged_progress_by_frame.keys()), dtype=np.float32)
    merged_progress = np.array(
        [merged_progress_by_frame[int(frame_idx)] for frame_idx in merged_frames],
        dtype=np.float32,
    )
    return merged_frames, merged_progress


def reconstruct_dense_progress(
    pair_indices: Sequence[Tuple[int, int]],
    cumulative_progress: Sequence[float],
    total_frames: int,
) -> np.ndarray:
    if total_frames <= 0:
        return np.array([], dtype=np.float32)
    if len(pair_indices) == 0:
        return np.zeros(total_frames, dtype=np.float32)

    anchor_frames = [int(pair_indices[0][0])] + [int(j) for _, j in pair_indices]
    anchor_progress = [0.0] + [float(value) for value in cumulative_progress]
    merged_frames, merged_progress = _merge_duplicate_anchor_frames(anchor_frames, anchor_progress)
    dense_frames = np.arange(total_frames, dtype=np.float32)
    return np.interp(dense_frames, merged_frames, merged_progress).astype(np.float32)


def compute_ground_truth_dense_curve(total_frames: int) -> np.ndarray:
    if total_frames <= 0:
        return np.array([], dtype=np.float32)
    if total_frames == 1:
        return np.array([0.0], dtype=np.float32)
    return np.linspace(0.0, 100.0, total_frames, dtype=np.float32)


def compute_ground_truth_delta_curve(
    total_frames: int,
    pair_offset: int,
    num_points: int,
    valid_pair_count: Optional[int] = None,
) -> np.ndarray:
    if num_points <= 0:
        return np.array([], dtype=np.float32)
    if total_frames <= 0:
        return np.zeros(num_points, dtype=np.float32)
    if valid_pair_count is None:
        valid_pair_count = num_points
    valid_pair_count = max(0, min(valid_pair_count, num_points))
    if valid_pair_count == 0:
        return np.zeros(num_points, dtype=np.float32)

    gt_delta = 100.0 * float(pair_offset) / float(total_frames)
    return np.full(num_points, gt_delta, dtype=np.float32)
