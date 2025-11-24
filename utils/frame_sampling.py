import os
import random
from typing import List, Dict, Tuple, Optional
from utils.utils import list_image_files
from utils.data_formatting import compute_abs_progress_from_index_int


def sample_pair_indices(
    T: int,
    max_delta_t: int,
    min_delta_t: int = 1,
    peak_distance: int = 3,
    rise_factor: float = 1.3,
    decay_factor: float = 0.8,
) -> Optional[Tuple[int, int]]:
    """
    在给定长度为 T 的帧序列里，按山峰分布随机采样 (i, j)
    
    山峰分布：在 peak_distance 位置达到最大权重，左侧上升，右侧衰减
    
    Args:
        T: 序列长度
        max_delta_t: 最大时间差
        min_delta_t: 最小时间差（硬限制）
        peak_distance: 峰值位置（权重最大的距离）
        rise_factor: 上升因子（峰值左侧，>1 表示上升速度）
        decay_factor: 衰减因子（峰值右侧，0 < decay_factor < 1），越小衰减越快
    
    Returns:
        (i, j) 或 None
    """
    if T < 2:
        return None
    
    i = random.randint(0, T - 2)
    
    # 计算所有可能的 delta_t 及其权重（山峰分布）
    candidates: List[Tuple[int, float]] = []
    
    # 计算权重函数
    def calculate_weight(dt: int) -> float:
        if dt < peak_distance:
            # 上升阶段：从 min_delta_t 到 peak_distance
            weight = rise_factor ** (dt - min_delta_t)
        elif dt == peak_distance:
            # 峰值位置：权重为 1.0
            weight = 1.0
        else:
            # 衰减阶段：从 peak_distance 到 max_delta_t
            weight = decay_factor ** (dt - peak_distance)
        return weight
    
    # 正向采样
    max_forward = min(max_delta_t, T - 1 - i)
    for dt in range(min_delta_t, max_forward + 1):
        weight = calculate_weight(dt)
        candidates.append((dt, weight))
    
    # 反向采样
    max_backward = min(max_delta_t, i)
    for dt in range(min_delta_t, max_backward + 1):
        weight = calculate_weight(dt)
        candidates.append((-dt, weight))
    
    if not candidates:
        return None
    
    # 按权重采样（加权随机选择）
    deltas, weights = zip(*candidates)
    total_weight = sum(weights)
    if total_weight == 0:
        return None
    
    r = random.uniform(0, total_weight)
    cumsum = 0
    for delta_t, weight in candidates:
        cumsum += weight
        if r <= cumsum:
            j = i + delta_t
            return i, j
    
    # 如果没选到（理论上不会发生），返回第一个
    delta_t = candidates[0][0]
    j = i + delta_t
    return i, j


def sample_reference_frames_from_demo(
    reference_demo_path: str,
    reference_views: List[str],
    num_ref_frames: int,
    ref_jitter: int,
) -> Tuple[List[str], List[int]]:
    """
    从给定的 reference demo 路径中采样 reference frames
    
    Args:
        reference_demo_path: reference demo 的路径
        reference_views: 需要采样的视角列表
        num_ref_frames: 需要采样的帧数量
        ref_jitter: 索引 jitter 范围
    
    Returns:
        (ref_img_paths, ref_progress_ints): 采样的图片路径列表和对应的进度整数列表
    """
    ref_img_paths: List[str] = []
    ref_progress_ints: List[int] = []
    
    ref_view_to_frames: Dict[str, List[str]] = {}
    for v in reference_views:
        v_path = os.path.join(reference_demo_path, v)
        frames = list_image_files(v_path)
        if len(frames) == 0:
            continue
        ref_view_to_frames[v] = frames
    
    if not ref_view_to_frames:
        return [], []
    
    T_ref = min(len(frames) for frames in ref_view_to_frames.values())
    if T_ref == 0:
        return [], []
    
    if T_ref <= num_ref_frames:
        base_indices = list(range(T_ref))
    else:
        base_indices = [int(round(i * (T_ref - 1) / (num_ref_frames - 1))) for i in range(num_ref_frames)]
    
    indices: List[int] = []
    for idx in base_indices:
        offset = random.randint(-ref_jitter, ref_jitter) if ref_jitter > 0 else 0
        j_idx = max(0, min(T_ref - 1, idx + offset))
        indices.append(j_idx)
    
    indices = sorted(set(indices))
    
    for idx in indices:
        prog_int = compute_abs_progress_from_index_int(idx, T_ref)
        ref_progress_ints.append(prog_int)
        for v in reference_views:
            v_path = os.path.join(reference_demo_path, v)
            frames_v = ref_view_to_frames[v]
            frame_name = frames_v[idx]
            img_abs = os.path.abspath(os.path.join(v_path, frame_name))
            ref_img_paths.append(img_abs)
    
    return ref_img_paths, ref_progress_ints

