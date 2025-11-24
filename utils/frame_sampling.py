import os
import random
from typing import List, Dict, Tuple, Optional
from utils.utils import list_image_files
from utils.data_formatting import compute_abs_progress_from_index_int


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

