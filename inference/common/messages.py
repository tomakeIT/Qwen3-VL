from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import random
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from tqdm import tqdm

from common.demo_scan import resolve_pair_frame_paths, scan_demo_frames
from utils.data_formatting import build_qwen_messages
from utils.frame_sampling import sample_reference_frames_from_demo
from utils.prompt import build_prompt


def sample_reference_demo_pack(
    reference_demo_path: Optional[str],
    reference_config,
    rng: Optional[random.Random] = None,
) -> Tuple[List[Any], List[int]]:
    """采样并缓存 reference demo，用于跨多个 pair 复用。"""
    if not reference_demo_path:
        return [], []

    return sample_reference_frames_from_demo(
        avg_frames=reference_config.avg_frames,
        min_frames=reference_config.frames_min,
        max_frames=reference_config.frames_max,
        std=reference_config.frames_std,
        reference_demo_path=reference_demo_path,
        reference_views=reference_config.views,
        ref_jitter=reference_config.jitter,
        rng=rng,
    )


def build_messages_from_inputs(
    target_inputs_t1: Sequence[Any],
    target_inputs_t2: Sequence[Any],
    reference_inputs: Sequence[Any],
    reference_progress_ints: Sequence[int],
    reference_view_names: Sequence[str],
    target_view_names: Sequence[str],
    task_desc: str,
) -> List[Dict[str, Any]]:
    """根据预加载图像对象或路径直接构造 Qwen messages。"""
    img_paths, human_str = build_prompt(
        ref_img_paths=list(reference_inputs),
        ref_progress_ints=list(reference_progress_ints),
        target_img_paths_t1=list(target_inputs_t1),
        target_img_paths_t2=list(target_inputs_t2),
        reference_view_names=list(reference_view_names),
        target_view_names=list(target_view_names),
        task_desc=task_desc,
    )
    return build_qwen_messages(human_str, img_paths)


def build_messages_from_demo(
    target_demo_path: str,
    i: int,
    j: int,
    reference_demo_path: Optional[str],
    task_desc: str,
    target_views: Sequence[str],
    reference_config,
) -> List[Dict[str, Any]]:
    """根据 demo 路径和参数直接构造 messages 格式。"""
    view_to_frames, total_frames = scan_demo_frames(target_demo_path, target_views)
    if total_frames < 2:
        raise ValueError(f"Target demo has insufficient frames: T={total_frames}")

    target_paths_t1, target_paths_t2 = resolve_pair_frame_paths(
        target_demo_path=target_demo_path,
        view_to_frames=view_to_frames,
        target_views=target_views,
        i=i,
        j=j,
    )
    ref_inputs, ref_progress_ints = sample_reference_demo_pack(
        reference_demo_path=reference_demo_path,
        reference_config=reference_config,
    )
    return build_messages_from_inputs(
        target_inputs_t1=target_paths_t1,
        target_inputs_t2=target_paths_t2,
        reference_inputs=ref_inputs,
        reference_progress_ints=ref_progress_ints,
        reference_view_names=reference_config.views,
        target_view_names=target_views,
        task_desc=task_desc,
    )


def build_messages_for_job_chunk(
    jobs: Sequence[Any],
    build_message_fn: Callable[[Any], Tuple[int, int, List[Dict[str, Any]]]],
    global_build_workers: int,
) -> Tuple[List[Tuple[int, int]], List[List[Dict[str, Any]]]]:
    """为一小块 jobs 并行构建 messages，控制内存峰值。"""
    effective_workers = max(1, global_build_workers)
    all_meta: List[Tuple[int, int]] = []
    all_messages: List[List[Dict[str, Any]]] = []

    if effective_workers == 1:
        iterator = map(build_message_fn, jobs)
        for episode_id, pair_idx, messages in tqdm(
            iterator,
            total=len(jobs),
            desc="构建messages",
        ):
            all_meta.append((episode_id, pair_idx))
            all_messages.append(messages)
        return all_meta, all_messages

    with ThreadPoolExecutor(max_workers=effective_workers) as executor:
        iterator = executor.map(build_message_fn, jobs)
        for episode_id, pair_idx, messages in tqdm(
            iterator,
            total=len(jobs),
            desc="构建messages",
        ):
            all_meta.append((episode_id, pair_idx))
            all_messages.append(messages)
    return all_meta, all_messages
