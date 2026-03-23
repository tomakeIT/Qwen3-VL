from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import os
import random
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from tqdm import tqdm

from utils.data_formatting import build_qwen_messages
from utils.frame_sampling import sample_reference_frames_from_demo
from utils.prompt import build_prompt
from utils.utils import list_image_files


def build_view_to_frames(
    target_demo_path: str,
    target_views: Sequence[str],
) -> Dict[str, List[str]]:
    view_to_frames: Dict[str, List[str]] = {}
    for view_name in target_views:
        view_to_frames[str(view_name)] = list_image_files(os.path.join(target_demo_path, view_name))
    return view_to_frames


def infer_demo_length(view_to_frames: Mapping[str, Sequence[str]]) -> int:
    if len(view_to_frames) == 0:
        return 0
    return min(len(frames) for frames in view_to_frames.values())


def scan_demo_frames(
    target_demo_path: str,
    target_views: Sequence[str],
) -> Tuple[Dict[str, List[str]], int]:
    view_to_frames = build_view_to_frames(target_demo_path, target_views)
    return view_to_frames, infer_demo_length(view_to_frames)


def clamp_frame_index(frame_index: int, total_frames: int) -> int:
    if total_frames <= 0:
        return 0
    return max(0, min(total_frames - 1, int(frame_index)))


def resolve_pair_frame_paths(
    target_demo_path: str,
    view_to_frames: Mapping[str, Sequence[str]],
    target_views: Sequence[str],
    i: int,
    j: int,
) -> Tuple[List[str], List[str]]:
    total_frames = infer_demo_length(view_to_frames)
    i = clamp_frame_index(i, total_frames)
    j = clamp_frame_index(j, total_frames)

    target_paths_t1: List[str] = []
    target_paths_t2: List[str] = []
    for view_name in target_views:
        view_name = str(view_name)
        view_path = os.path.join(target_demo_path, view_name)
        frames = view_to_frames[view_name]
        target_paths_t1.append(os.path.abspath(os.path.join(view_path, frames[i])))
        target_paths_t2.append(os.path.abspath(os.path.join(view_path, frames[j])))
    return target_paths_t1, target_paths_t2


def sample_reference_demo_pack(
    reference_demo_path: Optional[str],
    reference_config,
    rng: Optional[random.Random] = None,
) -> Tuple[List[Any], List[int]]:
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
    effective_workers = max(1, global_build_workers)
    all_meta: List[Tuple[int, int]] = []
    all_messages: List[List[Dict[str, Any]]] = []

    if effective_workers == 1:
        iterator = map(build_message_fn, jobs)
        for episode_id, pair_idx, messages in tqdm(
            iterator,
            total=len(jobs),
            desc="构建 messages",
        ):
            all_meta.append((episode_id, pair_idx))
            all_messages.append(messages)
        return all_meta, all_messages

    with ThreadPoolExecutor(max_workers=effective_workers) as executor:
        iterator = executor.map(build_message_fn, jobs)
        for episode_id, pair_idx, messages in tqdm(
            iterator,
            total=len(jobs),
            desc="构建 messages",
        ):
            all_meta.append((episode_id, pair_idx))
            all_messages.append(messages)
    return all_meta, all_messages
