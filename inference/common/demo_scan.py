from __future__ import annotations

import os
from typing import Dict, List, Mapping, Sequence, Tuple

from utils.utils import list_image_files


def build_view_to_frames(
    target_demo_path: str,
    target_views: Sequence[str],
) -> Dict[str, List[str]]:
    """Scan per-view frame file names for a demo."""
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
