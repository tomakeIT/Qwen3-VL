"""Per-demo sampling logic for dataset building."""

import os
import random
import logging
from typing import Dict, Any, List, Tuple, Optional

from utils.prompt import build_prompt
from utils.data_formatting import build_qwen_data_sample, compute_delta_progress_label_int
from utils.utils import abs_to_rel_path, compute_mean_pixel_diff, list_image_files
from utils.frame_sampling import sample_pair_indices

logger = logging.getLogger(__name__)


def sample_reference_multiview_frames(
    root: str,
    task_name: str,
    valid_demos: List[str],
    reference_views: List[str],
    avg_frames: int,
    frames_min: int,
    frames_max: int,
    frames_std: float,
    ref_jitter: int,
) -> Tuple[List[str], List[int]]:
    """Sample reference frames from a random demo in the same task."""
    from utils.frame_sampling import sample_reference_frames_from_demo
    from utils.utils import list_subdirs

    task_path = os.path.join(root, task_name)

    demo_name = random.choice(valid_demos)
    demo_path = os.path.join(task_path, demo_name)
    view_names = list_subdirs(demo_path)
    if all(v in view_names for v in reference_views):
        ref_demo_path = demo_path
    else:
        return [], []

    return sample_reference_frames_from_demo(
        avg_frames=avg_frames,
        min_frames=frames_min,
        max_frames=frames_max,
        std=frames_std,
        reference_demo_path=ref_demo_path,
        reference_views=reference_views,
        ref_jitter=ref_jitter,
    )


def generate_samples_for_demo(
    root: str,
    task_name: str,
    demo_name: str,
    task_desc: str,
    all_task_names: List[str],
    task_desc_map: Dict[str, str],
    reference_demos: List[str],
    required_views: List[str],
    sampling_cfg: Any,
    reference_cfg: Any,
    filtering_cfg: Any,
    seed_offset: int = 0,
) -> List[Dict[str, Any]]:
    """Generate samples for a single demo.

    This function can be called in parallel for different demos.

    Returns:
        List of samples for this demo
    """
    # Set random seed for reproducibility in this worker
    random.seed(seed_offset)

    demo_path = os.path.join(root, task_name, demo_name)
    samples = []

    # Get all frames for all views
    view_to_frames: Dict[str, List[str]] = {}
    for v in required_views:
        v_path = os.path.join(demo_path, v)
        frames = list_image_files(v_path)
        view_to_frames[v] = frames

    T = min(len(frames_list) for frames_list in view_to_frames.values())
    if T < 2:
        return samples

    ref_img_paths: List[str] = []
    ref_prog_ints: List[int] = []

    # For each pair in demo
    for pair_idx in range(sampling_cfg.pairs_per_demo):
        # Sample reference frames
        if reference_cfg.resample_every > 0 and (pair_idx % reference_cfg.resample_every == 0):
            ref_img_paths, ref_prog_ints = sample_reference_multiview_frames(
                root=root,
                task_name=task_name,
                valid_demos=reference_demos,
                reference_views=reference_cfg.views,
                avg_frames=reference_cfg.avg_frames,
                frames_min=reference_cfg.frames_min,
                frames_max=reference_cfg.frames_max,
                frames_std=reference_cfg.frames_std,
                ref_jitter=reference_cfg.jitter,
            )

        # Sample target i,j pairs
        pair = sample_pair_indices(
            T,
            sampling_cfg.max_delta_t,
            sampling_cfg.min_delta_t,
            sampling_cfg.peak_distance,
            sampling_cfg.rise_factor,
            sampling_cfg.decay_factor,
        )
        if pair is None:
            continue
        i, j = pair

        target_paths_t1: List[str] = []
        target_paths_t2: List[str] = []
        per_view_diffs: List[float] = []

        for v in required_views:
            frames_v = view_to_frames[v]
            frame_i_name = frames_v[i]
            frame_j_name = frames_v[j]
            v_path = os.path.join(demo_path, v)
            img_abs_1 = os.path.join(v_path, frame_i_name)
            img_abs_2 = os.path.join(v_path, frame_j_name)
            target_paths_t1.append(img_abs_1)
            target_paths_t2.append(img_abs_2)

            # Filter static diff
            if filtering_cfg.static_diff_threshold > 0:
                diff = compute_mean_pixel_diff(img_abs_1, img_abs_2)
                per_view_diffs.append(diff)

        # Compute delta progress label
        delta_progress_int = compute_delta_progress_label_int(i, j, T)
        if filtering_cfg.static_diff_threshold > 0 and per_view_diffs:
            if all(d < filtering_cfg.static_diff_threshold for d in per_view_diffs):
                delta_progress_int = 0

        # Random mismatch task description
        if random.random() < filtering_cfg.mismatch_prob and len(all_task_names) > 1:
            mismatch_candidates = [t for t in all_task_names if t != task_name]
            used_task_desc = task_desc_map[random.choice(mismatch_candidates)]
            delta_progress_int = 0
        else:
            used_task_desc = task_desc

        img_paths, human_str = build_prompt(
            ref_img_paths=ref_img_paths,
            ref_progress_ints=ref_prog_ints,
            target_img_paths_t1=target_paths_t1,
            target_img_paths_t2=target_paths_t2,
            reference_view_names=reference_cfg.views,
            target_view_names=required_views,
            task_desc=used_task_desc
        )

        assistant_answer = f"{delta_progress_int:+d}"
        images_rel = [abs_to_rel_path(root, img) for img in img_paths]
        data_sample = build_qwen_data_sample(images_rel, human_str, assistant_answer)
        samples.append(data_sample)

    return samples
