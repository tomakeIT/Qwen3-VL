"""Dataset statistics computation and management."""

import os
import json
import logging
from typing import Dict, Any, List
import numpy as np

from utils.utils import list_subdirs, list_image_files

logger = logging.getLogger(__name__)


def compute_episode_stats(
    root: str,
    task_name: str,
    required_views: List[str]
) -> Dict[str, Any]:
    """Compute frame count statistics for all episodes in a task.

    Returns:
        Dict with episode info including frame counts, paths, and task-level statistics.
    """
    task_path = os.path.join(root, task_name)
    episode_dirs = list_subdirs(task_path)

    episodes_info = []
    frame_counts = []

    for ep_name in sorted(episode_dirs):
        ep_path = os.path.join(task_path, ep_name)

        # Get frame count from required views
        view_frame_counts = {}
        valid_episode = True

        for view in required_views:
            view_path = os.path.join(ep_path, view)
            if os.path.exists(view_path):
                frames = list_image_files(view_path)
                view_frame_counts[view] = len(frames)
            else:
                valid_episode = False
                break

        if not valid_episode or not view_frame_counts:
            continue

        # Check consistency across views
        counts = list(view_frame_counts.values())
        is_consistent = len(set(counts)) == 1
        frame_count = counts[0] if is_consistent else min(counts)

        episodes_info.append({
            "episode_id": ep_name,
            "episode_path": ep_path,
            "frame_count": frame_count,
            "view_frame_counts": view_frame_counts,
            "is_consistent": is_consistent
        })
        frame_counts.append(frame_count)

    # Compute statistics
    if frame_counts:
        frame_counts_sorted = sorted(frame_counts)
        n = len(frame_counts_sorted)
        q1_idx = n // 4
        q3_idx = 3 * n // 4

        stats = {
            "total_episodes": len(episodes_info),
            "frame_counts": frame_counts,
            "min": min(frame_counts),
            "max": max(frame_counts),
            "mean": float(np.mean(frame_counts)),
            "median": float(np.median(frame_counts)),
            "q1": float(frame_counts_sorted[q1_idx]),
            "q3": float(frame_counts_sorted[q3_idx]),
            "std": float(np.std(frame_counts)),
        }
    else:
        stats = {
            "total_episodes": 0,
            "frame_counts": [],
            "min": 0, "max": 0, "mean": 0, "median": 0,
            "q1": 0, "q3": 0, "std": 0,
        }

    return {
        "task_name": task_name,
        "task_path": task_path,
        "episodes": episodes_info,
        "statistics": stats
    }


def filter_episodes_by_q1_q3(task_stats: Dict[str, Any]) -> List[str]:
    """Filter episodes to keep only those with frame count in [Q1, Q3] range."""
    stats = task_stats["statistics"]
    q1 = stats["q1"]
    q3 = stats["q3"]

    filtered = []
    for ep in task_stats["episodes"]:
        if q1 <= ep["frame_count"] <= q3:
            filtered.append(ep["episode_id"])

    logger.info(
        f"Task {task_stats['task_name']}: filtered {len(filtered)}/{stats['total_episodes']} "
        f"episodes in Q1-Q3 range [{q1:.0f}, {q3:.0f}]"
    )
    return filtered


def save_dataset_statistics(
    root: str,
    all_task_stats: Dict[str, Any]
) -> str:
    """Save dataset statistics to the root directory."""
    stats_path = os.path.join(root, "dataset_statistics.json")

    summary = {
        "dataset_root": root,
        "total_tasks": len(all_task_stats),
        "tasks": {}
    }

    for task_name, task_data in all_task_stats.items():
        summary["tasks"][task_name] = {
            "task_path": task_data["task_path"],
            "total_episodes": task_data["statistics"]["total_episodes"],
            "frame_statistics": {
                "min": task_data["statistics"]["min"],
                "max": task_data["statistics"]["max"],
                "mean": task_data["statistics"]["mean"],
                "median": task_data["statistics"]["median"],
                "q1": task_data["statistics"]["q1"],
                "q3": task_data["statistics"]["q3"],
                "std": task_data["statistics"]["std"],
            },
            "episodes": [
                {
                    "episode_id": ep["episode_id"],
                    "episode_path": ep["episode_path"],
                    "frame_count": ep["frame_count"],
                    "is_consistent": ep["is_consistent"]
                }
                for ep in task_data["episodes"]
            ]
        }

    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved dataset statistics to {stats_path}")
    return stats_path
