"""File IO operations and metadata management."""

import os
import json
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


def save_split_metadata(
    output_dir: str,
    split_name: str,
    task_samples_info: Dict[str, Any]
) -> str:
    """Save metadata about the generated samples.

    Args:
        output_dir: Directory to save metadata
        split_name: 'train' or 'eval'
        task_samples_info: Dict with task_name -> {episodes, sample_count, ...}

    Returns:
        Path to the saved metadata file
    """
    os.makedirs(output_dir, exist_ok=True)
    metadata_path = os.path.join(output_dir, f"{split_name}_metadata.json")

    total_samples = sum(info["sample_count"] for info in task_samples_info.values())
    total_episodes = sum(len(info["episodes"]) for info in task_samples_info.values())

    metadata = {
        "split": split_name,
        "total_samples": total_samples,
        "total_episodes": total_episodes,
        "total_tasks": len(task_samples_info),
        "tasks": {}
    }

    for task_name, info in task_samples_info.items():
        metadata["tasks"][task_name] = {
            "sample_count": info["sample_count"],
            "episode_count": len(info["episodes"]),
            "episodes": info["episodes"],
            "frame_count_range": info.get("frame_count_range", {}),
            "sample_to_episode_ratio": info["sample_count"] / len(info["episodes"]) if info["episodes"] else 0
        }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {split_name} metadata to {metadata_path}")
    return metadata_path


def save_task_samples(
    output_dir: str,
    task_name: str,
    split_name: str,
    samples: List[Dict[str, Any]]
) -> str:
    """Save samples for a specific task and split.

    Args:
        output_dir: Output directory
        task_name: Name of the task
        split_name: 'train' or 'eval'
        samples: List of samples to save

    Returns:
        Path to the saved JSON file
    """
    os.makedirs(output_dir, exist_ok=True)

    if split_name == "eval":
        filename = f"{task_name}_eval.json"
    else:
        filename = f"{task_name}.json"

    json_path = os.path.join(output_dir, filename)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved {split_name} JSON for task {task_name} to {json_path}")
    return json_path


def load_task_descriptions(root: str) -> Dict[str, str]:
    """Load task descriptions from the dataset root."""
    desc_path = os.path.join(root, "task_descriptions.json")
    with open(desc_path, "r", encoding="utf-8") as f:
        return json.load(f)
