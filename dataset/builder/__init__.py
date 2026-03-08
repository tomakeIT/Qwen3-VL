"""Dataset builder module for creating Qwen-style training data.

This module provides a modular, parallelized approach to building datasets
for VLAC-style delta-progress critic training.

Example usage:
    from builder import DatasetBuilder, process_all_tasks, load_task_descriptions
    from builder.io import save_split_metadata

    # Load configuration
    config = load_config("config.yaml")

    # Initialize builder and split tasks
    builder = DatasetBuilder(config)
    train_tasks, eval_tasks, stats = builder.split_tasks(task_desc_map)

    # Process all tasks with parallel demo processing
    train_info, eval_info = process_all_tasks(
        config, train_tasks, eval_tasks, task_desc_map, num_workers=16, seed=42
    )

    # Save metadata
    save_split_metadata(config.output_dir, "train", train_info)
    save_split_metadata(config.output_dir, "eval", eval_info)
"""

from .core import DatasetBuilder
from .parallel import process_all_tasks
from .io import load_task_descriptions, save_split_metadata, save_task_samples
from .stats import compute_episode_stats, filter_episodes_by_q1_q3, save_dataset_statistics

__all__ = [
    "DatasetBuilder",
    "process_all_tasks",
    "load_task_descriptions",
    "save_split_metadata",
    "save_task_samples",
    "compute_episode_stats",
    "filter_episodes_by_q1_q3",
    "save_dataset_statistics",
]
