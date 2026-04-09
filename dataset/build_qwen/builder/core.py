"""Core dataset builder logic - task splitting and coordination."""

import os
import logging
from typing import Dict, Any, List, Tuple
from types import SimpleNamespace

from .stats import compute_episode_stats, filter_episodes_by_q1_q3, save_dataset_statistics

logger = logging.getLogger(__name__)


class DatasetBuilder:
    """Dataset builder for creating Qwen-style training data.

    This class handles:
    - Task discovery and filtering
    - Episode statistics computation
    - Train/eval split
    - Dataset statistics saving
    """

    def __init__(self, config: SimpleNamespace):
        self.config = config
        self.root = config.root
        self.required_views = config.sampling.required_views
        self.sampling_cfg = config.sampling
        self.reference_cfg = config.reference
        self.filtering_cfg = config.filtering
        self.split_cfg = config.split

    def split_tasks(
        self,
        task_desc_map: Dict[str, str]
    ) -> Tuple[Dict[str, List[str]], Dict[str, List[str]], Dict[str, Any]]:
        """Split tasks into train and eval sets with optional Q1-Q3 filtering.

        Returns:
            (train_tasks, eval_tasks, all_task_stats)
            - train_tasks: Dict of task_name -> list of train demo names
            - eval_tasks: Dict of task_name -> list of eval demo names
            - all_task_stats: Dict of task_name -> task statistics
        """
        train_ratio = self.split_cfg.train_ratio
        min_demos_per_task = self.split_cfg.min_demos_per_task
        selected_tasks = self.split_cfg.selected_tasks
        use_q1_q3_only = getattr(self.split_cfg, 'use_q1_q3_only', False)

        # Get all task directories
        task_dirs = [d for d in os.listdir(self.root) if os.path.isdir(os.path.join(self.root, d))]

        # Filter by selected_tasks
        if selected_tasks is None:
            tasks_to_process = task_dirs
            logger.info(f"Processing all tasks (total: {len(tasks_to_process)})")
        else:
            tasks_to_process = selected_tasks
            logger.info(f"Processing selected tasks: {tasks_to_process} (total: {len(tasks_to_process)})")

        train_tasks: Dict[str, List[str]] = {}
        eval_tasks: Dict[str, List[str]] = {}
        all_task_stats: Dict[str, Any] = {}

        for task_name in tasks_to_process:
            if task_name not in task_desc_map:
                logger.warning(f"Task {task_name} not found in task_descriptions.json, skipping")
                continue

            # Compute episode statistics for this task
            task_stats = compute_episode_stats(self.root, task_name, self.required_views)
            all_task_stats[task_name] = task_stats

            # Get valid episode IDs
            if use_q1_q3_only:
                valid_episodes = filter_episodes_by_q1_q3(task_stats)
                total_eps = task_stats["statistics"]["total_episodes"]
                logger.info(
                    f"Task {task_name}: using Q1-Q3 filter, {len(valid_episodes)}/{total_eps} episodes retained"
                )
            else:
                valid_episodes = [ep["episode_id"] for ep in task_stats["episodes"]]

            # Skip tasks with too few episodes
            if len(valid_episodes) < min_demos_per_task:
                logger.warning(
                    f"Task {task_name} has only {len(valid_episodes)} valid episodes "
                    f"(min required: {min_demos_per_task}), skipping"
                )
                continue

            # Split into train and eval
            num_train = int(len(valid_episodes) * train_ratio)
            sorted_episodes = sorted(valid_episodes)
            train_tasks[task_name] = sorted_episodes[:num_train]
            eval_tasks[task_name] = sorted_episodes[num_train:]

            logger.info(
                f"Task {task_name}: {len(valid_episodes)} valid episodes total, "
                f"train={len(train_tasks[task_name])}, eval={len(eval_tasks[task_name])}"
            )

        # Save dataset statistics to root directory
        save_dataset_statistics(self.root, all_task_stats)

        return train_tasks, eval_tasks, all_task_stats
