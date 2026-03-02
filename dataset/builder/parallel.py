"""Multi-level parallel processing for dataset building.

Supports both task-level and demo-level parallelism to maximize speed.
"""

import os
import random
import logging
from typing import Dict, Any, List, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
from types import SimpleNamespace

from .sampling import generate_samples_for_demo
from .io import save_task_samples

logger = logging.getLogger(__name__)


def _process_demo_worker(
    worker_args: Tuple
) -> Tuple[str, str, List[Dict[str, Any]]]:
    """Worker function for processing a single demo.

    Args:
        worker_args: Tuple of (
            root, task_name, demo_name, task_desc, all_task_names,
            task_desc_map, reference_demos, required_views,
            sampling_cfg, reference_cfg, filtering_cfg, seed_offset
        )

    Returns:
        (task_name, demo_name, samples)
    """
    (
        root, task_name, demo_name, task_desc, all_task_names,
        task_desc_map, reference_demos, required_views,
        sampling_cfg, reference_cfg, filtering_cfg, seed_offset
    ) = worker_args

    samples = generate_samples_for_demo(
        root=root,
        task_name=task_name,
        demo_name=demo_name,
        task_desc=task_desc,
        all_task_names=all_task_names,
        task_desc_map=task_desc_map,
        reference_demos=reference_demos,
        required_views=required_views,
        sampling_cfg=sampling_cfg,
        reference_cfg=reference_cfg,
        filtering_cfg=filtering_cfg,
        seed_offset=seed_offset,
    )

    return task_name, demo_name, samples


def process_task_with_demo_parallelism(
    config: SimpleNamespace,
    task_name: str,
    train_demos: List[str],
    eval_demos: List[str],
    task_desc_map: Dict[str, str],
    all_task_names: List[str],
    num_workers: int,
    base_seed: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
    """Process a single task with demo-level parallelism.

    Uses ProcessPoolExecutor to parallelize across demos within a task.

    Returns:
        (train_samples, train_metadata, eval_samples, eval_metadata)
    """
    logger.info(f"Processing task: {task_name} with {num_workers} workers")

    task_desc = task_desc_map[task_name]
    sampling_cfg = config.sampling
    reference_cfg = config.reference
    filtering_cfg = config.filtering

    # Prepare work items for train demos
    train_work_items = []
    for idx, demo_name in enumerate(train_demos):
        seed_offset = base_seed + hash(f"{task_name}_{demo_name}_train") % 100000
        train_work_items.append((
            config.root, task_name, demo_name, task_desc, all_task_names,
            task_desc_map, train_demos, sampling_cfg.required_views,
            sampling_cfg, reference_cfg, filtering_cfg, seed_offset
        ))

    # Prepare work items for eval demos
    eval_work_items = []
    for idx, demo_name in enumerate(eval_demos):
        seed_offset = base_seed + hash(f"{task_name}_{demo_name}_eval") % 100000
        eval_work_items.append((
            config.root, task_name, demo_name, task_desc, all_task_names,
            task_desc_map, train_demos, sampling_cfg.required_views,
            sampling_cfg, reference_cfg, filtering_cfg, seed_offset
        ))

    train_samples = []
    eval_samples = []

    # Process train demos in parallel
    if train_work_items:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(_process_demo_worker, item): item for item in train_work_items}

            for future in as_completed(futures):
                task_name_ret, demo_name, samples = future.result()
                train_samples.extend(samples)
                logger.debug(f"  {task_name}/{demo_name} (train): generated {len(samples)} samples")

    # Process eval demos in parallel
    if eval_work_items:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(_process_demo_worker, item): item for item in eval_work_items}

            for future in as_completed(futures):
                task_name_ret, demo_name, samples = future.result()
                eval_samples.extend(samples)
                logger.debug(f"  {task_name}/{demo_name} (eval): generated {len(samples)} samples")

    # Build metadata
    train_metadata = {
        "task_name": task_name,
        "split": "train",
        "sample_count": len(train_samples),
        "episodes": train_demos,
    }

    eval_metadata = {
        "task_name": task_name,
        "split": "eval",
        "sample_count": len(eval_samples),
        "episodes": eval_demos,
    }

    logger.info(f"Task {task_name}: train={len(train_samples)}, eval={len(eval_samples)}")

    return train_samples, train_metadata, eval_samples, eval_metadata


def process_all_tasks(
    config: SimpleNamespace,
    train_tasks: Dict[str, List[str]],
    eval_tasks: Dict[str, List[str]],
    task_desc_map: Dict[str, str],
    num_workers: int,
    seed: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Process all tasks with full parallelism.

    For each task, uses demo-level parallelism. Tasks are processed sequentially
    but demos within each task are parallelized.

    Returns:
        (train_task_info, eval_task_info) - metadata for all tasks
    """
    all_task_names = list(train_tasks.keys())
    train_task_info: Dict[str, Any] = {}
    eval_task_info: Dict[str, Any] = {}

    os.makedirs(config.train_output_dir, exist_ok=True)
    if config.eval_output_dir:
        os.makedirs(config.eval_output_dir, exist_ok=True)

    total_tasks = len(train_tasks)
    logger.info(f"Processing {total_tasks} tasks with demo-level parallelism (workers={num_workers})")

    for task_idx, task_name in enumerate(sorted(train_tasks.keys())):
        train_demos = train_tasks[task_name]
        eval_demos = eval_tasks.get(task_name, [])

        logger.info(f"[{task_idx+1}/{total_tasks}] Processing task: {task_name}")

        # Process this task with demo-level parallelism
        train_samples, train_metadata, eval_samples, eval_metadata = process_task_with_demo_parallelism(
            config=config,
            task_name=task_name,
            train_demos=train_demos,
            eval_demos=eval_demos,
            task_desc_map=task_desc_map,
            all_task_names=all_task_names,
            num_workers=num_workers,
            base_seed=seed + task_idx,
        )

        # Save train samples immediately
        if train_samples:
            save_task_samples(config.train_output_dir, task_name, "train", train_samples)
            train_task_info[task_name] = train_metadata

        # Save eval samples immediately
        if eval_samples and config.eval_output_dir:
            save_task_samples(config.eval_output_dir, task_name, "eval", eval_samples)
            eval_task_info[task_name] = eval_metadata

    return train_task_info, eval_task_info
