"""Build Qwen-style JSON dataset for VLAC-style delta-progress critic.

This script builds training datasets with multi-level parallelism:
- Task-level: Tasks are processed sequentially but demos within each task are parallelized
- Demo-level: Multiple demos from the same task are processed concurrently

Usage:
    python build_qwen_dataset.py --config configs/build_config_15tasks.yaml --num_workers 16
"""

import os
import sys
import random
import argparse
import logging
import yaml
from types import SimpleNamespace

# Add parent directory (Qwen3-VL root) to path to import utils and builder modules
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

from dataset.builder import (
    DatasetBuilder,
    process_all_tasks,
    load_task_descriptions,
    save_split_metadata,
)
from utils.utils import dict_to_namespace


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def main(args: argparse.Namespace) -> None:
    """Main entry point for dataset building."""
    # Load configuration
    with open(args.config, "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)

    # Override with command line arguments
    if args.root:
        config_dict["root"] = args.root

    config = dict_to_namespace(config_dict)
    random.seed(config.seed)

    # Load task descriptions
    task_desc_map = load_task_descriptions(config.root)

    # Initialize builder and split tasks
    builder = DatasetBuilder(config)
    train_tasks, eval_tasks, all_task_stats = builder.split_tasks(task_desc_map)

    total_tasks = len(train_tasks)
    if total_tasks == 0:
        logger.error("No tasks to process! Check your configuration.")
        return

    logger.info(f"Processing {total_tasks} tasks with >={config.split.min_demos_per_task} demos")
    logger.info("-" * 60)

    # Process all tasks with demo-level parallelism
    num_workers = args.num_workers or os.cpu_count() or 1
    logger.info(f"Using {num_workers} workers for demo-level parallelism")

    train_task_info, eval_task_info = process_all_tasks(
        config=config,
        train_tasks=train_tasks,
        eval_tasks=eval_tasks,
        task_desc_map=task_desc_map,
        num_workers=num_workers,
        seed=config.seed,
    )

    # Save aggregated metadata
    if train_task_info:
        save_split_metadata(config.train_output_dir, "train", train_task_info)

    if eval_task_info and config.eval_output_dir:
        save_split_metadata(config.eval_output_dir, "eval", eval_task_info)

    logger.info("=" * 60)
    logger.info("Dataset building completed!")
    logger.info(f"Train tasks: {len(train_task_info)}")
    logger.info(f"Eval tasks: {len(eval_task_info)}")
    logger.info(f"Train output: {config.train_output_dir}")
    logger.info(f"Eval output: {config.eval_output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build Qwen-style JSON dataset for VLAC-style delta-progress critic"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file."
    )
    parser.add_argument(
        "--root",
        type=str,
        help="Root of dataset (overrides YAML config)."
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: CPU count)."
    )

    args = parser.parse_args()
    main(args)
