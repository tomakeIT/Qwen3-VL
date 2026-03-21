from __future__ import annotations

import os
import random
import re
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence, TypeVar

import numpy as np

T = TypeVar("T")


def safe_mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(np.mean(values)) if values else 0.0


def safe_median(values: Iterable[float]) -> float:
    values = list(values)
    return float(np.median(values)) if values else 0.0


def sanitize_filename(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return sanitized or "task"


def sample_items(items: Sequence[T], max_items: int, seed: int) -> List[T]:
    items = list(items)
    if max_items <= 0 or len(items) <= max_items:
        return items
    rng = random.Random(seed)
    sampled_indices = sorted(rng.sample(range(len(items)), max_items))
    return [items[idx] for idx in sampled_indices]


def group_items_by(items: Sequence[T], key_fn: Callable[[T], str]) -> Dict[str, List[T]]:
    grouped: Dict[str, List[T]] = {}
    for item in items:
        grouped.setdefault(key_fn(item), []).append(item)
    return grouped


def build_summary_payload(
    grouped_items: Mapping[str, Sequence[T]],
    plotted_items: Mapping[str, Sequence[T]],
    summarize_fn: Callable[[Sequence[T]], Dict[str, Any]],
) -> Dict[str, Any]:
    task_summaries: Dict[str, Any] = {}
    all_items: List[T] = []
    for task_desc, task_items in grouped_items.items():
        task_items = list(task_items)
        all_items.extend(task_items)
        metrics = summarize_fn(task_items)
        metrics["num_plotted_episodes"] = len(plotted_items.get(task_desc, []))
        task_summaries[task_desc] = metrics

    overall = summarize_fn(all_items) if all_items else {"num_episodes": 0}
    overall["num_tasks"] = len(grouped_items)
    return {
        "overall": overall,
        "tasks": task_summaries,
    }


def resolve_default_output_dir(input_path: str, suffix: str) -> str:
    input_dir = os.path.dirname(os.path.abspath(input_path))
    input_stem = os.path.splitext(os.path.basename(input_path))[0]
    return os.path.join(input_dir, f"{input_stem}{suffix}")
