import os
import re
import json
from typing import Dict, List, Optional, Set, Tuple


def parse_sampling_rate(dataset_spec: str) -> Tuple[str, float]:
    """Parse dataset spec with optional % sampling suffix.

    Examples:
        /path/to/train          -> ("/path/to/train", 1.0)
        /path/to/train%30       -> ("/path/to/train", 0.3)
    """
    spec = dataset_spec.strip()
    match = re.search(r"%(\d+)$", spec)
    if not match:
        return spec, 1.0

    rate = int(match.group(1))
    if rate <= 0 or rate > 100:
        raise ValueError(f"Invalid sampling rate in dataset spec `{dataset_spec}`. Use 1-100.")
    return re.sub(r"%(\d+)$", "", spec), rate / 100.0


def _parse_task_filter(path_spec: str) -> Tuple[str, Optional[Set[str]]]:
    """Parse optional task filter suffix.

    Syntax:
        /path/to/dataset_root::TaskA+TaskB
    """
    if "::" not in path_spec:
        return path_spec, None

    base_path, task_expr = path_spec.split("::", 1)
    task_names = {item.strip() for item in task_expr.split("+") if item.strip()}
    if not task_names:
        raise ValueError(f"Invalid task filter in dataset spec `{path_spec}`.")
    return base_path.strip(), task_names


def _list_annotation_files(directory: str, task_filter: Optional[Set[str]] = None) -> List[str]:
    files = []
    for name in sorted(os.listdir(directory)):
        path = os.path.join(directory, name)
        if os.path.isfile(path) and name.lower().endswith((".json", ".jsonl")):
            if task_filter is not None:
                task_name = os.path.splitext(name)[0]
                if task_name not in task_filter:
                    continue
            files.append(path)
    return files


def _load_data_path_from_metadata(dataset_root: str, split_name: str) -> Optional[str]:
    metadata_path = os.path.join(dataset_root, f"{split_name}_metadata.json")
    if not os.path.isfile(metadata_path):
        return None

    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    except Exception:
        return None

    data_path = metadata.get("data_path")
    if isinstance(data_path, str) and data_path.strip():
        return os.path.abspath(data_path)
    return None


def _configs_from_dataset_root(
    dataset_root: str,
    sampling_rate: float,
    task_filter: Optional[Set[str]] = None,
) -> List[Dict[str, str]]:
    train_dir = os.path.join(dataset_root, "train")
    if not os.path.isdir(train_dir):
        raise ValueError(
            f"Dataset root `{dataset_root}` must contain a `train` directory."
        )

    annotation_files = _list_annotation_files(train_dir, task_filter=task_filter)
    if not annotation_files:
        raise ValueError(f"No annotation files found in `{train_dir}`.")

    data_path = _load_data_path_from_metadata(dataset_root, "train")
    if data_path is None:
        raise ValueError(
            f"Missing required `data_path` in `{os.path.join(dataset_root, 'train_metadata.json')}`."
        )
    return [
        {
            "annotation_path": os.path.abspath(annotation_path),
            "data_path": os.path.abspath(data_path),
            "sampling_rate": sampling_rate,
        }
        for annotation_path in annotation_files
    ]


def _configs_from_split_dir(
    split_dir: str,
    sampling_rate: float,
    task_filter: Optional[Set[str]] = None,
) -> List[Dict[str, str]]:
    annotation_files = _list_annotation_files(split_dir, task_filter=task_filter)
    if not annotation_files:
        raise ValueError(f"No annotation files found in `{split_dir}`.")

    # Expected layout: <dataset_output_root>/train|eval/*.json
    split_name = os.path.basename(split_dir)
    dataset_root = os.path.dirname(split_dir)
    data_path = _load_data_path_from_metadata(dataset_root, split_name)
    if data_path is None:
        raise ValueError(
            f"Missing required `data_path` in `{os.path.join(dataset_root, f'{split_name}_metadata.json')}`."
        )
    return [
        {
            "annotation_path": os.path.abspath(annotation_path),
            "data_path": os.path.abspath(data_path),
            "sampling_rate": sampling_rate,
        }
        for annotation_path in annotation_files
    ]


def _config_from_annotation_file(annotation_path: str, sampling_rate: float) -> Dict[str, str]:
    ext = os.path.splitext(annotation_path)[1].lower()
    if ext not in (".json", ".jsonl"):
        raise ValueError(f"Unsupported annotation file type: `{annotation_path}`")

    ann_abs = os.path.abspath(annotation_path)
    parent = os.path.dirname(ann_abs)
    parent_name = os.path.basename(parent)

    # Expected layout: <dataset_output_root>/train|eval/<task>.json
    if parent_name in ("train", "eval"):
        dataset_root = os.path.dirname(parent)
        data_path = _load_data_path_from_metadata(dataset_root, parent_name)
        if data_path is None:
            raise ValueError(
                f"Missing required `data_path` in `{os.path.join(dataset_root, f'{parent_name}_metadata.json')}`."
            )
    else:
        data_path = os.path.dirname(parent)

    return {
        "annotation_path": ann_abs,
        "data_path": os.path.abspath(data_path),
        "sampling_rate": sampling_rate,
    }


def _resolve_dataset_spec(path_spec: str, sampling_rate: float) -> List[Dict[str, str]]:
    base_path, task_filter = _parse_task_filter(path_spec)
    path = os.path.abspath(base_path)
    if not os.path.exists(path):
        raise ValueError(f"Dataset path does not exist: `{base_path}`")

    if os.path.isdir(path):
        base = os.path.basename(path)
        if base in ("train", "eval"):
            return _configs_from_split_dir(path, sampling_rate, task_filter=task_filter)
        if os.path.isdir(os.path.join(path, "train")):
            return _configs_from_dataset_root(path, sampling_rate, task_filter=task_filter)
        return _configs_from_split_dir(path, sampling_rate, task_filter=task_filter)

    if task_filter is not None:
        raise ValueError("Task filter `::TaskA+TaskB` only supports directory dataset specs.")
    return [_config_from_annotation_file(path, sampling_rate)]


def data_list(dataset_names: List[str]) -> List[Dict[str, str]]:
    """Build dataset configs from path-like dataset specs.

    `dataset_names` now expects paths, not registered aliases.
    """
    config_list: List[Dict[str, str]] = []
    for dataset_name in dataset_names:
        if not dataset_name or not dataset_name.strip():
            continue
        path_spec, sampling_rate = parse_sampling_rate(dataset_name)
        config_list.extend(_resolve_dataset_spec(path_spec, sampling_rate))

    if not config_list:
        raise ValueError("No valid dataset specs found in `dataset_use`.")
    return config_list
