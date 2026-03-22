import copy
import json
import os
import re
import shutil
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


DEFAULT_TARGET_VIEW_TO_VIDEO_KEY: Dict[str, str] = {
    "first_person_camera": "observation.images.first_person",
    "left_hand_camera": "observation.images.left_hand",
    "right_hand_camera": "observation.images.right_hand",
    "left_shoulder_camera": "observation.images.left_shoulder",
    "right_shoulder_camera": "observation.images.right_shoulder",
    "top_view_camera": "observation.images.top",
}


@dataclass
class LeRobotVideoSource:
    target_view: str
    video_key: str
    image_dir: Optional[str]
    video_path: Optional[str] = None
    height: Optional[int] = None
    width: Optional[int] = None


@dataclass
class LeRobotEpisodeRecord:
    episode_index: int
    task_desc: str
    task_index: Optional[int]
    length: int
    parquet_path: str
    video_sources: Dict[str, LeRobotVideoSource]


def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def write_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_lerobot_info(dataset_root: str) -> Dict[str, Any]:
    return _read_json(os.path.join(dataset_root, "meta", "info.json"))


def load_lerobot_tasks(dataset_root: str) -> Dict[int, str]:
    tasks_path = os.path.join(dataset_root, "meta", "tasks.jsonl")
    tasks_map: Dict[int, str] = {}
    for row in _read_jsonl(tasks_path):
        tasks_map[int(row["task_index"])] = row["task"]
    return tasks_map


def load_lerobot_episode_rows(dataset_root: str) -> List[Dict[str, Any]]:
    return _read_jsonl(os.path.join(dataset_root, "meta", "episodes.jsonl"))


def load_lerobot_episode_stats_rows(dataset_root: str) -> List[Dict[str, Any]]:
    return _read_jsonl(os.path.join(dataset_root, "meta", "episodes_stats.jsonl"))


def resolve_target_view_to_video_key(
    target_views: Sequence[str],
    info_features: Mapping[str, Dict[str, Any]],
    explicit_map: Optional[Mapping[str, str]] = None,
) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    feature_keys = {
        key for key, feature in info_features.items() if feature.get("dtype") == "video"
    }

    for target_view in target_views:
        video_key = None
        if explicit_map is not None:
            video_key = explicit_map.get(target_view)
        if video_key is None:
            video_key = DEFAULT_TARGET_VIEW_TO_VIDEO_KEY.get(target_view)
        if video_key is None:
            raise KeyError(f"找不到 target view `{target_view}` 对应的 LeRobot video key")
        if video_key not in feature_keys:
            raise KeyError(f"LeRobot 数据集缺少视频特征 `{video_key}`，无法映射 `{target_view}`")
        mapping[target_view] = video_key

    return mapping


def resolve_episode_chunk(info: Mapping[str, Any], episode_index: int) -> int:
    chunk_size = int(info.get("chunks_size", 1000))
    if chunk_size <= 0:
        raise ValueError(f"chunks_size 非法: {chunk_size}")
    return episode_index // chunk_size


def resolve_episode_parquet_path(
    dataset_root: str,
    info: Mapping[str, Any],
    episode_index: int,
) -> str:
    rel_path = info["data_path"].format(
        episode_chunk=resolve_episode_chunk(info, episode_index),
        episode_index=episode_index,
    )
    return os.path.join(dataset_root, rel_path)


def resolve_episode_video_path(
    dataset_root: str,
    info: Mapping[str, Any],
    episode_index: int,
    video_key: str,
) -> str:
    rel_path = info["video_path"].format(
        episode_chunk=resolve_episode_chunk(info, episode_index),
        episode_index=episode_index,
        video_key=video_key,
    )
    return os.path.join(dataset_root, rel_path)


def resolve_episode_image_dir(dataset_root: str, video_path: str) -> str:
    if os.path.isabs(video_path):
        rel_video_path = os.path.relpath(video_path, dataset_root)
        if rel_video_path.startswith(".."):
            raise ValueError(f"video_path 不在 dataset_root 下: dataset_root={dataset_root}, video_path={video_path}")
    else:
        rel_video_path = video_path

    video_root, ext = os.path.splitext(rel_video_path)
    if ext.lower() != ".mp4":
        raise ValueError(f"当前只支持 mp4 视频路径镜像到 images: {video_path}")

    rel_parts = video_root.split(os.sep)
    if not rel_parts or rel_parts[0] != "videos":
        raise ValueError(f"video_path 必须位于 videos/ 下，当前为: {video_path}")
    rel_parts[0] = "images"
    return os.path.join(dataset_root, *rel_parts)


def resolve_episode_image_dir_from_video_key(
    dataset_root: str,
    info: Mapping[str, Any],
    episode_index: int,
    video_key: str,
) -> str:
    rel_video_path = info["video_path"].format(
        episode_chunk=resolve_episode_chunk(info, episode_index),
        episode_index=episode_index,
        video_key=video_key,
    )
    return resolve_episode_image_dir(dataset_root, rel_video_path)


def load_lerobot_episodes(
    dataset_root: str,
    target_views: Sequence[str],
    explicit_view_map: Optional[Mapping[str, str]] = None,
    include_video_metadata: bool = False,
) -> Tuple[Dict[str, Any], Dict[int, str], List[LeRobotEpisodeRecord], Dict[str, str]]:
    info = load_lerobot_info(dataset_root)
    tasks_map = load_lerobot_tasks(dataset_root)
    episode_rows = load_lerobot_episode_rows(dataset_root)
    view_mapping = resolve_target_view_to_video_key(
        target_views=target_views,
        info_features=info["features"],
        explicit_map=explicit_view_map,
    )
    task_to_index = {task_desc: task_index for task_index, task_desc in tasks_map.items()}

    episodes: List[LeRobotEpisodeRecord] = []
    for row in episode_rows:
        episode_index = int(row["episode_index"])
        task_list = row.get("tasks", [])
        if not task_list:
            raise ValueError(f"episode {episode_index} 缺少 task 描述")
        task_desc = task_list[0]
        task_index = task_to_index.get(task_desc)
        parquet_path = resolve_episode_parquet_path(dataset_root, info, episode_index)

        video_sources: Dict[str, LeRobotVideoSource] = {}
        for target_view, video_key in view_mapping.items():
            image_dir = resolve_episode_image_dir_from_video_key(
                dataset_root=dataset_root,
                info=info,
                episode_index=episode_index,
                video_key=video_key,
            )
            video_path = None
            height = None
            width = None
            if include_video_metadata:
                feature = info["features"][video_key]
                shape = feature.get("shape", [])
                if len(shape) < 2:
                    raise ValueError(f"视频特征 `{video_key}` 缺少有效 shape: {shape}")
                video_path = resolve_episode_video_path(dataset_root, info, episode_index, video_key)
                height = int(shape[0])
                width = int(shape[1])
            video_sources[target_view] = LeRobotVideoSource(
                target_view=target_view,
                video_key=video_key,
                image_dir=image_dir,
                video_path=video_path,
                height=height,
                width=width,
            )

        episodes.append(
            LeRobotEpisodeRecord(
                episode_index=episode_index,
                task_desc=task_desc,
                task_index=task_index,
                length=int(row["length"]),
                parquet_path=parquet_path,
                video_sources=video_sources,
            )
        )

    return info, tasks_map, episodes, view_mapping


def find_orphan_episode_files(
    dataset_root: str,
    valid_episode_indices: Sequence[int],
) -> List[str]:
    valid_episode_set = set(int(idx) for idx in valid_episode_indices)
    pattern = re.compile(r"episode_(\d+)\.parquet$")
    orphan_paths: List[str] = []

    data_root = os.path.join(dataset_root, "data")
    if not os.path.isdir(data_root):
        return orphan_paths

    for chunk_name in sorted(os.listdir(data_root)):
        chunk_path = os.path.join(data_root, chunk_name)
        if not os.path.isdir(chunk_path):
            continue
        for file_name in sorted(os.listdir(chunk_path)):
            match = pattern.match(file_name)
            if match is None:
                continue
            episode_index = int(match.group(1))
            if episode_index not in valid_episode_set:
                orphan_paths.append(os.path.join(chunk_path, file_name))

    return orphan_paths


def ensure_parent_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def prepare_output_dataset(
    input_root: str,
    output_root: str,
) -> None:
    if os.path.exists(output_root):
        raise FileExistsError(f"输出目录已存在: {output_root}")

    os.makedirs(output_root, exist_ok=False)
    os.makedirs(os.path.join(output_root, "meta"), exist_ok=True)
    os.makedirs(os.path.join(output_root, "data"), exist_ok=True)

    norm_stats_path = os.path.join(input_root, "norm_stats.json")
    if os.path.exists(norm_stats_path):
        shutil.copy2(norm_stats_path, os.path.join(output_root, "norm_stats.json"))

    for file_name in ("tasks.jsonl", "episodes.jsonl"):
        src_path = os.path.join(input_root, "meta", file_name)
        if os.path.exists(src_path):
            shutil.copy2(src_path, os.path.join(output_root, "meta", file_name))


def clone_info_with_new_float_features(
    info: Mapping[str, Any],
    feature_names: Sequence[str],
) -> Dict[str, Any]:
    info_copy = copy.deepcopy(dict(info))
    features = info_copy.setdefault("features", {})
    for feature_name in feature_names:
        features[feature_name] = {
            "dtype": "float32",
            "shape": [1],
            "names": None,
        }
    return info_copy


def update_huggingface_metadata(
    metadata: Optional[Mapping[bytes, bytes]],
    feature_names: Sequence[str],
) -> Dict[bytes, bytes]:
    metadata_dict = dict(metadata or {})
    raw_payload = metadata_dict.get(b"huggingface")
    if raw_payload is None:
        hf_payload: Dict[str, Any] = {"info": {"features": {}}}
    else:
        hf_payload = json.loads(raw_payload.decode("utf-8"))

    feature_dict = hf_payload.setdefault("info", {}).setdefault("features", {})
    for feature_name in feature_names:
        feature_dict[feature_name] = {"dtype": "float32", "_type": "Value"}

    metadata_dict[b"huggingface"] = json.dumps(hf_payload, ensure_ascii=False).encode("utf-8")
    return metadata_dict
