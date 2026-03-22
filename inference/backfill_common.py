from __future__ import annotations

import argparse
import json
import os
import random
import time
from functools import partial
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from inference.demo_utils import (
    build_messages_for_job_chunk,
    build_messages_from_inputs,
    sample_reference_demo_pack,
)
from inference.io_utils import load_json_or_yaml
from inference.lerobot_io import (
    ensure_parent_dir,
    load_lerobot_info,
    resolve_episode_parquet_path,
    update_huggingface_metadata,
)
from inference.video_frame_reader import load_episode_image_dirs, resolve_frame_path

DELTA_FEATURE_NAME = "prediction.progress_delta"
TASK_MAP_CANDIDATE_FILES = ("task_descriptions.json", "task_map.json")


def _print_block(title: str, rows: Sequence[Tuple[str, Any]]) -> None:
    print(f"[{title}]")
    for key, value in rows:
        print(f"  {key}: {value}")


class BackfillWandbTracker:
    def __init__(
        self,
        *,
        enabled: bool,
        project: Optional[str],
        run_name: Optional[str],
        group: Optional[str],
        tags: Optional[Sequence[str]],
        total_episodes: int,
        total_pairs: int,
        total_tasks: int,
        args: argparse.Namespace,
        image_transport: str,
        dispatch_mode: str,
    ) -> None:
        self.enabled = bool(enabled)
        self.total_episodes = int(total_episodes)
        self.total_pairs = int(total_pairs)
        self.total_tasks = int(total_tasks)
        self._start_time = time.time()
        self._wandb = None
        self._run = None

        if not self.enabled:
            return

        if not project:
            raise ValueError("启用 --wandb 时，必须通过 --wandb-project 或环境变量 WANDB_PROJECT 指定 project")

        try:
            import wandb  # type: ignore
        except ImportError as exc:
            raise RuntimeError("启用 --wandb 失败：当前环境未安装 wandb") from exc

        config = {
            "dataset_root": getattr(args, "dataset_root", None),
            "output_root": getattr(args, "output_root", None),
            "pair_interval": getattr(args, "pair_interval", None),
            "batch_size": getattr(args, "batch_size", None),
            "num_gpus": getattr(args, "num_gpus", None),
            "episode_chunk_size": getattr(args, "episode_chunk_size", None),
            "global_build_workers": getattr(args, "global_build_workers", None),
            "seed": getattr(args, "seed", None),
            "dry_run": getattr(args, "dry_run", None),
            "delta_feature_name": DELTA_FEATURE_NAME,
            "image_transport": image_transport,
            "dispatch_mode": dispatch_mode,
            "total_episodes": self.total_episodes,
            "total_pairs": self.total_pairs,
            "total_tasks": self.total_tasks,
        }
        self._wandb = wandb
        self._run = wandb.init(
            project=project,
            name=run_name,
            group=group,
            tags=list(tags or []),
            job_type="backfill",
            config=config,
        )

        run_url = None
        get_url = getattr(self._run, "get_url", None)
        if callable(get_url):
            run_url = get_url()
        if not run_url:
            run_url = getattr(self._run, "url", None)
        _print_block("wandb", [
            ("project", project),
            ("run_name", getattr(self._run, "name", run_name or "<auto>")),
            ("run_url", run_url or "<unavailable>"),
        ])

    def _progress_payload(self, *, pairs_completed: int, episodes_completed: int) -> Dict[str, Any]:
        elapsed_sec = max(time.time() - self._start_time, 0.0)
        pairs_per_sec = float(pairs_completed) / elapsed_sec if elapsed_sec > 1e-8 else 0.0
        payload: Dict[str, Any] = {
            "progress/pairs_completed": int(pairs_completed),
            "progress/pairs_total": self.total_pairs,
            "progress/episodes_completed": int(episodes_completed),
            "progress/episodes_total": self.total_episodes,
            "progress/tasks_total": self.total_tasks,
            "runtime/elapsed_sec": elapsed_sec,
            "runtime/pairs_per_sec": pairs_per_sec,
        }
        payload["progress/pairs_ratio"] = (
            float(pairs_completed) / float(self.total_pairs) if self.total_pairs > 0 else 1.0
        )
        payload["progress/episodes_ratio"] = (
            float(episodes_completed) / float(self.total_episodes) if self.total_episodes > 0 else 1.0
        )
        if pairs_per_sec > 1e-8 and pairs_completed < self.total_pairs:
            payload["runtime/eta_sec"] = float(self.total_pairs - pairs_completed) / pairs_per_sec
        return payload

    def _log(self, payload: Dict[str, Any]) -> None:
        if self._wandb is None or self._run is None:
            return
        self._wandb.log(payload)

    def log_start(
        self,
        *,
        target_views: Sequence[str],
        view_mapping: Mapping[str, str],
        source_task_map_path: Optional[str],
        reference_tasks: int,
        orphan_parquet_count: int,
        image_transport: str,
        dispatch_mode: str,
    ) -> None:
        self._log({
            "status/event": "start",
            "meta/target_views": ",".join(target_views),
            "meta/view_mapping": dict(view_mapping),
            "meta/source_task_map_path": source_task_map_path or "<not-found>",
            "meta/reference_tasks": int(reference_tasks),
            "meta/orphan_parquet_count": int(orphan_parquet_count),
            "meta/image_transport": image_transport,
            "meta/dispatch_mode": dispatch_mode,
            **self._progress_payload(pairs_completed=0, episodes_completed=0),
        })

    def log_dry_run(self, *, dry_run_stats: Mapping[str, Any]) -> None:
        self._log({
            "status/event": "dry_run",
            **{f"dry_run/{key}": value for key, value in dry_run_stats.items()},
            **self._progress_payload(pairs_completed=0, episodes_completed=0),
        })

    def log_episode_chunk(
        self,
        *,
        episode_chunk_index: int,
        total_episode_chunks: int,
        episodes_in_chunk: int,
        pairs_in_chunk: int,
        missing_pairs_in_chunk: int,
        manifest_rows: int,
        pairs_completed: int,
        episodes_completed: int,
    ) -> None:
        self._log({
            "status/event": "episode_chunk",
            "chunk/episode_chunk_index": int(episode_chunk_index),
            "chunk/episode_chunks_total": int(total_episode_chunks),
            "chunk/episodes_in_chunk": int(episodes_in_chunk),
            "chunk/pairs_in_chunk": int(pairs_in_chunk),
            "chunk/missing_pairs_in_chunk": int(missing_pairs_in_chunk),
            "output/manifest_rows": int(manifest_rows),
            **self._progress_payload(
                pairs_completed=pairs_completed,
                episodes_completed=episodes_completed,
            ),
        })

    def log_finish(
        self,
        *,
        status: str,
        manifest_rows: int,
        pairs_completed: int,
        episodes_completed: int,
    ) -> None:
        self._log({
            "status/event": status,
            "output/manifest_rows": int(manifest_rows),
            **self._progress_payload(
                pairs_completed=pairs_completed,
                episodes_completed=episodes_completed,
            ),
        })

    def log_failure(
        self,
        *,
        error_type: str,
        error_message: str,
        pairs_completed: int,
        episodes_completed: int,
    ) -> None:
        self._log({
            "status/event": "failed",
            "error/type": error_type,
            "error/message": error_message[:1000],
            **self._progress_payload(
                pairs_completed=pairs_completed,
                episodes_completed=episodes_completed,
            ),
        })

    def finish(self, exit_code: int = 0) -> None:
        if self._wandb is None:
            return
        self._wandb.finish(exit_code=exit_code)


def _load_mapping_file(path: str) -> Any:
    return load_json_or_yaml(path)


def load_reference_map(path: str) -> Dict[str, str]:
    payload = _load_mapping_file(path)
    if not isinstance(payload, dict):
        raise ValueError(
            "reference_map 只支持一种格式：顶层字典，"
            "形如 {\"ArrangeVegetables\": \"/abs/path/to/episode\", ...}"
        )

    normalized: Dict[str, str] = {}
    for task_name, reference_path in payload.items():
        if not isinstance(task_name, str) or not isinstance(reference_path, str):
            raise ValueError(
                "reference_map 的每一项都必须是字符串到字符串，"
                f"当前为: {task_name} -> {reference_path}"
            )
        normalized[task_name] = reference_path
    return normalized


def load_task_description_map(path: str) -> Dict[str, str]:
    payload = _load_mapping_file(path)
    if not isinstance(payload, dict):
        raise ValueError(f"任务映射文件必须是 JSON/YAML 字典: {path}")

    normalized: Dict[str, str] = {}
    for task_name, task_desc in payload.items():
        if not isinstance(task_name, str) or not isinstance(task_desc, str):
            raise ValueError(f"任务映射项必须是字符串到字符串: {task_name} -> {task_desc}")
        normalized[task_name] = task_desc
    return normalized


def _iter_parent_dirs(path: str) -> Iterable[str]:
    current = os.path.abspath(path)
    if os.path.isfile(current):
        current = os.path.dirname(current)

    while True:
        yield current
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent


def auto_discover_task_description_map(reference_paths: Sequence[str]) -> Tuple[Optional[str], Optional[Dict[str, str]]]:
    seen_candidates = set()
    for reference_path in reference_paths:
        for parent_dir in _iter_parent_dirs(reference_path):
            for candidate_name in TASK_MAP_CANDIDATE_FILES:
                candidate_path = os.path.join(parent_dir, candidate_name)
                if candidate_path in seen_candidates:
                    continue
                seen_candidates.add(candidate_path)
                if not os.path.exists(candidate_path):
                    continue
                try:
                    task_map = load_task_description_map(candidate_path)
                except Exception:
                    continue
                if len(task_map) > 0:
                    return candidate_path, task_map

    return None, None


def resolve_reference_map_by_task_desc(
    raw_reference_map: Mapping[str, str],
    target_task_descs: Sequence[str],
    source_task_map: Optional[Mapping[str, str]] = None,
) -> Dict[str, str]:
    target_task_desc_set = set(target_task_descs)
    resolved_reference_map: Dict[str, str] = {}
    if not source_task_map:
        raise ValueError(
            "reference_map 现在只支持 `task_name -> reference_demo_path` 格式，"
            "因此必须提供或自动发现 source task map。"
        )

    for task_name, reference_path in raw_reference_map.items():
        if task_name not in source_task_map:
            raise KeyError(f"reference_map 中的 task_name `{task_name}` 不在 source task map 里。")
        resolved_task_desc = source_task_map[task_name]
        if resolved_task_desc not in target_task_desc_set:
            continue
        if resolved_task_desc in resolved_reference_map and resolved_reference_map[resolved_task_desc] != reference_path:
            raise ValueError(
                f"reference map 对任务 `{resolved_task_desc}` 解析出多个路径: "
                f"{resolved_reference_map[resolved_task_desc]} vs {reference_path}"
            )
        resolved_reference_map[resolved_task_desc] = reference_path

    missing_task_descs = sorted(target_task_desc_set - set(resolved_reference_map.keys()))
    if missing_task_descs:
        available_keys = sorted(raw_reference_map.keys())
        alias_pairs = [
            f"{task_name} -> {task_desc}"
            for task_name, task_desc in sorted(source_task_map.items())
            if task_desc in missing_task_descs or task_desc in target_task_desc_set
        ]
        alias_hint = ""
        if alias_pairs:
            alias_hint = "；可用 task_name 映射示例: " + ", ".join(alias_pairs[:8])
        raise KeyError(
            "reference_map 缺少以下任务的映射: "
            + ", ".join(missing_task_descs)
            + f"。当前 reference_map keys: {available_keys}"
            + alias_hint
        )
    return resolved_reference_map


def build_dense_window_pairs(total_frames: int, pair_interval: int) -> List[Tuple[int, int]]:
    if total_frames <= 0:
        return []
    if pair_interval <= 0:
        raise ValueError(f"pair_interval 必须为正整数，当前为 {pair_interval}")
    if total_frames <= pair_interval:
        return []
    return [(i, i + pair_interval) for i in range(0, total_frames - pair_interval)]


def chunked(seq: Sequence[Any], chunk_size: int) -> Iterable[Sequence[Any]]:
    for start_idx in range(0, len(seq), chunk_size):
        yield seq[start_idx:start_idx + chunk_size]


def build_reference_packs(
    task_descs: Sequence[str],
    reference_map: Mapping[str, str],
    reference_config,
    seed: int,
) -> Dict[str, Dict[str, Any]]:
    reference_packs: Dict[str, Dict[str, Any]] = {}
    for task_offset, task_desc in enumerate(sorted(set(task_descs))):
        reference_demo_path = reference_map.get(task_desc)
        if reference_demo_path is None:
            raise KeyError(f"reference map 缺少任务 `{task_desc}`")
        if not os.path.isdir(reference_demo_path):
            raise FileNotFoundError(f"reference demo 不存在: {reference_demo_path}")

        rng = random.Random(seed + task_offset)
        missing_reference_views = [
            reference_view
            for reference_view in reference_config.views
            if not os.path.isdir(os.path.join(reference_demo_path, reference_view))
        ]
        if missing_reference_views:
            raise FileNotFoundError(
                "reference demo 必须是图片切分版目录结构，缺少以下视角目录: "
                + ", ".join(missing_reference_views)
                + f"；reference_demo_path={reference_demo_path}"
            )

        reference_inputs, reference_progress_ints = sample_reference_demo_pack(
            reference_demo_path=reference_demo_path,
            reference_config=reference_config,
            rng=rng,
        )
        if len(reference_inputs) == 0:
            raise RuntimeError(f"reference demo 采样为空: task={task_desc}, path={reference_demo_path}")

        reference_packs[task_desc] = {
            "reference_demo_path": reference_demo_path,
            "reference_inputs": reference_inputs,
            "reference_progress_ints": reference_progress_ints,
            "reference_view_names": list(reference_config.views),
        }
    return reference_packs


def build_episode_metas(
    episode_records: Sequence[Any],
    reference_packs: Mapping[str, Dict[str, Any]],
    pair_interval: int,
) -> List[Dict[str, Any]]:
    episode_metas: List[Dict[str, Any]] = []
    for episode_id, episode_record in enumerate(episode_records):
        pair_indices = build_dense_window_pairs(
            total_frames=episode_record.length,
            pair_interval=pair_interval,
        )
        episode_metas.append({
            "episode_id": episode_id,
            "episode_index": episode_record.episode_index,
            "task_desc": episode_record.task_desc,
            "task_index": episode_record.task_index,
            "parquet_path": episode_record.parquet_path,
            "video_sources": episode_record.video_sources,
            "reference_demo_path": reference_packs[episode_record.task_desc]["reference_demo_path"],
            "ij_pairs": pair_indices,
            "num_pairs": len(pair_indices),
            "pair_offset": pair_interval,
            "T": episode_record.length,
        })
    return episode_metas


def build_episode_jobs(episode_meta: Mapping[str, Any]) -> List[Dict[str, Any]]:
    return [
        {
            "episode_id": episode_meta["episode_id"],
            "pair_idx": pair_idx,
            "i": i,
            "j": j,
            "task_desc": episode_meta["task_desc"],
        }
        for pair_idx, (i, j) in enumerate(episode_meta["ij_pairs"])
    ]


def plan_episode_shards(
    episode_metas: Sequence[Dict[str, Any]],
    num_shards: int,
) -> List[List[Dict[str, Any]]]:
    if num_shards <= 0:
        raise ValueError(f"num_shards 必须为正整数，当前为 {num_shards}")
    shards: List[List[Dict[str, Any]]] = [[] for _ in range(num_shards)]
    shard_loads = [0 for _ in range(num_shards)]
    sorted_episode_metas = sorted(
        episode_metas,
        key=lambda meta: (int(meta["num_pairs"]), int(meta["T"])),
        reverse=True,
    )
    for episode_meta in sorted_episode_metas:
        shard_idx = min(range(num_shards), key=lambda idx: (shard_loads[idx], len(shards[idx]), idx))
        shards[shard_idx].append(episode_meta)
        shard_loads[shard_idx] += int(episode_meta["num_pairs"])
    return shards


def build_episode_image_dirs(
    episode_metas: Sequence[Dict[str, Any]],
) -> Dict[int, Dict[str, str]]:
    image_dirs_by_episode: Dict[int, Dict[str, str]] = {}
    for episode_meta in tqdm(episode_metas, desc="解析 episode 图片目录"):
        image_dirs_by_episode[episode_meta["episode_id"]] = load_episode_image_dirs(
            video_sources=episode_meta["video_sources"],
        )
    return image_dirs_by_episode


def build_lerobot_job_message(
    job: Dict[str, Any],
    image_dirs_by_episode: Mapping[int, Dict[str, str]],
    reference_packs: Mapping[str, Dict[str, Any]],
    target_views: Sequence[str],
) -> Tuple[int, int, List[Dict[str, Any]]]:
    episode_id = job["episode_id"]
    task_desc = job["task_desc"]
    image_dirs = image_dirs_by_episode[episode_id]
    reference_pack = reference_packs[task_desc]

    target_inputs_t1 = [resolve_frame_path(image_dirs[target_view], job["i"]) for target_view in target_views]
    target_inputs_t2 = [resolve_frame_path(image_dirs[target_view], job["j"]) for target_view in target_views]
    messages = build_messages_from_inputs(
        target_inputs_t1=target_inputs_t1,
        target_inputs_t2=target_inputs_t2,
        reference_inputs=reference_pack["reference_inputs"],
        reference_progress_ints=reference_pack["reference_progress_ints"],
        reference_view_names=reference_pack["reference_view_names"],
        target_view_names=list(target_views),
        task_desc=task_desc,
    )
    return episode_id, job["pair_idx"], messages


def infer_dense_delta_predictions(
    inference,
    episode_metas: Sequence[Dict[str, Any]],
    global_jobs: Sequence[Dict[str, Any]],
    reference_packs: Mapping[str, Dict[str, Any]],
    target_views: Sequence[str],
    batch_size: int,
    global_build_workers: int,
    desc: str,
) -> Dict[int, List[Optional[int]]]:
    episode_predictions: Dict[int, List[Optional[int]]] = {
        meta["episode_id"]: [None] * meta["num_pairs"] for meta in episode_metas
    }
    if len(global_jobs) == 0:
        return episode_predictions

    image_dirs_by_episode = build_episode_image_dirs(episode_metas=episode_metas)
    build_message_fn = partial(
        build_lerobot_job_message,
        image_dirs_by_episode=image_dirs_by_episode,
        reference_packs=reference_packs,
        target_views=target_views,
    )
    all_meta, all_messages = build_messages_for_job_chunk(
        jobs=global_jobs,
        build_message_fn=build_message_fn,
        global_build_workers=global_build_workers,
    )
    if len(all_messages) == 0:
        return episode_predictions

    all_predictions = inference.infer_from_messages_batch(
        all_messages,
        batch_size=batch_size,
    )
    for (episode_id, pair_idx), pred in zip(all_meta, all_predictions):
        episode_predictions[episode_id][pair_idx] = pred
    return episode_predictions


def build_dense_delta_column(
    total_frames: int,
    pair_indices: Sequence[Tuple[int, int]],
    delta_progress: Sequence[float],
) -> np.ndarray:
    dense_delta = np.zeros(total_frames, dtype=np.float32)
    last_filled_index: Optional[int] = None
    last_filled_value = 0.0
    for (start_frame, _), delta_value in zip(pair_indices, delta_progress):
        start_frame = int(start_frame)
        last_filled_index = start_frame
        last_filled_value = float(delta_value)
        dense_delta[start_frame] = last_filled_value
    if last_filled_index is not None and last_filled_index + 1 < total_frames:
        dense_delta[last_filled_index + 1:] = last_filled_value
    return dense_delta


def build_dense_delta_results(
    episode_metas: Sequence[Dict[str, Any]],
    episode_predictions: Mapping[int, Sequence[Optional[int]]],
    fill_missing_with_zero: bool = True,
) -> List[Dict[str, Any]]:
    dense_results: List[Dict[str, Any]] = []
    for episode_meta in episode_metas:
        episode_id = episode_meta["episode_id"]
        preds = list(episode_predictions[episode_id])
        pair_indices_all = [tuple(int(v) for v in pair) for pair in episode_meta["ij_pairs"]]
        delta_values: List[float] = []
        pair_indices_valid: List[Tuple[int, int]] = []
        missing_pair_indices: List[Tuple[int, int]] = []

        for idx, pred in enumerate(preds):
            pair = pair_indices_all[idx]
            if pred is None:
                missing_pair_indices.append(pair)
                if not fill_missing_with_zero:
                    continue
                pred = 0
            pair_indices_valid.append(pair)
            delta_values.append(float(pred))

        dense_delta = build_dense_delta_column(
            total_frames=episode_meta["T"],
            pair_indices=pair_indices_valid,
            delta_progress=delta_values,
        )
        dense_results.append({
            "episode_id": episode_id,
            "episode_index": episode_meta["episode_index"],
            "task_desc": episode_meta["task_desc"],
            "reference_demo_path": episode_meta["reference_demo_path"],
            "total_frames": episode_meta["T"],
            "pair_offset": episode_meta["pair_offset"],
            "pair_indices": pair_indices_valid,
            "frame_indices": list(range(episode_meta["T"])),
            "delta_progress": dense_delta,
            "missing_pair_indices": missing_pair_indices,
        })
    return dense_results


def build_empty_dense_result(episode_meta: Mapping[str, Any]) -> Dict[str, Any]:
    dense_delta = np.zeros(int(episode_meta["T"]), dtype=np.float32)
    return {
        "episode_index": episode_meta["episode_index"],
        "task_desc": episode_meta["task_desc"],
        "reference_demo_path": episode_meta["reference_demo_path"],
        "total_frames": episode_meta["T"],
        "pair_offset": episode_meta["pair_offset"],
        "pair_indices": [],
        "delta_progress": dense_delta,
        "missing_pair_indices": list(episode_meta["ij_pairs"]),
        "frame_indices": list(range(int(episode_meta["T"]))),
    }


def compute_scalar_stats(values: np.ndarray) -> Dict[str, List[float]]:
    valid_values = values[np.isfinite(values)]
    if valid_values.size == 0:
        return {
            "min": [0.0],
            "max": [0.0],
            "mean": [0.0],
            "std": [0.0],
            "count": [0],
        }
    return {
        "min": [float(np.min(valid_values))],
        "max": [float(np.max(valid_values))],
        "mean": [float(np.mean(valid_values))],
        "std": [float(np.std(valid_values))],
        "count": [int(valid_values.size)],
    }


def write_augmented_parquet(
    input_path: str,
    output_path: str,
    dense_delta: np.ndarray,
    delta_feature_name: str,
) -> None:
    table = pq.read_table(input_path)
    if table.num_rows != len(dense_delta):
        raise ValueError(
            f"parquet 行数与写回序列长度不一致: path={input_path}, rows={table.num_rows}, "
            f"delta={len(dense_delta)}"
        )
    if delta_feature_name in table.column_names:
        raise ValueError(f"输出 parquet 已包含列 `{delta_feature_name}`: {input_path}")

    table = table.append_column(
        delta_feature_name,
        pa.array(np.asarray(dense_delta, dtype=np.float32), type=pa.float32()),
    )
    metadata = update_huggingface_metadata(table.schema.metadata, [delta_feature_name])
    table = table.replace_schema_metadata(metadata)
    ensure_parent_dir(output_path)
    pq.write_table(table, output_path, compression="snappy")


def build_output_stats_row(
    base_stats_row: Optional[Mapping[str, Any]],
    episode_index: int,
    dense_delta: np.ndarray,
    delta_feature_name: str,
) -> Dict[str, Any]:
    base_row = dict(base_stats_row or {"episode_index": episode_index, "stats": {}})
    output_stats_row = {
        "episode_index": int(base_row["episode_index"]),
        "stats": dict(base_row.get("stats", {})),
    }
    output_stats_row["stats"][delta_feature_name] = compute_scalar_stats(dense_delta)
    return output_stats_row


def build_manifest_row(dense_result: Mapping[str, Any], dense_delta: np.ndarray) -> Dict[str, Any]:
    return {
        "episode_index": dense_result["episode_index"],
        "task_desc": dense_result["task_desc"],
        "reference_demo_path": dense_result["reference_demo_path"],
        "total_frames": dense_result["total_frames"],
        "pair_offset": dense_result["pair_offset"],
        "pair_indices": [list(pair) for pair in dense_result["pair_indices"]],
        "frame_indices": list(dense_result["frame_indices"]),
        "delta_progress": dense_delta.tolist(),
        "missing_pair_indices": [list(pair) for pair in dense_result["missing_pair_indices"]],
    }


def append_jsonl_rows(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    ensure_parent_dir(path)
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def validate_output_dataset(
    output_root: str,
    episode_metas: Sequence[Dict[str, Any]],
    delta_feature_name: str,
    verify_samples: int,
) -> None:
    if verify_samples <= 0:
        return
    output_info = load_lerobot_info(output_root)
    if delta_feature_name not in output_info.get("features", {}):
        raise RuntimeError(f"输出 info.json 缺少特征 `{delta_feature_name}`")

    for episode_meta in list(episode_metas[:verify_samples]):
        output_parquet_path = resolve_episode_parquet_path(
            output_root,
            output_info,
            episode_meta["episode_index"],
        )
        table = pq.read_table(output_parquet_path, columns=[delta_feature_name])
        if table.num_rows != episode_meta["T"]:
            raise RuntimeError(
                f"输出 parquet 行数校验失败: path={output_parquet_path}, "
                f"rows={table.num_rows}, expected={episode_meta['T']}"
            )
