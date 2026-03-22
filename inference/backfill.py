from __future__ import annotations

import argparse
import cProfile
import os
import pstats
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
from inference.io_utils import load_config_namespace, load_json_or_yaml
from inference.lerobot_io import (
    clone_info_with_new_float_features,
    ensure_parent_dir,
    find_orphan_episode_files,
    load_lerobot_episode_stats_rows,
    load_lerobot_episodes,
    load_lerobot_info,
    prepare_output_dataset,
    resolve_episode_parquet_path,
    update_huggingface_metadata,
    write_json,
    write_jsonl,
)
from inference.video_frame_reader import load_episode_frame_cache

DEFAULT_DELTA_FEATURE = "prediction.progress_delta"
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
            "dataset_root": args.dataset_root,
            "output_root": args.output_root,
            "pair_interval": args.pair_interval,
            "batch_size": args.batch_size,
            "num_gpus": args.num_gpus,
            "episode_chunk_size": args.episode_chunk_size,
            "ffmpeg_workers": args.ffmpeg_workers,
            "global_build_workers": args.global_build_workers,
            "seed": args.seed,
            "dry_run": args.dry_run,
            "delta_feature_name": args.delta_feature_name,
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
            float(pairs_completed) / float(self.total_pairs)
            if self.total_pairs > 0 else 1.0
        )
        payload["progress/episodes_ratio"] = (
            float(episodes_completed) / float(self.total_episodes)
            if self.total_episodes > 0 else 1.0
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
        image_cache_enabled: bool,
    ) -> None:
        self._log({
            "status/event": "start",
            "meta/target_views": ",".join(target_views),
            "meta/view_mapping": dict(view_mapping),
            "meta/source_task_map_path": source_task_map_path or "<not-found>",
            "meta/reference_tasks": int(reference_tasks),
            "meta/orphan_parquet_count": int(orphan_parquet_count),
            "meta/image_cache_enabled": bool(image_cache_enabled),
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


def build_dense_window_pairs(total_frames: int, pair_interval: int) -> Tuple[List[Tuple[int, int]], np.ndarray]:
    if total_frames <= 0:
        return [], np.array([], dtype=np.int64)
    if pair_interval <= 0:
        raise ValueError(f"pair_interval 必须为正整数，当前为 {pair_interval}")
    if total_frames <= pair_interval:
        return [], np.array([], dtype=np.int64)

    pair_indices = [(i, i + pair_interval) for i in range(0, total_frames - pair_interval)]
    frame_indices = np.array([i for i, _ in pair_indices], dtype=np.int64)
    return pair_indices, frame_indices


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


def decode_episode_chunk(
    episode_metas: Sequence[Dict[str, Any]],
    ffmpeg_workers: int,
    prefer_image_cache: bool,
    ffmpeg_bin: str,
) -> Dict[int, Dict[str, Dict[int, Any]]]:
    frame_caches: Dict[int, Dict[str, Dict[int, Any]]] = {}
    for episode_meta in tqdm(episode_metas, desc="解码 episode 帧"):
        total_frames = int(episode_meta["T"])
        if total_frames <= 0:
            continue
        frame_caches[episode_meta["episode_id"]] = load_episode_frame_cache(
            video_sources=episode_meta["video_sources"],
            frame_indices=list(range(total_frames)),
            ffmpeg_workers=ffmpeg_workers,
            prefer_image_cache=prefer_image_cache,
            ffmpeg_bin=ffmpeg_bin,
        )
    return frame_caches


def build_lerobot_job_message(
    job: Dict[str, Any],
    frame_caches: Mapping[int, Dict[str, Dict[int, Any]]],
    reference_packs: Mapping[str, Dict[str, Any]],
    target_views: Sequence[str],
) -> Tuple[int, int, List[Dict[str, Any]]]:
    episode_id = job["episode_id"]
    task_desc = job["task_desc"]
    frame_cache = frame_caches[episode_id]
    reference_pack = reference_packs[task_desc]

    target_inputs_t1 = [frame_cache[target_view][job["i"]] for target_view in target_views]
    target_inputs_t2 = [frame_cache[target_view][job["j"]] for target_view in target_views]
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
    ffmpeg_workers: int,
    prefer_image_cache: bool,
    ffmpeg_bin: str,
    batch_size: int,
    global_build_workers: int,
    desc: str,
) -> Dict[int, List[Optional[int]]]:
    episode_predictions: Dict[int, List[Optional[int]]] = {
        meta["episode_id"]: [None] * meta["num_pairs"] for meta in episode_metas
    }
    if len(global_jobs) == 0:
        return episode_predictions

    frame_caches = decode_episode_chunk(
        episode_metas=episode_metas,
        ffmpeg_workers=ffmpeg_workers,
        prefer_image_cache=prefer_image_cache,
        ffmpeg_bin=ffmpeg_bin,
    )
    build_message_fn = partial(
        build_lerobot_job_message,
        frame_caches=frame_caches,
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
        desc=desc,
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
            "delta_progress": dense_delta.tolist(),
            "missing_pair_indices": missing_pair_indices,
        })
    return dense_results


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


def run_dry_run(
    episode_metas: Sequence[Dict[str, Any]],
    reference_packs: Mapping[str, Dict[str, Any]],
    target_views: Sequence[str],
    ffmpeg_workers: int,
    prefer_image_cache: bool,
    ffmpeg_bin: str,
) -> Dict[str, Any]:
    if len(episode_metas) == 0:
        payload = {"episodes": 0, "pairs": 0, "sample_message_items": 0}
        _print_block("dry_run", list(payload.items()))
        return payload

    first_chunk = list(episode_metas)
    frame_caches = decode_episode_chunk(
        episode_metas=first_chunk,
        ffmpeg_workers=ffmpeg_workers,
        prefer_image_cache=prefer_image_cache,
        ffmpeg_bin=ffmpeg_bin,
    )

    sample_job = None
    for episode_meta in first_chunk:
        if episode_meta["ij_pairs"]:
            i, j = episode_meta["ij_pairs"][0]
            sample_job = {
                "episode_id": episode_meta["episode_id"],
                "pair_idx": 0,
                "i": i,
                "j": j,
                "task_desc": episode_meta["task_desc"],
            }
            break

    sample_message_items = 0
    if sample_job is not None:
        _, _, messages = build_lerobot_job_message(
            sample_job,
            frame_caches=frame_caches,
            reference_packs=reference_packs,
            target_views=target_views,
        )
        sample_message_items = len(messages[0]["content"]) if messages else 0

    total_pairs = sum(meta["num_pairs"] for meta in first_chunk)
    payload = {
        "episodes": len(first_chunk),
        "pairs": total_pairs,
        "sample_message_items": sample_message_items,
    }
    _print_block("dry_run", [
        ("episodes", len(first_chunk)),
        ("pairs", total_pairs),
        ("sample_message_items", sample_message_items),
    ])
    return payload


def _emit_profile_report(
    profiler: cProfile.Profile,
    *,
    profile_output: Optional[str],
    sort_key: str,
    top_k: int,
) -> None:
    if profile_output:
        output_dir = os.path.dirname(profile_output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        profiler.dump_stats(profile_output)

    stats = pstats.Stats(profiler)
    stats.calc_callees()
    filtered_rows: List[Tuple[str, int, int, float, float]] = []
    current_file = os.path.abspath(__file__)
    for (file_name, line_no, func_name), stat in stats.stats.items():
        if os.path.abspath(file_name) != current_file:
            continue
        primitive_calls, total_calls, total_time, cumulative_time, _ = stat
        filtered_rows.append((
            func_name,
            int(primitive_calls),
            int(total_calls),
            float(total_time),
            float(cumulative_time),
        ))

    sort_key = sort_key.lower()
    if sort_key not in {"cumulative", "time"}:
        raise ValueError(f"不支持的 --profile-sort: {sort_key}")

    sort_index = 4 if sort_key == "cumulative" else 3
    filtered_rows.sort(key=lambda row: row[sort_index], reverse=True)
    filtered_rows = filtered_rows[: max(1, int(top_k))]

    _print_block("profile", [
        ("scope", "backfill.py"),
        ("sort", sort_key),
        ("top_k", len(filtered_rows)),
        ("stats_file", profile_output or "<not-saved>"),
    ])
    for func_name, primitive_calls, total_calls, total_time, cumulative_time in filtered_rows:
        print(
            "  "
            f"{func_name}: primitive_calls={primitive_calls}, total_calls={total_calls}, "
            f"self_time={total_time:.3f}s, cumulative_time={cumulative_time:.3f}s"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="对 LeRobot v2.1 数据集回填 dense delta progress")
    parser.add_argument("--base-model", type=str, default="models/Qwen-VL-2B-Instruct")
    parser.add_argument("--adapter", type=str, required=True)
    parser.add_argument("--dataset-root", type=str, required=True, help="LeRobot v2.1 数据集根目录")
    parser.add_argument("--output-root", type=str, default=None, help="输出数据集根目录；dry-run 时可不传")
    parser.add_argument(
        "--reference-map",
        type=str,
        required=True,
        help="reference demo 路径映射；只支持 {task_name: reference_demo_path}",
    )
    parser.add_argument(
        "--source-task-map",
        type=str,
        default=None,
        help="可选：源数据集的 task_name -> task_description 映射文件；不传则尝试自动发现",
    )
    parser.add_argument("--config", type=str, default="dataset/configs/build_config_15tasks.yaml")
    parser.add_argument("--pair-interval", type=int, default=50, help="dense pair 时间间隔")
    parser.add_argument("--batch-size", type=int, default=8, help="每张 GPU 的子 batch 大小")
    parser.add_argument("--num-gpus", type=int, default=1, help="使用的 GPU 数量")
    parser.add_argument("--global-build-workers", type=int, default=8, help="构建 Qwen messages 的线程数")
    parser.add_argument("--episode-chunk-size", type=int, default=1, help="每次处理多少个 episode，并在 chunk 内整体构造 messages")
    parser.add_argument("--ffmpeg-workers", type=int, default=6, help="单个 episode 内并行解码多少路视频")
    parser.add_argument("--ffmpeg-bin", type=str, default="ffmpeg")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--delta-feature-name", type=str, default=DEFAULT_DELTA_FEATURE)
    parser.add_argument("--dry-run", action="store_true", help="只校验路径/解码/message 构造，不加载模型、不写文件")
    parser.add_argument("--limit-episodes", type=int, default=None, help="仅与 --dry-run 配合，用于快速抽样校验")
    parser.add_argument("--no-image-cache", action="store_true", help="禁用 images/ cache，强制从视频解码")
    parser.add_argument("--verify-samples", type=int, default=3, help="写回完成后随机校验的 parquet 数量")
    parser.add_argument("--wandb", action="store_true", help="启用 Weights & Biases 进度跟踪")
    parser.add_argument("--wandb-project", type=str, default=os.environ.get("WANDB_PROJECT"), help="W&B project 名称")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="W&B run name")
    parser.add_argument("--wandb-group", type=str, default=None, help="W&B group")
    parser.add_argument("--wandb-tags", nargs="*", default=None, help="W&B tags")
    parser.add_argument("--profile", action="store_true", help="启用最基础的 cProfile profiling")
    parser.add_argument("--profile-output", type=str, default=None, help="可选：保存 .prof stats 文件路径")
    parser.add_argument("--profile-sort", type=str, default="cumulative", help="profile 排序方式：cumulative 或 time")
    parser.add_argument("--profile-top-k", type=int, default=20, help="终端打印多少条 backfill.py 内部 profiling 结果")
    return parser


def _run_backfill(args: argparse.Namespace) -> None:
    if args.limit_episodes is not None and not args.dry_run:
        raise ValueError("--limit-episodes 仅支持与 --dry-run 一起使用，避免生成不完整数据集")
    if not args.dry_run and not args.output_root:
        raise ValueError("非 dry-run 模式必须提供 --output-root")
    if args.episode_chunk_size <= 0:
        raise ValueError("--episode-chunk-size 必须为正整数")

    config = load_config_namespace(args.config)
    target_views = list(config.sampling.required_views)

    info, _, episode_records, view_mapping = load_lerobot_episodes(
        dataset_root=args.dataset_root,
        target_views=target_views,
    )
    orphan_paths = find_orphan_episode_files(
        dataset_root=args.dataset_root,
        valid_episode_indices=[episode_record.episode_index for episode_record in episode_records],
    )
    if args.limit_episodes is not None:
        episode_records = episode_records[:args.limit_episodes]

    raw_reference_map = load_reference_map(args.reference_map)
    source_task_map = None
    source_task_map_path = None
    if args.source_task_map:
        source_task_map_path = args.source_task_map
        source_task_map = load_task_description_map(args.source_task_map)
    else:
        source_task_map_path, source_task_map = auto_discover_task_description_map(list(raw_reference_map.values()))

    reference_map = resolve_reference_map_by_task_desc(
        raw_reference_map=raw_reference_map,
        target_task_descs=[episode_record.task_desc for episode_record in episode_records],
        source_task_map=source_task_map,
    )
    reference_packs = build_reference_packs(
        task_descs=[episode_record.task_desc for episode_record in episode_records],
        reference_map=reference_map,
        reference_config=config.reference,
        seed=args.seed,
    )

    episode_metas: List[Dict[str, Any]] = []
    for episode_id, episode_record in enumerate(episode_records):
        pair_indices, frame_indices = build_dense_window_pairs(
            total_frames=episode_record.length,
            pair_interval=args.pair_interval,
        )
        episode_metas.append({
            "episode_id": episode_id,
            "episode_index": episode_record.episode_index,
            "task_desc": episode_record.task_desc,
            "task_index": episode_record.task_index,
            "parquet_path": episode_record.parquet_path,
            "video_sources": episode_record.video_sources,
            "reference_demo_path": reference_packs[episode_record.task_desc]["reference_demo_path"],
            "frame_indices": frame_indices,
            "ij_pairs": pair_indices,
            "num_pairs": len(pair_indices),
            "pair_offset": args.pair_interval,
            "T": episode_record.length,
            "target_demo_path": f"episode_{episode_record.episode_index:06d}",
        })

    num_tasks = len({episode_record.task_desc for episode_record in episode_records})
    total_pairs = sum(meta["num_pairs"] for meta in episode_metas)
    _print_block("backfill_start", [
        ("dataset_root", args.dataset_root),
        ("output_root", args.output_root if args.output_root else "<dry-run>"),
        ("dry_run", args.dry_run),
        ("episodes", len(episode_metas)),
        ("pairs_total", total_pairs),
        ("tasks", num_tasks),
        ("views", ",".join(target_views)),
        ("pair_interval", args.pair_interval),
        ("batch_size", args.batch_size),
        ("num_gpus", args.num_gpus),
        ("episode_chunk_size", args.episode_chunk_size),
        ("frame_loading", "full_episode"),
        ("image_cache", not args.no_image_cache),
        ("view_mapping", view_mapping),
    ])
    _print_block("references", [
        ("source_task_map", source_task_map_path if source_task_map_path else "<not-found>"),
        ("reference_tasks", len(reference_packs)),
    ])
    if orphan_paths:
        preview = ", ".join(orphan_paths[:3])
        if len(orphan_paths) > 3:
            preview += ", ..."
        _print_block("warnings", [("orphan_parquet_count", len(orphan_paths)), ("examples", preview)])

    tracker = BackfillWandbTracker(
        enabled=args.wandb,
        project=args.wandb_project,
        run_name=args.wandb_run_name,
        group=args.wandb_group,
        tags=args.wandb_tags,
        total_episodes=len(episode_metas),
        total_pairs=total_pairs,
        total_tasks=num_tasks,
        args=args,
    )
    tracker.log_start(
        target_views=target_views,
        view_mapping=view_mapping,
        source_task_map_path=source_task_map_path,
        reference_tasks=len(reference_packs),
        orphan_parquet_count=len(orphan_paths),
        image_cache_enabled=not args.no_image_cache,
    )

    pairs_completed = 0
    episodes_completed = 0
    exit_code = 0
    try:
        if args.dry_run:
            dry_run_metas = episode_metas[: max(1, min(len(episode_metas), args.episode_chunk_size))]
            dry_run_stats = run_dry_run(
                episode_metas=dry_run_metas,
                reference_packs=reference_packs,
                target_views=target_views,
                ffmpeg_workers=args.ffmpeg_workers,
                prefer_image_cache=not args.no_image_cache,
                ffmpeg_bin=args.ffmpeg_bin,
            )
            tracker.log_dry_run(dry_run_stats=dry_run_stats)
            tracker.log_finish(
                status="dry_run_done",
                manifest_rows=0,
                pairs_completed=0,
                episodes_completed=0,
            )
            return

        prepare_output_dataset(
            input_root=args.dataset_root,
            output_root=args.output_root,
        )
        output_info = clone_info_with_new_float_features(
            info=info,
            feature_names=[args.delta_feature_name],
        )

        existing_stats_rows = load_lerobot_episode_stats_rows(args.dataset_root)
        existing_stats_by_episode = {int(row["episode_index"]): row for row in existing_stats_rows}
        output_episode_stats_rows: List[Dict[str, Any]] = []
        delta_manifest_rows: List[Dict[str, Any]] = []

        from inference.multi_gpu_inferencer import MultiGPUDeltaProgressInference

        inference = MultiGPUDeltaProgressInference(
            base_model_path=args.base_model,
            adapter_path=args.adapter,
            num_gpus=args.num_gpus,
        )

        chunk_list = list(chunked(episode_metas, args.episode_chunk_size))
        try:
            for chunk_idx, episode_chunk in enumerate(tqdm(chunk_list, desc="回填 chunks"), start=1):
                global_jobs: List[Dict[str, Any]] = []
                for episode_meta in episode_chunk:
                    for pair_idx, (i, j) in enumerate(episode_meta["ij_pairs"]):
                        global_jobs.append({
                            "episode_id": episode_meta["episode_id"],
                            "pair_idx": pair_idx,
                            "i": i,
                            "j": j,
                            "task_desc": episode_meta["task_desc"],
                        })

                tqdm.write(
                    f"[chunk {chunk_idx}/{len(chunk_list)}] episodes={len(episode_chunk)} pairs={len(global_jobs)}"
                )
                episode_predictions = infer_dense_delta_predictions(
                    inference=inference,
                    episode_metas=episode_chunk,
                    global_jobs=global_jobs,
                    reference_packs=reference_packs,
                    target_views=target_views,
                    ffmpeg_workers=args.ffmpeg_workers,
                    prefer_image_cache=not args.no_image_cache,
                    ffmpeg_bin=args.ffmpeg_bin,
                    batch_size=args.batch_size,
                    global_build_workers=args.global_build_workers,
                    desc="LeRobot dense delta inference",
                )
                dense_delta_results = build_dense_delta_results(
                    episode_metas=episode_chunk,
                    episode_predictions=episode_predictions,
                    fill_missing_with_zero=True,
                )
                dense_result_by_episode_id = {
                    dense_result["episode_id"]: dense_result for dense_result in dense_delta_results
                }

                for episode_meta in episode_chunk:
                    dense_result = dense_result_by_episode_id.get(episode_meta["episode_id"])
                    if dense_result is None:
                        dense_delta = np.zeros(episode_meta["T"], dtype=np.float32)
                        dense_result = {
                            "episode_index": episode_meta["episode_index"],
                            "task_desc": episode_meta["task_desc"],
                            "reference_demo_path": episode_meta["reference_demo_path"],
                            "total_frames": episode_meta["T"],
                            "pair_offset": episode_meta["pair_offset"],
                            "pair_indices": [],
                            "delta_progress": dense_delta.tolist(),
                            "missing_pair_indices": episode_meta["ij_pairs"],
                            "frame_indices": list(range(episode_meta["T"])),
                        }
                    else:
                        dense_delta = np.asarray(dense_result["delta_progress"], dtype=np.float32)

                    output_parquet_path = resolve_episode_parquet_path(
                        args.output_root,
                        output_info,
                        episode_meta["episode_index"],
                    )
                    write_augmented_parquet(
                        input_path=episode_meta["parquet_path"],
                        output_path=output_parquet_path,
                        dense_delta=dense_delta,
                        delta_feature_name=args.delta_feature_name,
                    )

                    base_stats_row = existing_stats_by_episode.get(episode_meta["episode_index"], {
                        "episode_index": episode_meta["episode_index"],
                        "stats": {},
                    })
                    output_stats_row = {
                        "episode_index": base_stats_row["episode_index"],
                        "stats": dict(base_stats_row.get("stats", {})),
                    }
                    output_stats_row["stats"][args.delta_feature_name] = compute_scalar_stats(dense_delta)
                    output_episode_stats_rows.append(output_stats_row)

                    delta_manifest_rows.append({
                        "episode_index": dense_result["episode_index"],
                        "task_desc": dense_result["task_desc"],
                        "reference_demo_path": dense_result["reference_demo_path"],
                        "total_frames": dense_result["total_frames"],
                        "pair_offset": dense_result["pair_offset"],
                        "pair_indices": [list(pair) for pair in dense_result["pair_indices"]],
                        "frame_indices": list(dense_result["frame_indices"]),
                        "delta_progress": list(dense_result["delta_progress"]),
                        "missing_pair_indices": [list(pair) for pair in dense_result["missing_pair_indices"]],
                    })

                pairs_completed += len(global_jobs)
                episodes_completed += len(episode_chunk)
                missing_pairs_in_chunk = sum(
                    len(dense_result["missing_pair_indices"]) for dense_result in dense_delta_results
                )
                tracker.log_episode_chunk(
                    episode_chunk_index=chunk_idx,
                    total_episode_chunks=len(chunk_list),
                    episodes_in_chunk=len(episode_chunk),
                    pairs_in_chunk=len(global_jobs),
                    missing_pairs_in_chunk=missing_pairs_in_chunk,
                    manifest_rows=len(delta_manifest_rows),
                    pairs_completed=pairs_completed,
                    episodes_completed=episodes_completed,
                )
                tqdm.write(
                    f"[chunk {chunk_idx}/{len(chunk_list)}] wrote_episodes={len(episode_chunk)}"
                )
        finally:
            inference.close()

        write_json(os.path.join(args.output_root, "meta", "info.json"), output_info)
        write_jsonl(os.path.join(args.output_root, "meta", "episodes_stats.jsonl"), output_episode_stats_rows)
        write_jsonl(os.path.join(args.output_root, "meta", "progress_sparse_predictions.jsonl"), delta_manifest_rows)
        validate_output_dataset(
            output_root=args.output_root,
            episode_metas=episode_metas,
            delta_feature_name=args.delta_feature_name,
            verify_samples=args.verify_samples,
        )

        _print_block("backfill_done", [
            ("output_root", args.output_root),
            ("episodes", len(episode_metas)),
            ("manifest_rows", len(delta_manifest_rows)),
            ("verify_samples", args.verify_samples),
        ])
        tracker.log_finish(
            status="completed",
            manifest_rows=len(delta_manifest_rows),
            pairs_completed=pairs_completed,
            episodes_completed=episodes_completed,
        )
    except Exception as exc:
        exit_code = 1
        tracker.log_failure(
            error_type=type(exc).__name__,
            error_message=str(exc),
            pairs_completed=pairs_completed,
            episodes_completed=episodes_completed,
        )
        raise
    finally:
        tracker.finish(exit_code=exit_code)


def main(args: argparse.Namespace) -> None:
    if not args.profile:
        _run_backfill(args)
        return

    profiler = cProfile.Profile()
    try:
        profiler.enable()
        _run_backfill(args)
    finally:
        profiler.disable()
        _emit_profile_report(
            profiler,
            profile_output=args.profile_output,
            sort_key=args.profile_sort,
            top_k=args.profile_top_k,
        )


if __name__ == "__main__":
    main(build_parser().parse_args())
