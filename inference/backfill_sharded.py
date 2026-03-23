from __future__ import annotations

import argparse
import cProfile
import multiprocessing as mp
import os
import time
import traceback
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from inference.backfill_common import (
    DELTA_FEATURE_NAME,
    BackfillWandbTracker,
    _print_block,
    append_jsonl_rows,
    auto_discover_task_description_map,
    build_dense_delta_results,
    build_empty_dense_result,
    build_episode_image_dirs,
    build_episode_jobs,
    build_episode_metas,
    build_lerobot_job_message,
    build_manifest_row,
    build_output_stats_row,
    build_reference_packs,
    chunked,
    infer_dense_delta_predictions,
    load_reference_map,
    load_task_description_map,
    plan_episode_shards,
    resolve_reference_map_by_task_desc,
    validate_output_dataset,
    write_augmented_parquet,
)
from inference.demo_utils import build_messages_for_job_chunk, build_messages_from_inputs
from inference.io_utils import load_config_namespace, load_jsonl_rows
from inference.lerobot_io import (
    clone_info_with_new_float_features,
    find_orphan_episode_files,
    load_lerobot_episode_stats_rows,
    load_lerobot_episodes,
    prepare_output_dataset,
    resolve_episode_parquet_path,
    write_json,
    write_jsonl,
)
from inference.video_frame_reader import (
    load_episode_image_frame_cache,
    load_episode_video_frame_cache,
    load_image_inputs_as_objects,
)


def _dump_profile_stats(profiler: cProfile.Profile, *, profile_output: str) -> None:
    output_dir = os.path.dirname(profile_output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    profiler.dump_stats(profile_output)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="对 LeRobot v2.1 数据集回填 dense delta progress（episode-sharded 多 GPU 版）")
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
    parser.add_argument("--global-build-workers", type=int, default=8, help="单个 worker 内构建 Qwen messages 的线程数")
    parser.add_argument("--episode-chunk-size", type=int, default=1, help="单个 worker 每轮处理多少个 episode")
    parser.add_argument(
        "--input-mode",
        choices=("images", "images_cached", "video_local"),
        default="images",
        help="worker 本地读取模式",
    )
    parser.add_argument("--ffmpeg-bin", type=str, default="ffmpeg", help="video_local 模式使用的 ffmpeg 可执行文件")
    parser.add_argument("--ffmpeg-workers", type=int, default=2, help="video_local 模式单个 worker 内并行解码视角数")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true", help="只校验 shard 规划与 message 构造，不加载模型、不写文件")
    parser.add_argument("--limit-episodes", type=int, default=None, help="可选：仅处理前若干个 episode")
    parser.add_argument("--verify-samples", type=int, default=3, help="写回完成后随机校验的 parquet 数量")
    parser.add_argument("--wandb-project", type=str, default=os.environ.get("WANDB_PROJECT"), help="W&B project 名称；传入或设置环境变量即启用")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="W&B run name")
    parser.add_argument("--profile-output", type=str, default=None, help="可选：保存 cProfile stats 文件路径，传入即启用 profiling")
    return parser


def _normalize_num_gpus(requested_num_gpus: int) -> int:
    import torch

    available_gpus = torch.cuda.device_count()
    if available_gpus <= 0:
        raise RuntimeError("未检测到可用 CUDA GPU，无法运行 episode-sharded backfill")
    if requested_num_gpus <= 0:
        raise ValueError(f"--num-gpus 必须为正整数，当前为 {requested_num_gpus}")
    return min(requested_num_gpus, available_gpus)


def _preload_reference_packs_as_objects(
    reference_packs: Mapping[str, Dict[str, Any]],
    task_descs: Sequence[str],
) -> Dict[str, Dict[str, Any]]:
    preloaded_reference_packs: Dict[str, Dict[str, Any]] = {}
    for task_desc in sorted(set(task_descs)):
        reference_pack = reference_packs[task_desc]
        preloaded_reference_packs[task_desc] = {
            **reference_pack,
            "reference_inputs": load_image_inputs_as_objects(reference_pack["reference_inputs"]),
        }
    return preloaded_reference_packs


def _build_object_job_message(
    job: Dict[str, Any],
    frame_caches_by_episode: Mapping[int, Dict[str, Dict[int, Any]]],
    reference_packs: Mapping[str, Dict[str, Any]],
    target_views: Sequence[str],
) -> Tuple[int, int, List[Dict[str, Any]]]:
    frame_caches = frame_caches_by_episode[job["episode_id"]]
    reference_pack = reference_packs[job["task_desc"]]
    target_inputs_t1 = [frame_caches[target_view][job["i"]] for target_view in target_views]
    target_inputs_t2 = [frame_caches[target_view][job["j"]] for target_view in target_views]
    messages = build_messages_from_inputs(
        target_inputs_t1=target_inputs_t1,
        target_inputs_t2=target_inputs_t2,
        reference_inputs=reference_pack["reference_inputs"],
        reference_progress_ints=reference_pack["reference_progress_ints"],
        reference_view_names=reference_pack["reference_view_names"],
        target_view_names=list(target_views),
        task_desc=job["task_desc"],
    )
    return job["episode_id"], job["pair_idx"], messages


def infer_dense_delta_predictions_images_cached(
    inference,
    episode_metas: Sequence[Dict[str, Any]],
    reference_packs: Mapping[str, Dict[str, Any]],
    target_views: Sequence[str],
    batch_size: int,
    global_build_workers: int,
) -> Tuple[Dict[int, List[int | None]], Dict[str, float]]:
    episode_predictions: Dict[int, List[int | None]] = {
        meta["episode_id"]: [None] * meta["num_pairs"] for meta in episode_metas
    }
    if len(episode_metas) == 0:
        return episode_predictions, {"load_sec": 0.0, "build_sec": 0.0, "infer_sec": 0.0}

    load_start = time.perf_counter()
    frame_caches_by_episode: Dict[int, Dict[str, Dict[int, Any]]] = {}
    jobs: List[Dict[str, Any]] = []
    image_workers = max(1, min(int(global_build_workers), len(target_views)))
    for episode_meta in episode_metas:
        jobs.extend(build_episode_jobs(episode_meta))
        frame_caches_by_episode[episode_meta["episode_id"]] = load_episode_image_frame_cache(
            video_sources=episode_meta["video_sources"],
            total_frames=int(episode_meta["T"]),
            image_workers=image_workers,
        )
    load_sec = time.perf_counter() - load_start

    build_start = time.perf_counter()
    build_message_fn = lambda job: _build_object_job_message(
        job,
        frame_caches_by_episode=frame_caches_by_episode,
        reference_packs=reference_packs,
        target_views=target_views,
    )
    all_meta, all_messages = build_messages_for_job_chunk(
        jobs=jobs,
        build_message_fn=build_message_fn,
        global_build_workers=global_build_workers,
    )
    build_sec = time.perf_counter() - build_start
    if len(all_messages) == 0:
        return episode_predictions, {"load_sec": load_sec, "build_sec": build_sec, "infer_sec": 0.0}

    infer_start = time.perf_counter()
    all_predictions = inference.infer_from_messages_batch(
        all_messages,
        batch_size=batch_size,
    )
    infer_sec = time.perf_counter() - infer_start
    for (episode_id, pair_idx), pred in zip(all_meta, all_predictions):
        episode_predictions[episode_id][pair_idx] = pred
    return episode_predictions, {
        "load_sec": load_sec,
        "build_sec": build_sec,
        "infer_sec": infer_sec,
    }


def infer_dense_delta_predictions_video_local(
    inference,
    episode_metas: Sequence[Dict[str, Any]],
    reference_packs: Mapping[str, Dict[str, Any]],
    target_views: Sequence[str],
    batch_size: int,
    global_build_workers: int,
    ffmpeg_bin: str,
    ffmpeg_workers: int,
) -> Tuple[Dict[int, List[int | None]], Dict[str, float]]:
    episode_predictions: Dict[int, List[int | None]] = {
        meta["episode_id"]: [None] * meta["num_pairs"] for meta in episode_metas
    }
    if len(episode_metas) == 0:
        return episode_predictions, {"decode_sec": 0.0, "build_sec": 0.0, "infer_sec": 0.0}

    decode_start = time.perf_counter()
    frame_caches_by_episode: Dict[int, Dict[str, Dict[int, Any]]] = {}
    jobs: List[Dict[str, Any]] = []
    for episode_meta in episode_metas:
        jobs.extend(build_episode_jobs(episode_meta))
        frame_caches_by_episode[episode_meta["episode_id"]] = load_episode_video_frame_cache(
            video_sources=episode_meta["video_sources"],
            total_frames=int(episode_meta["T"]),
            ffmpeg_workers=ffmpeg_workers,
            ffmpeg_bin=ffmpeg_bin,
        )
    decode_sec = time.perf_counter() - decode_start

    build_start = time.perf_counter()
    build_message_fn = lambda job: _build_object_job_message(
        job,
        frame_caches_by_episode=frame_caches_by_episode,
        reference_packs=reference_packs,
        target_views=target_views,
    )
    all_meta, all_messages = build_messages_for_job_chunk(
        jobs=jobs,
        build_message_fn=build_message_fn,
        global_build_workers=global_build_workers,
    )
    build_sec = time.perf_counter() - build_start

    infer_start = time.perf_counter()
    all_predictions = inference.infer_from_messages_batch(
        all_messages,
        batch_size=batch_size,
    )
    infer_sec = time.perf_counter() - infer_start
    for (episode_id, pair_idx), pred in zip(all_meta, all_predictions):
        episode_predictions[episode_id][pair_idx] = pred
    return episode_predictions, {
        "decode_sec": decode_sec,
        "build_sec": build_sec,
        "infer_sec": infer_sec,
    }


def run_dry_run_sharded(
    episode_metas: Sequence[Dict[str, Any]],
    reference_packs: Mapping[str, Dict[str, Any]],
    target_views: Sequence[str],
    input_mode: str,
    ffmpeg_bin: str,
    ffmpeg_workers: int,
) -> Dict[str, Any]:
    if len(episode_metas) == 0:
        payload = {"episodes": 0, "pairs": 0, "sample_message_items": 0, "sample_image_type": None}
        _print_block("dry_run", list(payload.items()))
        return payload

    first_episode = episode_metas[0]
    if not first_episode["ij_pairs"]:
        payload = {"episodes": 1, "pairs": 0, "sample_message_items": 0, "sample_image_type": None}
        _print_block("dry_run", list(payload.items()))
        return payload

    i, j = first_episode["ij_pairs"][0]
    sample_job = {
        "episode_id": first_episode["episode_id"],
        "pair_idx": 0,
        "i": i,
        "j": j,
        "task_desc": first_episode["task_desc"],
    }

    if input_mode == "images":
        image_dirs_by_episode = build_episode_image_dirs([first_episode])
        sample_target_input_type = "str"
        _, _, messages = build_lerobot_job_message(
            sample_job,
            image_dirs_by_episode=image_dirs_by_episode,
            reference_packs=reference_packs,
            target_views=target_views,
        )
    elif input_mode == "images_cached":
        sample_target_input_type = "Image"
        frame_caches_by_episode = {
            first_episode["episode_id"]: load_episode_image_frame_cache(
                video_sources=first_episode["video_sources"],
                total_frames=int(first_episode["T"]),
                image_workers=len(target_views),
            )
        }
        cached_reference_packs = _preload_reference_packs_as_objects(
            reference_packs=reference_packs,
            task_descs=[first_episode["task_desc"]],
        )
        _, _, messages = _build_object_job_message(
            sample_job,
            frame_caches_by_episode=frame_caches_by_episode,
            reference_packs=cached_reference_packs,
            target_views=target_views,
        )
    else:
        sample_target_input_type = "Image"
        frame_caches_by_episode = {
            first_episode["episode_id"]: load_episode_video_frame_cache(
                video_sources=first_episode["video_sources"],
                total_frames=int(first_episode["T"]),
                ffmpeg_workers=ffmpeg_workers,
                ffmpeg_bin=ffmpeg_bin,
            )
        }
        _, _, messages = _build_object_job_message(
            sample_job,
            frame_caches_by_episode=frame_caches_by_episode,
            reference_packs=reference_packs,
            target_views=target_views,
        )

    sample_message_items = len(messages[0]["content"]) if messages else 0
    sample_image_type = None
    for item in messages[0]["content"]:
        if item.get("type") == "image":
            sample_image_type = type(item.get("image")).__name__
            break
    payload = {
        "episodes": len(episode_metas),
        "pairs": sum(meta["num_pairs"] for meta in episode_metas),
        "sample_message_items": sample_message_items,
        "sample_image_type": sample_image_type,
        "sample_target_input_type": sample_target_input_type,
        "input_mode": input_mode,
    }
    _print_block("dry_run", list(payload.items()))
    return payload


def _write_worker_chunk_outputs(
    *,
    episode_chunk: Sequence[Dict[str, Any]],
    dense_delta_results: Sequence[Dict[str, Any]],
    existing_stats_by_episode: Mapping[int, Dict[str, Any]],
    output_root: str,
    output_info: Mapping[str, Any],
    stats_path: str,
    manifest_path: str,
) -> int:
    dense_result_by_episode_id = {
        dense_result["episode_id"]: dense_result for dense_result in dense_delta_results
    }
    chunk_output_stats_rows: List[Dict[str, Any]] = []
    chunk_delta_manifest_rows: List[Dict[str, Any]] = []
    for episode_meta in episode_chunk:
        dense_result = dense_result_by_episode_id.get(episode_meta["episode_id"])
        if dense_result is None:
            dense_result = build_empty_dense_result(episode_meta)
        dense_delta = np.asarray(dense_result["delta_progress"], dtype=np.float32)
        output_parquet_path = resolve_episode_parquet_path(
            output_root,
            output_info,
            episode_meta["episode_index"],
        )
        write_augmented_parquet(
            input_path=episode_meta["parquet_path"],
            output_path=output_parquet_path,
            dense_delta=dense_delta,
            delta_feature_name=DELTA_FEATURE_NAME,
        )
        chunk_output_stats_rows.append(
            build_output_stats_row(
                base_stats_row=existing_stats_by_episode.get(episode_meta["episode_index"]),
                episode_index=episode_meta["episode_index"],
                dense_delta=dense_delta,
                delta_feature_name=DELTA_FEATURE_NAME,
            )
        )
        chunk_delta_manifest_rows.append(build_manifest_row(dense_result, dense_delta))

    append_jsonl_rows(stats_path, chunk_output_stats_rows)
    append_jsonl_rows(manifest_path, chunk_delta_manifest_rows)
    return len(chunk_delta_manifest_rows)


def _worker_loop(
    rank: int,
    gpu_id: int,
    episode_shard: Sequence[Dict[str, Any]],
    reference_packs: Mapping[str, Dict[str, Any]],
    existing_stats_by_episode: Mapping[int, Dict[str, Any]],
    output_root: str,
    output_info: Mapping[str, Any],
    target_views: Sequence[str],
    args_dict: Mapping[str, Any],
    result_queue,
) -> None:
    worker_start = time.perf_counter()
    stats_path = os.path.join(output_root, ".backfill_sharded", f"rank_{rank:02d}", "episodes_stats.jsonl")
    manifest_path = os.path.join(output_root, ".backfill_sharded", f"rank_{rank:02d}", "progress_sparse_predictions.jsonl")
    write_jsonl(stats_path, [])
    write_jsonl(manifest_path, [])

    try:
        from inference.inferencer import DeltaProgressInference

        inference = DeltaProgressInference(
            base_model_path=str(args_dict["base_model"]),
            adapter_path=str(args_dict["adapter"]),
            device=f"cuda:{gpu_id}",
        )
        timings = {
            "predict_sec": 0.0,
            "reference_sec": 0.0,
            "load_sec": 0.0,
            "decode_sec": 0.0,
            "build_sec": 0.0,
            "infer_sec": 0.0,
            "write_sec": 0.0,
        }
        pairs_completed = 0
        episodes_completed = 0
        manifest_rows_written = 0
        event_chunk_index = 0

        input_mode = str(args_dict["input_mode"])
        if input_mode in ("images", "images_cached"):
            shard_chunks = list(chunked(list(episode_shard), int(args_dict["episode_chunk_size"])))
        else:
            shard_chunks = [[episode_meta] for episode_meta in episode_shard]

        if input_mode == "images_cached":
            reference_start = time.perf_counter()
            cached_reference_packs = _preload_reference_packs_as_objects(
                reference_packs=reference_packs,
                task_descs=[episode_meta["task_desc"] for episode_meta in episode_shard],
            )
            timings["reference_sec"] += time.perf_counter() - reference_start
        else:
            cached_reference_packs = reference_packs

        for episode_chunk in shard_chunks:
            event_chunk_index += 1
            predict_start = time.perf_counter()
            if input_mode == "images":
                global_jobs: List[Dict[str, Any]] = []
                for episode_meta in episode_chunk:
                    global_jobs.extend(build_episode_jobs(episode_meta))
                episode_predictions = infer_dense_delta_predictions(
                    inference=inference,
                    episode_metas=episode_chunk,
                    global_jobs=global_jobs,
                    reference_packs=reference_packs,
                    target_views=target_views,
                    batch_size=int(args_dict["batch_size"]),
                    global_build_workers=int(args_dict["global_build_workers"]),
                    desc=f"worker{rank}-images",
                )
                chunk_pairs = len(global_jobs)
                predict_elapsed = time.perf_counter() - predict_start
                timings["predict_sec"] += predict_elapsed
            elif input_mode == "images_cached":
                episode_predictions, mode_timings = infer_dense_delta_predictions_images_cached(
                    inference=inference,
                    episode_metas=episode_chunk,
                    reference_packs=cached_reference_packs,
                    target_views=target_views,
                    batch_size=int(args_dict["batch_size"]),
                    global_build_workers=int(args_dict["global_build_workers"]),
                )
                chunk_pairs = sum(meta["num_pairs"] for meta in episode_chunk)
                timings["load_sec"] += mode_timings["load_sec"]
                timings["build_sec"] += mode_timings["build_sec"]
                timings["infer_sec"] += mode_timings["infer_sec"]
                timings["predict_sec"] += time.perf_counter() - predict_start
            else:
                episode_predictions, mode_timings = infer_dense_delta_predictions_video_local(
                    inference=inference,
                    episode_metas=episode_chunk,
                    reference_packs=reference_packs,
                    target_views=target_views,
                    batch_size=int(args_dict["batch_size"]),
                    global_build_workers=int(args_dict["global_build_workers"]),
                    ffmpeg_bin=str(args_dict["ffmpeg_bin"]),
                    ffmpeg_workers=int(args_dict["ffmpeg_workers"]),
                )
                chunk_pairs = sum(meta["num_pairs"] for meta in episode_chunk)
                timings["decode_sec"] += mode_timings["decode_sec"]
                timings["build_sec"] += mode_timings["build_sec"]
                timings["infer_sec"] += mode_timings["infer_sec"]
                timings["predict_sec"] += time.perf_counter() - predict_start

            dense_delta_results = build_dense_delta_results(
                episode_metas=episode_chunk,
                episode_predictions=episode_predictions,
                fill_missing_with_zero=True,
            )

            write_start = time.perf_counter()
            manifest_rows_added = _write_worker_chunk_outputs(
                episode_chunk=episode_chunk,
                dense_delta_results=dense_delta_results,
                existing_stats_by_episode=existing_stats_by_episode,
                output_root=output_root,
                output_info=output_info,
                stats_path=stats_path,
                manifest_path=manifest_path,
            )
            timings["write_sec"] += time.perf_counter() - write_start

            pairs_completed += chunk_pairs
            episodes_completed += len(episode_chunk)
            manifest_rows_written += manifest_rows_added
            result_queue.put((
                "progress",
                rank,
                {
                    "chunk_index": event_chunk_index,
                    "episodes_completed": episodes_completed,
                    "pairs_completed": pairs_completed,
                    "episodes_in_chunk": len(episode_chunk),
                    "pairs_in_chunk": chunk_pairs,
                    "manifest_rows_added": manifest_rows_added,
                    "timings": dict(timings),
                },
            ))

        inference = None
        result_queue.put((
            "done",
            rank,
            {
                "episodes_completed": episodes_completed,
                "pairs_completed": pairs_completed,
                "manifest_rows_written": manifest_rows_written,
                "stats_path": stats_path,
                "manifest_path": manifest_path,
                "timings": {
                    **timings,
                    "total_sec": time.perf_counter() - worker_start,
                },
            },
        ))
    except Exception:
        result_queue.put(("error", rank, traceback.format_exc()))


def _merge_shard_outputs(
    *,
    shard_stats_paths: Sequence[str],
    shard_manifest_paths: Sequence[str],
    output_stats_path: str,
    output_manifest_path: str,
) -> int:
    stats_rows: List[Dict[str, Any]] = []
    for shard_stats_path in shard_stats_paths:
        stats_rows.extend(load_jsonl_rows(shard_stats_path))
    stats_rows.sort(key=lambda row: int(row["episode_index"]))
    write_jsonl(output_stats_path, stats_rows)

    manifest_rows: List[Dict[str, Any]] = []
    for shard_manifest_path in shard_manifest_paths:
        manifest_rows.extend(load_jsonl_rows(shard_manifest_path))
    manifest_rows.sort(key=lambda row: int(row["episode_index"]))
    write_jsonl(output_manifest_path, manifest_rows)
    return len(manifest_rows)


def _run_backfill_sharded(args: argparse.Namespace) -> None:
    if not args.dry_run and not args.output_root:
        raise ValueError("非 dry-run 模式必须提供 --output-root")
    if args.episode_chunk_size <= 0:
        raise ValueError("--episode-chunk-size 必须为正整数")

    actual_num_gpus = args.num_gpus if args.dry_run else _normalize_num_gpus(args.num_gpus)
    config = load_config_namespace(args.config)
    target_views = list(config.sampling.required_views)
    include_video_metadata = args.input_mode == "video_local"

    info, _, episode_records, view_mapping = load_lerobot_episodes(
        dataset_root=args.dataset_root,
        target_views=target_views,
        include_video_metadata=include_video_metadata,
    )
    if args.limit_episodes is not None:
        episode_records = episode_records[:args.limit_episodes]
    if args.dry_run or args.limit_episodes is not None:
        orphan_paths: List[str] = []
    else:
        orphan_paths = find_orphan_episode_files(
            dataset_root=args.dataset_root,
            valid_episode_indices=[episode_record.episode_index for episode_record in episode_records],
        )

    raw_reference_map = load_reference_map(args.reference_map)
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
    episode_metas = build_episode_metas(
        episode_records=episode_records,
        reference_packs=reference_packs,
        pair_interval=args.pair_interval,
    )
    shard_plan = plan_episode_shards(episode_metas, actual_num_gpus)
    active_shards = [shard for shard in shard_plan if shard]

    num_tasks = len({episode_record.task_desc for episode_record in episode_records})
    total_pairs = sum(meta["num_pairs"] for meta in episode_metas)
    shard_summary = ", ".join(
        f"gpu{idx}=episodes:{len(shard)},pairs:{sum(meta['num_pairs'] for meta in shard)}"
        for idx, shard in enumerate(active_shards)
    )
    _print_block("backfill_sharded_start", [
        ("dataset_root", args.dataset_root),
        ("output_root", args.output_root if args.output_root else "<dry-run>"),
        ("dry_run", args.dry_run),
        ("episodes", len(episode_metas)),
        ("pairs_total", total_pairs),
        ("tasks", num_tasks),
        ("views", ",".join(target_views)),
        ("pair_interval", args.pair_interval),
        ("batch_size", args.batch_size),
        ("num_gpus", actual_num_gpus),
        ("episode_chunk_size", args.episode_chunk_size),
        ("input_mode", args.input_mode),
        ("dispatch_mode", "episode_sharded"),
        ("shards", shard_summary),
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
        enabled=bool(args.wandb_project or args.wandb_run_name),
        project=args.wandb_project,
        run_name=args.wandb_run_name,
        group=None,
        tags=None,
        total_episodes=len(episode_metas),
        total_pairs=total_pairs,
        total_tasks=num_tasks,
        args=args,
        image_transport=args.input_mode,
        dispatch_mode="episode_sharded",
    )
    tracker.log_start(
        target_views=target_views,
        view_mapping=view_mapping,
        source_task_map_path=source_task_map_path,
        reference_tasks=len(reference_packs),
        orphan_parquet_count=len(orphan_paths),
        image_transport=args.input_mode,
        dispatch_mode="episode_sharded",
    )

    if args.dry_run:
        dry_run_stats = run_dry_run_sharded(
            episode_metas=episode_metas[: max(1, min(len(episode_metas), args.episode_chunk_size))],
            reference_packs=reference_packs,
            target_views=target_views,
            input_mode=args.input_mode,
            ffmpeg_bin=args.ffmpeg_bin,
            ffmpeg_workers=args.ffmpeg_workers,
        )
        tracker.log_dry_run(dry_run_stats=dry_run_stats)
        tracker.log_finish(
            status="dry_run_done",
            manifest_rows=0,
            pairs_completed=0,
            episodes_completed=0,
        )
        tracker.finish(exit_code=0)
        return

    prepare_output_dataset(
        input_root=args.dataset_root,
        output_root=args.output_root,
    )
    output_info = clone_info_with_new_float_features(
        info=info,
        feature_names=[DELTA_FEATURE_NAME],
    )
    existing_stats_rows = load_lerobot_episode_stats_rows(args.dataset_root)
    existing_stats_by_episode = {int(row["episode_index"]): row for row in existing_stats_rows}

    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    workers = []
    args_dict = {
        "base_model": args.base_model,
        "adapter": args.adapter,
        "batch_size": args.batch_size,
        "global_build_workers": args.global_build_workers,
        "episode_chunk_size": args.episode_chunk_size,
        "input_mode": args.input_mode,
        "ffmpeg_bin": args.ffmpeg_bin,
        "ffmpeg_workers": args.ffmpeg_workers,
    }
    for rank, episode_shard in enumerate(active_shards):
        worker = ctx.Process(
            target=_worker_loop,
            args=(
                rank,
                rank,
                episode_shard,
                reference_packs,
                existing_stats_by_episode,
                args.output_root,
                output_info,
                target_views,
                args_dict,
                result_queue,
            ),
            daemon=True,
        )
        worker.start()
        workers.append(worker)

    manifest_rows_written = 0
    pairs_completed = 0
    episodes_completed = 0
    completed_workers = 0
    total_progress_events = sum(
        len(list(chunked(list(shard), args.episode_chunk_size)))
        if args.input_mode in ("images", "images_cached")
        else len(shard)
        for shard in active_shards
    )
    progress_event_index = 0
    shard_stats_paths: List[str] = []
    shard_manifest_paths: List[str] = []
    worker_summaries: Dict[int, Dict[str, Any]] = {}
    exit_code = 0

    try:
        while completed_workers < len(workers):
            event_type, rank, payload = result_queue.get()
            if event_type == "error":
                exit_code = 1
                raise RuntimeError(f"[worker {rank}] 失败:\n{payload}")
            if event_type == "progress":
                progress_event_index += 1
                pairs_completed += int(payload["pairs_in_chunk"])
                episodes_completed += int(payload["episodes_in_chunk"])
                manifest_rows_written += int(payload["manifest_rows_added"])
                tracker.log_episode_chunk(
                    episode_chunk_index=progress_event_index,
                    total_episode_chunks=total_progress_events,
                    episodes_in_chunk=int(payload["episodes_in_chunk"]),
                    pairs_in_chunk=int(payload["pairs_in_chunk"]),
                    missing_pairs_in_chunk=0,
                    manifest_rows=manifest_rows_written,
                    pairs_completed=pairs_completed,
                    episodes_completed=episodes_completed,
                )
                tqdm_message = (
                    f"[worker {rank}] episodes={payload['episodes_completed']} "
                    f"pairs={payload['pairs_completed']} "
                    f"timing={payload['timings']}"
                )
                print(tqdm_message)
                continue
            if event_type == "done":
                completed_workers += 1
                worker_summaries[rank] = payload
                shard_stats_paths.append(payload["stats_path"])
                shard_manifest_paths.append(payload["manifest_path"])

        for worker in workers:
            worker.join()
            if worker.exitcode not in (0, None):
                exit_code = 1
                raise RuntimeError(f"worker 退出异常: pid={worker.pid}, exitcode={worker.exitcode}")

        output_stats_path = os.path.join(args.output_root, "meta", "episodes_stats.jsonl")
        output_manifest_path = os.path.join(args.output_root, "meta", "progress_sparse_predictions.jsonl")
        manifest_rows_written = _merge_shard_outputs(
            shard_stats_paths=shard_stats_paths,
            shard_manifest_paths=shard_manifest_paths,
            output_stats_path=output_stats_path,
            output_manifest_path=output_manifest_path,
        )
        write_json(os.path.join(args.output_root, "meta", "info.json"), output_info)
        validate_output_dataset(
            output_root=args.output_root,
            episode_metas=episode_metas,
            delta_feature_name=DELTA_FEATURE_NAME,
            verify_samples=args.verify_samples,
        )

        timing_rows = []
        aggregate_total_sec = 0.0
        for rank in sorted(worker_summaries):
            summary = worker_summaries[rank]
            timing_rows.append((f"worker_{rank}", summary["timings"]))
            aggregate_total_sec += float(summary["timings"]["total_sec"])
        _print_block("worker_timing_summary", timing_rows)
        _print_block("backfill_sharded_done", [
            ("output_root", args.output_root),
            ("episodes", len(episode_metas)),
            ("manifest_rows", manifest_rows_written),
            ("verify_samples", args.verify_samples),
            ("aggregate_worker_total_sec", round(aggregate_total_sec, 3)),
        ])
        tracker.log_finish(
            status="completed",
            manifest_rows=manifest_rows_written,
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
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
        raise
    finally:
        for worker in workers:
            worker.join(timeout=5)
        tracker.finish(exit_code=exit_code)


def main(args: argparse.Namespace) -> None:
    if not args.profile_output:
        _run_backfill_sharded(args)
        return

    profiler = cProfile.Profile()
    try:
        profiler.enable()
        _run_backfill_sharded(args)
    finally:
        profiler.disable()
        _dump_profile_stats(profiler, profile_output=args.profile_output)


if __name__ == "__main__":
    main(build_parser().parse_args())
