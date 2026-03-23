from __future__ import annotations

import argparse
import cProfile
import os
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
from tqdm import tqdm

from inference.backfill.common import (
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
    resolve_reference_map_by_task_desc,
    validate_output_dataset,
    write_augmented_parquet,
)
from inference.core.io_utils import load_config_namespace
from inference.core.lerobot_io import (
    clone_info_with_new_float_features,
    find_orphan_episode_files,
    load_lerobot_episode_stats_rows,
    load_lerobot_episodes,
    prepare_output_dataset,
    resolve_episode_parquet_path,
    write_json,
    write_jsonl,
)


def run_dry_run(
    episode_metas: Sequence[Dict[str, Any]],
    reference_packs: Mapping[str, Dict[str, Any]],
    target_views: Sequence[str],
) -> Dict[str, Any]:
    if len(episode_metas) == 0:
        payload = {"episodes": 0, "pairs": 0, "sample_message_items": 0, "sample_image_type": None}
        _print_block("dry_run", list(payload.items()))
        return payload

    first_chunk = list(episode_metas)
    image_dirs_by_episode = build_episode_image_dirs(episode_metas=first_chunk)

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
    sample_image_type = None
    sample_target_input_type = None
    if sample_job is not None:
        sample_target_input_type = "str"
        _, _, messages = build_lerobot_job_message(
            sample_job,
            image_dirs_by_episode=image_dirs_by_episode,
            reference_packs=reference_packs,
            target_views=target_views,
        )
        sample_message_items = len(messages[0]["content"]) if messages else 0
        for item in messages[0]["content"]:
            if item.get("type") == "image":
                sample_image_type = type(item.get("image")).__name__
                break

    total_pairs = sum(meta["num_pairs"] for meta in first_chunk)
    payload = {
        "episodes": len(first_chunk),
        "pairs": total_pairs,
        "sample_message_items": sample_message_items,
        "sample_image_type": sample_image_type,
        "sample_target_input_type": sample_target_input_type,
    }
    _print_block("dry_run", list(payload.items()))
    return payload


def _dump_profile_stats(profiler: cProfile.Profile, *, profile_output: str) -> None:
    output_dir = os.path.dirname(profile_output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    profiler.dump_stats(profile_output)


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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true", help="只校验路径/message 构造，不加载模型、不写文件")
    parser.add_argument("--limit-episodes", type=int, default=None, help="可选：仅处理前若干个 episode")
    parser.add_argument("--verify-samples", type=int, default=3, help="写回完成后随机校验的 parquet 数量")
    parser.add_argument("--wandb-project", type=str, default=os.environ.get("WANDB_PROJECT"), help="W&B project 名称；传入或设置环境变量即启用")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="W&B run name")
    parser.add_argument("--profile-output", type=str, default=None, help="可选：保存 cProfile stats 文件路径，传入即启用 profiling")
    return parser


def _run_backfill(args: argparse.Namespace) -> None:
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
        ("image_transport", "path_only"),
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
        enabled=bool(args.wandb_project or args.wandb_run_name),
        project=args.wandb_project,
        run_name=args.wandb_run_name,
        total_episodes=len(episode_metas),
        total_pairs=total_pairs,
    )
    tracker.log_start()

    pairs_completed = 0
    episodes_completed = 0
    exit_code = 0
    try:
        if args.dry_run:
            dry_run_metas = episode_metas[: max(1, min(len(episode_metas), args.episode_chunk_size))]
            run_dry_run(
                episode_metas=dry_run_metas,
                reference_packs=reference_packs,
                target_views=target_views,
            )
            tracker.log_dry_run()
            tracker.log_finish(
                status="dry_run_done",
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
            feature_names=[DELTA_FEATURE_NAME],
        )
        existing_stats_rows = load_lerobot_episode_stats_rows(args.dataset_root)
        existing_stats_by_episode = {int(row["episode_index"]): row for row in existing_stats_rows}
        output_stats_path = os.path.join(args.output_root, "meta", "episodes_stats.jsonl")
        delta_manifest_path = os.path.join(args.output_root, "meta", "progress_sparse_predictions.jsonl")
        write_jsonl(output_stats_path, [])
        write_jsonl(delta_manifest_path, [])
        manifest_rows_written = 0

        from inference.core.multi_gpu_inferencer import MultiGPUDeltaProgressInference

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
                    global_jobs.extend(build_episode_jobs(episode_meta))

                tqdm.write(
                    f"[chunk {chunk_idx}/{len(chunk_list)}] episodes={len(episode_chunk)} pairs={len(global_jobs)}"
                )
                episode_predictions = infer_dense_delta_predictions(
                    inference=inference,
                    episode_metas=episode_chunk,
                    global_jobs=global_jobs,
                    reference_packs=reference_packs,
                    target_views=target_views,
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
                chunk_output_stats_rows: List[Dict[str, Any]] = []
                chunk_delta_manifest_rows: List[Dict[str, Any]] = []

                for episode_meta in episode_chunk:
                    dense_result = dense_result_by_episode_id.get(episode_meta["episode_id"])
                    if dense_result is None:
                        dense_result = build_empty_dense_result(episode_meta)
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

                append_jsonl_rows(output_stats_path, chunk_output_stats_rows)
                append_jsonl_rows(delta_manifest_path, chunk_delta_manifest_rows)
                manifest_rows_written += len(chunk_delta_manifest_rows)

                pairs_completed += len(global_jobs)
                episodes_completed += len(episode_chunk)
                tracker.log_episode_chunk(
                    episodes_in_chunk=len(episode_chunk),
                    pairs_in_chunk=len(global_jobs),
                    pairs_completed=pairs_completed,
                    episodes_completed=episodes_completed,
                )
                tqdm.write(
                    f"[chunk {chunk_idx}/{len(chunk_list)}] wrote_episodes={len(episode_chunk)}"
                )
        finally:
            inference.close()

        write_json(os.path.join(args.output_root, "meta", "info.json"), output_info)
        validate_output_dataset(
            output_root=args.output_root,
            episode_metas=episode_metas,
            delta_feature_name=DELTA_FEATURE_NAME,
            verify_samples=args.verify_samples,
        )
        _print_block("backfill_done", [
            ("output_root", args.output_root),
            ("episodes", len(episode_metas)),
            ("manifest_rows", manifest_rows_written),
            ("verify_samples", args.verify_samples),
        ])
        tracker.log_finish(
            status="completed",
            pairs_completed=pairs_completed,
            episodes_completed=episodes_completed,
        )
    except Exception as exc:
        exit_code = 1
        tracker.log_failure(
            pairs_completed=pairs_completed,
            episodes_completed=episodes_completed,
        )
        raise
    finally:
        tracker.finish(exit_code=exit_code)


def main(args: argparse.Namespace) -> None:
    if not args.profile_output:
        _run_backfill(args)
        return

    profiler = cProfile.Profile()
    try:
        profiler.enable()
        _run_backfill(args)
    finally:
        profiler.disable()
        _dump_profile_stats(profiler, profile_output=args.profile_output)


if __name__ == "__main__":
    main(build_parser().parse_args())
