import argparse
import json
import os
import random
import sys
from functools import partial
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(CURRENT_DIR)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from tqdm import tqdm

from eval_curves_from_batch_demos import build_sparse_curve_results, infer_job_predictions
from inference_pairwise_from_demo import build_messages_from_inputs, sample_reference_demo_pack
from lerobot_io import (
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
from utils.data_formatting import compute_abs_progress_from_index_int
from utils.utils import dict_to_namespace
from video_frame_reader import decode_video_frames, load_episode_frame_cache, probe_video_info


DEFAULT_CUMULATIVE_FEATURE = "prediction.progress_cumulative"
DEFAULT_DELTA_FEATURE = "prediction.progress_delta"
REFERENCE_VIEW_TO_VIDEO_FILE = {
    "first_person_camera": "isaac_replay_state_first_person.mp4",
    "left_hand_camera": "isaac_replay_state_left_hand.mp4",
    "right_hand_camera": "isaac_replay_state_right_hand.mp4",
    "left_shoulder_camera": "isaac_replay_state_left_shoulder.mp4",
    "right_shoulder_camera": "isaac_replay_state_right_shoulder.mp4",
    "top_view_camera": "isaac_replay_state_top.mp4",
}


def _load_mapping_file(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        if path.endswith((".yaml", ".yml")):
            return yaml.safe_load(f)
        return json.load(f)


def load_reference_map(path: str) -> Dict[str, str]:
    payload = _load_mapping_file(path)

    if isinstance(payload, dict):
        if "tasks" in payload and isinstance(payload["tasks"], dict):
            payload = payload["tasks"]
        elif "references" in payload and isinstance(payload["references"], dict):
            payload = payload["references"]

    if isinstance(payload, dict):
        normalized: Dict[str, str] = {}
        for task_desc, value in payload.items():
            if isinstance(value, str):
                normalized[str(task_desc)] = value
                continue
            if isinstance(value, dict):
                reference_path = (
                    value.get("reference_demo")
                    or value.get("reference_demo_path")
                    or value.get("path")
                )
                if isinstance(reference_path, str):
                    normalized[str(task_desc)] = reference_path
                    continue
            raise ValueError(f"无法解析 reference map 条目: {task_desc} -> {value}")
        return normalized

    if isinstance(payload, list):
        normalized = {}
        for item in payload:
            if not isinstance(item, dict):
                raise ValueError(f"reference map 列表项必须是字典，当前为: {item}")
            task_desc = item.get("task") or item.get("task_desc")
            reference_path = (
                item.get("reference_demo")
                or item.get("reference_demo_path")
                or item.get("path")
            )
            if not isinstance(task_desc, str) or not isinstance(reference_path, str):
                raise ValueError(f"reference map 列表项缺少 task/path: {item}")
            normalized[task_desc] = reference_path
        return normalized

    raise ValueError(f"不支持的 reference map 格式: {type(payload)}")


def build_anchor_pairs(total_frames: int, pair_interval: int) -> Tuple[List[int], List[Tuple[int, int]], np.ndarray]:
    if total_frames <= 0:
        return [], [], np.array([], dtype=np.int64)
    if pair_interval <= 0:
        raise ValueError(f"pair_interval 必须为正整数，当前为 {pair_interval}")

    anchors = list(range(0, total_frames, pair_interval))
    if not anchors:
        anchors = [0]
    if anchors[-1] != total_frames - 1:
        anchors.append(total_frames - 1)

    pair_indices = list(zip(anchors[:-1], anchors[1:]))
    frame_indices = np.array([j for _, j in pair_indices], dtype=np.int64)
    return anchors, pair_indices, frame_indices


def chunked(seq: Sequence[Any], chunk_size: int) -> Iterable[Sequence[Any]]:
    for start_idx in range(0, len(seq), chunk_size):
        yield seq[start_idx:start_idx + chunk_size]


def _sample_reference_indices(
    total_frames: int,
    avg_frames: int,
    min_frames: int,
    max_frames: int,
    std: float,
    jitter: int,
    rng: random.Random,
) -> List[int]:
    if total_frames <= 0:
        return []

    num_ref_frames = int(round(rng.gauss(avg_frames, std)))
    num_ref_frames = max(min_frames, min(max_frames, num_ref_frames))

    if total_frames <= num_ref_frames:
        base_indices = list(range(total_frames))
    else:
        base_indices = [
            int(round(i * (total_frames - 1) / (num_ref_frames - 1)))
            for i in range(num_ref_frames)
        ]

    indices: List[int] = []
    for index in base_indices:
        offset = rng.randint(-jitter, jitter) if jitter > 0 else 0
        indices.append(max(0, min(total_frames - 1, index + offset)))

    return sorted(set(indices))


def sample_reference_video_pack(
    reference_demo_path: str,
    reference_config,
    rng: random.Random,
    ffmpeg_bin: str,
    ffprobe_bin: str,
) -> Tuple[List[Any], List[int]]:
    reference_views = list(reference_config.views)
    video_infos = {}
    for reference_view in reference_views:
        video_file_name = REFERENCE_VIEW_TO_VIDEO_FILE.get(reference_view)
        if video_file_name is None:
            raise KeyError(f"reference 视角 `{reference_view}` 没有对应的视频文件名映射")
        video_path = os.path.join(reference_demo_path, video_file_name)
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"reference 视频不存在: {video_path}")
        video_infos[reference_view] = {
            "video_path": video_path,
            "info": probe_video_info(video_path, ffprobe_bin=ffprobe_bin),
        }

    total_frames = min(item["info"]["total_frames"] for item in video_infos.values())
    sampled_indices = _sample_reference_indices(
        total_frames=total_frames,
        avg_frames=reference_config.avg_frames,
        min_frames=reference_config.frames_min,
        max_frames=reference_config.frames_max,
        std=reference_config.frames_std,
        jitter=reference_config.jitter,
        rng=rng,
    )

    reference_progress_ints = [
        compute_abs_progress_from_index_int(frame_index, total_frames)
        for frame_index in sampled_indices
    ]

    view_to_frames = {}
    for reference_view in reference_views:
        info = video_infos[reference_view]["info"]
        view_to_frames[reference_view] = decode_video_frames(
            video_path=video_infos[reference_view]["video_path"],
            frame_indices=sampled_indices,
            width=info["width"],
            height=info["height"],
            ffmpeg_bin=ffmpeg_bin,
        )

    reference_inputs: List[Any] = []
    for frame_index in sampled_indices:
        for reference_view in reference_views:
            reference_inputs.append(view_to_frames[reference_view][frame_index])

    return reference_inputs, reference_progress_ints


def build_reference_packs(
    task_descs: Sequence[str],
    reference_map: Mapping[str, str],
    reference_config,
    seed: int,
    ffmpeg_bin: str,
    ffprobe_bin: str,
) -> Dict[str, Dict[str, Any]]:
    reference_packs: Dict[str, Dict[str, Any]] = {}
    for task_offset, task_desc in enumerate(sorted(set(task_descs))):
        reference_demo_path = reference_map.get(task_desc)
        if reference_demo_path is None:
            raise KeyError(f"reference map 缺少任务 `{task_desc}`")
        if not os.path.isdir(reference_demo_path):
            raise FileNotFoundError(f"reference demo 不存在: {reference_demo_path}")

        rng = random.Random(seed + task_offset)
        if all(
            os.path.isdir(os.path.join(reference_demo_path, reference_view))
            for reference_view in reference_config.views
        ):
            reference_inputs, reference_progress_ints = sample_reference_demo_pack(
                reference_demo_path=reference_demo_path,
                reference_config=reference_config,
                rng=rng,
            )
        else:
            reference_inputs, reference_progress_ints = sample_reference_video_pack(
                reference_demo_path=reference_demo_path,
                reference_config=reference_config,
                rng=rng,
                ffmpeg_bin=ffmpeg_bin,
                ffprobe_bin=ffprobe_bin,
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
    for episode_meta in tqdm(episode_metas, desc="解码episode帧缓存"):
        if episode_meta["num_pairs"] == 0:
            continue
        anchor_indices = sorted({frame_idx for pair in episode_meta["ij_pairs"] for frame_idx in pair})
        frame_caches[episode_meta["episode_id"]] = load_episode_frame_cache(
            video_sources=episode_meta["video_sources"],
            frame_indices=anchor_indices,
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


def reconstruct_dense_sequences(
    total_frames: int,
    pair_indices: Sequence[Tuple[int, int]],
    cumulative_progress: Sequence[float],
) -> Tuple[np.ndarray, np.ndarray]:
    if total_frames <= 0:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)
    if len(pair_indices) == 0:
        zeros = np.zeros(total_frames, dtype=np.float32)
        return zeros, zeros.copy()

    anchor_frames = [int(pair_indices[0][0])] + [int(j) for _, j in pair_indices]
    anchor_progress = [0.0] + [float(value) for value in cumulative_progress]
    dense_cumulative = np.interp(
        np.arange(total_frames, dtype=np.float32),
        np.array(anchor_frames, dtype=np.float32),
        np.array(anchor_progress, dtype=np.float32),
    ).astype(np.float32)
    dense_delta = np.diff(dense_cumulative, prepend=dense_cumulative[:1]).astype(np.float32)
    dense_delta[0] = 0.0
    return dense_cumulative, dense_delta


def compute_scalar_stats(values: np.ndarray) -> Dict[str, List[float]]:
    if values.size == 0:
        return {
            "min": [0.0],
            "max": [0.0],
            "mean": [0.0],
            "std": [0.0],
            "count": [0],
        }

    return {
        "min": [float(np.min(values))],
        "max": [float(np.max(values))],
        "mean": [float(np.mean(values))],
        "std": [float(np.std(values))],
        "count": [int(values.size)],
    }


def write_augmented_parquet(
    input_path: str,
    output_path: str,
    dense_cumulative: np.ndarray,
    dense_delta: np.ndarray,
    cumulative_feature_name: str,
    delta_feature_name: str,
) -> None:
    table = pq.read_table(input_path)
    if table.num_rows != len(dense_cumulative) or table.num_rows != len(dense_delta):
        raise ValueError(
            f"parquet 行数与写回序列长度不一致: path={input_path}, rows={table.num_rows}, "
            f"cum={len(dense_cumulative)}, delta={len(dense_delta)}"
        )

    for feature_name in (cumulative_feature_name, delta_feature_name):
        if feature_name in table.column_names:
            raise ValueError(f"输出 parquet 已包含列 `{feature_name}`: {input_path}")

    table = table.append_column(
        cumulative_feature_name,
        pa.array(np.asarray(dense_cumulative, dtype=np.float32), type=pa.float32()),
    )
    table = table.append_column(
        delta_feature_name,
        pa.array(np.asarray(dense_delta, dtype=np.float32), type=pa.float32()),
    )

    metadata = update_huggingface_metadata(
        table.schema.metadata,
        [cumulative_feature_name, delta_feature_name],
    )
    table = table.replace_schema_metadata(metadata)
    ensure_parent_dir(output_path)
    pq.write_table(table, output_path, compression="snappy")


def validate_output_dataset(
    output_root: str,
    episode_metas: Sequence[Dict[str, Any]],
    cumulative_feature_name: str,
    delta_feature_name: str,
    verify_samples: int,
) -> None:
    if verify_samples <= 0:
        return

    output_info = load_lerobot_info(output_root)
    for feature_name in (cumulative_feature_name, delta_feature_name):
        if feature_name not in output_info.get("features", {}):
            raise RuntimeError(f"输出 info.json 缺少特征 `{feature_name}`")

    sample_episode_metas = list(episode_metas[:verify_samples])
    for episode_meta in sample_episode_metas:
        output_parquet_path = resolve_episode_parquet_path(
            output_root,
            output_info,
            episode_meta["episode_index"],
        )
        table = pq.read_table(output_parquet_path, columns=[cumulative_feature_name, delta_feature_name])
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
) -> None:
    if len(episode_metas) == 0:
        print("dry-run: 没有 episode 可供检查")
        return

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

    if sample_job is not None:
        _, _, messages = build_lerobot_job_message(
            sample_job,
            frame_caches=frame_caches,
            reference_packs=reference_packs,
            target_views=target_views,
        )
        content_length = len(messages[0]["content"]) if messages else 0
        print(f"dry-run: 成功构建示例 message，content items={content_length}")

    total_pairs = sum(meta["num_pairs"] for meta in first_chunk)
    print(f"dry-run: 校验 episode 数={len(first_chunk)}，总 pair 数={total_pairs}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="对 LeRobot v2.1 数据集回填 pairwise progress 推理结果")
    parser.add_argument("--base-model", type=str, default="models/Qwen-VL-2B-Instruct")
    parser.add_argument("--adapter", type=str, required=True)
    parser.add_argument("--dataset-root", type=str, required=True, help="LeRobot v2.1 数据集根目录")
    parser.add_argument("--output-root", type=str, default=None, help="输出数据集根目录；dry-run 时可不传")
    parser.add_argument("--reference-map", type=str, required=True, help="任务描述到 reference demo 路径的 JSON/YAML 映射")
    parser.add_argument("--config", type=str, required=True, help="训练/推理使用的 YAML 配置")
    parser.add_argument("--pair-interval", type=int, default=50, help="两次稀疏推理之间的帧间隔")
    parser.add_argument("--batch-size", type=int, default=8, help="每张 GPU 的子 batch 大小")
    parser.add_argument("--num-gpus", type=int, default=1, help="使用的 GPU 数量")
    parser.add_argument("--global-build-workers", type=int, default=8, help="构建 Qwen messages 的线程数")
    parser.add_argument("--message-chunk-size", type=int, default=128, help="每次送入多 GPU 推理的 message 数量")
    parser.add_argument("--episode-chunk-size", type=int, default=4, help="每次处理多少个 episode，控制解码和内存峰值")
    parser.add_argument("--ffmpeg-workers", type=int, default=6, help="单个 episode 内并行解码多少路视频")
    parser.add_argument("--ffmpeg-bin", type=str, default="ffmpeg")
    parser.add_argument("--ffprobe-bin", type=str, default="ffprobe")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cumulative-feature-name", type=str, default=DEFAULT_CUMULATIVE_FEATURE)
    parser.add_argument("--delta-feature-name", type=str, default=DEFAULT_DELTA_FEATURE)
    parser.add_argument("--dry-run", action="store_true", help="只校验路径/解码/message 构造，不加载模型、不写文件")
    parser.add_argument("--limit-episodes", type=int, default=None, help="仅与 --dry-run 配合，用于快速抽样校验")
    parser.add_argument("--copy-large-dirs", action="store_true", help="默认复用 videos/images 的符号链接，设置此项则完整拷贝")
    parser.add_argument("--no-image-cache", action="store_true", help="禁用 images/ cache，强制从视频解码")
    parser.add_argument("--verify-samples", type=int, default=3, help="写回完成后随机校验的 parquet 数量")
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    if args.limit_episodes is not None and not args.dry_run:
        raise ValueError("--limit-episodes 仅支持与 --dry-run 一起使用，避免生成不完整数据集")
    if not args.dry_run and not args.output_root:
        raise ValueError("非 dry-run 模式必须提供 --output-root")
    if args.episode_chunk_size <= 0:
        raise ValueError("--episode-chunk-size 必须为正整数")

    with open(args.config, "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)
    config = dict_to_namespace(config_dict)
    target_views = list(config.sampling.required_views)

    info, _, episode_records, view_mapping = load_lerobot_episodes(
        dataset_root=args.dataset_root,
        target_views=target_views,
    )
    orphan_paths = find_orphan_episode_files(
        dataset_root=args.dataset_root,
        valid_episode_indices=[episode_record.episode_index for episode_record in episode_records],
    )
    for orphan_path in orphan_paths:
        print(f"警告：发现未被 episodes.jsonl 声明的孤儿 parquet，输出时将跳过: {orphan_path}")

    if args.limit_episodes is not None:
        episode_records = episode_records[:args.limit_episodes]

    reference_map = load_reference_map(args.reference_map)
    reference_packs = build_reference_packs(
        task_descs=[episode_record.task_desc for episode_record in episode_records],
        reference_map=reference_map,
        reference_config=config.reference,
        seed=args.seed,
        ffmpeg_bin=args.ffmpeg_bin,
        ffprobe_bin=args.ffprobe_bin,
    )

    episode_metas: List[Dict[str, Any]] = []
    for episode_id, episode_record in enumerate(episode_records):
        _, pair_indices, frame_indices = build_anchor_pairs(
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
            "T": episode_record.length,
            "target_demo_path": f"episode_{episode_record.episode_index:06d}",
        })

    print(f"目标视角映射: {view_mapping}")
    print(f"待处理 episode 数量: {len(episode_metas)}")

    if args.dry_run:
        dry_run_metas = episode_metas[: max(1, min(len(episode_metas), args.episode_chunk_size))]
        run_dry_run(
            episode_metas=dry_run_metas,
            reference_packs=reference_packs,
            target_views=target_views,
            ffmpeg_workers=args.ffmpeg_workers,
            prefer_image_cache=not args.no_image_cache,
            ffmpeg_bin=args.ffmpeg_bin,
        )
        return

    prepare_output_dataset(
        input_root=args.dataset_root,
        output_root=args.output_root,
        link_large_dirs=not args.copy_large_dirs,
    )
    output_info = clone_info_with_new_float_features(
        info=info,
        feature_names=[args.cumulative_feature_name, args.delta_feature_name],
    )

    existing_stats_rows = load_lerobot_episode_stats_rows(args.dataset_root)
    existing_stats_by_episode = {
        int(row["episode_index"]): row for row in existing_stats_rows
    }
    output_episode_stats_rows: List[Dict[str, Any]] = []
    sparse_manifest_rows: List[Dict[str, Any]] = []

    from multi_gpu_inferencer import MultiGPUDeltaProgressInference

    inference = MultiGPUDeltaProgressInference(
        base_model_path=args.base_model,
        adapter_path=args.adapter,
        num_gpus=args.num_gpus,
    )
    try:
        for episode_chunk in tqdm(
            list(chunked(episode_metas, args.episode_chunk_size)),
            desc="处理episode chunks",
        ):
            frame_caches = decode_episode_chunk(
                episode_metas=episode_chunk,
                ffmpeg_workers=args.ffmpeg_workers,
                prefer_image_cache=not args.no_image_cache,
                ffmpeg_bin=args.ffmpeg_bin,
            )

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

            build_message_fn = partial(
                build_lerobot_job_message,
                frame_caches=frame_caches,
                reference_packs=reference_packs,
                target_views=target_views,
            )
            episode_predictions = infer_job_predictions(
                inference=inference,
                episode_metas=list(episode_chunk),
                global_jobs=global_jobs,
                build_message_fn=build_message_fn,
                batch_size=args.batch_size,
                global_build_workers=args.global_build_workers,
                message_chunk_size=args.message_chunk_size,
                desc="LeRobot backfill inference",
            )
            sparse_results = build_sparse_curve_results(
                episode_metas=list(episode_chunk),
                episode_predictions=episode_predictions,
                fill_missing_with_zero=True,
            )
            sparse_result_by_episode_id = {
                sparse_result["episode_id"]: sparse_result for sparse_result in sparse_results
            }

            for episode_meta in episode_chunk:
                sparse_result = sparse_result_by_episode_id.get(episode_meta["episode_id"])
                if sparse_result is None:
                    dense_cumulative = np.zeros(episode_meta["T"], dtype=np.float32)
                    dense_delta = np.zeros(episode_meta["T"], dtype=np.float32)
                    sparse_result = {
                        "pair_indices": [],
                        "delta_progress": [],
                        "cumulative_progress": [],
                        "missing_pair_indices": episode_meta["ij_pairs"],
                        "frame_indices": [],
                    }
                else:
                    dense_cumulative, dense_delta = reconstruct_dense_sequences(
                        total_frames=episode_meta["T"],
                        pair_indices=sparse_result["pair_indices"],
                        cumulative_progress=sparse_result["cumulative_progress"],
                    )

                output_parquet_path = resolve_episode_parquet_path(
                    args.output_root,
                    output_info,
                    episode_meta["episode_index"],
                )
                write_augmented_parquet(
                    input_path=episode_meta["parquet_path"],
                    output_path=output_parquet_path,
                    dense_cumulative=dense_cumulative,
                    dense_delta=dense_delta,
                    cumulative_feature_name=args.cumulative_feature_name,
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
                output_stats_row["stats"][args.cumulative_feature_name] = compute_scalar_stats(dense_cumulative)
                output_stats_row["stats"][args.delta_feature_name] = compute_scalar_stats(dense_delta)
                output_episode_stats_rows.append(output_stats_row)

                sparse_manifest_rows.append({
                    "episode_index": episode_meta["episode_index"],
                    "task_desc": episode_meta["task_desc"],
                    "reference_demo_path": episode_meta["reference_demo_path"],
                    "pair_indices": [list(pair) for pair in sparse_result["pair_indices"]],
                    "frame_indices": list(sparse_result["frame_indices"]),
                    "delta_progress": list(sparse_result["delta_progress"]),
                    "cumulative_progress": list(sparse_result["cumulative_progress"]),
                    "missing_pair_indices": [list(pair) for pair in sparse_result["missing_pair_indices"]],
                })
    finally:
        inference.close()

    write_json(os.path.join(args.output_root, "meta", "info.json"), output_info)
    write_jsonl(os.path.join(args.output_root, "meta", "episodes_stats.jsonl"), output_episode_stats_rows)
    write_jsonl(os.path.join(args.output_root, "meta", "progress_sparse_predictions.jsonl"), sparse_manifest_rows)
    validate_output_dataset(
        output_root=args.output_root,
        episode_metas=episode_metas,
        cumulative_feature_name=args.cumulative_feature_name,
        delta_feature_name=args.delta_feature_name,
        verify_samples=args.verify_samples,
    )
    print(f"完成，输出数据集写入: {args.output_root}")


if __name__ == "__main__":
    main(parse_args())
