import argparse
import os
import shutil
import subprocess
import tempfile
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

from tqdm import tqdm

from inference.core.lerobot_io import (
    load_lerobot_episode_rows,
    load_lerobot_info,
    resolve_episode_image_dir,
    resolve_episode_video_path,
)


@dataclass(frozen=True)
class VideoExtractJob:
    episode_index: int
    video_key: str
    video_path: str
    image_dir: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="将 LeRobot videos/ 并行预处理为 images/ 镜像目录。"
    )
    parser.add_argument("--dataset-root", required=True, help="LeRobot 数据集根目录")
    parser.add_argument("--workers", type=int, default=8, help="并行 ffmpeg worker 数")
    parser.add_argument("--start-episode", type=int, default=0, help="起始 episode_index")
    parser.add_argument("--limit-episodes", type=int, default=None, help="最多处理多少个 episode")
    parser.add_argument("--ffmpeg-bin", default="ffmpeg", help="ffmpeg 可执行文件路径")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已存在的 images 目录")
    return parser.parse_args()


def discover_video_keys(info_features: Dict[str, Dict[str, object]]) -> List[str]:
    return sorted(
        key for key, feature in info_features.items() if feature.get("dtype") == "video"
    )


def iter_video_jobs(
    dataset_root: str,
    start_episode: int,
    limit_episodes: Optional[int],
) -> Iterable[VideoExtractJob]:
    info = load_lerobot_info(dataset_root)
    episode_rows = load_lerobot_episode_rows(dataset_root)
    video_keys = discover_video_keys(info["features"])

    selected_rows = [
        row for row in episode_rows if int(row["episode_index"]) >= int(start_episode)
    ]
    if limit_episodes is not None:
        selected_rows = selected_rows[: max(0, int(limit_episodes))]

    for row in selected_rows:
        episode_index = int(row["episode_index"])
        for video_key in video_keys:
            video_path = resolve_episode_video_path(dataset_root, info, episode_index, video_key)
            image_dir = resolve_episode_image_dir(dataset_root, video_path)
            yield VideoExtractJob(
                episode_index=episode_index,
                video_key=video_key,
                video_path=video_path,
                image_dir=image_dir,
            )


def image_dir_ready(image_dir: str) -> bool:
    return os.path.isfile(os.path.join(image_dir, "frame_000000.png"))


def extract_video_to_images(
    job: VideoExtractJob,
    ffmpeg_bin: str,
    overwrite: bool,
) -> str:
    if image_dir_ready(job.image_dir) and not overwrite:
        return "skipped"

    if os.path.isdir(job.image_dir) and not overwrite:
        raise RuntimeError(
            f"目标 images 目录已存在但看起来不完整；如需重建请加 --overwrite: {job.image_dir}"
        )

    parent_dir = os.path.dirname(job.image_dir)
    os.makedirs(parent_dir, exist_ok=True)
    temp_dir = tempfile.mkdtemp(
        prefix=os.path.basename(job.image_dir) + ".tmp.",
        dir=parent_dir,
    )
    try:
        output_pattern = os.path.join(temp_dir, "frame_%06d.png")
        command = [
            ffmpeg_bin,
            "-v",
            "error",
            "-i",
            job.video_path,
            "-start_number",
            "0",
            output_pattern,
        ]
        completed = subprocess.run(command, capture_output=True, check=False)
        if completed.returncode != 0:
            stderr = completed.stderr.decode("utf-8", errors="replace")
            raise RuntimeError(
                f"ffmpeg 拆帧失败: episode={job.episode_index}, video_key={job.video_key}, "
                f"video_path={job.video_path}\n{stderr}"
            )
        if not image_dir_ready(temp_dir):
            raise RuntimeError(
                f"ffmpeg 拆帧后缺少首帧输出: episode={job.episode_index}, video_key={job.video_key}, "
                f"video_path={job.video_path}"
            )

        if os.path.isdir(job.image_dir):
            shutil.rmtree(job.image_dir)
        os.replace(temp_dir, job.image_dir)
        temp_dir = ""
        return "written"
    finally:
        if temp_dir and os.path.isdir(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)


def main() -> None:
    args = parse_args()
    jobs = list(
        iter_video_jobs(
            dataset_root=args.dataset_root,
            start_episode=args.start_episode,
            limit_episodes=args.limit_episodes,
        )
    )
    if not jobs:
        print("[precompute_images] no_jobs")
        return

    total_episodes = len({job.episode_index for job in jobs})
    print(
        "[precompute_images] "
        f"dataset_root={args.dataset_root} "
        f"episodes={total_episodes} "
        f"videos={len(jobs)} "
        f"workers={args.workers} "
        f"overwrite={args.overwrite}"
    )

    written = 0
    skipped = 0
    max_workers = max(1, min(int(args.workers), len(jobs)))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures: Dict[Future[str], VideoExtractJob] = {
            executor.submit(
                extract_video_to_images,
                job=job,
                ffmpeg_bin=args.ffmpeg_bin,
                overwrite=args.overwrite,
            ): job
            for job in jobs
        }
        progress = tqdm(total=len(futures), desc="拆帧视频")
        try:
            for future in as_completed(futures):
                job = futures[future]
                status = future.result()
                if status == "written":
                    written += 1
                elif status == "skipped":
                    skipped += 1
                else:
                    raise RuntimeError(f"未知任务状态: {status}")
                progress.update(1)
                progress.set_postfix(written=written, skipped=skipped, episode=job.episode_index)
        finally:
            progress.close()

    print(
        "[precompute_images_done] "
        f"videos={len(jobs)} "
        f"written={written} "
        f"skipped={skipped}"
    )


if __name__ == "__main__":
    main()
