#!/usr/bin/env python3
"""
Extract frames from videos at specified FPS for lerobot dataset format.
Processes videos in videos/chunk-*/observation.images.*/ directory and saves frames to task/episode/view directories.
"""

import os
import json
import subprocess
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count
from typing import Tuple, Optional, Dict
import time


def extract_frames_from_video(
    video_path: str,
    output_dir: str,
    fps: float = 2.0,
    scale: float = 1.0
) -> Tuple[bool, str]:
    """
    Extract frames from a video at specified FPS.
    
    Args:
        video_path: Path to input video
        output_dir: Directory to save frames
        fps: Target frame rate (default 2.0 Hz)
        scale: Resolution scale ratio (1.0 = original size, 0.5 = half, etc.)
    
    Returns:
        (success, error_message)
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    filter_str = f"fps={fps}"
    if abs(scale - 1.0) > 1e-4:
        filter_str += f",scale=iw*{scale}:ih*{scale}"
    
    cmd = [
        "ffmpeg",
        "-i", str(video_path),
        "-vf", filter_str,
        "-y",
        "-loglevel", "error",
        str(output_dir / "frame_%06d.png")
    ]
    
    try:
        subprocess.run(cmd, capture_output=True, check=True)
        return (True, "成功")
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr.decode() if e.stderr else str(e)
        return (False, f"ffmpeg错误: {error_msg}")


def _process_single_video(args: Tuple[str, str, str, str, str, float, float]) -> Tuple[str, str, str, bool, str]:
    """
    Process a single video file.
    
    Args:
        (video_path, output_dir, task_name, episode_name, view_name, fps, scale)
    
    Returns:
        (task_name, episode_name, view_name, success, status_message)
    """
    video_path_str, output_dir_str, task_name, episode_name, view_name, fps, scale = args
    
    success, status_msg = extract_frames_from_video(video_path_str, output_dir_str, fps, scale)
    return (task_name, episode_name, view_name, success, status_msg)


def load_task_name(dataset_path: Path) -> str:
    """
    Load task name from meta/tasks.jsonl.
    Returns the first task name found, or 'default_task' if not found.
    """
    tasks_file = dataset_path / "meta" / "tasks.jsonl"
    if tasks_file.exists():
        try:
            with open(tasks_file, "r", encoding="utf-8") as f:
                first_line = f.readline().strip()
                if first_line:
                    task_data = json.loads(first_line)
                    return task_data.get("task", "default_task")
        except Exception as e:
            print(f"警告: 无法读取任务文件 {tasks_file}: {e}")
    
    return "default_task"


def extract_view_name_from_path(view_dir_name: str) -> str:
    """
    Extract clean view name from observation.images.* directory name.
    Example: 'observation.images.first_person' -> 'first_person'
    """
    if view_dir_name.startswith("observation.images."):
        return view_dir_name[len("observation.images."):]
    return view_dir_name


def process_dataset(
    dataset_path: str,
    fps: float = 2.0,
    max_workers: Optional[int] = None,
    scale: float = 1.0,
    output_root: Optional[str] = None
):
    """
    Process all videos in the lerobot dataset to extract frames.
    
    Args:
        dataset_path: Root path of the lerobot dataset (should contain videos/ and meta/ directories)
        fps: Target frame rate (default 2.0 Hz)
        max_workers: Maximum number of parallel workers
        scale: Resolution scale ratio (1.0 = original, 0.5 = half)
        output_root: Root directory for output frames. If None, uses dataset_path.
    """
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        print(f"错误: 数据集路径不存在: {dataset_path}")
        return
    
    videos_dir = dataset_path / "videos"
    if not videos_dir.exists():
        print(f"错误: 视频目录不存在: {videos_dir}")
        return
    
    # Load task name from meta/tasks.jsonl
    task_name = load_task_name(dataset_path)
    print(f"任务名称: {task_name}")
    
    # Determine output root
    if output_root is None:
        output_root = dataset_path
    else:
        output_root = Path(output_root)
    
    if max_workers is None:
        max_workers = max(1, cpu_count() - 1)
    
    print(f"开始提取帧: {dataset_path}")
    print(f"目标帧率: {fps} Hz")
    print(f"分辨率压缩比率(scale): {scale}")
    print(f"并行进程数: {max_workers}")
    print(f"输出根目录: {output_root}")
    print("-" * 60)
    
    # Collect all video files
    video_tasks = []
    
    # Iterate through chunk directories
    for chunk_dir in sorted(videos_dir.iterdir()):
        if not chunk_dir.is_dir() or not chunk_dir.name.startswith("chunk-"):
            continue
        
        # Iterate through view directories (observation.images.*)
        for view_dir in sorted(chunk_dir.iterdir()):
            if not view_dir.is_dir() or not view_dir.name.startswith("observation.images."):
                continue
            
            view_name = extract_view_name_from_path(view_dir.name)
            
            # Find all video files in this view directory
            for video_file in sorted(view_dir.glob("episode_*.mp4")):
                # Extract episode name from filename (e.g., episode_000000.mp4 -> episode_000000)
                episode_name = video_file.stem
                
                # Output directory: {output_root}/{task_name}/{episode_name}/{view_name}/
                output_dir = output_root / task_name / episode_name / view_name
                
                video_tasks.append((
                    str(video_file),
                    str(output_dir),
                    task_name,
                    episode_name,
                    view_name,
                    fps,
                    scale
                ))
    
    total_videos = len(video_tasks)
    print(f"找到 {total_videos} 个视频文件需要处理")
    print("-" * 60)
    
    if total_videos == 0:
        print("没有找到需要处理的视频文件")
        return
    
    processed = 0
    failed = 0
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_process_single_video, task): task
            for task in video_tasks
        }
        
        for future in as_completed(futures):
            task_name, episode_name, view_name, success, status_msg = future.result()
            
            if success:
                processed += 1
                if processed % 10 == 0:
                    elapsed = time.time() - start_time
                    rate = processed / elapsed if elapsed > 0 else 0
                    remaining = (total_videos - processed - failed) / rate if rate > 0 else 0
                    print(f"进度: {processed}/{total_videos} (速度: {rate:.1f}个/秒, 预计剩余: {remaining:.0f}秒)")
            else:
                failed += 1
                print(f"[{task_name}/{episode_name}/{view_name}] ✗ {status_msg}")
    
    elapsed_time = time.time() - start_time
    print("-" * 60)
    print(f"处理完成! 成功: {processed}, 失败: {failed}, 耗时: {elapsed_time:.1f}秒")


def main():
    parser = argparse.ArgumentParser(description="从lerobot格式数据集的视频中提取帧")
    parser.add_argument("--dataset-path", type=str, required=True, help="lerobot数据集根目录路径（应包含videos/和meta/目录）")
    parser.add_argument("--fps", type=float, default=2.0, help="目标帧率，默认2.0 Hz")
    parser.add_argument("--max-workers", type=int, help="最大并行进程数，默认为CPU核心数-1")
    parser.add_argument("--scale", type=float, default=1.0, help="分辨率压缩比率(0-1]，默认1.0")
    parser.add_argument("--output-root", type=str, help="输出根目录，默认为数据集根目录")
    
    args = parser.parse_args()
    
    if not (0.0 < args.scale <= 1.0):
        parser.error("scale必须为大于0且不超过1的浮点数")
    
    process_dataset(args.dataset_path, args.fps, args.max_workers, args.scale, args.output_root)


if __name__ == "__main__":
    main()

