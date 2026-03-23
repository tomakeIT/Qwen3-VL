#!/usr/bin/env python3
"""
Extract frames from videos at specified FPS.
Processes videos in replay_results/ directory and saves frames to view directories.
"""

import os
import subprocess
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count
from typing import Tuple, Optional
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


def _process_single_video(args: Tuple[str, str, str, float, float]) -> Tuple[str, str, str, bool, str]:
    """
    Process a single video file.
    
    Args:
        (video_path, output_dir, view_name, fps, scale)
    
    Returns:
        (task_name, demo_name, view_name, success, status_message)
    """
    video_path_str, output_dir_str, view_name, fps, scale = args
    video_path = Path(video_path_str)
    
    # Extract task and demo names from path
    # Path structure: dataset/task/demo/replay_results/video.mp4
    parts = video_path.parts
    if len(parts) < 4:
        return ("unknown", "unknown", view_name, False, "Invalid path structure")
    
    demo_name = parts[-3]  # demo directory name
    task_name = parts[-4]  # task directory name
    
    success, status_msg = extract_frames_from_video(video_path_str, output_dir_str, fps, scale)
    return (task_name, demo_name, view_name, success, status_msg)


def process_dataset(
    dataset_path: str,
    fps: float = 2.0,
    max_workers: Optional[int] = None,
    scale: float = 1.0
):
    """
    Process all videos in the dataset to extract frames.
    
    Args:
        dataset_path: Root path of the dataset
        fps: Target frame rate (default 2.0 Hz)
        max_workers: Maximum number of parallel workers
        scale: Resolution scale ratio (1.0 = original, 0.5 = half)
    """
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        print(f"错误: 数据集路径不存在: {dataset_path}")
        return
    
    if max_workers is None:
        max_workers = max(1, cpu_count() - 1)
    
    print(f"开始提取帧: {dataset_path}")
    print(f"目标帧率: {fps} Hz")
    print(f"分辨率压缩比率(scale): {scale}")
    print(f"并行进程数: {max_workers}")
    print("-" * 60)
    
    # Collect all video files
    video_tasks = []
    
    # Map video filename patterns to view names
    video_to_view = {
        "isaac_replay_action_state_left_hand_camera.mp4": "left_hand_camera",
        "isaac_replay_action_state_first_person_camera.mp4": "first_person_camera",
        "isaac_replay_action_state_right_hand_camera.mp4": "right_hand_camera",
        "isaac_replay_action_state_left_shoulder_camera.mp4": "left_shoulder_camera",
        "isaac_replay_action_state_eye_in_hand_camera.mp4": "eye_in_hand_camera",
        "isaac_replay_action_state_right_shoulder_camera.mp4": "right_shoulder_camera",
    }
    
    for task_dir in sorted(dataset_path.iterdir()):
        if not task_dir.is_dir():
            continue
        
        for demo_dir in sorted(task_dir.iterdir()):
            if not demo_dir.is_dir():
                continue
            
            replay_results_dir = demo_dir / "replay_results"
            if not replay_results_dir.exists():
                continue
            
            # Find all video files in replay_results
            for video_file in replay_results_dir.glob("*.mp4"):
                video_name = video_file.name
                
                # Determine view name from video filename
                view_name = None
                for pattern, view in video_to_view.items():
                    if pattern in video_name:
                        view_name = view
                        break
                
                if view_name is None:
                    # Fallback: use video name without extension
                    view_name = video_file.stem
                
                # Output directory: task/demo/view_name/
                output_dir = demo_dir / view_name
                
                video_tasks.append((str(video_file), str(output_dir), view_name, fps, scale))
    
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
            task_name, demo_name, view_name, success, status_msg = future.result()
            
            if success:
                processed += 1
                if processed % 10 == 0:
                    elapsed = time.time() - start_time
                    rate = processed / elapsed if elapsed > 0 else 0
                    remaining = (total_videos - processed - failed) / rate if rate > 0 else 0
                    print(f"进度: {processed}/{total_videos} (速度: {rate:.1f}个/秒, 预计剩余: {remaining:.0f}秒)")
            else:
                failed += 1
                print(f"[{task_name}/{demo_name}/{view_name}] ✗ {status_msg}")
    
    elapsed_time = time.time() - start_time
    print("-" * 60)
    print(f"处理完成! 成功: {processed}, 失败: {failed}, 耗时: {elapsed_time:.1f}秒")


def main():
    parser = argparse.ArgumentParser(description="从视频中提取帧")
    parser.add_argument("--dataset-path", type=str, required=True, help="数据集根目录路径")
    parser.add_argument("--fps", type=float, default=2.0, help="目标帧率，默认2.0 Hz")
    parser.add_argument("--max-workers", type=int, help="最大并行进程数，默认为CPU核心数-1")
    parser.add_argument("--scale", type=float, default=1.0, help="分辨率压缩比率(0-1]，默认1.0")
    
    args = parser.parse_args()
    
    if not (0.0 < args.scale <= 1.0):
        parser.error("scale必须为大于0且不超过1的浮点数")
    
    process_dataset(args.dataset_path, args.fps, args.max_workers, args.scale)


if __name__ == "__main__":
    main()

