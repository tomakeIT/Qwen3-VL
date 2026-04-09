#!/usr/bin/env python3
"""
视频拆分脚本
将isaac_replay_state.mp4拆分成多个视角的视频
"""

import os
import subprocess
import argparse
from pathlib import Path
from typing import Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count
import time


def _process_single_crop(args: Tuple[str, str, int, int, int, int]) -> Tuple[str, bool, str]:
    """
    处理单个视频裁剪任务（用于并行处理）
    返回: (output_file, success, error_message)
    """
    input_video, output_file, x, y, width, height = args
    try:
        cmd = [
            "ffmpeg",
            "-i", input_video,
            "-vf", f"crop={width}:{height}:{x}:{y}",
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "23",
            "-y",
            "-loglevel", "error",  # 减少输出
            output_file
        ]
        
        subprocess.run(cmd, capture_output=True, check=True)
        return (output_file, True, "")
    except subprocess.CalledProcessError as e:
        return (output_file, False, e.stderr.decode() if e.stderr else str(e))


def split_3view_video(input_video: str, output_dir: str, parallel: bool = True) -> bool:
    """
    拆分3视角视频 (3840×720 -> 3个1280×720)
    从左到右：left_hand, first_person, right_hand
    """
    os.makedirs(output_dir, exist_ok=True)
    
    output_files = [
        os.path.join(output_dir, "isaac_replay_action_state_left_hand_camera.mp4"),
        os.path.join(output_dir, "isaac_replay_action_state_first_person_camera.mp4"),
        os.path.join(output_dir, "isaac_replay_action_state_right_hand_camera.mp4"),
    ]
    
    if all(os.path.exists(f) for f in output_files):
        return True
    
    crop_tasks = [
        (input_video, output_file, i * 1280, 0, 1280, 720)
        for i, output_file in enumerate(output_files)
    ]
    
    if parallel:
        with ThreadPoolExecutor(max_workers=min(3, cpu_count())) as executor:
            futures = [executor.submit(_process_single_crop, task) for task in crop_tasks]
            results = [future.result() for future in as_completed(futures)]
        
        for output_file, success, error_msg in results:
            if not success:
                return False
    else:
        for task in crop_tasks:
            output_file, success, error_msg = _process_single_crop(task)
            if not success:
                return False
    
    return True


def split_6view_video(input_video: str, output_dir: str, parallel: bool = True) -> bool:
    """
    拆分6视角视频 (3840×1440 -> 6个1280×720)
    2行3列布局：
    左上: left_hand, 上: first_person, 右上: right_hand
    左下: left_shoulder, 下: eye_in_hand, 右下: right_shoulder
    """
    os.makedirs(output_dir, exist_ok=True)
    
    output_files = [
        os.path.join(output_dir, "isaac_replay_action_state_left_hand_camera.mp4"),      # 左上 (0, 0)
        os.path.join(output_dir, "isaac_replay_action_state_first_person_camera.mp4"),    # 上 (1280, 0)
        os.path.join(output_dir, "isaac_replay_action_state_right_hand_camera.mp4"),     # 右上 (2560, 0)
        os.path.join(output_dir, "isaac_replay_action_state_left_shoulder_camera.mp4"),  # 左下 (0, 720)
        os.path.join(output_dir, "isaac_replay_action_state_eye_in_hand_camera.mp4"),   # 下 (1280, 720)
        os.path.join(output_dir, "isaac_replay_action_state_right_shoulder_camera.mp4"), # 右下 (2560, 720)
    ]
    
    if all(os.path.exists(f) for f in output_files):
        return True
    
    positions = [
        (0, 0), (1280, 0), (2560, 0),
        (0, 720), (1280, 720), (2560, 720),
    ]
    
    crop_tasks = [
        (input_video, output_file, x, y, 1280, 720)
        for (x, y), output_file in zip(positions, output_files)
    ]
    
    if parallel:
        with ThreadPoolExecutor(max_workers=min(6, cpu_count())) as executor:
            futures = [executor.submit(_process_single_crop, task) for task in crop_tasks]
            results = [future.result() for future in as_completed(futures)]
        
        for output_file, success, error_msg in results:
            if not success:
                return False
    else:
        for task in crop_tasks:
            output_file, success, error_msg = _process_single_crop(task)
            if not success:
                return False
    
    return True


def _process_single_demo(args: Tuple[str, str, int]) -> Tuple[str, str, bool, str]:
    """
    处理单个demo（用于并行处理）
    返回: (task_name, demo_name, success, status_message)
    """
    task_dir_str, demo_dir_str, num_views = args
    task_dir = Path(task_dir_str)
    demo_dir = Path(demo_dir_str)
    task_name = task_dir.name
    demo_name = demo_dir.name
    
    video_file = demo_dir / "isaac_replay_state.mp4"
    if not video_file.exists():
        return (task_name, demo_name, False, "未找到视频文件")
    
    output_dir = demo_dir / "replay_results"
    
    # 定义视角文件名
    view_files_3 = [
        "isaac_replay_action_state_left_hand_camera.mp4",
        "isaac_replay_action_state_first_person_camera.mp4",
        "isaac_replay_action_state_right_hand_camera.mp4",
    ]
    view_files_6 = view_files_3 + [
        "isaac_replay_action_state_left_shoulder_camera.mp4",
        "isaac_replay_action_state_eye_in_hand_camera.mp4",
        "isaac_replay_action_state_right_shoulder_camera.mp4",
    ]
    
    output_files = [output_dir / f for f in (view_files_6 if num_views == 6 else view_files_3)]
    
    if all(f.exists() for f in output_files):
        return (task_name, demo_name, True, "已存在")
    
    if num_views == 3:
        success = split_3view_video(str(video_file), str(output_dir), parallel=True)
    elif num_views == 6:
        success = split_6view_video(str(video_file), str(output_dir), parallel=True)
    else:
        return (task_name, demo_name, False, f"不支持的视角数: {num_views}")
    
    return (task_name, demo_name, success, "成功" if success else "处理失败")


def process_dataset(dataset_path: str, num_views: int, max_workers: Optional[int] = None):
    dataset_path = Path(dataset_path)
    if max_workers is None:
        max_workers = max(1, cpu_count() - 1)
    
    print(f"开始处理数据集: {dataset_path}, 视角数: {num_views}, 并行进程数: {max_workers}")
    print("-" * 60)
    
    demo_tasks = []
    for task_dir in sorted(dataset_path.iterdir()):
        if not task_dir.is_dir():
            continue
        for demo_dir in sorted(task_dir.iterdir()):
            if not demo_dir.is_dir():
                continue
            video_file = demo_dir / "isaac_replay_state.mp4"
            if video_file.exists():
                demo_tasks.append((str(task_dir), str(demo_dir), num_views))
    
    total_demos = len(demo_tasks)
    print(f"找到 {total_demos} 个需要处理的demo")
    print("-" * 60)
    
    if total_demos == 0:
        return
    
    processed_demos = 0
    failed_demos = 0
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_demo = {
            executor.submit(_process_single_demo, task): task 
            for task in demo_tasks
        }
        
        for future in as_completed(future_to_demo):
            task_name, demo_name, success, status_msg = future.result()
            
            if success:
                processed_demos += 1
                if processed_demos % 10 == 0:
                    elapsed = time.time() - start_time
                    rate = processed_demos / elapsed if elapsed > 0 else 0
                    remaining = (total_demos - processed_demos - failed_demos) / rate if rate > 0 else 0
                    print(f"进度: {processed_demos}/{total_demos} (速度: {rate:.1f}个/秒, 预计剩余: {remaining:.0f}秒)")
            else:
                failed_demos += 1
                if status_msg != "已存在":
                    print(f"[{task_name}/{demo_name}] ✗ {status_msg}")
    
    elapsed_time = time.time() - start_time
    print("-" * 60)
    print(f"处理完成! 成功: {processed_demos}, 失败: {failed_demos}, 耗时: {elapsed_time:.1f}秒")


def main():
    parser = argparse.ArgumentParser(description="拆分视频为多个视角")
    parser.add_argument("--dataset-path", type=str, required=True, help="数据集根目录路径")
    parser.add_argument("--num-views", type=int, choices=[3, 6], required=True, help="初始视频包含的视角数量（3或6）")
    parser.add_argument("--max-workers", type=int, help="最大并行进程数，默认为CPU核心数-1")
    
    args = parser.parse_args()
    process_dataset(args.dataset_path, args.num_views, args.max_workers)


if __name__ == "__main__":
    main()

