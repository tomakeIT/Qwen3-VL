#!/usr/bin/env python3
"""
视频采样测试脚本
从replay_results文件夹中的视频按指定fps采样率提取图片
"""

import cv2
import os
from pathlib import Path
import argparse


def extract_frames_from_video(video_path: str, output_dir: str, target_fps: float = 1.0):
    """
    从视频中按指定fps采样率提取帧并保存为图片
    
    Args:
        video_path: 输入视频路径
        output_dir: 输出目录
        target_fps: 目标采样率（fps），例如 1.0 表示每秒提取1帧
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"错误: 无法打开视频文件 {video_path}")
        return False
    
    # 获取视频的原始fps
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / original_fps if original_fps > 0 else 0
    
    print(f"\n处理视频: {os.path.basename(video_path)}")
    print(f"  原始fps: {original_fps:.2f}")
    print(f"  总帧数: {total_frames}")
    print(f"  时长: {duration:.2f}秒")
    print(f"  目标采样率: {target_fps} fps")
    
    # 计算采样间隔（每隔多少帧提取一帧）
    if target_fps >= original_fps:
        frame_interval = 1  # 如果目标fps大于等于原始fps，每帧都提取
        print(f"  警告: 目标fps ({target_fps}) >= 原始fps ({original_fps:.2f})，将提取所有帧")
    else:
        frame_interval = int(original_fps / target_fps)
        print(f"  采样间隔: 每 {frame_interval} 帧提取1帧")
    
    # 提取帧
    frame_count = 0
    saved_count = 0
    video_name = Path(video_path).stem  # 获取不带扩展名的文件名
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 按间隔采样
        if frame_count % frame_interval == 0:
            # 计算时间戳（秒）
            timestamp = frame_count / original_fps if original_fps > 0 else 0
            
            # 生成输出文件名：视频名_帧序号_时间戳.jpg
            output_filename = f"{video_name}_frame_{frame_count:06d}_t{timestamp:.3f}s.jpg"
            output_path = os.path.join(output_dir, output_filename)
            
            # 保存图片
            cv2.imwrite(output_path, frame)
            saved_count += 1
        
        frame_count += 1
    
    cap.release()
    
    print(f"  提取完成: 共保存 {saved_count} 张图片到 {output_dir}")
    return True


def process_replay_results(replay_results_dir: str, output_base_dir: str, target_fps: float = 1.0):
    """
    处理replay_results文件夹中的所有视频文件
    
    Args:
        replay_results_dir: replay_results文件夹路径
        output_base_dir: 输出基础目录
        target_fps: 目标采样率（fps）
    """
    replay_results_path = Path(replay_results_dir)
    if not replay_results_path.exists():
        print(f"错误: 目录不存在 {replay_results_dir}")
        return
    
    # 查找所有mp4视频文件
    video_files = list(replay_results_path.glob("*.mp4"))
    
    if not video_files:
        print(f"未找到视频文件 in {replay_results_dir}")
        return
    
    print(f"找到 {len(video_files)} 个视频文件")
    print(f"目标采样率: {target_fps} fps")
    print("=" * 60)
    
    # 为每个视频创建对应的输出目录
    for video_file in sorted(video_files):
        # 从文件名提取视角名称（例如：isaac_replay_action_state_first_person_camera）
        camera_name = video_file.stem.replace("isaac_replay_action_state_", "").replace("isaac_replay_", "")
        
        # 创建输出目录：output_base_dir/视角名称/
        output_dir = os.path.join(output_base_dir, camera_name)
        
        # 提取帧
        extract_frames_from_video(str(video_file), output_dir, target_fps)
    
    print("=" * 60)
    print(f"所有视频处理完成！输出目录: {output_base_dir}")


def main():
    parser = argparse.ArgumentParser(description="从视频中按指定fps采样率提取图片")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="1W_Libero_Piper/L10K3TurnOnTheStoveAndPutTheMokaPotOnIt/L10K3TurnOnTheStoveAndPutTheMokaPotOnIt_1758520720868771/replay_results",
        help="replay_results文件夹路径（相对或绝对路径）"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="test_sampled_images",
        help="输出目录"
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=1.0,
        help="目标采样率（fps），例如 1.0 表示每秒提取1帧"
    )
    
    args = parser.parse_args()
    
    # 如果输入路径是相对路径，转换为绝对路径
    if not os.path.isabs(args.input_dir):
        base_path = Path(__file__).parent
        input_dir = base_path / args.input_dir
    else:
        input_dir = Path(args.input_dir)
    
    # 如果输出路径是相对路径，转换为绝对路径
    if not os.path.isabs(args.output_dir):
        base_path = Path(__file__).parent
        output_dir = base_path / args.output_dir
    else:
        output_dir = Path(args.output_dir)
    
    # 处理视频
    process_replay_results(str(input_dir), str(output_dir), args.fps)


if __name__ == "__main__":
    main()

