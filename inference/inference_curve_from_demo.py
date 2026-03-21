from __future__ import annotations

"""
demo_path, reference_demo_path -> progress curve
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from PIL import Image
from types import SimpleNamespace
from typing import List, Optional, Tuple, Dict
from tqdm import tqdm

from common.demo_scan import scan_demo_frames
from common.io_utils import load_config_namespace
from common.messages import build_messages_from_demo


def load_frames_for_indices(target_demo_path: str, target_views: List[str], frame_indices: np.ndarray) -> List[np.ndarray]:
    """加载指定帧索引的图片（多视角拼接）"""
    view_to_frames, _ = scan_demo_frames(target_demo_path, target_views)
    
    loaded_frames = []
    for idx in frame_indices:
        view_images = []
        for v in target_views:
            frame_path = os.path.join(target_demo_path, v, view_to_frames[v][int(idx)])
            img = Image.open(frame_path).convert("RGB")
            view_images.append(np.array(img))
        
        # 水平拼接多视角
        if len(view_images) == 1:
            combined = view_images[0]
        else:
            combined = np.hstack(view_images)
        loaded_frames.append(combined)
    
    return loaded_frames

def infer_progress_curve(
    inference: MultiGPUDeltaProgressInference,
    target_demo_path: str,
    reference_demo_path: Optional[str],
    task_desc: str,
    target_views: List[str],
    reference_config: SimpleNamespace,
    step_interval: int = 1,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
    batch_size: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """给定target_demo路径，逐步采样并生成progress曲线
    
    Args:
        inference: DeltaProgressInference实例
        target_demo_path: target demo路径
        reference_demo_path: reference demo路径
        task_desc: 任务描述
        target_views: target视角列表
        reference_config: reference配置
        step_interval: 采样间隔
        start_frame: 起始帧
        end_frame: 结束帧
        batch_size: batch大小，大于1时使用batch推理
        
    Returns:
        frame_indices: 帧索引数组
        progress_values: progress值数组
    """
    _, T = scan_demo_frames(target_demo_path, target_views)
    if T < 2:
        raise ValueError(f"Target demo has insufficient frames: T={T}")

    if end_frame is None:
        end_frame = T - 1
    else:
        end_frame = min(end_frame, T - 1)
    
    # 预生成所有 (i, j) 对
    progress_range = range(start_frame, end_frame, step_interval)
    ij_pairs = []
    for i in progress_range:
        j = i + step_interval
        if j > end_frame:
            break
        ij_pairs.append((i, j))
    
    if len(ij_pairs) == 0:
        return np.array([]), np.array([])
    
    frame_indices = []
    progress_values = []
    current_progress = 0
    
    if batch_size > 1:
        # Batch 推理模式（支持多GPU）
        print(f"使用 batch 推理，batch_size={batch_size}")
        
        # 预生成所有 messages
        messages_list = []
        for i, j in tqdm(ij_pairs, desc="Building messages"):
            messages = build_messages_from_demo(
                target_demo_path=target_demo_path,
                i=i,
                j=j,
                reference_demo_path=reference_demo_path,
                task_desc=task_desc,
                target_views=target_views,
                reference_config=reference_config,
            )
            messages_list.append(messages)
        
        # Batch 推理（自动多GPU并行）
        all_results = inference.infer_from_messages_batch(
            messages_list, 
            batch_size=batch_size,
            desc="Batch inference"
        )
        
        # 累加 progress
        for idx, (i, j) in enumerate(ij_pairs):
            delta_progress = all_results[idx]
            if delta_progress is not None:
                current_progress += delta_progress
                frame_indices.append(j)
                progress_values.append(current_progress)
    else:
        # 单条推理模式（原始逻辑）
        for i, j in tqdm(ij_pairs, desc="Progress curve inference"):
            messages = build_messages_from_demo(
                target_demo_path=target_demo_path,
                i=i,
                j=j,
                reference_demo_path=reference_demo_path,
                task_desc=task_desc,
                target_views=target_views,
                reference_config=reference_config,
            )
            
            delta_progress = inference.infer_from_messages(messages)
            
            if delta_progress is not None:
                current_progress += delta_progress
                frame_indices.append(j)
                progress_values.append(current_progress)
    
    return np.array(frame_indices), np.array(progress_values)


def save_curve_plot(frame_indices: np.ndarray, progress_values: np.ndarray, output_path: str, task_name: Optional[str] = None):
    """保存progress曲线图"""
    plt.figure(figsize=(18, 3))
    plt.plot(frame_indices, progress_values, 'b-', linewidth=2, marker='o', markersize=4)
    plt.xlabel('Frame Index', fontsize=12)
    plt.ylabel('Progress (integer)', fontsize=12)
    plt.title(f'Task Progress Curve{f" - {task_name}" if task_name else ""}', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    plt.close()


def visualize_video_with_curve(video_frames: List[np.ndarray], frame_indices: np.ndarray, progress_values: np.ndarray, 
                               output_path: str, task_name: str, output_fps: float = 5.0):
    """创建可视化视频，包含原视频帧和progress曲线"""
    num_frames = len(video_frames)
    
    Writer = plt.matplotlib.animation.writers['ffmpeg']
    writer = Writer(fps=output_fps, metadata=dict(artist='Qwen3-VL'), bitrate=1800)
    
    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(2, 1, height_ratios=[2, 1], hspace=0.3)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    
    fig.suptitle(f'Task: {task_name}', fontsize=14, fontweight='bold')
    
    ax1.axis('off')
    ax1.set_title('Video Frame', fontsize=12)
    im1 = ax1.imshow(video_frames[0])
    
    ax2.set_xlim(frame_indices[0], frame_indices[-1])
    ax2.set_xlabel('Frame Index', fontsize=12)
    ax2.set_ylabel('Progress (integer)', fontsize=12)
    ax2.set_title('Progress Curve', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    ax2.plot(frame_indices, progress_values, 'lightgray', linewidth=1, alpha=0.5, label='Full Curve')
    
    line, = ax2.plot([], [], 'b-', linewidth=2, label='Progress')
    point, = ax2.plot([], [], 'ro', markersize=8, label='Current')
    ax2.legend(loc='upper right')
    
    x_data, y_data = [], []
    
    def animate(frame_idx):
        im1.set_array(video_frames[frame_idx])
        
        x_data.append(frame_indices[frame_idx])
        y_data.append(progress_values[frame_idx])
        line.set_data(x_data, y_data)
        point.set_data([frame_indices[frame_idx]], [progress_values[frame_idx]])
        
        if frame_idx > 0:
            margin = max(10, num_frames // 20)
            ax2.set_xlim(max(frame_indices[0], frame_indices[frame_idx] - margin), 
                        min(frame_indices[-1], frame_indices[frame_idx] + margin))
        
        return [im1, line, point]
    
    interval = int(1000 / output_fps) if output_fps > 0 else 200
    anim = FuncAnimation(fig, animate, frames=num_frames, interval=interval, blit=True, repeat=True)
    anim.save(output_path, writer=writer)
    plt.close()


def save_curve_data(frame_indices: np.ndarray, progress_values: np.ndarray, output_path: str, task_name: str):
    """保存曲线数值"""
    with open(output_path, 'w') as f:
        f.write(f"Task: {task_name}\n")
        f.write(f"Total frames: {len(frame_indices)}\n")
        f.write(f"Progress range: [{progress_values.min():.2f}, {progress_values.max():.2f}]\n\n")
        f.write("Frame Index\tProgress (integer)\n")
        for idx, prog in zip(frame_indices, progress_values):
            f.write(f"{int(idx)}\t{prog:.2f}\n")


def main(args):
    
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    config = load_config_namespace(args.config)
    from multi_gpu_inferencer import MultiGPUDeltaProgressInference
    
    # 初始化推理器（自动处理单/多 GPU）
    inference = MultiGPUDeltaProgressInference(
        base_model_path=args.base_model,
        adapter_path=args.adapter,
        num_gpus=args.num_gpus,
    )
    
    target_views = config.sampling.required_views
    frame_indices, progress_values = infer_progress_curve(
        inference=inference,
        target_demo_path=args.target_demo,
        reference_demo_path=args.reference_demo,
        task_desc=args.task_desc,
        target_views=target_views,
        reference_config=config.reference,
        step_interval=args.step_interval,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        batch_size=args.batch_size,
    )
    
    # 加载视频帧
    video_frames = load_frames_for_indices(args.target_demo, target_views, frame_indices)
    
    # 保存curve图
    curve_plot_path = os.path.join(args.output_dir, "progress_curve.png")
    save_curve_plot(frame_indices, progress_values, curve_plot_path, args.task_desc)
    print(f"Curve plot saved to {curve_plot_path}")
    
    # 保存可视化视频
    video_path = os.path.join(args.output_dir, "progress_curve_video.mp4")
    visualize_video_with_curve(video_frames, frame_indices, progress_values, video_path, args.task_desc, args.output_fps)
    print(f"Video saved to {video_path}")
    
    # 保存曲线数值
    data_path = os.path.join(args.output_dir, "progress_curve_data.txt")
    save_curve_data(frame_indices, progress_values, data_path, args.task_desc)
    print(f"Curve data saved to {data_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=str, default="models/Qwen-VL-2B-Instruct", help="基础模型路径")
    parser.add_argument("--adapter", type=str, required=True, help="LoRA适配器路径")
    parser.add_argument("--target-demo", type=str, required=True, help="target demo路径")
    parser.add_argument("--reference-demo", type=str, help="reference demo路径")
    parser.add_argument("--task-desc", type=str, required=True, help="任务描述")
    parser.add_argument("--config", type=str, required=True, help="YAML配置文件路径")
    parser.add_argument("--step-interval", type=int, default=1, help="采样间隔")
    parser.add_argument("--start-frame", type=int, default=0, help="起始帧")
    parser.add_argument("--end-frame", type=int, default=None, help="结束帧")
    parser.add_argument("--output-dir", type=str, default="outputs/inference_progress_curve", help="输出目录")
    parser.add_argument("--output-fps", type=float, default=5.0, help="输出视频fps")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch推理大小，大于1时使用batch推理加速")
    parser.add_argument("--num-gpus", type=int, default=1, help="使用的GPU数量（默认1，设为-1使用所有可用GPU）")
    
    args = parser.parse_args()
    main(args)

