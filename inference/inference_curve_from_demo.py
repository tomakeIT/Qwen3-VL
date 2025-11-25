"""
demo_path, reference_demo_path -> progress curve
"""

import os
import argparse
import yaml
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from PIL import Image
from types import SimpleNamespace
from typing import List, Optional, Tuple, Dict
from tqdm import tqdm

from inferencer import DeltaProgressInference
from utils.utils import list_image_files, dict_to_namespace
from inference_pairwise_from_demo import build_messages_from_demo


def load_frames_for_indices(target_demo_path: str, target_views: List[str], frame_indices: np.ndarray) -> List[np.ndarray]:
    """加载指定帧索引的图片（多视角拼接）"""
    view_to_frames: Dict[str, List[str]] = {}
    for v in target_views:
        v_path = os.path.join(target_demo_path, v)
        frames = list_image_files(v_path)
        view_to_frames[v] = frames
    
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
    inference: DeltaProgressInference,
    target_demo_path: str,
    reference_demo_path: Optional[str],
    task_desc: str,
    target_views: List[str],
    reference_config: SimpleNamespace,
    step_interval: int = 1,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """给定target_demo路径，逐步采样并生成progress曲线"""
    view_to_frames: Dict[str, List[str]] = {}
    for v in target_views:
        v_path = os.path.join(target_demo_path, v)
        frames = list_image_files(v_path)
        view_to_frames[v] = frames

    T = min(len(frames) for frames in view_to_frames.values())
    if T < 2:
        raise ValueError(f"Target demo has insufficient frames: T={T}")

    if end_frame is None:
        end_frame = T - 1
    else:
        end_frame = min(end_frame, T - 1)
    
    frame_indices = []
    progress_values = []
    current_progress = 0

    progress_range = range(start_frame, end_frame, step_interval)
    for i in tqdm(progress_range, desc="Progress curve inference"):
        j = i + step_interval
        if j > end_frame:
            break
        
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
            current_progress = max(0, min(100, current_progress))
            frame_indices.append(j)
            progress_values.append(current_progress)
    
    return np.array(frame_indices), np.array(progress_values)


def save_curve_plot(frame_indices: np.ndarray, progress_values: np.ndarray, output_path: str, task_name: Optional[str] = None):
    """保存progress曲线图"""
    plt.figure(figsize=(12, 6))
    plt.plot(frame_indices, progress_values, 'b-', linewidth=2, marker='o', markersize=4)
    plt.xlabel('Frame Index', fontsize=12)
    plt.ylabel('Progress (%)', fontsize=12)
    plt.title(f'Task Progress Curve{f" - {task_name}" if task_name else ""}', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
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
    ax2.set_ylim(0, 100)
    ax2.set_xlabel('Frame Index', fontsize=12)
    ax2.set_ylabel('Progress (%)', fontsize=12)
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
        f.write("Frame Index\tProgress (%)\n")
        for idx, prog in zip(frame_indices, progress_values):
            f.write(f"{int(idx)}\t{prog:.2f}\n")


def main(args):
    
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    with open(args.config, "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)
    
    config = dict_to_namespace(config_dict)
    
    inference = DeltaProgressInference(
        base_model_path=args.base_model,
        adapter_path=args.adapter,
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
    
    args = parser.parse_args()
    main(args)

