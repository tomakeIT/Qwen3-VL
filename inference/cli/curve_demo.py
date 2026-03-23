from __future__ import annotations

"""
对单个 demo 生成稀疏 progress curve 及可视化视频。
"""

import argparse
import os
from types import SimpleNamespace
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from PIL import Image
from tqdm import tqdm

from inference.core.demo_utils import build_messages_from_demo, scan_demo_frames
from inference.core.io_utils import load_config_namespace


def load_frames_for_indices(target_demo_path: str, target_views: List[str], frame_indices: np.ndarray) -> List[np.ndarray]:
    view_to_frames, _ = scan_demo_frames(target_demo_path, target_views)

    loaded_frames = []
    for idx in frame_indices:
        view_images = []
        for view_name in target_views:
            frame_path = os.path.join(target_demo_path, view_name, view_to_frames[view_name][int(idx)])
            img = Image.open(frame_path).convert("RGB")
            view_images.append(np.array(img))
        loaded_frames.append(view_images[0] if len(view_images) == 1 else np.hstack(view_images))
    return loaded_frames


def infer_progress_curve(
    inference,
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
    _, total_frames = scan_demo_frames(target_demo_path, target_views)
    if total_frames < 2:
        raise ValueError(f"Target demo has insufficient frames: T={total_frames}")

    end_frame = total_frames - 1 if end_frame is None else min(end_frame, total_frames - 1)
    ij_pairs = []
    for i in range(start_frame, end_frame, step_interval):
        j = i + step_interval
        if j > end_frame:
            break
        ij_pairs.append((i, j))

    if len(ij_pairs) == 0:
        return np.array([]), np.array([])

    frame_indices: List[int] = []
    progress_values: List[int] = []
    current_progress = 0

    if batch_size > 1:
        messages_list = []
        for i, j in tqdm(ij_pairs, desc="构建 messages"):
            messages_list.append(build_messages_from_demo(
                target_demo_path=target_demo_path,
                i=i,
                j=j,
                reference_demo_path=reference_demo_path,
                task_desc=task_desc,
                target_views=target_views,
                reference_config=reference_config,
            ))

        all_results = inference.infer_from_messages_batch(
            messages_list,
            batch_size=batch_size,
            desc="Curve inference",
        )
        for idx, (_, j) in enumerate(ij_pairs):
            delta_progress = all_results[idx]
            if delta_progress is not None:
                current_progress += delta_progress
                frame_indices.append(j)
                progress_values.append(current_progress)
    else:
        for i, j in tqdm(ij_pairs, desc="推理 progress curve"):
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


def save_curve_plot(frame_indices: np.ndarray, progress_values: np.ndarray, output_path: str, task_name: Optional[str] = None) -> None:
    plt.figure(figsize=(18, 3))
    plt.plot(frame_indices, progress_values, "b-", linewidth=2, marker="o", markersize=4)
    plt.xlabel("Frame Index", fontsize=12)
    plt.ylabel("Progress (integer)", fontsize=12)
    plt.title(f"Task Progress Curve{f' - {task_name}' if task_name else ''}", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close()


def visualize_video_with_curve(
    video_frames: List[np.ndarray],
    frame_indices: np.ndarray,
    progress_values: np.ndarray,
    output_path: str,
    task_name: str,
    output_fps: float = 5.0,
) -> None:
    writer_cls = plt.matplotlib.animation.writers["ffmpeg"]
    writer = writer_cls(fps=output_fps, metadata={"artist": "Qwen3-VL"}, bitrate=1800)

    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(2, 1, height_ratios=[2, 1], hspace=0.3)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    fig.suptitle(f"Task: {task_name}", fontsize=14, fontweight="bold")
    ax1.axis("off")
    ax1.set_title("Video Frame", fontsize=12)
    im1 = ax1.imshow(video_frames[0])

    ax2.set_xlim(frame_indices[0], frame_indices[-1])
    ax2.set_xlabel("Frame Index", fontsize=12)
    ax2.set_ylabel("Progress (integer)", fontsize=12)
    ax2.set_title("Progress Curve", fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.plot(frame_indices, progress_values, "lightgray", linewidth=1, alpha=0.5, label="Full Curve")

    line, = ax2.plot([], [], "b-", linewidth=2, label="Progress")
    point, = ax2.plot([], [], "ro", markersize=8, label="Current")
    ax2.legend(loc="upper right")

    x_data: List[float] = []
    y_data: List[float] = []

    def animate(frame_idx):
        im1.set_array(video_frames[frame_idx])
        x_data.append(frame_indices[frame_idx])
        y_data.append(progress_values[frame_idx])
        line.set_data(x_data, y_data)
        point.set_data([frame_indices[frame_idx]], [progress_values[frame_idx]])

        if frame_idx > 0:
            margin = max(10, len(video_frames) // 20)
            ax2.set_xlim(
                max(frame_indices[0], frame_indices[frame_idx] - margin),
                min(frame_indices[-1], frame_indices[frame_idx] + margin),
            )
        return [im1, line, point]

    interval = int(1000 / output_fps) if output_fps > 0 else 200
    anim = FuncAnimation(fig, animate, frames=len(video_frames), interval=interval, blit=True, repeat=True)
    anim.save(output_path, writer=writer)
    plt.close()


def save_curve_data(frame_indices: np.ndarray, progress_values: np.ndarray, output_path: str, task_name: str) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"Task: {task_name}\n")
        f.write(f"Total frames: {len(frame_indices)}\n")
        f.write(f"Progress range: [{progress_values.min():.2f}, {progress_values.max():.2f}]\n\n")
        f.write("Frame Index\tProgress (integer)\n")
        for idx, prog in zip(frame_indices, progress_values):
            f.write(f"{int(idx)}\t{prog:.2f}\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="对单个 demo 生成 progress curve")
    parser.add_argument("--base-model", type=str, default="models/Qwen-VL-2B-Instruct", help="基础模型路径")
    parser.add_argument("--adapter", type=str, required=True, help="LoRA 适配器路径")
    parser.add_argument("--target-demo", type=str, required=True, help="target demo 路径")
    parser.add_argument("--reference-demo", type=str, help="reference demo 路径")
    parser.add_argument("--task-desc", type=str, required=True, help="任务描述")
    parser.add_argument("--config", type=str, required=True, help="YAML 配置文件路径")
    parser.add_argument("--step-interval", type=int, default=1, help="采样间隔")
    parser.add_argument("--start-frame", type=int, default=0, help="起始帧")
    parser.add_argument("--end-frame", type=int, default=None, help="结束帧")
    parser.add_argument("--output-dir", type=str, default="outputs/inference_progress_curve", help="输出目录")
    parser.add_argument("--output-fps", type=float, default=5.0, help="输出视频 fps")
    parser.add_argument("--batch-size", type=int, default=1, help="batch 推理大小")
    parser.add_argument("--num-gpus", type=int, default=1, help="使用的 GPU 数量")
    return parser


def main(args: argparse.Namespace) -> None:
    from inference.core.multi_gpu_inferencer import MultiGPUDeltaProgressInference

    os.makedirs(args.output_dir, exist_ok=True)
    config = load_config_namespace(args.config)
    inference = MultiGPUDeltaProgressInference(
        base_model_path=args.base_model,
        adapter_path=args.adapter,
        num_gpus=args.num_gpus,
    )
    try:
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
    finally:
        inference.close()

    video_frames = load_frames_for_indices(args.target_demo, target_views, frame_indices)

    curve_plot_path = os.path.join(args.output_dir, "progress_curve.png")
    save_curve_plot(frame_indices, progress_values, curve_plot_path, args.task_desc)

    video_path = os.path.join(args.output_dir, "progress_curve_video.mp4")
    visualize_video_with_curve(video_frames, frame_indices, progress_values, video_path, args.task_desc, args.output_fps)

    data_path = os.path.join(args.output_dir, "progress_curve_data.txt")
    save_curve_data(frame_indices, progress_values, data_path, args.task_desc)

    print(f"output_dir: {args.output_dir}")
    print(f"curve_plot: {curve_plot_path}")
    print(f"curve_video: {video_path}")
    print(f"curve_data: {data_path}")


if __name__ == "__main__":
    main(build_parser().parse_args())
