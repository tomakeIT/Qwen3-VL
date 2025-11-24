"""
Level 3: 给定demo路径，逐步采样并生成progress曲线
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple, Dict

from inferencer import DeltaProgressInference
from utils.utils import list_image_files
from inference_pair import infer_pairwise_delta_progress


def infer_progress_curve(
    inference: DeltaProgressInference,
    target_demo_path: str,
    reference_demo_path: Optional[str],
    task_desc: str,
    reference_views: List[str],
    target_views: List[str],
    step_interval: int = 1,
    num_ref_frames: int = 4,
    ref_jitter: int = 2,
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
    
    for i in range(start_frame, end_frame, step_interval):
        if i + step_interval > end_frame:
            break
        
        delta_t = step_interval
        delta_progress = infer_pairwise_delta_progress(
            inference=inference,
            target_demo_path=target_demo_path,
            i=i,
            delta_t=delta_t,
            reference_demo_path=reference_demo_path,
            task_desc=task_desc,
            reference_views=reference_views,
            target_views=target_views,
            num_ref_frames=num_ref_frames,
            ref_jitter=ref_jitter,
        )
        
        if delta_progress is not None:
            current_progress += delta_progress
            current_progress = max(0, min(100, current_progress))
            frame_indices.append(i + step_interval)
            progress_values.append(current_progress)
    
    return np.array(frame_indices), np.array(progress_values)


def visualize_progress_curve(
    frame_indices: np.ndarray,
    progress_values: np.ndarray,
    output_path: str,
    task_name: Optional[str] = None,
):
    """可视化progress曲线并保存"""
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
    print(f"Progress curve saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Level 3: Progress曲线推理")
    parser.add_argument("--base-model", type=str, required=True, help="基础模型路径")
    parser.add_argument("--adapter", type=str, required=True, help="LoRA适配器路径")
    parser.add_argument("--target-demo", type=str, required=True, help="target demo路径")
    parser.add_argument("--reference-demo", type=str, help="reference demo路径")
    parser.add_argument("--task-desc", type=str, required=True, help="任务描述")
    parser.add_argument("--step-interval", type=int, default=1, help="采样间隔")
    parser.add_argument("--start-frame", type=int, default=0, help="起始帧")
    parser.add_argument("--end-frame", type=int, default=None, help="结束帧")
    parser.add_argument("--output", type=str, required=True, help="输出图片路径")
    parser.add_argument("--reference-views", type=str, nargs="+", 
                       default=["first_person_camera"], help="reference视角列表")
    parser.add_argument("--target-views", type=str, nargs="+",
                       default=["first_person_camera", "left_hand_camera", "right_hand_camera"],
                       help="target视角列表")
    parser.add_argument("--num-ref-frames", type=int, default=4, help="reference帧数量")
    parser.add_argument("--ref-jitter", type=int, default=2, help="reference索引jitter范围")
    
    args = parser.parse_args()
    
    inference = DeltaProgressInference(
        base_model_path=args.base_model,
        adapter_path=args.adapter,
    )
    
    frame_indices, progress_values = infer_progress_curve(
        inference=inference,
        target_demo_path=args.target_demo,
        reference_demo_path=args.reference_demo,
        task_desc=args.task_desc,
        reference_views=args.reference_views,
        target_views=args.target_views,
        step_interval=args.step_interval,
        num_ref_frames=args.num_ref_frames,
        ref_jitter=args.ref_jitter,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
    )
    
    visualize_progress_curve(frame_indices, progress_values, args.output, args.task_desc)


if __name__ == "__main__":
    main()

