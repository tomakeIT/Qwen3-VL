"""
demo_path, i, delta_t, reference_demo_path -> inference result
"""

import os
import argparse
from typing import List, Dict, Any, Optional

from inferencer import DeltaProgressInference
from utils.data_formatting import build_qwen_messages
from utils.utils import list_image_files
from utils.frame_sampling import sample_reference_frames_from_demo


def build_messages_from_demos(
    target_demo_path: str,
    i: int,
    delta_t: int,
    reference_demo_path: Optional[str],
    task_desc: str,
    reference_views: List[str],
    target_views: List[str],
    num_ref_frames: int = 4,
    ref_jitter: int = 2,
) -> List[Dict[str, Any]]:
    """根据demo路径和参数直接构造messages格式"""
    view_to_frames: Dict[str, List[str]] = {}
    for v in target_views:
        v_path = os.path.join(target_demo_path, v)
        frames = list_image_files(v_path)
        view_to_frames[v] = frames
    
    T = min(len(frames) for frames in view_to_frames.values())
    if T < 2:
        raise ValueError(f"Target demo has insufficient frames: T={T}")
    
    j = i + delta_t
    j = max(0, min(T - 1, j))
    
    target_paths_t1: List[str] = []
    target_paths_t2: List[str] = []
    for v in target_views:
        v_path = os.path.join(target_demo_path, v)
        frames_v = view_to_frames[v]
        frame_i_name = frames_v[i]
        frame_j_name = frames_v[j]
        img_abs_1 = os.path.abspath(os.path.join(v_path, frame_i_name))
        img_abs_2 = os.path.abspath(os.path.join(v_path, frame_j_name))
        target_paths_t1.append(img_abs_1)
        target_paths_t2.append(img_abs_2)
    
    if reference_demo_path:
        ref_img_paths, ref_progress_ints = sample_reference_frames_from_demo(
            reference_demo_path=reference_demo_path,
            reference_views=reference_views,
            num_ref_frames=num_ref_frames,
            ref_jitter=ref_jitter,
        )
    else:
        ref_img_paths = []
        ref_progress_ints = []
    
    images, human_str = build_prompt_with_reference_multiview(
        ref_img_paths=ref_img_paths,
        ref_progress_ints=ref_progress_ints,
        target_img_paths_t1=target_paths_t1,
        target_img_paths_t2=target_paths_t2,
        reference_view_names=reference_views,
        target_view_names=target_views,
        task_desc=task_desc,
    )

    messages = build_qwen_messages(human_str, images)
    return messages


def infer_pairwise_delta_progress(
    inference: DeltaProgressInference,
    root: str,
    target_demo_path: str,
    i: int,
    delta_t: int,
    reference_demo_path: Optional[str],
    task_desc: str,
    reference_views: List[str],
    target_views: List[str],
    num_ref_frames: int = 4,
    ref_jitter: int = 2,
) -> Optional[int]:
    """给定target_demo路径、i、delta_t、reference_demo路径，采样构造并推理"""
    messages = build_messages_from_demos(
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
    return inference.infer_from_messages(messages)


def main():
    parser = argparse.ArgumentParser(description="Level 2: Pair-wise推理")
    parser.add_argument("--base-model", type=str, required=True, help="基础模型路径")
    parser.add_argument("--adapter", type=str, required=True, help="LoRA适配器路径")
    parser.add_argument("--root", type=str, required=True, help="数据集根目录")
    parser.add_argument("--target-demo", type=str, required=True, help="target demo路径")
    parser.add_argument("--i", type=int, required=True, help="起始帧索引")
    parser.add_argument("--delta-t", type=int, required=True, help="时间差")
    parser.add_argument("--reference-demo", type=str, help="reference demo路径")
    parser.add_argument("--task-desc", type=str, required=True, help="任务描述")
    parser.add_argument("--reference-views", type=str, nargs="+", 
                       default=["first_person_camera"], help="reference视角列表")
    parser.add_argument("--target-views", type=str, nargs="+",
                       default=["first_person_camera", "left_hand_camera", "right_hand_camera"],
                       help="target视角列表")
    parser.add_argument("--num-ref-frames", type=int, default=4, help="reference帧数量")
    parser.add_argument("--ref-jitter", type=int, default=2, help="reference索引jitter范围")
    parser.add_argument("--no-flash-attn", action="store_true", help="不使用flash attention")
    
    args = parser.parse_args()
    
    inference = DeltaProgressInference(
        base_model_path=args.base_model,
        adapter_path=args.adapter,
        use_flash_attention=not args.no_flash_attn
    )
    
    result = infer_pairwise_delta_progress(
        inference=inference,
        root=args.root,
        target_demo_path=args.target_demo,
        i=args.i,
        delta_t=args.delta_t,
        reference_demo_path=args.reference_demo,
        task_desc=args.task_desc,
        reference_views=args.reference_views,
        target_views=args.target_views,
        num_ref_frames=args.num_ref_frames,
        ref_jitter=args.ref_jitter,
    )
    print(f"Delta Progress: {result}")


if __name__ == "__main__":
    main()

