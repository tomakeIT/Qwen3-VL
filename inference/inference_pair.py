"""
demo_path, i, j, reference_demo_path -> inference result
"""

import os
import argparse
import yaml
from typing import List, Dict, Any, Optional
from types import SimpleNamespace

from inferencer import DeltaProgressInference
from utils.data_formatting import build_qwen_messages
from utils.utils import list_image_files, dict_to_namespace
from utils.frame_sampling import sample_reference_frames_from_demo
from utils.prompt import build_prompt_with_reference_multiview


def build_messages_from_demos(
    target_demo_path: str,
    i: int,
    j: int,
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
    
    # 确保 i 和 j 在有效范围内
    i = max(0, min(T - 1, i))
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
    
    img_paths, human_str = build_prompt_with_reference_multiview(
        ref_img_paths=ref_img_paths,
        ref_progress_ints=ref_progress_ints,
        target_img_paths_t1=target_paths_t1,
        target_img_paths_t2=target_paths_t2,
        reference_view_names=reference_views,
        target_view_names=target_views,
        task_desc=task_desc,
    )

    messages = build_qwen_messages(human_str, img_paths)
    return messages


def infer_pairwise_delta_progress(
    inference: DeltaProgressInference,
    target_demo_path: str,
    i: int,
    j: int,
    reference_demo_path: Optional[str],
    task_desc: str,
    reference_views: List[str],
    target_views: List[str],
    num_ref_frames: int = 4,
    ref_jitter: int = 2,
) -> Optional[int]:
    """给定target_demo路径、i、j、reference_demo路径，采样构造并推理"""
    messages = build_messages_from_demos(
        target_demo_path=target_demo_path,
        i=i,
        j=j,
        reference_demo_path=reference_demo_path,
        task_desc=task_desc,
        reference_views=reference_views,
        target_views=target_views,
        num_ref_frames=num_ref_frames,
        ref_jitter=ref_jitter,
    )
    return inference.infer_from_messages(messages)


def main():
    parser = argparse.ArgumentParser(description="Pair-wise推理")
    parser.add_argument("--base-model", type=str, required=True, help="基础模型路径")
    parser.add_argument("--adapter", type=str, required=True, help="LoRA适配器路径")
    parser.add_argument("--target-demo", type=str, required=True, help="target demo路径")
    parser.add_argument("--i", type=int, required=True, help="起始帧索引")
    parser.add_argument("--j", type=int, required=True, help="结束帧索引")
    parser.add_argument("--reference-demo", type=str, help="reference demo路径")
    parser.add_argument("--task-desc", type=str, required=True, help="任务描述")
    parser.add_argument("--config", type=str, required=True, help="YAML配置文件路径")
    
    args = parser.parse_args()
    
    # 从 yaml 配置文件读取采样参数
    with open(args.config, "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)
    
    config = dict_to_namespace(config_dict)
    
    # 从配置中读取参数
    reference_views = config.reference.views
    target_views = config.sampling.required_views
    num_ref_frames = config.reference.frames
    ref_jitter = config.reference.jitter
    
    inference = DeltaProgressInference(
        base_model_path=args.base_model,
        adapter_path=args.adapter,
    )
    
    result = infer_pairwise_delta_progress(
        inference=inference,
        target_demo_path=args.target_demo,
        i=args.i,
        j=args.j,
        reference_demo_path=args.reference_demo,
        task_desc=args.task_desc,
        reference_views=reference_views,
        target_views=target_views,
        num_ref_frames=num_ref_frames,
        ref_jitter=ref_jitter,
    )
    print(f"Delta Progress: {result}")


if __name__ == "__main__":
    main()

