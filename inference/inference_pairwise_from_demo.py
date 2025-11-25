"""
demo_path, i, j, reference_demo_path -> inference result
"""

import os
import argparse
import yaml
from typing import List, Dict, Any, Optional
from inferencer import DeltaProgressInference
from utils.data_formatting import build_qwen_messages, compute_delta_progress_label_int
from utils.utils import list_image_files, dict_to_namespace
from utils.frame_sampling import sample_reference_frames_from_demo
from utils.prompt import build_prompt


def build_messages_from_demo(
    target_demo_path: str,
    i: int,
    j: int,
    reference_demo_path: Optional[str],
    task_desc: str,
    target_views: List[str],
    reference_config
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
    

    ref_img_paths, ref_progress_ints = sample_reference_frames_from_demo(
        avg_frames=reference_config.avg_frames,
        min_frames=reference_config.frames_min,
        max_frames=reference_config.frames_max,
        std=reference_config.frames_std,
        reference_demo_path=reference_demo_path,
        reference_views=reference_config.views,
        ref_jitter=reference_config.jitter,
    )
    
    img_paths, human_str = build_prompt(
        ref_img_paths=ref_img_paths,
        ref_progress_ints=ref_progress_ints,
        target_img_paths_t1=target_paths_t1,
        target_img_paths_t2=target_paths_t2,
        reference_view_names=reference_config.views,
        target_view_names=target_views,
        task_desc=task_desc,
    )

    messages = build_qwen_messages(human_str, img_paths)
    return messages


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=str, default="models/Qwen-VL-2B-Instruct", help="基础模型路径")
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
    
    target_views = config.sampling.required_views
    inference = DeltaProgressInference(
        base_model_path=args.base_model,
        adapter_path=args.adapter,
    )
    messages = build_messages_from_demo(
        target_demo_path=args.target_demo,
        i=args.i,
        j=args.j,
        reference_demo_path=args.reference_demo,
        task_desc=args.task_desc,
        target_views=target_views,
        reference_config=config.reference,
    )
    predicted_delta_progress = inference.infer_from_messages(messages)

    T = min(len(list_image_files(os.path.join(args.target_demo, v))) for v in target_views)
    gt_delta_progress = compute_delta_progress_label_int(args.i, args.j, T)
    print(f"Predicted Delta Progress: {predicted_delta_progress}")
    print(f"Ground Truth Delta Progress: {gt_delta_progress} (only if it is a successful demo)")


if __name__ == "__main__":
    main()

