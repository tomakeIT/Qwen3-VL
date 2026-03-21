from __future__ import annotations

"""
对单个demo进行密集采样推理，输出JSON格式的progress数据
采样方式：(0, delta_t), (1, delta_t+1), (2, delta_t+2), ..., (T-1, min(T-1+delta_t, T-1))
始终输出T个点
"""

import os
import json
import argparse
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from types import SimpleNamespace

from common.demo_scan import scan_demo_frames
from common.io_utils import load_config_namespace
from common.messages import build_messages_from_demo


def infer_dense_progress_curve(
    inference: DeltaProgressInference,
    target_demo_path: str,
    reference_demo_path: Optional[str],
    task_desc: str,
    target_views: List[str],
    reference_config: SimpleNamespace,
    delta_t: int,
) -> Dict[str, Any]:
    """
    对target_demo进行密集采样推理，返回JSON格式的数据
    
    Args:
        delta_t: 窗口大小，对于每个i，采样(i, min(i+delta_t, T-1))
    
    Returns:
        包含推理结果的字典
    """
    _, T = scan_demo_frames(target_demo_path, target_views)
    if T < 2:
        raise ValueError(f"Target demo has insufficient frames: T={T}")

    # 初始化结果结构
    result = {
        "demo_name": os.path.basename(target_demo_path),
        "total_frames": T,
        "delta_t": delta_t,
        "target_views": target_views,
        "delta_progress": [],
        "cumulative_progress": []
    }

    current_progress = 0
    delta_progress_list = []
    cumulative_progress_list = []

    # 对每个i从0到T-1进行采样
    for i in tqdm(range(T), desc=f"推理进度 (T={T})"):
        j = min(i + delta_t, T - 1)
        
        # 如果i >= j（即i + delta_t >= T-1的情况），delta_progress设为0，不进行推理
        if i >= j:
            delta_progress = 0
        else:
            # 构建消息并进行推理
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
            
            # 如果推理失败，设为0
            if delta_progress is None:
                delta_progress = 0
        
        # 累积progress
        current_progress += delta_progress
        
        # 记录到列表（始终记录，确保有T个点）
        delta_progress_list.append(delta_progress)
        cumulative_progress_list.append(current_progress)

    result["delta_progress"] = delta_progress_list
    result["cumulative_progress"] = cumulative_progress_list
    return result


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    
    config = load_config_namespace(args.config)
    from inferencer import DeltaProgressInference
    
    # 初始化推理器
    inference = DeltaProgressInference(
        base_model_path=args.base_model,
        adapter_path=args.adapter,
    )
    
    target_views = config.sampling.required_views
    
    # 进行推理
    result = infer_dense_progress_curve(
        inference=inference,
        target_demo_path=args.target_demo,
        reference_demo_path=args.reference_demo,
        task_desc=args.task_desc,
        target_views=target_views,
        reference_config=config.reference,
        delta_t=args.delta_t,
    )
    
    # 保存JSON结果
    demo_name = os.path.basename(args.target_demo)
    output_json_path = os.path.join(args.output_dir, f"{demo_name}.json")
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"结果已保存到: {output_json_path}")
    print(f"共推理 {len(result['delta_progress'])} 个样本")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="对单个demo进行密集采样推理，输出JSON格式的progress数据")
    parser.add_argument("--base-model", type=str, default="models/Qwen-VL-2B-Instruct", help="基础模型路径")
    parser.add_argument("--adapter", type=str, required=True, help="LoRA适配器路径")
    parser.add_argument("--target-demo", type=str, required=True, help="target demo路径")
    parser.add_argument("--reference-demo", type=str, help="reference demo路径")
    parser.add_argument("--task-desc", type=str, required=True, help="任务描述")
    parser.add_argument("--config", type=str, required=True, help="YAML配置文件路径")
    parser.add_argument("--delta-t", type=int, required=True, help="窗口大小，对于每个i，采样(i, min(i+delta_t, T-1))")
    parser.add_argument("--output-dir", type=str, default="outputs/dense_curves", help="输出目录")
    
    args = parser.parse_args()
    main(args)

