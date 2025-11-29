"""
批量推理多个demo的整条progress curve，并计算curve level的evaluation指标
定位：inference/eval whole progress curves from validation demo list
"""

import os
import argparse
import yaml
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from types import SimpleNamespace
from typing import List, Optional, Dict, Any, Tuple
from tqdm import tqdm

from inferencer import DeltaProgressInference
from utils.utils import list_image_files, dict_to_namespace
from inference_curve_from_demo import infer_progress_curve


def calc_correlation(pred: np.ndarray, gt: np.ndarray) -> Tuple[float, float]:
    """Calculate Pearson and Spearman correlation coefficients.
    
    Args:
        pred: Predicted sequence.
        gt: Ground truth sequence.
        
    Returns:
        Tuple of (pearson, spearman) correlation coefficients.
    """
    if pred.size < 2 or np.allclose(pred, pred[0]):
        return 0.0, 0.0
    pearson, _ = pearsonr(pred, gt)
    spearman, _ = spearmanr(pred, gt)
    return float(np.nan_to_num(pearson)), float(np.nan_to_num(spearman))


def calc_total_variation(pred: np.ndarray) -> Tuple[float, float]:
    """Calculate total variation and normalized total variation.
    
    Args:
        pred: Predicted sequence.
        
    Returns:
        Tuple of (total_variation, normalized_total_variation).
    """
    if pred.size < 2:
        return 0.0, 0.0
    diffs = np.abs(np.diff(pred))
    tv = float(np.sum(diffs))
    denom = float(abs(pred[-1] - pred[0]))
    norm_tv = tv / denom if denom > 1e-8 else 0.0
    return tv, norm_tv


def calc_monotonicity_rate(pred: np.ndarray) -> float:
    """Calculate the percentage of timesteps where progress increases.
    
    Args:
        pred: Predicted sequence.
    
    Returns:
        A float between 0.0 and 1.0 representing the fraction of timesteps
        where the predicted sequence increases.
    """
    if pred.size < 2:
        return 0.0
    diffs = np.diff(pred)
    increasing_steps = np.sum(diffs > 0)
    return float(increasing_steps / len(diffs))


def compute_ground_truth_curve(frame_indices: np.ndarray) -> np.ndarray:
    """计算ground truth progress curve（0-100的直线）
    
    Args:
        frame_indices: 帧索引数组
        
    Returns:
        Ground truth progress值数组（0-100）
    """
    if frame_indices.size == 0:
        return np.array([])
    if frame_indices.size == 1:
        return np.array([0.0])
    
    # 线性插值从0到100
    min_frame = float(frame_indices[0])
    max_frame = float(frame_indices[-1])
    
    if max_frame - min_frame < 1e-8:
        return np.zeros_like(frame_indices, dtype=np.float32)
    
    # 线性映射：frame -> progress (0-100)
    progress = ((frame_indices - min_frame) / (max_frame - min_frame)) * 100.0
    return progress.astype(np.float32)


def evaluate_curves(
    inference: DeltaProgressInference,
    demo_list: List[Dict[str, Any]],
    reference_demo_path: Optional[str],
    task_desc: str,
    target_views: List[str],
    reference_config: SimpleNamespace,
    step_interval: int = 1,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
) -> Tuple[Dict[str, float], List[Tuple[np.ndarray, np.ndarray]]]:
    """批量推理多个demo的progress curve并计算评估指标
    
    Args:
        inference: DeltaProgressInference实例
        demo_list: demo列表，每个元素是 {"target_demo": path} 或字符串路径
        reference_demo_path: 全局reference demo路径（如果demo_list中没有指定）
        task_desc: 任务描述
        target_views: target视角列表
        reference_config: reference配置
        step_interval: 采样间隔
        start_frame: 起始帧
        end_frame: 结束帧
        
    Returns:
        包含平均指标的字典和所有curve数据列表（每个元素为(frame_indices, progress_values, T)）
    """
    pearsons, spearmans = [], []
    norm_total_vars = []
    monotonicity_rates = []
    all_curves = []
    
    print(f"正在批量推理 {len(demo_list)} 个demo的progress curve...")
    
    for demo_item in tqdm(demo_list, desc="推理progress curves"):
        # 解析demo路径
        if isinstance(demo_item, str):
            target_demo_path = demo_item
            demo_reference_demo_path = reference_demo_path
        elif isinstance(demo_item, dict):
            target_demo_path = demo_item.get("target_demo", demo_item.get("demo", ""))
            demo_reference_demo_path = demo_item.get("reference_demo", reference_demo_path)
        else:
            print(f"警告：跳过无效的demo项: {demo_item}")
            continue
        
        if not target_demo_path or not os.path.exists(target_demo_path):
            print(f"警告：demo路径不存在，跳过: {target_demo_path}")
            continue
        
        try:
            # 计算T值（总帧数）
            T = min(len(list_image_files(os.path.join(target_demo_path, v))) for v in target_views)
            
            # 推理progress curve
            frame_indices, progress_values = infer_progress_curve(
                inference=inference,
                target_demo_path=target_demo_path,
                reference_demo_path=demo_reference_demo_path,
                task_desc=task_desc,
                target_views=target_views,
                reference_config=reference_config,
                step_interval=step_interval,
                start_frame=start_frame,
                end_frame=end_frame,
            )
            
            if frame_indices.size == 0 or progress_values.size == 0:
                print(f"警告：demo {target_demo_path} 没有有效的progress值，跳过")
                continue
            
            # 计算ground truth curve（0-100的直线）
            gt_progress = compute_ground_truth_curve(frame_indices)
            
            if gt_progress.size != progress_values.size:
                print(f"警告：demo {target_demo_path} 的frame_indices和progress_values长度不匹配，跳过")
                continue
            
            # 计算指标
            pearson, spearman = calc_correlation(progress_values, gt_progress)
            _, norm_tv = calc_total_variation(progress_values)
            monotonicity_rate = calc_monotonicity_rate(progress_values)
            
            pearsons.append(pearson)
            spearmans.append(spearman)
            norm_total_vars.append(norm_tv)
            monotonicity_rates.append(monotonicity_rate)
            all_curves.append((frame_indices, progress_values, T))
            
        except Exception as e:
            print(f"警告：处理demo {target_demo_path} 时出错: {e}")
            continue
    
    # 计算平均指标
    def _safe_mean(values: List[float]) -> float:
        return float(np.mean(values)) if len(values) > 0 else 0.0
    
    return {
        "pearson": _safe_mean(pearsons),
        "spearman": _safe_mean(spearmans),
        "norm_total_variation": _safe_mean(norm_total_vars),
        "monotonicity_rate": _safe_mean(monotonicity_rates),
        "num_valid_demos": len(pearsons),
    }, all_curves


def main(args):

    # 加载配置
    with open(args.config, "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)
    config = dict_to_namespace(config_dict)
    
    # 加载demo列表
    print(f"正在加载demo列表: {args.demo_list}")
    with open(args.demo_list, "r", encoding="utf-8") as f:
        demo_list = json.load(f)

    demo_list = list(demo_list["eval"].values())
    
    print(f"共 {len(demo_list)} 个demo")
    
    # 加载模型
    inference = DeltaProgressInference(
        base_model_path=args.base_model,
        adapter_path=args.adapter,
    )
    
    # 批量推理并计算指标
    target_views = config.sampling.required_views
    metrics, all_curves = evaluate_curves(
        inference=inference,
        demo_list=demo_list,
        reference_demo_path=args.reference_demo,
        task_desc=args.task_desc,
        target_views=target_views,
        reference_config=config.reference,
        step_interval=args.step_interval,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
    )
    
    # 绘制所有curve
    if all_curves:
        plt.figure(figsize=(12, 8))
        for frame_indices, progress_values, T in all_curves:
            normalized_frames = (frame_indices - frame_indices[0]) / (frame_indices[-1] - frame_indices[0]) if len(frame_indices) > 1 and frame_indices[-1] != frame_indices[0] else np.linspace(0, 1, len(frame_indices))
            plt.plot(normalized_frames, progress_values, alpha=0.6, linewidth=1, label=f"T={T}")
        plt.xlabel("Normalized Frame Index")
        plt.ylabel("Progress (%)")
        plt.title(f"All Progress Curves (n={len(all_curves)})")
        plt.legend(loc='best', fontsize=8, ncol=2)
        plt.grid(True, alpha=0.3)
        plot_path = args.plot_output if args.plot_output else "all_curves.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\n所有curve图已保存到: {plot_path}")
    
    # 打印结果
    print("\n" + "="*50)
    print("评估结果:")
    print(f"有效demo数量: {metrics['num_valid_demos']}")
    print(f"Pearson Correlation: {metrics['pearson']:.4f}")
    print(f"Spearman Correlation: {metrics['spearman']:.4f}")
    print(f"Normalized Total Variation: {metrics['norm_total_variation']:.4f}")
    print(f"Monotonicity Rate: {metrics['monotonicity_rate']:.4f}")
    print("="*50)
    
    # 可选：保存结果
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存到: {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="批量推理多个demo的progress curve并计算评估指标")
    parser.add_argument("--base-model", type=str, default="models/Qwen-VL-2B-Instruct", help="基础模型路径")
    parser.add_argument("--adapter", type=str, required=True, help="LoRA适配器路径")
    parser.add_argument("--demo-list", type=str, required=True, help="validation demo列表文件路径（JSON格式）")
    parser.add_argument("--reference-demo", type=str, default=None, help="全局reference demo路径（如果demo列表中没有指定）")
    parser.add_argument("--task-desc", type=str, required=True, help="任务描述")
    parser.add_argument("--config", type=str, required=True, help="YAML配置文件路径")
    parser.add_argument("--step-interval", type=int, default=1, help="采样间隔")
    parser.add_argument("--start-frame", type=int, default=0, help="起始帧")
    parser.add_argument("--end-frame", type=int, default=None, help="结束帧")
    parser.add_argument("--output", type=str, default=None, help="可选：保存结果到JSON文件")
    parser.add_argument("--plot-output", type=str, default="./curves.png", help="可选：保存curve图路径")
    args = parser.parse_args()
    
    main(args)

