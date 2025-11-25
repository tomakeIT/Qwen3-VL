"""
批量推理 data_samples 并计算 MSE 指标
定位：inference/eval pair-wise from datasamples file
"""

import json
import argparse
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm
import numpy as np

from inferencer import DeltaProgressInference
from utils.data_formatting import data_sample_to_messages_and_answer


def evaluate_data_samples(
    inference: DeltaProgressInference,
    data_samples: List[Dict[str, Any]],
    data_root: str,
    batch_size: int = 8,
    max_new_tokens: int = 128,
) -> Tuple[List[Optional[int]], List[Optional[int]], float]:

    # 转换所有data_samples为messages格式
    messages_list = []
    ground_truths = []
    
    print("正在转换data_samples为messages格式...")
    for data_sample in tqdm(data_samples, desc="转换格式"):
        messages, gt_delta_progress = data_sample_to_messages_and_answer(
            data_sample, data_root=data_root
        )
        messages_list.append(messages)
        ground_truths.append(gt_delta_progress)
    
    # 批量推理
    predictions = []
    print(f"正在批量推理（batch_size={batch_size}）...")
    for i in tqdm(range(0, len(messages_list), batch_size), desc="批量推理"):
        batch_messages = messages_list[i:i + batch_size]
        batch_predictions = inference.infer_from_messages_batch(
            batch_messages, max_new_tokens=max_new_tokens
        )
        predictions.extend(batch_predictions)
    
    # 计算MSE（只计算有效样本）
    valid_pairs = [
        (p, gt) for p, gt in zip(predictions, ground_truths)
        if p is not None and gt is not None
    ]
    
    if valid_pairs:
        valid_preds, valid_gts = zip(*valid_pairs)
        mse = np.square(np.array(valid_preds) - np.array(valid_gts)).mean()
    else:
        mse = float('inf')
    
    return predictions, ground_truths, mse


def main(args):
    # 加载模型
    inference = DeltaProgressInference(
        base_model_path=args.base_model,
        adapter_path=args.adapter,
    )
    
    # 加载data_samples
    print(f"正在加载data_samples: {args.data_samples}")
    with open(args.data_samples, "r", encoding="utf-8") as f:
        data_samples = json.load(f)
    
    print(f"共 {len(data_samples)} 个样本")
    
    # 批量推理并计算MSE
    predictions, ground_truths, mse = evaluate_data_samples(
        inference=inference,
        data_samples=data_samples,
        data_root=args.data_root,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )
    
    # 统计信息
    valid_count = sum(1 for p, gt in zip(predictions, ground_truths) if p is not None and gt is not None)
    total_count = len(predictions)
    
    print("\n" + "="*50)
    print("评估结果:")
    print(f"总样本数: {total_count}")
    print(f"有效样本数: {valid_count}")
    print(f"MSE: {mse:.4f}")
    print("="*50)
    
    # 可选：保存结果
    if args.output:
        results = [
            {
                "predicted_delta_progress": pred,
                "ground_truth_delta_progress": gt,
            }
            for pred, gt in zip(predictions, ground_truths)
        ]
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump({
                "mse": mse,
                "valid_count": valid_count,
                "total_count": total_count,
                "results": results
            }, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存到: {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="批量推理data_samples并计算MSE")
    parser.add_argument("--base-model", type=str, default="models/Qwen-VL-2B-Instruct", help="基础模型路径")
    parser.add_argument("--adapter", type=str, required=True, help="LoRA适配器路径")
    parser.add_argument("--data-samples", type=str, required=True, help="data sample json路径")
    parser.add_argument("--data-root", type=str, default="", help="data sample里所有图片路径的根目录")
    parser.add_argument("--batch-size", type=int, default=32, help="批处理大小")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="最大生成token数")
    parser.add_argument("--output", type=str, default=None, help="可选：保存预测结果到JSON文件")
    
    args = parser.parse_args()
    main(args)

