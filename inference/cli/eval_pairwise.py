"""
批量推理 data_samples 并计算 pairwise MSE。
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from inference.core.io_utils import load_json
from utils.data_formatting import data_sample_to_messages_and_answer


def evaluate_data_samples(
    inference: MultiGPUDeltaProgressInference,
    data_samples: List[Dict[str, Any]],
    data_root: str,
    batch_size: int = 8,
    max_new_tokens: int = 128,
) -> Tuple[List[Optional[int]], List[Optional[int]], float]:
    messages_list = []
    ground_truths = []

    for data_sample in tqdm(data_samples, desc="构建 samples"):
        messages, gt_delta_progress = data_sample_to_messages_and_answer(
            data_sample,
            data_root=data_root,
        )
        messages_list.append(messages)
        ground_truths.append(gt_delta_progress)

    predictions = inference.infer_from_messages_batch(
        messages_list,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        desc="Pairwise evaluation",
    )

    valid_pairs = [
        (pred, gt)
        for pred, gt in zip(predictions, ground_truths)
        if pred is not None and gt is not None
    ]
    if valid_pairs:
        valid_preds, valid_gts = zip(*valid_pairs)
        mse = np.square(np.array(valid_preds) - np.array(valid_gts)).mean()
    else:
        mse = float("inf")
    return predictions, ground_truths, float(mse)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="批量推理 data_samples 并计算 MSE")
    parser.add_argument("--base-model", type=str, default="models/Qwen-VL-2B-Instruct", help="基础模型路径")
    parser.add_argument("--adapter", type=str, required=True, help="LoRA 适配器路径")
    parser.add_argument("--data-samples", type=str, required=True, help="data sample JSON 路径")
    parser.add_argument("--data-root", type=str, default="", help="data sample 中所有图片路径的根目录")
    parser.add_argument("--batch-size", type=int, default=32, help="批处理大小")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="最大生成 token 数")
    parser.add_argument("--output", type=str, default=None, help="可选：保存预测结果到 JSON")
    parser.add_argument("--num-gpus", type=int, default=1, help="使用的 GPU 数量")
    return parser


def main(args: argparse.Namespace) -> None:
    from inference.core.multi_gpu_inferencer import MultiGPUDeltaProgressInference

    inference = MultiGPUDeltaProgressInference(
        base_model_path=args.base_model,
        adapter_path=args.adapter,
        num_gpus=args.num_gpus,
    )
    try:
        data_samples = load_json(args.data_samples)
        predictions, ground_truths, mse = evaluate_data_samples(
            inference=inference,
            data_samples=data_samples,
            data_root=args.data_root,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
        )
    finally:
        inference.close()

    valid_count = sum(1 for pred, gt in zip(predictions, ground_truths) if pred is not None and gt is not None)
    total_count = len(predictions)
    print(f"total_samples: {total_count}")
    print(f"valid_samples: {valid_count}")
    print(f"mse: {mse:.4f}")

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
                "results": results,
            }, f, indent=2, ensure_ascii=False)
        print(f"output_path: {args.output}")


if __name__ == "__main__":
    main(build_parser().parse_args())
