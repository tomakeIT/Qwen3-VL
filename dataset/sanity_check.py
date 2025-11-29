#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据 sanity check 脚本，用来在训练前排查潜在“坏样本”。

检查内容（针对 build_qwen_dataset.py 生成的 JSON）：
1. image 路径是否存在 / 是否能被正常打开（PIL.verify）
2. human 部分 `<image>` 占位符数量是否与 `image` 列表长度一致
3. conversations 结构是否为 [human, gpt] 且字段完整
4. gpt 的 `value` 是否能解析出整数标签（-100 ~ 100）

用法示例：
  python dataset/sanity_check.py \\
    --json /path/to/train.json \\
    --data_root /home/lightwheel/erdao.liang/Qwen3-VL
"""

import argparse
import json
import os
from typing import Any, Dict, List, Optional, Set

from tqdm import tqdm
from utils.data_formatting import parse_delta_progress_int

try:
    from PIL import Image
except Exception:  # Pillow 未安装时，只做存在性检查
    Image = None


def check_sample(
    idx: int,
    sample: Dict[str, Any],
    data_root: Optional[str],
) -> List[str]:
    """对单个样本做多项检查，返回错误信息列表（空列表表示通过）。"""
    errors: List[str] = []

    images = sample.get("image", [])
    if isinstance(images, str):
        images = [images]
    if not isinstance(images, list):
        errors.append("`image` 字段类型不是 list/str")
        images = []

    conversations = sample.get("conversations", [])
    if not isinstance(conversations, list) or len(conversations) == 0:
        errors.append("`conversations` 为空或类型错误")
        return errors

    # 1) conversations 结构 & human/gpt 顺序
    if len(conversations) < 2:
        errors.append("conversations 长度 < 2，期望 [human, gpt]")
    else:
        if conversations[0].get("from") != "human":
            errors.append("conversations[0].from 不是 'human'")
        if conversations[1].get("from") != "gpt":
            errors.append("conversations[1].from 不是 'gpt'")

    # 2) `<image>` 占位符数量与 image 数量是否一致
    num_placeholders = 0
    for conv in conversations:
        if conv.get("from") == "human":
            text = str(conv.get("value", ""))
            num_placeholders += text.count("<image>")
    if num_placeholders != len(images):
        errors.append(
            f"<image> 占位符数量({num_placeholders}) 与 image 列表长度({len(images)}) 不一致"
        )

    # 3) gpt 的 value 能否解析出整数标签
    gpt_values = [
        str(c.get("value", ""))
        for c in conversations
        if c.get("from") == "gpt"
    ]
    if not gpt_values:
        errors.append("找不到 gpt 回复（from='gpt'）")
    else:
        label_int = parse_delta_progress_int(gpt_values[0])
        if label_int is None:
            errors.append(f"gpt value 无法解析为整数标签: {gpt_values[0]!r}")

    # 4) 图像存在性 & 能否被打开
    if data_root is not None:
        for rel_path in images:
            abs_path = os.path.join(data_root, rel_path)
            if not os.path.exists(abs_path):
                errors.append(f"图像不存在: {abs_path}")
                continue
            if Image is not None:
                try:
                    with Image.open(abs_path) as im:
                        im.verify()
                except Exception as e:  # noqa: BLE001
                    errors.append(f"图像无法打开或已损坏: {abs_path} ({e})")

    return errors


def main() -> None:
    parser = argparse.ArgumentParser(
        description="检查 JSON 数据中的样本结构与图像文件是否正常"
    )
    parser.add_argument(
        "--json",
        type=str,
        required=True,
        help="build_qwen_dataset.py 生成的 JSON 数据文件路径",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="/home/lightwheel/erdao.liang/LightwheelData",
        help="图片相对路径对应的根目录（通常是 YAML 里的 root）",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="最多检查多少条样本（默认检查全部）",
    )
    parser.add_argument(
        "--only-images",
        action="store_true",
        help="只检查图像是否能被打开：从 JSON 中收集所有 image 路径，去重后逐个 verify()",
    )
    args = parser.parse_args()

    with open(args.json, "r", encoding="utf-8") as f:
        data: List[Dict[str, Any]] = json.load(f)

    # 模式一：只检查所有图像文件是否正常
    if args.only_images:
        if args.data_root is None:
            raise ValueError("--only-images 模式下必须提供 --data_root")

        # 收集所有 image 相对路径并去重
        rel_paths: Set[str] = set()
        for sample in data:
            imgs = sample.get("image", [])
            if isinstance(imgs, str):
                imgs = [imgs]
            if isinstance(imgs, list):
                for p in imgs:
                    if isinstance(p, str):
                        rel_paths.add(p)

        print(f"[INFO] Collected {len(rel_paths)} unique image paths from JSON.")

        bad_files: List[str] = []
        total_imgs = len(rel_paths)
        for rel in tqdm(
            sorted(rel_paths),
            desc="Checking images",
            total=total_imgs,
        ):
            abs_path = os.path.join(args.data_root, rel)
            if not os.path.exists(abs_path):
                tqdm.write(f"[BAD] Missing image: {abs_path}")
                bad_files.append(f"missing::{abs_path}")
                continue
            if Image is not None:
                try:
                    with Image.open(abs_path) as im:
                        im.verify()
                except Exception as e:  # noqa: BLE001
                    tqdm.write(
                        f"[BAD] Corrupted image: {abs_path} ({e})"
                    )
                    bad_files.append(f"corrupted::{abs_path}")

        print(
            f"[INFO] Image-only check done. {len(bad_files)} problematic images out of {total_imgs}."
        )
        if bad_files:
            print("[INFO] Example problematic images (up to 50):")
            for f in bad_files[:50]:
                print("  -", f)
        return

    # 模式二：原来的按样本逐条综合检查
    total = len(data)
    if args.max_samples is not None:
        total = min(total, args.max_samples)

    print(f"[INFO] Loaded {len(data)} samples, will check first {total} samples.")
    bad_indices: List[int] = []
    bad_details: Dict[int, List[str]] = {}

    for idx, sample in enumerate(data[:total]):
        errs = check_sample(idx, sample, args.data_root)
        if errs:
            bad_indices.append(idx)
            bad_details[idx] = errs

    print(f"[INFO] Checked {total} samples, found {len(bad_indices)} bad samples.")
    if bad_indices:
        print("[INFO] Example bad samples (up to 20):")
        for i in bad_indices[:20]:
            print(f"  - idx={i}:")
            for msg in bad_details[i]:
                print(f"      * {msg}")


if __name__ == "__main__":
    main()


