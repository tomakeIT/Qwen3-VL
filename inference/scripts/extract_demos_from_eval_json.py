#!/usr/bin/env python3
"""
从验证集 JSON 文件中提取 demo 列表，生成供 eval_curves_from_batch_demos.py 使用的 demo list JSON
"""

import json
import argparse
import os
from collections import defaultdict
from pathlib import Path


def extract_demos_from_eval_json(eval_json_path: str, data_root: str, max_demos: int = None):
    """
    从验证集 JSON 文件中提取所有 demo 路径

    Args:
        eval_json_path: 验证集 JSON 文件路径 (如 ArrangeVegetables_eval.json)
        data_root: 数据根目录 (如 /home/lightwheel/erdao.liang/LightwheelData/slowdata/)
        max_demos: 最多提取的 demo 数量 (None 表示全部)

    Returns:
        dict: {"eval": {demo_name: demo_path, ...}}
    """
    print(f"正在读取验证集文件: {eval_json_path}")

    with open(eval_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 从样本中提取 demo 路径
    demo_paths = set()

    for item in data:
        if 'image' not in item:
            continue

        # 从 image 路径中提取 demo 路径
        # 路径格式: "1W_Robocasa_X7s_More/ArrangeVegetables/ArrangeVegetables_xxx/camera/frame_xxx.png"
        for img_path in item['image']:
            parts = img_path.split('/')
            if len(parts) >= 3:
                # 提取 task_name 和 demo_id
                task_name = parts[1]  # ArrangeVegetables
                demo_id = parts[2]    # ArrangeVegetables_1761706678200653

                # 构建完整路径
                demo_path = os.path.join(data_root, parts[0], task_name, demo_id)
                demo_paths.add((demo_id, demo_path))

    # 转换为字典格式
    demo_dict = {}
    for idx, (demo_id, demo_path) in enumerate(sorted(demo_paths)):
        if max_demos is not None and idx >= max_demos:
            break
        demo_dict[demo_id] = demo_path

    print(f"共提取到 {len(demo_dict)} 个唯一的 demo")

    return {"eval": demo_dict}


def main():
    parser = argparse.ArgumentParser(description="从验证集 JSON 中提取 demo 列表")
    parser.add_argument("--eval-json", type=str, required=True,
                        help="验证集 JSON 文件路径 (如 ArrangeVegetables_eval.json)")
    parser.add_argument("--data-root", type=str,
                        default="/home/lightwheel/erdao.liang/LightwheelData/slowdata/",
                        help="数据根目录")
    parser.add_argument("--output", type=str, required=True,
                        help="输出 demo list JSON 文件路径")
    parser.add_argument("--max-demos", type=int, default=None,
                        help="最多提取的 demo 数量 (默认全部)")

    args = parser.parse_args()

    # 提取 demo 列表
    demo_list = extract_demos_from_eval_json(
        eval_json_path=args.eval_json,
        data_root=args.data_root,
        max_demos=args.max_demos
    )

    # 保存到文件
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(demo_list, f, indent=2, ensure_ascii=False)

    print(f"\nDemo list 已保存至: {args.output}")
    print(f"包含 {len(demo_list['eval'])} 个验证集 demo")


if __name__ == "__main__":
    main()
