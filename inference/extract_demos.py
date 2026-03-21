"""
从验证集 JSON 中提取 demo 列表。
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Optional


def extract_demos(eval_json_path: str, data_root: str, max_demos: Optional[int] = None) -> Dict[str, Dict[str, str]]:
    with open(eval_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    demo_paths = set()
    for item in data:
        if "image" not in item:
            continue
        for img_path in item["image"]:
            parts = img_path.split("/")
            if len(parts) < 3:
                continue
            task_name = parts[1]
            demo_id = parts[2]
            demo_path = os.path.join(data_root, parts[0], task_name, demo_id)
            demo_paths.add((demo_id, demo_path))

    demo_dict: Dict[str, str] = {}
    for idx, (demo_id, demo_path) in enumerate(sorted(demo_paths)):
        if max_demos is not None and idx >= max_demos:
            break
        demo_dict[demo_id] = demo_path
    return {"eval": demo_dict}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="从验证集 JSON 中提取 demo 列表")
    parser.add_argument("--eval-json", type=str, required=True, help="验证集 JSON 文件路径")
    parser.add_argument("--data-root", type=str, default="/home/lightwheel/erdao.liang/LightwheelData/slowdata/", help="数据根目录")
    parser.add_argument("--output", type=str, required=True, help="输出 demo list JSON 文件路径")
    parser.add_argument("--max-demos", type=int, default=None, help="最多提取的 demo 数量")
    return parser


def main(args: argparse.Namespace) -> None:
    demo_list = extract_demos(
        eval_json_path=args.eval_json,
        data_root=args.data_root,
        max_demos=args.max_demos,
    )
    output_dir = os.path.dirname(args.output) or "."
    os.makedirs(output_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(demo_list, f, indent=2, ensure_ascii=False)
    print(f"output_path: {args.output}")
    print(f"num_eval_demos: {len(demo_list['eval'])}")


if __name__ == "__main__":
    main(build_parser().parse_args())
