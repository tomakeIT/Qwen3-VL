#!/usr/bin/env python3
import json
import re
import argparse

def main(args):
    with open(args.json, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"[INFO] Loaded {len(data)} samples from {args.json}")

    bad_indices = []
    for idx, sample in enumerate(data):
        images = sample.get("images", [])
        convs = sample.get("conversations", [])
        text = "".join(c.get("value", "") for c in convs)

        num_images = len(images)
        num_placeholders = text.count("<image>")

        if num_placeholders > num_images:
            print(
                f"[BAD] idx={idx}, <image>={num_placeholders}, "
                f"images={num_images}"
            )
            bad_indices.append(idx)

    print(f"\n[INFO] Found {len(bad_indices)} bad samples.")

    # 可选：导出一个干净版 JSON（只保留好的样本）
    if args.save_clean and bad_indices:
        good_data = [
            s for i, s in enumerate(data) if i not in bad_indices
        ]
        out_path = args.output or (args.json.replace(".json", "_clean.json"))
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(good_data, f, ensure_ascii=False, indent=2)
        print(f"[INFO] Saved cleaned dataset with {len(good_data)} samples to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=str, required=True)
    parser.add_argument("--save-clean", action="store_true")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    main(args)
