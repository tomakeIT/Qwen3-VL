#!/usr/bin/env python3
from pathlib import Path

QWEN_DATASET_PATH = Path("/data/local/erdao.liang/LightwheelData/qwen_libero_piper_all")
ORIGIN_DATA_PATH = Path("/data/local/erdao.liang/LightwheelData/1W_Libero_Piper")
SPLITS = ("train", "eval")


def collect_configs():
    configs = {}
    for split in SPLITS:
        split_dir = QWEN_DATASET_PATH / split
        if not split_dir.exists():
            continue
        for json_path in sorted(split_dir.glob("*.json")):
            key = json_path.stem
            if key in configs:
                key = f"{key}_{split}"
            if key in configs:
                raise ValueError(f"Duplicate key even after suffix: {key}")
            configs[key] = {
                "annotation_path": str(json_path),
                "data_path": ORIGIN_DATA_PATH,
            }
    return configs


def emit(configs):
    print("{")
    for name, cfg in configs.items():
        print(f'    "{name}": {{')
        print(f'        "annotation_path": "{cfg["annotation_path"]}",')
        print(f'        "data_path": "{cfg["data_path"]}"')
        print("    },")
    print("}")


if __name__ == "__main__":
    emit(collect_configs())