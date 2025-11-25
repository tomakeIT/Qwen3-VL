"""
临时脚本：将L90L6PutTheWhiteMugOnThePlate文件夹分成85%训练和15%评估
输出文件夹名到绝对路径的索引
"""

import os
import json
from pathlib import Path

def create_train_eval_split(base_dir, train_ratio=0.85):
    """
    创建训练/评估分割
    
    Args:
        base_dir: 基础目录路径
        train_ratio: 训练集比例（默认0.85）
    
    Returns:
        dict: 包含train和eval的索引字典
    """
    base_path = Path(base_dir)
    
    # 获取所有子文件夹（只包含目录）
    subdirs = [d for d in base_path.iterdir() if d.is_dir()]
    
    # 按名称排序以确保一致性
    subdirs.sort(key=lambda x: x.name)
    
    total = len(subdirs)
    train_count = int(total * train_ratio)
    
    train_dirs = subdirs[:train_count]
    eval_dirs = subdirs[train_count:]
    
    # 创建索引：文件夹名 -> 绝对路径
    train_index = {d.name: str(d.absolute()) for d in train_dirs}
    eval_index = {d.name: str(d.absolute()) for d in eval_dirs}
    
    return {
        "train": train_index,
        "eval": eval_index,
        "stats": {
            "total": total,
            "train_count": len(train_dirs),
            "eval_count": len(eval_dirs),
            "train_ratio": len(train_dirs) / total if total > 0 else 0,
            "eval_ratio": len(eval_dirs) / total if total > 0 else 0
        }
    }


def main():
    # 基础目录
    base_dir = "/home/lightwheel/erdao.liang/LightwheelData/1W_Libero_Piper/L90L6PutTheWhiteMugOnThePlate"
    
    # 创建分割
    split = create_train_eval_split(base_dir, train_ratio=0.85)
    
    # 输出结果
    output_file = "train_eval_split_index.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(split, f, indent=2, ensure_ascii=False)
    
    print(f"分割完成！")
    print(f"总文件夹数: {split['stats']['total']}")
    print(f"训练集: {split['stats']['train_count']} ({split['stats']['train_ratio']*100:.1f}%)")
    print(f"评估集: {split['stats']['eval_count']} ({split['stats']['eval_ratio']*100:.1f}%)")
    print(f"\n索引已保存到: {output_file}")
    
    # 打印前几个示例
    print("\n训练集示例（前5个）:")
    for i, (name, path) in enumerate(list(split['train'].items())[:5]):
        print(f"  {name} -> {path}")
    
    print("\n评估集示例（前5个）:")
    for i, (name, path) in enumerate(list(split['eval'].items())[:5]):
        print(f"  {name} -> {path}")


if __name__ == "__main__":
    main()

