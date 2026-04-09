#!/usr/bin/env python3
"""
从文件夹名中提取任务描述并生成JSON文件
"""
import os
import re
import json
import argparse
from pathlib import Path
from collections import OrderedDict


def fix_common_typos(text):
    """修复常见的拼写错误"""
    fixes = {
        'Closelt': 'CloseIt',
        'Placelt': 'PlaceIt',
        'Pub': 'Put',  # PubBookInLeftCaddy -> PutBookInLeftCaddy
        'Bottm': 'Bottom',  # PubBowlInBottmCabinet -> PutBowlInBottomCabinet
    }
    for typo, correct in fixes.items():
        text = text.replace(typo, correct)
    return text


def camel_to_sentence(camel_str):
    """将驼峰命名转换为正常句子"""
    # 先修复常见拼写错误
    camel_str = fix_common_typos(camel_str)
    
    # 在大写字母前插入空格（除了第一个字符）
    # 处理连续大写字母的情况（如 "ABC" -> "A B C"）
    result = re.sub(r'(?<!^)(?=[A-Z])', ' ', camel_str)
    # 将多个连续空格替换为单个空格
    result = re.sub(r'\s+', ' ', result)
    return result.strip()


def remove_prefix_code(folder_name):
    """移除文件夹名前缀代号"""
    # 常见的代号模式：
    # L10K3, L90K1, L10L1, L90S1 等（字母+数字+字母+数字）
    # Libero10, Libero90, LiberoGoal 等
    # LO, LS, LG 等（两个字母）
    
    # 先尝试匹配常见的代号模式（按优先级排序）
    patterns = [
        r'^L\d+[KLS]\d+',    # L10K3, L90K1, L10L1, L90S1 等
        r'^LiberoGoal',      # LiberoGoal
        r'^Libero\d+',       # Libero10, Libero90 等
        r'^L[OPSG]',         # LO, LS, LP, LG 等
        r'^L\d+',            # L10, L90 等
    ]
    
    for pattern in patterns:
        match = re.match(pattern, folder_name)
        if match:
            return folder_name[match.end():]
    
    return folder_name


def remove_timestamp(folder_name):
    """移除时间戳（下划线后的数字）"""
    # 匹配下划线后的数字（可能是时间戳）
    # 例如: L90L6PutTheWhiteMugOnThePlate_1762235317475676
    parts = folder_name.split('_')
    if len(parts) > 1:
        # 检查最后一部分是否是纯数字（时间戳）
        if parts[-1].isdigit() and len(parts[-1]) > 10:
            return '_'.join(parts[:-1])
    return folder_name


def extract_task_description(folder_name):
    """从文件夹名中提取任务描述"""
    # 1. 移除时间戳
    name = remove_timestamp(folder_name)
    
    # 2. 移除前缀代号
    name = remove_prefix_code(name)
    
    # 3. 如果为空，返回原文件夹名（可能是没有代号的情况）
    if not name:
        name = folder_name.split('_')[0]  # 至少移除时间戳部分
    
    # 4. 移除末尾的数字后缀（如 PutButterInBasket2 -> PutButterInBasket）
    name = re.sub(r'\d+$', '', name)
    
    # 5. 将驼峰命名转换为句子
    sentence = camel_to_sentence(name)
    
    # 6. 转换为小写，然后首字母大写
    sentence = sentence.lower()
    if sentence:
        sentence = sentence[0].upper() + sentence[1:]
    
    return sentence


def scan_directories(base_paths):
    """扫描指定目录下的所有文件夹"""
    all_tasks = OrderedDict()
    
    for base_path in base_paths:
        if not os.path.exists(base_path):
            print(f"警告: 目录不存在: {base_path}")
            continue
        
        print(f"扫描目录: {base_path}")
        for item in os.listdir(base_path):
            item_path = os.path.join(base_path, item)
            if os.path.isdir(item_path):
                # 提取任务描述
                description = extract_task_description(item)
                # 使用原始文件夹名作为key（去除时间戳部分）
                key = remove_timestamp(item)
                
                # 如果已经有相同的key，保留更完整的描述
                if key not in all_tasks or len(description) > len(all_tasks[key]):
                    all_tasks[key] = description
                    print(f"  找到任务: {key} -> {description}")
    
    return all_tasks


def main():
    parser = argparse.ArgumentParser(description="从文件夹名中提取任务描述并生成JSON文件")
    parser.add_argument("--dataset-path", type=str, required=True, help="数据集根目录路径")
    parser.add_argument("--output", type=str, default="task_descriptions.json", help="输出JSON文件路径")
    
    args = parser.parse_args()
    
    print("开始提取任务描述...")
    tasks = scan_directories([args.dataset_path])
    
    sorted_tasks = OrderedDict(sorted(tasks.items()))
    
    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(sorted_tasks, f, indent=2, ensure_ascii=False)
    
    print(f"\n完成! 共提取 {len(sorted_tasks)} 个任务描述")
    print(f"结果已保存到: {output_file}")


if __name__ == "__main__":
    main()

