#!/usr/bin/env python3
import json
import os
import re
from collections import Counter
import matplotlib.pyplot as plt

def extract_frame_num(img_path):
    """从路径中提取帧号，如 frame_000003.png -> 3"""
    match = re.search(r'frame_(\d+)\.png', img_path)
    return int(match.group(1)) if match else None

def get_total_frames(base_dir, img_path):
    """从图片路径获取对应目录的总帧数"""
    # 构建完整路径：base_dir + 相对路径（去掉文件名）
    dir_path = os.path.join(base_dir, os.path.dirname(img_path))
    if not os.path.isdir(dir_path):
        return None
    # 统计该目录下的 frame_*.png 文件数量
    frames = [f for f in os.listdir(dir_path)]
    return len(frames)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--json', required=True, help='JSON data samples file')
    parser.add_argument('--base_dir', default="/home/lightwheel/erdao.liang/LightwheelData", help='Base directory for images')
    parser.add_argument('--output', default='frame_indices_stats.png', help='Output plot file')
    args = parser.parse_args()

    with open(args.json, 'r') as f:
        samples = json.load(f)

    i_list, j_list = [], []
    i_norm_list, j_norm_list = [], []
    j_minus_i_list, j_minus_i_norm_list = [], []
    delta_progress_list = []

    for sample in samples:
        images = sample['image']
        if len(images) < 6:
            continue
        
        # 倒数第1-3个是 j (frame_000020.png)
        # 倒数第4-6个是 i (frame_000003.png)
        # 从倒数第4个提取 i，倒数第1个提取 j
        img_i = images[-4]  # 倒数第4个
        img_j = images[-1]  # 倒数第1个
        
        i = extract_frame_num(img_i)
        j = extract_frame_num(img_j)
        
        if i is None or j is None:
            continue
        
        # 获取总帧数 T（直接使用 img_i 所在的目录）
        T = get_total_frames(args.base_dir, img_i)

        if T is None or T == 0:
            continue
        
        # 提取 delta_progress 从 conversations[1]["value"]
        delta_progress = None
        if 'conversations' in sample and len(sample['conversations']) >= 2:
            gpt_value = sample['conversations'][1].get('value', '').strip()
            # 解析 "+35", "-13", "0" 等格式
            try:
                delta_progress = int(gpt_value)
            except ValueError:
                # 如果解析失败，跳过这个样本
                continue
        
        if delta_progress is None:
            continue
        
        i_list.append(i)
        j_list.append(j)
        i_norm_list.append(i / T)
        j_norm_list.append(j / T)
        j_minus_i_list.append(j - i)
        j_minus_i_norm_list.append((j - i) / T)
        delta_progress_list.append(delta_progress)

    # 统计并画图
    fig, axes = plt.subplots(3, 2, figsize=(12, 15))
    
    # i 和 j 的统计
    i_counter = Counter(i_list)
    j_counter = Counter(j_list)
    
    axes[0, 0].bar(i_counter.keys(), i_counter.values())
    axes[0, 0].set_xlabel('i')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title(f'Occurrences of i (Total {len(i_list)} samples)')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].bar(j_counter.keys(), j_counter.values())
    axes[0, 1].set_xlabel('j')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title(f'Occurrences of j (Total {len(j_list)} samples)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Statistics for normalized values
    axes[1, 0].hist(i_norm_list, bins=50, edgecolor='black', alpha=0.7)
    axes[1, 0].set_xlabel('i/T')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('Distribution of Normalized i/T')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].hist(j_norm_list, bins=50, edgecolor='black', alpha=0.7)
    axes[1, 1].set_xlabel('j/T')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('Distribution of Normalized j/T')
    axes[1, 1].grid(True, alpha=0.3)
    
    # j-i 和 (j-i)/T 的统计
    j_minus_i_counter = Counter(j_minus_i_list)
    axes[2, 0].bar(j_minus_i_counter.keys(), j_minus_i_counter.values())
    axes[2, 0].set_xlabel('j-i')
    axes[2, 0].set_ylabel('Count')
    axes[2, 0].set_title(f'Occurrences of j-i (Total {len(j_minus_i_list)} samples)')
    axes[2, 0].grid(True, alpha=0.3)
    
    axes[2, 1].hist(j_minus_i_norm_list, bins=50, edgecolor='black', alpha=0.7)
    axes[2, 1].set_xlabel('(j-i)/T')
    axes[2, 1].set_ylabel('Count')
    axes[2, 1].set_title('Distribution of Normalized (j-i)/T')
    axes[2, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    
    # 创建新图：散点图
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))
    
    # i/T vs j/T
    axes2[0].scatter(i_norm_list, j_norm_list, alpha=0.5, s=20)
    axes2[0].set_xlabel('i/T')
    axes2[0].set_ylabel('j/T')
    axes2[0].set_title('i/T vs j/T')
    axes2[0].grid(True, alpha=0.3)
    
    # i/T vs delta_progress
    axes2[1].scatter(i_norm_list, delta_progress_list, alpha=0.5, s=20)
    axes2[1].set_xlabel('i/T')
    axes2[1].set_ylabel('delta_progress')
    axes2[1].set_title('i/T vs delta_progress')
    axes2[1].grid(True, alpha=0.3)
    
    # j/T vs delta_progress
    axes2[2].scatter(j_norm_list, delta_progress_list, alpha=0.5, s=20)
    axes2[2].set_xlabel('j/T')
    axes2[2].set_ylabel('delta_progress')
    axes2[2].set_title('j/T vs delta_progress')
    axes2[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    scatter_output = args.output.replace('.png', '_scatter.png')
    plt.savefig(scatter_output, dpi=150)
    
    print(f"Statistics completed! Processed {len(i_list)} samples in total")
    print(f"i range: {min(i_list)} - {max(i_list)}, j range: {min(j_list)} - {max(j_list)}")
    print(f"j-i range: {min(j_minus_i_list)} - {max(j_minus_i_list)}")
    print(f"i/T range: {min(i_norm_list):.3f} - {max(i_norm_list):.3f}")
    print(f"j/T range: {min(j_norm_list):.3f} - {max(j_norm_list):.3f}")
    print(f"(j-i)/T range: {min(j_minus_i_norm_list):.3f} - {max(j_minus_i_norm_list):.3f}")
    print(f"delta_progress range: {min(delta_progress_list)} - {max(delta_progress_list)}")
    print(f"Plot saved to: {args.output}")
    print(f"Scatter plot saved to: {scatter_output}")

if __name__ == '__main__':
    main()

