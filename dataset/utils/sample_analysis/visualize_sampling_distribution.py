#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可视化不同 decay_factor 下 50 次采样的实际分布
"""

import matplotlib.pyplot as plt
import numpy as np
import random
from collections import Counter
from typing import List, Tuple, Optional

def sample_pair_indices(T: int, max_delta_t: int, min_delta_t: int = 1, decay_factor: float = 0.8) -> Optional[Tuple[int, int]]:
    """在给定长度为 T 的帧序列里，按距离衰减分布随机采样 (i, j)"""
    if T < 2:
        return None
    
    i = random.randint(0, T - 2)
    
    candidates: List[Tuple[int, float]] = []
    
    # 正向采样
    max_forward = min(max_delta_t, T - 1 - i)
    for dt in range(min_delta_t, max_forward + 1):
        weight = decay_factor ** (dt - min_delta_t)
        candidates.append((dt, weight))
    
    # 反向采样
    max_backward = min(max_delta_t, i)
    for dt in range(min_delta_t, max_backward + 1):
        weight = decay_factor ** (dt - min_delta_t)
        candidates.append((-dt, weight))
    
    if not candidates:
        return None
    
    deltas, weights = zip(*candidates)
    total_weight = sum(weights)
    if total_weight == 0:
        return None
    
    r = random.uniform(0, total_weight)
    cumsum = 0
    for delta_t, weight in candidates:
        cumsum += weight
        if r <= cumsum:
            j = i + delta_t
            return i, j
    
    delta_t = candidates[0][0]
    j = i + delta_t
    return i, j

def simulate_sampling(num_samples: int, T: int, max_delta_t: int, min_delta_t: int, decay_factor: float) -> List[int]:
    """模拟多次采样，返回采样到的距离列表"""
    distances = []
    for _ in range(num_samples):
        pair = sample_pair_indices(T, max_delta_t, min_delta_t, decay_factor)
        if pair is not None:
            i, j = pair
            distances.append(abs(j - i))
    return distances

# 配置参数
T = 100  # 假设序列长度为 100
max_delta_t = 40
min_delta_t = 2
num_samples = 50
decay_factors = [0.5, 0.6, 0.7, 0.8, 0.9]

# 设置随机种子以便复现
random.seed(42)
np.random.seed(42)

# 创建图表
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# 子图1：采样分布直方图
ax1 = axes[0]
bins = range(min_delta_t, max_delta_t + 2)
width = 0.15
x_positions = np.arange(min_delta_t, max_delta_t + 1)

for idx, decay_factor in enumerate(decay_factors):
    distances = simulate_sampling(num_samples, T, max_delta_t, min_delta_t, decay_factor)
    counter = Counter(distances)
    counts = [counter.get(d, 0) for d in range(min_delta_t, max_delta_t + 1)]
    
    x = x_positions + (idx - len(decay_factors) / 2) * width
    ax1.bar(x, counts, width, label=f'decay_factor={decay_factor}', alpha=0.8)

ax1.set_xlabel('Distance (|Δt|)', fontsize=12)
ax1.set_ylabel('Sampling Count', fontsize=12)
ax1.set_title(f'Actual Sampling Distribution (50 samples, min={min_delta_t}, max={max_delta_t})', fontsize=14, fontweight='bold')
ax1.set_xticks(x_positions[::2])
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3, axis='y')

# 子图2：累积分布
ax2 = axes[1]
for decay_factor in decay_factors:
    distances = simulate_sampling(num_samples, T, max_delta_t, min_delta_t, decay_factor)
    counter = Counter(distances)
    cumulative = []
    cumsum = 0
    for d in range(min_delta_t, max_delta_t + 1):
        cumsum += counter.get(d, 0)
        cumulative.append(cumsum)
    
    cumulative_pct = [c / num_samples * 100 for c in cumulative]
    ax2.plot(x_positions, cumulative_pct, marker='o', label=f'decay_factor={decay_factor}', linewidth=2, markersize=4)

ax2.set_xlabel('Distance (|Δt|)', fontsize=12)
ax2.set_ylabel('Cumulative Percentage (%)', fontsize=12)
ax2.set_title('Cumulative Distribution', fontsize=14, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0, 105)

plt.tight_layout()
output_path = '/home/erdao/Documents/LightwheelData/preprocessing/decay_analysis/sampling_distribution_visualization.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"可视化图表已保存到: {output_path}")

# 打印统计信息
print("\n" + "="*70)
print(f"不同 decay_factor 下的采样统计（{num_samples} 次采样，min={min_delta_t}, max={max_delta_t}）")
print("="*70)

for decay_factor in decay_factors:
    distances = simulate_sampling(num_samples, T, max_delta_t, min_delta_t, decay_factor)
    counter = Counter(distances)
    
    print(f"\ndecay_factor = {decay_factor}:")
    print(f"{'Distance':<10} {'Count':<10} {'Percentage':<12} {'Cumulative %':<12}")
    print("-" * 50)
    cumsum = 0
    for d in sorted(counter.keys())[:15]:  # 显示前15个距离
        count = counter[d]
        cumsum += count
        pct = count / num_samples * 100
        cum_pct = cumsum / num_samples * 100
        print(f"{d:<10} {count:<10} {pct:<12.2f} {cum_pct:<12.2f}")
    
    # 统计信息
    mean_dist = np.mean(distances)
    std_dist = np.std(distances)
    print(f"\n  Mean distance: {mean_dist:.2f}, Std: {std_dist:.2f}")
    print(f"  Most frequent: {counter.most_common(1)[0][0]} (appeared {counter.most_common(1)[0][1]} times)")

# 显示图表
plt.show()

