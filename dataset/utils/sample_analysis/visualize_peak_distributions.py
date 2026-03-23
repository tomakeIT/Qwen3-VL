#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可视化几种先上升再衰减的分布选项
"""

import matplotlib.pyplot as plt
import numpy as np
import random
from collections import Counter
from typing import List, Tuple, Optional

def calculate_peak_weights(
    max_delta_t: int,
    min_delta_t: int,
    peak_distance: int,
    decay_factor: float = 0.8,
    rise_factor: float = 1.2,
) -> List[Tuple[int, float]]:
    """
    计算山峰分布的权重
    
    Args:
        max_delta_t: 最大距离
        min_delta_t: 最小距离
        peak_distance: 峰值位置
        decay_factor: 衰减因子（峰值右侧）
        rise_factor: 上升因子（峰值左侧，>1 表示上升速度）
    
    Returns:
        [(distance, weight), ...]
    """
    weights = []
    
    for dt in range(min_delta_t, max_delta_t + 1):
        if dt < peak_distance:
            # 上升阶段：从 min_delta_t 到 peak_distance
            # 使用指数上升：weight = rise_factor^(dt - min_delta_t)
            weight = rise_factor ** (dt - min_delta_t)
        elif dt == peak_distance:
            # 峰值位置：权重为 1.0
            weight = 1.0
        else:
            # 衰减阶段：从 peak_distance 到 max_delta_t
            # 使用指数衰减：weight = decay_factor^(dt - peak_distance)
            weight = decay_factor ** (dt - peak_distance)
        
        weights.append((dt, weight))
    
    return weights

def sample_with_peak_distribution(
    T: int,
    max_delta_t: int,
    min_delta_t: int,
    peak_distance: int,
    decay_factor: float = 0.8,
    rise_factor: float = 1.2,
) -> Optional[Tuple[int, int]]:
    """使用山峰分布采样 (i, j)"""
    if T < 2:
        return None
    
    i = random.randint(0, T - 2)
    
    candidates: List[Tuple[int, float]] = []
    
    # 计算权重
    weight_map = dict(calculate_peak_weights(max_delta_t, min_delta_t, peak_distance, decay_factor, rise_factor))
    
    # 正向采样
    max_forward = min(max_delta_t, T - 1 - i)
    for dt in range(min_delta_t, max_forward + 1):
        weight = weight_map[dt]
        candidates.append((dt, weight))
    
    # 反向采样
    max_backward = min(max_delta_t, i)
    for dt in range(min_delta_t, max_backward + 1):
        weight = weight_map[dt]
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

def simulate_sampling(
    num_samples: int,
    T: int,
    max_delta_t: int,
    min_delta_t: int,
    peak_distance: int,
    decay_factor: float = 0.8,
    rise_factor: float = 1.2,
) -> List[int]:
    """模拟多次采样"""
    distances = []
    for _ in range(num_samples):
        pair = sample_with_peak_distribution(T, max_delta_t, min_delta_t, peak_distance, decay_factor, rise_factor)
        if pair is not None:
            i, j = pair
            distances.append(abs(j - i))
    return distances

# 配置参数
T = 100
max_delta_t = 80
min_delta_t = 10
num_samples = 50

# 测试不同的峰值位置和参数组合
test_configs = [
    {"peak_distance": 5, "decay_factor": 0.8, "rise_factor": 1.2, "label": "Peak@5, rise=1.2, decay=0.8"},
    {"peak_distance": 8, "decay_factor": 0.8, "rise_factor": 1.2, "label": "Peak@8, rise=1.2, decay=0.8"},
    {"peak_distance": 10, "decay_factor": 0.8, "rise_factor": 1.2, "label": "Peak@10, rise=1.2, decay=0.8"},
    {"peak_distance": 8, "decay_factor": 0.7, "rise_factor": 1.3, "label": "Peak@8, rise=1.3, decay=0.7"},
    {"peak_distance": 8, "decay_factor": 0.9, "rise_factor": 1.1, "label": "Peak@8, rise=1.1, decay=0.9"},
    {"peak_distance": 3, "decay_factor": 0.8, "rise_factor": 1.3, "label": "Peak@2, rise=1.5, decay=0.8"},
]

random.seed(42)
np.random.seed(42)

# 创建图表
fig, axes = plt.subplots(3, 1, figsize=(14, 12))

# 子图1：权重分布（理论值）
ax1 = axes[0]
x_positions = np.arange(min_delta_t, max_delta_t + 1)

for config in test_configs:
    weights = calculate_peak_weights(
        max_delta_t, min_delta_t,
        config["peak_distance"],
        config["decay_factor"],
        config["rise_factor"],
    )
    distances, weight_vals = zip(*weights)
    # 归一化以便比较
    total = sum(weight_vals)
    normalized = [w / total for w in weight_vals]
    ax1.plot(x_positions, normalized, marker='o', label=config["label"], linewidth=2, markersize=3)

ax1.set_xlabel('Distance (|Δt|)', fontsize=12)
ax1.set_ylabel('Normalized Weight', fontsize=12)
ax1.set_title('Theoretical Weight Distribution (Peak-based)', fontsize=14, fontweight='bold')
ax1.legend(fontsize=9, loc='upper right')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(min_delta_t - 1, max_delta_t + 1)

# 子图2：实际采样分布（直方图）
ax2 = axes[1]
bins = range(min_delta_t, max_delta_t + 2)
width = 0.15
x_positions = np.arange(min_delta_t, max_delta_t + 1)

for idx, config in enumerate(test_configs):
    distances = simulate_sampling(
        num_samples, T, max_delta_t, min_delta_t,
        config["peak_distance"],
        config["decay_factor"],
        config["rise_factor"],
    )
    counter = Counter(distances)
    counts = [counter.get(d, 0) for d in range(min_delta_t, max_delta_t + 1)]
    
    x = x_positions + (idx - len(test_configs) / 2) * width
    ax2.bar(x, counts, width, label=config["label"], alpha=0.7)

ax2.set_xlabel('Distance (|Δt|)', fontsize=12)
ax2.set_ylabel('Sampling Count', fontsize=12)
ax2.set_title(f'Actual Sampling Distribution ({num_samples} samples)', fontsize=14, fontweight='bold')
ax2.set_xticks(x_positions[::3])
ax2.legend(fontsize=8, ncol=1, loc='upper right')
ax2.grid(True, alpha=0.3, axis='y')

# 子图3：累积分布
ax3 = axes[2]
for config in test_configs:
    distances = simulate_sampling(
        num_samples, T, max_delta_t, min_delta_t,
        config["peak_distance"],
        config["decay_factor"],
        config["rise_factor"],
    )
    counter = Counter(distances)
    cumulative = []
    cumsum = 0
    for d in range(min_delta_t, max_delta_t + 1):
        cumsum += counter.get(d, 0)
        cumulative.append(cumsum)
    
    cumulative_pct = [c / num_samples * 100 for c in cumulative]
    ax3.plot(x_positions, cumulative_pct, marker='o', label=config["label"], linewidth=2, markersize=3)

ax3.set_xlabel('Distance (|Δt|)', fontsize=12)
ax3.set_ylabel('Cumulative Percentage (%)', fontsize=12)
ax3.set_title('Cumulative Distribution', fontsize=14, fontweight='bold')
ax3.legend(fontsize=8, ncol=1, loc='lower right')
ax3.grid(True, alpha=0.3)
ax3.set_ylim(0, 105)

plt.tight_layout()
output_path = '/home/erdao/Documents/LightwheelData/utils/sample_analysis/peak_distribution_visualization.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"可视化图表已保存到: {output_path}")

# 打印统计信息
print("\n" + "="*80)
print(f"不同峰值配置下的采样统计（{num_samples} 次采样，min={min_delta_t}, max={max_delta_t}）")
print("="*80)

for config in test_configs:
    distances = simulate_sampling(
        num_samples, T, max_delta_t, min_delta_t,
        config["peak_distance"],
        config["decay_factor"],
        config["rise_factor"],
    )
    counter = Counter(distances)
    
    print(f"\n{config['label']}:")
    print(f"{'Distance':<10} {'Count':<10} {'Percentage':<12} {'Cumulative %':<12}")
    print("-" * 50)
    cumsum = 0
    for d in sorted(counter.keys())[:20]:
        count = counter[d]
        cumsum += count
        pct = count / num_samples * 100
        cum_pct = cumsum / num_samples * 100
        print(f"{d:<10} {count:<10} {pct:<12.2f} {cum_pct:<12.2f}")
    
    mean_dist = np.mean(distances)
    std_dist = np.std(distances)
    most_freq = counter.most_common(1)[0] if counter else (0, 0)
    print(f"\n  Mean: {mean_dist:.2f}, Std: {std_dist:.2f}, Most frequent: {most_freq[0]} ({most_freq[1]} times)")

# 显示图表
plt.show()

