#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
临时脚本：可视化 decay_factor 对采样比例的影响
"""

import matplotlib.pyplot as plt
import numpy as np

def calculate_weights(max_delta_t: int, min_delta_t: int, decay_factor: float):
    """计算不同距离的权重"""
    distances = list(range(min_delta_t, max_delta_t + 1))
    weights = [decay_factor ** (dt - min_delta_t) for dt in distances]
    # 归一化概率
    total_weight = sum(weights)
    probabilities = [w / total_weight for w in weights]
    return distances, weights, probabilities

# 配置参数
max_delta_t = 40
min_delta_t = 2
decay_factors = [0.5, 0.6, 0.7, 0.8, 0.9]

# 创建图表
fig, axes = plt.subplots(2, 1, figsize=(12, 10))

# 子图1：权重分布
ax1 = axes[0]
for decay_factor in decay_factors:
    distances, weights, probabilities = calculate_weights(max_delta_t, min_delta_t, decay_factor)
    ax1.plot(distances, weights, marker='o', label=f'decay_factor={decay_factor}', linewidth=2, markersize=4)

ax1.set_xlabel('Distance (|Δt|)', fontsize=12)
ax1.set_ylabel('Weight', fontsize=12)
ax1.set_title('Weight Distribution for Different decay_factor', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(min_delta_t - 1, max_delta_t + 1)

# 子图2：归一化后的概率分布
ax2 = axes[1]
for decay_factor in decay_factors:
    distances, weights, probabilities = calculate_weights(max_delta_t, min_delta_t, decay_factor)
    ax2.plot(distances, probabilities, marker='o', label=f'decay_factor={decay_factor}', linewidth=2, markersize=4)

ax2.set_xlabel('Distance (|Δt|)', fontsize=12)
ax2.set_ylabel('Sampling Probability', fontsize=12)
ax2.set_title('Sampling Probability Distribution (Normalized)', fontsize=14, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(min_delta_t - 1, max_delta_t + 1)

plt.tight_layout()
plt.savefig('/home/erdao/Documents/LightwheelData/preprocessing/decay_factor_visualization.png', dpi=150, bbox_inches='tight')
print("可视化图表已保存到: preprocessing/decay_factor_visualization.png")

# 打印一些统计信息
print("\n" + "="*60)
print("不同 decay_factor 下的采样统计（前10个距离）")
print("="*60)
for decay_factor in decay_factors:
    distances, weights, probabilities = calculate_weights(max_delta_t, min_delta_t, decay_factor)
    print(f"\ndecay_factor = {decay_factor}:")
    print(f"{'距离':<8} {'权重':<12} {'概率':<12} {'累积概率':<12}")
    print("-" * 50)
    cumsum = 0
    for i, (dt, w, p) in enumerate(zip(distances[:10], weights[:10], probabilities[:10])):
        cumsum += p
        print(f"{dt:<8} {w:<12.6f} {p:<12.4%} {cumsum:<12.4%}")

# 显示图表
plt.show()

