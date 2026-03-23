#!/usr/bin/env python3
"""
机器人操作数据分析脚本
分析HDF5格式的机器人操作数据集
"""

import os
import h5py
import numpy as np
from typing import Dict, Any
import argparse


def analyze_hdf5_file(filepath: str) -> Dict[str, Any]:
    """分析HDF5文件"""
    print(f"\n{'='*60}")
    print(f"分析HDF5文件: {os.path.basename(filepath)}")
    print(f"{'='*60}")
    
    info = {
        'file_size_mb': os.path.getsize(filepath) / (1024 * 1024),
        'datasets': {},
        'groups': []
    }
    
    with h5py.File(filepath, 'r') as f:
        print(f"文件大小: {info['file_size_mb']:.2f} MB")
        print(f"\n文件结构:")
        
        def print_structure(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"  Dataset: {name}")
                print(f"    Shape: {obj.shape}")
                print(f"    Dtype: {obj.dtype}")
                print(f"    Size: {obj.size:,} 元素")
                
                # 计算内存大小
                size_bytes = obj.nbytes
                if size_bytes < 1024:
                    size_str = f"{size_bytes} B"
                elif size_bytes < 1024 * 1024:
                    size_str = f"{size_bytes / 1024:.2f} KB"
                else:
                    size_str = f"{size_bytes / (1024 * 1024):.2f} MB"
                print(f"    内存大小: {size_str}")
                
                # 显示数据统计信息（如果数据不太大）
                if obj.size < 1000000:  # 小于100万个元素
                    data = obj[:]
                    if np.issubdtype(obj.dtype, np.floating):
                        print(f"    范围: [{np.min(data):.4f}, {np.max(data):.4f}]")
                        print(f"    均值: {np.mean(data):.4f}")
                        print(f"    标准差: {np.std(data):.4f}")
                    elif np.issubdtype(obj.dtype, np.integer):
                        print(f"    范围: [{np.min(data)}, {np.max(data)}]")
                        print(f"    均值: {np.mean(data):.4f}")
                
                # 保存数据集信息
                info['datasets'][name] = {
                    'shape': obj.shape,
                    'dtype': str(obj.dtype),
                    'size': obj.size
                }
                
            elif isinstance(obj, h5py.Group):
                print(f"  Group: {name}")
                info['groups'].append(name)
        
        f.visititems(print_structure)
        
        # 尝试读取一些常见的数据集
        print(f"\n详细数据内容:")
        if 'observations' in f:
            print(f"\n  Observations:")
            obs_group = f['observations']
            for key in obs_group.keys():
                dataset = obs_group[key]
                print(f"    {key}: shape={dataset.shape}, dtype={dataset.dtype}")
                if dataset.size < 10000:
                    sample = dataset[:min(5, len(dataset))]
                    print(f"      样本数据: {sample}")
        
        if 'actions' in f:
            print(f"\n  Actions:")
            actions_group = f['actions']
            for key in actions_group.keys():
                dataset = actions_group[key]
                print(f"    {key}: shape={dataset.shape}, dtype={dataset.dtype}")
                if dataset.size < 10000:
                    sample = dataset[:min(5, len(dataset))]
                    print(f"      样本数据: {sample}")
        
        if 'states' in f:
            print(f"\n  States:")
            states_group = f['states']
            for key in states_group.keys():
                dataset = states_group[key]
                print(f"    {key}: shape={dataset.shape}, dtype={dataset.dtype}")
        
        # 查找所有顶层数据集
        print(f"\n顶层数据集:")
        for key in f.keys():
            if isinstance(f[key], h5py.Dataset):
                dataset = f[key]
                print(f"  {key}: shape={dataset.shape}, dtype={dataset.dtype}")
                if dataset.size < 100:
                    print(f"    数据: {dataset[:]}")
    
    return info




def main():
    parser = argparse.ArgumentParser(description='分析HDF5格式的机器人操作数据')
    parser.add_argument('hdf5_file', nargs='?', 
                       default='/home/erdao/Documents/LightwheelData/x7s_1/L90L6PutTheWhiteMugOnThePlate_1762235317475676/dataset_success.hdf5',
                       help='要分析的HDF5文件路径')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.hdf5_file):
        print(f"错误: 文件不存在: {args.hdf5_file}")
        return
    
    if not args.hdf5_file.endswith('.hdf5'):
        print(f"错误: 不是HDF5文件: {args.hdf5_file}")
        return
    
    analyze_hdf5_file(args.hdf5_file)


if __name__ == '__main__':
    main()

