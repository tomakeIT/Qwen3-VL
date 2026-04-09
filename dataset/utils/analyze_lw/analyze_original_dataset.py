#!/usr/bin/env python3
"""
Dataset Analysis Script: Count image frames per episode, check multi-view consistency, and visualize results.
"""

import os
import json
import glob
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def analyze_dataset(root_path):
    """
    Analyze dataset structure, count frames per task and episode.

    Structure: root/task_name/episode_name/view_name/*.png
    """
    root = Path(root_path)
    stats = {}

    # Get all task folders
    tasks = [d for d in root.iterdir() if d.is_dir()]
    print(f"Found {len(tasks)} tasks")

    for task_dir in sorted(tasks):
        task_name = task_dir.name
        print(f"\nProcessing task: {task_name}")

        task_stats = {
            'episodes': {},
            'episode_count': 0,
            'frame_count_distribution': [],
            'inconsistent_episodes': []
        }

        # Get all episodes
        episodes = [d for d in task_dir.iterdir() if d.is_dir()]
        task_stats['episode_count'] = len(episodes)

        for episode_dir in sorted(episodes):
            episode_name = episode_dir.name

            # Get all view folders
            view_dirs = [d for d in episode_dir.iterdir() if d.is_dir()]

            # Filter out image views (exclude non-image folders)
            image_extensions = ['.png', '.jpg', '.jpeg']
            view_frame_counts = {}

            # Exclude known non-image folders
            excluded_dirs = {'isaac_replay_state', 'lwlab_logs'}

            for view_dir in view_dirs:
                if view_dir.name in excluded_dirs:
                    continue
                view_name = view_dir.name

                # Check if folder contains image files
                image_files = []
                for ext in image_extensions:
                    image_files.extend(glob.glob(str(view_dir / f"*{ext}")))

                if image_files:
                    view_frame_counts[view_name] = len(image_files)

            if view_frame_counts:
                # Check if frame counts are consistent across views
                frame_counts = list(view_frame_counts.values())
                is_consistent = len(set(frame_counts)) == 1

                # Use the first view's frame count as episode frame count
                frame_count = frame_counts[0]

                task_stats['episodes'][episode_name] = {
                    'frame_count': frame_count,
                    'views': view_frame_counts,
                    'is_consistent': is_consistent
                }

                task_stats['frame_count_distribution'].append(frame_count)

                if not is_consistent:
                    task_stats['inconsistent_episodes'].append(episode_name)

        stats[task_name] = task_stats

    return stats


def print_summary(stats):
    """Print statistics summary"""
    print("\n" + "=" * 80)
    print("Dataset Statistics Summary")
    print("=" * 80)

    for task_name, task_stats in sorted(stats.items()):
        episode_count = task_stats['episode_count']
        frame_counts = task_stats['frame_count_distribution']

        print(f"\n[{task_name}]")
        print(f"  Episode Count: {episode_count}")

        if frame_counts:
            print(f"  Frame Statistics:")
            print(f"    - Mean: {np.mean(frame_counts):.1f}")
            print(f"    - Median: {np.median(frame_counts):.1f}")
            print(f"    - Min: {min(frame_counts)}")
            print(f"    - Max: {max(frame_counts)}")

            # Show distribution of different frame counts
            unique_counts = sorted(set(frame_counts))
            if len(unique_counts) <= 5:
                for count in unique_counts:
                    num_eps = frame_counts.count(count)
                    print(f"    - {count} frames: {num_eps} episodes")

        inconsistent = task_stats['inconsistent_episodes']
        if inconsistent:
            print(f"  ⚠️  Inconsistent Episodes: {len(inconsistent)}")
            for ep in inconsistent[:3]:  # Show first 3 only
                ep_data = task_stats['episodes'][ep]
                print(f"      {ep}: {ep_data['views']}")
            if len(inconsistent) > 3:
                print(f"      ... and {len(inconsistent) - 3} more")
        else:
            print(f"  ✓ All episodes have consistent frame counts across views")


def visualize(stats, output_path="dataset_analysis.png"):
    """Visualize statistics"""
    tasks = sorted(stats.keys())
    num_tasks = len(tasks)

    # Create subplots
    fig = plt.figure(figsize=(16, 12))

    # 1. Episode count per task
    ax1 = plt.subplot(2, 3, 1)
    episode_counts = [stats[t]['episode_count'] for t in tasks]
    bars = ax1.bar(range(num_tasks), episode_counts, color='steelblue')
    ax1.set_xlabel('Task')
    ax1.set_ylabel('Number of Episodes')
    ax1.set_title('Episodes per Task')
    ax1.set_xticks(range(num_tasks))
    ax1.set_xticklabels(tasks, rotation=45, ha='right', fontsize=8)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontsize=7)

    # 2. Frame count distribution boxplot per task
    ax2 = plt.subplot(2, 3, 2)
    frame_data = [stats[t]['frame_count_distribution'] for t in tasks]
    bp = ax2.boxplot(frame_data, tick_labels=[t[:10] for t in tasks])
    ax2.set_xlabel('Task')
    ax2.set_ylabel('Frame Count')
    ax2.set_title('Frame Count Distribution per Task')
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right', fontsize=8)

    # 3. Average frame count per task
    ax3 = plt.subplot(2, 3, 3)
    avg_frames = [np.mean(stats[t]['frame_count_distribution']) if stats[t]['frame_count_distribution'] else 0 for t in tasks]
    bars = ax3.bar(range(num_tasks), avg_frames, color='coral')
    ax3.set_xlabel('Task')
    ax3.set_ylabel('Average Frame Count')
    ax3.set_title('Average Frame Count per Task')
    ax3.set_xticks(range(num_tasks))
    ax3.set_xticklabels(tasks, rotation=45, ha='right', fontsize=8)

    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.0f}', ha='center', va='bottom', fontsize=7)

    # 4. Frame count histogram (all tasks combined)
    ax4 = plt.subplot(2, 3, 4)
    all_frames = []
    for t in tasks:
        all_frames.extend(stats[t]['frame_count_distribution'])
    ax4.hist(all_frames, bins=30, color='mediumseagreen', edgecolor='black', alpha=0.7)
    ax4.set_xlabel('Frame Count')
    ax4.set_ylabel('Number of Episodes')
    ax4.set_title(f'Frame Count Distribution (Total {len(all_frames)} episodes)')
    ax4.axvline(np.mean(all_frames), color='red', linestyle='--', label=f'Mean: {np.mean(all_frames):.1f}')
    ax4.legend()

    # 5. Inconsistent episodes per task
    ax5 = plt.subplot(2, 3, 5)
    inconsistent_counts = [len(stats[t]['inconsistent_episodes']) for t in tasks]
    colors = ['red' if c > 0 else 'lightgreen' for c in inconsistent_counts]
    bars = ax5.bar(range(num_tasks), inconsistent_counts, color=colors)
    ax5.set_xlabel('Task')
    ax5.set_ylabel('Inconsistent Episodes')
    ax5.set_title('Inconsistent Episodes per Task')
    ax5.set_xticks(range(num_tasks))
    ax5.set_xticklabels(tasks, rotation=45, ha='right', fontsize=8)

    # 6. Total frame count per task
    ax6 = plt.subplot(2, 3, 6)
    total_frames = [sum(stats[t]['frame_count_distribution']) for t in tasks]
    bars = ax6.bar(range(num_tasks), total_frames, color='mediumpurple')
    ax6.set_xlabel('Task')
    ax6.set_ylabel('Total Frame Count')
    ax6.set_title('Total Frame Count per Task')
    ax6.set_xticks(range(num_tasks))
    ax6.set_xticklabels(tasks, rotation=45, ha='right', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")
    plt.show()


def export_to_json(stats, output_path="dataset_stats.json"):
    """Export statistics to JSON"""
    # Convert numpy types to Python native types
    export_stats = {}
    for task_name, task_stats in stats.items():
        export_stats[task_name] = {
            'episode_count': task_stats['episode_count'],
            'frame_count_distribution': task_stats['frame_count_distribution'],
            'inconsistent_episodes': task_stats['inconsistent_episodes'],
            'episodes': task_stats['episodes']
        }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(export_stats, f, indent=2, ensure_ascii=False)
    print(f"Statistics exported to: {output_path}")


def main():
    # Dataset path
    dataset_path = "/home/erdao.liang/LightwheelData/slowdata/1W_Robocasa_X7s_More"

    print(f"Analyzing dataset: {dataset_path}")
    print("-" * 80)

    # Analyze dataset
    stats = analyze_dataset(dataset_path)

    # Print summary
    print_summary(stats)

    # Export to JSON
    export_to_json(stats, "dataset_stats.json")

    # Visualize
    visualize(stats, "dataset_analysis.png")


if __name__ == "__main__":
    main()
