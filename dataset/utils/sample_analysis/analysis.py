#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analyze VLAC-style delta-progress dataset.

需要输入：
  --root  数据集根目录（和构建 JSON 时使用的一样）
  --json  构建好的 JSON 文件路径
  --out   输出可视化和统计结果目录

统计内容：
  1) reference demo frames 的绝对 progress 分布 (0~100 int)
  2) target image 的 delta progress 分布 (-100~100 int)
  3) target image 首帧 (Image-1) 绝对 progress 分布
  4) target image 末帧 (Image-2) 绝对 progress 分布
  5) 可视化若干直方图和散点图
"""

import os
import re
import json
import argparse
from typing import List, Dict, Any, Tuple
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


def list_image_files(path: str) -> List[str]:
    files = [
        f for f in os.listdir(path)
        if os.path.isfile(os.path.join(path, f)) and f.lower().endswith(IMG_EXTS)
    ]
    return sorted(files)


def resolve_image_path(root: str, img_rel: str) -> str:
    """
    将 JSON 中的相对路径（形如 1W_Libero_Piper/task/demo/view/frame.png）
    还原为文件系统中的绝对路径。

    假设构建 JSON 时使用的 root 是:
        /.../LightwheelData/1W_Libero_Piper
    则 JSON 里的路径会是:
        1W_Libero_Piper/task/demo/view/frame.png
    因此解析时需要在 root 的上一级拼接:
        dirname(root) + img_rel
    """
    root_abs = os.path.abspath(root)
    parent_dir = os.path.dirname(root_abs)
    return os.path.join(parent_dir, img_rel)


def compute_abs_progress_from_index(idx: int, total: int) -> float:
    """绝对进度：0~100 (float)，和你之前 demo 中的公式一致。"""
    if total <= 1:
        return 0.0
    return 100.0 * idx / float(total - 1)


def ensure_dir(path: str) -> None:
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def basic_stats(name: str, arr: np.ndarray) -> None:
    """打印基本统计信息。"""
    if arr.size == 0:
        print(f"[STATS] {name}: no data.")
        return
    print(f"\n[STATS] {name}")
    print(f"  count = {arr.size}")
    print(f"  min   = {np.nanmin(arr):.3f}")
    print(f"  max   = {np.nanmax(arr):.3f}")
    print(f"  mean  = {np.nanmean(arr):.3f}")
    print(f"  std   = {np.nanstd(arr):.3f}")
    print(
        "  quantiles "
        f"5/25/50/75/95 = "
        f"{np.nanpercentile(arr, 5):.3f}, "
        f"{np.nanpercentile(arr, 25):.3f}, "
        f"{np.nanpercentile(arr, 50):.3f}, "
        f"{np.nanpercentile(arr, 75):.3f}, "
        f"{np.nanpercentile(arr, 95):.3f}"
    )


def plot_hist(
    data: np.ndarray,
    out_path: str,
    title: str,
    xlabel: str,
    bins: int = 50,
    range_: Tuple[float, float] = None,
) -> None:
    if data.size == 0:
        print(f"[WARN] No data for histogram: {title}")
        return
    plt.figure(figsize=(6, 4))
    plt.hist(data, bins=bins, range=range_, alpha=0.75)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    # 提高分辨率
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[PLOT] Saved {out_path}")


def plot_scatter(
    x: np.ndarray,
    y: np.ndarray,
    out_path: str,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    if x.size == 0 or y.size == 0:
        print(f"[WARN] No data for scatter: {title}")
        return
    plt.figure(figsize=(5, 5))
    plt.scatter(x, y, s=5, alpha=0.5)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    # 提高分辨率
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[PLOT] Saved {out_path}")


def main(args):
    root = args.root
    json_path = args.json
    out_dir = args.out
    ensure_dir(out_dir)

    print(f"[INFO] Loading dataset JSON: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        samples: List[Dict[str, Any]] = json.load(f)
    print(f"[INFO] Total samples: {len(samples)}")

    # 收集统计量
    ref_progress_list: List[float] = []     # reference demo frames 绝对进度
    delta_progress_list: List[float] = []   # delta progress labels (int)
    start_progress_list: List[float] = []   # Image-1 绝对进度
    end_progress_list: List[float] = []     # Image-2 绝对进度

    # reference diversity 统计：有多少不同的 reference demo，以及各自被用多少次
    # key: "task_name/demo_name"（从第一张 reference 图像的相对路径推断）
    ref_demo_counter: Counter[str] = Counter()

    # 新增：占位符与图片数量一致性检查
    # 记录问题样本：(sample_idx, num_images, num_placeholders)
    placeholder_gt_images: List[Tuple[int, int, int]] = []  # <image> 多于 images
    placeholder_lt_images: List[Tuple[int, int, int]] = []  # <image> 少于 images

    # 正则表达式：解析 human prompt 里的 reference 进度
    # 兼容旧版和新版两种文案：
    #   旧版: "The task completion percentage in this frame is {prog:d}%."
    #   新版: "The task completion percentage for this time step is {prog:d}%."
    ref_prog_pattern = re.compile(
        r"The task completion percentage (?:in this frame|for this time step) is\s+(-?\d+)\s*%",
        flags=re.IGNORECASE,
    )

    for idx_sample, sample in enumerate(samples):
        try:
            images = sample.get("images", [])
            conv = sample.get("conversations", [])
            if len(images) < 2 or len(conv) < 2:
                continue

            # ---- 0) reference demo diversity 统计 ----
            # 当前构建脚本中：images = [ref_frames..., target_t1_views..., target_t2_views...]
            # 每个 time step 有 3 个视角 (first_person / left_hand / right_hand)，
            # target 有 2 个 time step，因此 target 部分共有 3*2 = 6 张图。
            # 剩余部分都是 reference 多视角图片。
            num_imgs = len(images)
            NUM_VIEWS = 3
            NUM_TARGET_IMGS = NUM_VIEWS * 2  # 6
            if num_imgs > NUM_TARGET_IMGS:
                num_ref_imgs = num_imgs - NUM_TARGET_IMGS
                if num_ref_imgs > 0:
                    first_ref_rel = images[0]
                    # 从相对路径中抽取 reference demo ID
                    # 现在的路径形式一般为：
                    #   <root_name>/<task_name>/<demo_name>/view/frame.png
                    # 为了把每个 episode 当成独立 reference，
                    # 我们使用前三级目录作为 ID：root/task/demo
                    parts = first_ref_rel.split("/")
                    if len(parts) >= 3:
                        ref_demo_id = "/".join(parts[:3])
                    elif len(parts) >= 2:
                        # 兼容旧格式：task/demo
                        ref_demo_id = "/".join(parts[:2])
                    else:
                        # 兜底：取两级上层目录
                        ref_demo_id = os.path.dirname(os.path.dirname(first_ref_rel))
                    ref_demo_counter[ref_demo_id] += 1

            # ---- 1) delta progress (assistant label) ----
            # 构建时是类似 "+23" 或 "-5" 或 "0"
            assistant_value = conv[1]["value"].strip()
            try:
                delta_int = int(assistant_value)
                delta_progress_list.append(delta_int)
            except Exception:
                # 如果解析失败就忽略
                continue

            # ---- 2) reference progress from human prompt ----
            human_value = conv[0]["value"]

            # 新增：检查 <image> 占位符数量与 images 数量是否一致
            num_placeholders = human_value.count("<image>")
            num_images = len(images)
            if num_placeholders > num_images:
                placeholder_gt_images.append((idx_sample, num_images, num_placeholders))
            elif num_placeholders < num_images:
                placeholder_lt_images.append((idx_sample, num_images, num_placeholders))

            matches = ref_prog_pattern.findall(human_value)
            for m in matches:
                try:
                    p = int(m)
                    ref_progress_list.append(p)
                except Exception:
                    pass

            # ---- 3) 首帧 / 末帧 绝对 progress ----
            # 最后两张是 target Image-1, Image-2
            img_rel_1 = images[-2]
            img_rel_2 = images[-1]

            # 还原绝对路径、view 目录、frame 文件列表
            abs_path_1 = resolve_image_path(root, img_rel_1)
            abs_path_2 = resolve_image_path(root, img_rel_2)

            view_dir_1 = os.path.dirname(abs_path_1)
            view_dir_2 = os.path.dirname(abs_path_2)

            # 正常情况 view_dir_1 == view_dir_2，因为来自同一 view
            # 这里我们各自计算一遍，兼容奇怪情况
            if os.path.isdir(view_dir_1):
                frame_files_1 = list_image_files(view_dir_1)
                base_name_1 = os.path.basename(abs_path_1)
                if base_name_1 in frame_files_1:
                    idx_1 = frame_files_1.index(base_name_1)
                    T1 = len(frame_files_1)
                    prog1 = compute_abs_progress_from_index(idx_1, T1)
                    start_progress_list.append(prog1)

            if os.path.isdir(view_dir_2):
                frame_files_2 = list_image_files(view_dir_2)
                base_name_2 = os.path.basename(abs_path_2)
                if base_name_2 in frame_files_2:
                    idx_2 = frame_files_2.index(base_name_2)
                    T2 = len(frame_files_2)
                    prog2 = compute_abs_progress_from_index(idx_2, T2)
                    end_progress_list.append(prog2)

        except Exception as e:
            # 避免单条样本 crash 整个统计
            if args.verbose:
                print(f"[WARN] Error processing sample {idx_sample}: {e}")
            continue

    # 转成 numpy 方便统计
    ref_arr = np.array(ref_progress_list, dtype=np.float32)
    delta_arr = np.array(delta_progress_list, dtype=np.float32)
    start_arr = np.array(start_progress_list, dtype=np.float32)
    end_arr = np.array(end_progress_list, dtype=np.float32)

    # ---- Reference demo diversity 统计输出 ----
    if len(ref_demo_counter) > 0:
        print("\n[REF DIVERSITY] 不同 reference demo 的数量:", len(ref_demo_counter))
        print("[REF DIVERSITY] 每个 reference demo 被使用的次数（按次数降序）：")
        for demo_id, cnt in ref_demo_counter.most_common():
            print(f"  {demo_id}: {cnt}")
    else:
        print("\n[REF DIVERSITY] 数据集中没有检测到任何 reference demo。")

    # ---- 打印基本统计 ----
    basic_stats("Reference absolute progress", ref_arr)
    basic_stats("Delta progress (labels)", delta_arr)
    basic_stats("Start frame absolute progress (Image-1)", start_arr)
    basic_stats("End frame absolute progress (Image-2)", end_arr)

    # ---- 画图 ----
    plot_hist(
        ref_arr,
        out_path=os.path.join(out_dir, "ref_progress_hist.png"),
        title="Reference Demo Progress Distribution",
        xlabel="Absolute progress (%)",
        bins=21,
        range_=(0, 100),
    )

    plot_hist(
        delta_arr,
        out_path=os.path.join(out_dir, "delta_progress_hist.png"),
        title="Delta Progress Label Distribution",
        xlabel="Delta progress (int, -100 ~ 100)",
        # 只统计 [-50, 50] 区间，并进一步加细 bin
        bins=101,
        range_=(-50, 50),
    )

    # 首帧/末帧各自的直方图
    if start_arr.size > 0 or end_arr.size > 0:
        plt.figure(figsize=(6, 4))
        if start_arr.size > 0:
            plt.hist(start_arr, bins=21, range=(0, 100),
                     alpha=0.6, label="Start (Image-1)")
        if end_arr.size > 0:
            plt.hist(end_arr, bins=21, range=(0, 100),
                     alpha=0.6, label="End (Image-2)")
        plt.title("Start vs End Absolute Progress")
        plt.xlabel("Absolute progress (%)")
        plt.ylabel("Count")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.tight_layout()
        out_path = os.path.join(out_dir, "start_end_progress_hist.png")
        # 提高分辨率
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"[PLOT] Saved {out_path}")

    # 首帧 vs 末帧的散点图
    n = min(start_arr.size, end_arr.size)
    if n > 0:
        plot_scatter(
            x=start_arr[:n],
            y=end_arr[:n],
            out_path=os.path.join(out_dir, "start_vs_end_scatter.png"),
            title="Start vs End Absolute Progress",
            xlabel="Start progress (%)",
            ylabel="End progress (%)",
        )
    
    # -----------------------------
    # 🔵 新增内容：单独绘制 start / end histograms
    # -----------------------------

    # Start frame histogram (Image-1 absolute progress)
    if start_arr.size > 0:
        plot_hist(
            start_arr,
            out_path=os.path.join(out_dir, "start_progress_hist.png"),
            title="Start Frame Absolute Progress Distribution",
            xlabel="Start progress (%)",
            bins=21,
            range_=(0, 100),
        )

    # End frame histogram (Image-2 absolute progress)
    if end_arr.size > 0:
        plot_hist(
            end_arr,
            out_path=os.path.join(out_dir, "end_progress_hist.png"),
            title="End Frame Absolute Progress Distribution",
            xlabel="End progress (%)",
            bins=21,
            range_=(0, 100),
        )
    # -----------------------------
    # 🔵 新增内容结束
    # -----------------------------

    # -----------------------------
    # 占位符与图片数量一致性检查报告
    # -----------------------------
    if placeholder_gt_images or placeholder_lt_images:
        print("\n[CHECK] <image> 占位符与 images 数量一致性检查")
        if placeholder_gt_images:
            print(f"  - 占位符数量 > 图片数量 的样本数: {len(placeholder_gt_images)}")
            for (idx_sample, num_imgs, num_ph) in placeholder_gt_images[:10]:
                print(f"    sample {idx_sample}: images={num_imgs}, <image>={num_ph}")
            if len(placeholder_gt_images) > 10:
                print(f"    ... 共 {len(placeholder_gt_images)} 条，仅展示前 10 条")
        else:
            print("  - 无 占位符数量 > 图片数量 的样本。")

        if placeholder_lt_images:
            print(f"  - 占位符数量 < 图片数量 的样本数: {len(placeholder_lt_images)}")
            for (idx_sample, num_imgs, num_ph) in placeholder_lt_images[:10]:
                print(f"    sample {idx_sample}: images={num_imgs}, <image>={num_ph}")
            if len(placeholder_lt_images) > 10:
                print(f"    ... 共 {len(placeholder_lt_images)} 条，仅展示前 10 条")
        else:
            print("  - 无 占位符数量 < 图片数量 的样本。")
    else:
        print("\n[CHECK] <image> 占位符检查：所有样本的占位符数量与 images 数量一致。")


    print("\n[INFO] Analysis done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True,
                        help="Root of original dataset (same as used when building JSON).")
    parser.add_argument("--json", type=str, required=True,
                        help="Path to the built JSON dataset file.")
    parser.add_argument("--out", type=str, default="analysis_outputs",
                        help="Directory to save plots and statistics.")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-sample warnings.")
    args = parser.parse_args()
    main(args)
