from PIL.Image import Image
import os
import re
import math
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional

from PIL import Image
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor


# ==========================
# 1. 基础工具函数
# ==========================

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_image_paths(folder: str) -> List[Path]:
    folder = Path(folder)
    paths = [
        p for p in sorted(folder.iterdir())
        if p.suffix.lower() in IMG_EXTS and p.is_file()
    ]
    if not paths:
        raise ValueError(f"No images found in {folder}")
    return paths


def subsample_indices(num_frames: int, max_frames: int) -> List[int]:
    """在 [0, num_frames-1] 中等间距子采样 max_frames 个 index."""
    if num_frames <= max_frames:
        return list(range(num_frames))
    # 等间隔采样
    return [int(round(i * (num_frames - 1) / (max_frames - 1))) for i in range(max_frames)]


def compute_progress_from_index(idx: int, total: int) -> float:
    """用帧序号给一个 0~100 的线性进度（demo 用的伪 GT）"""
    if total <= 1:
        return 0.0
    return 100.0 * idx / (total - 1)


# ==========================
# 2. 构造 In-Context 示例
# ==========================

def build_demo_examples(
    demo_folder: str,
    num_example_frames: int = 12,
    seed: int = 0,
) -> List[Tuple[Image.Image, float]]:
    """
    从 demo 轨迹文件夹中抽 num_example_frames 帧,
    返回 [(PIL.Image, progress_pct), ...]，顺序已随机打乱。
    """
    rng = random.Random(seed)

    img_paths = list_image_paths(demo_folder)
    total = len(img_paths)

    # 均匀sample num_example_frames个帧
    if total <= num_example_frames:
        sel_indices = list(range(total))
    else:
        sel_indices = [int(round(i * (total - 1) / (num_example_frames - 1))) for i in range(num_example_frames)]
    print(sel_indices)
    # 为了和 GVL 设定一致，先保持时间顺序，再随机打乱
    frames = []
    for idx in sel_indices:
        img = Image.open(img_paths[idx]).convert("RGB")
        prog = compute_progress_from_index(idx, total)
        frames.append((img, prog))

    # rng.shuffle(frames)
    return frames


# ==========================
# 3. 构造目标轨迹帧（包含 shuffle）
# ==========================

def build_target_frames(
    traj_folder: str,
    frame_id: int,
    seed: int = 0,
) -> Tuple[List[Image.Image], List[int], List[int]]:
    """
    """

    img_paths = list_image_paths(traj_folder)
    # 子采样后的原始时间顺序帧
    frames_subsampled = [
        Image.open(img_paths[idx]).convert("RGB")
        for idx in [0, frame_id]
    ]
    return frames_subsampled


# ==========================
# 4. 构造 Qwen 消息
# ==========================
from typing import List, Tuple, Optional
from PIL import Image


def build_qwen_messages(
    task_description: str,
    demo_examples: Optional[List[Tuple[Image.Image, float]]],
    target_frames: List[Image.Image],
) -> list:
    """
    利用 reference demo + (initial_frame, target_frames) 构造 Qwen 的 messages。

    target_frames[0] 视为 initial scene (0%)，
    target_frames[1:] 为需要预测进度的目标帧（可以只有 1 个，也可以多张）。
    """
    assert len(target_frames) >= 2, "target_frames 至少需要包含 initial + 1 个 target frame"

    initial_frame = target_frames[0]
    query_frames = target_frames[1:]

    content = []

    # ---- 任务说明（强调是连续数值，不要只用 0 或 100）----
    intro_text = (
        f"You are an expert roboticist tasked with estimating continuous task completion percentages "
        f"for a robot performing the task: {task_description}.\n"
        "The task completion percentage is a real-valued number between 0 and 100 (inclusive).\n"
        "Values like 3, 17, 42, 68, 91 are all valid. "
        "Do not restrict yourself to only 0 or 100 unless you are very certain the task is exactly at the start or fully completed.\n"
        "100 means the task is fully completed. 0 means the task has not started relative to the initial scene.\n\n"
        "We will first show you some example frames with their ground-truth task completion percentages, "
        "then an initial scene of a new episode and one or more target frames from the same episode. "
        "Your job is to estimate the task completion percentage for each target frame relative to the initial scene.\n\n"
    )
    content.append({"type": "text", "text": intro_text})

    # ---- In-context 示例部分（demo 轨迹）----
    if demo_examples is not None and len(demo_examples) > 0:
        content.append({
            "type": "text",
            "text": "Here are example frames with their task completion percentages:\n"
        })
        for i, (img, prog) in enumerate(demo_examples, start=1):
            # 文字 + 图片
            text = (
                f"\nExample Frame {i}: "
                f"At this time, the task completion percentage is {prog:.1f}%.\n"
            )
            content.append({"type": "text", "text": text})
            content.append({"type": "image", "image": img})

        # 让模型先分析这些 demo 为什么是这样的百分比
        analysis_instruction = (
            "\nBefore predicting completion for the new episode, analyze the above example frames.\n"
            "Explain in concise bullet points WHY the given task completion percentages for the example frames "
            "are reasonable, focusing on visual cues such as:\n"
            "- relative positions and distances between key objects,\n"
            "- orientation or alignment of objects,\n"
            "- whether the goal object is approaching or already at the target location,\n"
            "- partial vs. full contact or placement.\n\n"
            "This analysis should only refer to the EXAMPLE frames and their given percentages.\n"
        )
        content.append({"type": "text", "text": analysis_instruction})

    # ---- Initial scene（target 轨迹的第一帧）----
    content.append({"type": "text", "text": "\nAfter you finish your analysis of the example frames, you will estimate the completion for the new episode.\n\n"})
    content.append({"type": "text", "text": "\nInitial robot scene (new episode):\n"})
    content.append({"type": "image", "image": initial_frame})
    content.append({
        "type": "text",
        "text": (
            "In this initial robot scene, the task completion percentage is defined to be 0%.\n\n"
        ),
    })

    # ---- 说明如何预测 target frame ----
    query_text = (
        f"For the task of {task_description}, we now provide one target image from the same episode.\n"
        "Estimate its task completion percentage relative to the initial scene.\n\n"
        "Finally, after your analysis, you MUST provide your numeric answer in the following EXACT format: "
        "Frame 1: Frame Description: <short description>, Task Completion Percentages: <number between 0 and 100>%\n\n"
        "Make sure the percentage is NOT limited to only 0 or 100. Use intermediate values whenever appropriate.\n\n"
        "Now, here is the target image:\n"
    )
    content.append({"type": "text", "text": query_text})

    # ---- 逐个插入 target frame ----
    for i, img in enumerate(query_frames, start=1):
        content.append({"type": "text", "text": f"\nFrame {i} (target frame):\n"})
        content.append({"type": "image", "image": img})

    messages = [
        {
            "role": "user",
            "content": content,
        }
    ]
    return messages




# ==========================
# 6. 完整流程：给一个目标轨迹跑一次
# ==========================

def run_gvl_qwen_for_trajectory(
    model,
    processor,
    task_description: str,
    target_traj_folder: str,
    demo_traj_folder: Optional[str] = None,
    seed: int = 0,
    frame_id: int = 20,
    num_demo_frames: int = 12,
    max_new_tokens: int = 256,
):
    # 准备 demo in-context
    if demo_traj_folder is not None:
        demo_examples = build_demo_examples(
            demo_traj_folder,
            num_example_frames=num_demo_frames,
            seed=seed,
        )
    else:
        demo_examples = None
    # 准备目标轨迹的打乱帧
    target_frames = build_target_frames(
        target_traj_folder,
        frame_id=frame_id
    )

    messages = build_qwen_messages(
        task_description=task_description,
        demo_examples=demo_examples,
        target_frames=target_frames,
    )

    # 用 Qwen 的 processor 构造输入
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    # 推理
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
        )

    # 截掉 prompt 部分，只保留新生成的 token
    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_texts = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    output_text = output_texts[0]
    return output_text


# ==========================
# 7. main 示例
# ==========================

def main():
    # ===== 需要你修改的部分 =====
    task_description = "put the white mug on the plate"
    demo_traj_folder = "data/1/right_shoulder"
    target_traj_folder = "data/2/right_shoulder"
    frame_id = 32
    expected_progress = 100.0 * frame_id / (len(list_image_paths(target_traj_folder)) - 1)
    # ===== 加载模型 =====
    model_name = "Qwen/Qwen3-VL-2B-Instruct"
    print(f"Loading model: {model_name}")
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        dtype="auto",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_name)

    output_text = run_gvl_qwen_for_trajectory(
        model=model,
        processor=processor,
        task_description=task_description,
        target_traj_folder=target_traj_folder,
        demo_traj_folder=demo_traj_folder,
        seed=0,
        frame_id=frame_id,
        num_demo_frames=12,
        max_new_tokens=2048,
    )
    print("=== Raw model output ===")
    print(output_text)
    print("========================")
    print(f"Expected progress: {expected_progress:.2f}%")
if __name__ == "__main__":
    main()
