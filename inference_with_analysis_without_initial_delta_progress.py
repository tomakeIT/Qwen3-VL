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

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
try:
    import cv2  # type: ignore
except Exception:
    cv2 = None
try:
    import imageio  # type: ignore
except Exception:
    imageio = None
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

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
# 5. 可视化与视频生成
# ==========================

def _to_nan_array(values: List[Optional[float]]) -> np.ndarray:
    arr = np.array([np.nan if v is None else float(v) for v in values], dtype=np.float32)
    return arr


def render_progress_frame(
    current_image: Image.Image,
    predicted_progress_list: List[Optional[float]],
    expected_progress_list: List[Optional[float]],
    dpi: int = 120,
) -> np.ndarray:
    """
    生成一张复合图：左侧为当前帧图像，右侧为进度曲线（预测 vs 期望），并高亮当前点。
    返回 RGB 的 numpy 数组 (H, W, 3)。
    """
    img_np = np.asarray(current_image)
    img_h, img_w = img_np.shape[0], img_np.shape[1]

    # 右侧曲线区域宽度按图像宽度的 0.8 估算
    plot_w = int(round(img_w * 0.8))
    fig_w_px = img_w + plot_w
    fig_h_px = img_h
    figsize = (fig_w_px / dpi, fig_h_px / dpi)

    fig = plt.figure(figsize=figsize, dpi=dpi)
    gs = fig.add_gridspec(1, 2, width_ratios=[img_w, plot_w])

    # 左侧：图像
    ax_img = fig.add_subplot(gs[0, 0])
    ax_img.imshow(img_np)
    ax_img.axis("off")
    ax_img.set_title("Current Frame", fontsize=10)

    # 右侧：曲线
    ax_plot = fig.add_subplot(gs[0, 1])
    ax_plot.set_title("Task Progress (%)", fontsize=10)
    ax_plot.set_ylim(0, 100)
    ax_plot.set_xlim(1, max(2, len(predicted_progress_list)))
    ax_plot.set_xlabel("Frame Index")
    ax_plot.set_ylabel("Progress (%)")
    ax_plot.grid(True, linestyle="--", alpha=0.3)

    xs = np.arange(1, len(predicted_progress_list) + 1, dtype=np.int32)
    pred_arr = _to_nan_array(predicted_progress_list)
    exp_arr = _to_nan_array(expected_progress_list)

    # 画曲线（跳过 NaN 的段）
    if np.any(~np.isnan(exp_arr)):
        ax_plot.plot(xs, exp_arr, color="#1f77b4", label="Expected", linewidth=2)
    if np.any(~np.isnan(pred_arr)):
        ax_plot.plot(xs, pred_arr, color="#d62728", label="Predicted", linewidth=2)

    # 高亮当前点
    if len(xs) > 0:
        x_cur = xs[-1]
        if not np.isnan(exp_arr[-1]):
            ax_plot.scatter([x_cur], [exp_arr[-1]], color="#1f77b4", s=40, zorder=5)
        if not np.isnan(pred_arr[-1]):
            ax_plot.scatter([x_cur], [pred_arr[-1]], color="#d62728", s=40, zorder=6)

    ax_plot.legend(loc="lower right", fontsize=9)

    canvas = FigureCanvas(fig)
    canvas.draw()
    width, height = canvas.get_width_height()
    frame = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8).reshape((height, width, 3))
    plt.close(fig)
    return frame


def save_video(frames: List[np.ndarray], out_path: str, fps: int = 4) -> None:
    """
    将一组 RGB 帧写成 MP4 视频。优先使用 OpenCV；若不可用则尝试 imageio；
    若都不可用则导出为 PNG 序列。
    """
    if not frames:
        return
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    h, w = frames[0].shape[0], frames[0].shape[1]

    if cv2 is not None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
        for fr in frames:
            if fr.shape[0] != h or fr.shape[1] != w:
                fr = cv2.resize(fr, (w, h))
            bgr = fr[:, :, ::-1]
            writer.write(bgr)
        writer.release()
        return

    if imageio is not None:
        with imageio.get_writer(out_path, fps=fps, codec="libx264") as writer:
            for fr in frames:
                writer.append_data(fr)
        return

    # 回退：导出 PNG 序列
    frames_dir = os.path.splitext(out_path)[0] + "_frames"
    os.makedirs(frames_dir, exist_ok=True)
    for i, fr in enumerate(frames):
        Image.fromarray(fr).save(os.path.join(frames_dir, f"frame_{i:05d}.png"))


# ==========================
# 5.1 设定确定性与随机种子
# ==========================
def set_global_determinism(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        try:
            torch.use_deterministic_algorithms(True)  # 可能在少数算子上抛异常
        except Exception:
            pass
        try:
            import torch.backends.cudnn as cudnn  # type: ignore
            cudnn.deterministic = True
            cudnn.benchmark = False
        except Exception:
            pass
    except Exception:
        pass


# ==========================
# 4. 构造 Qwen 消息
# ==========================
from typing import List, Tuple, Optional
from PIL import Image


def build_qwen_messages(
    task_description: str,
    demo_examples: Optional[List[Tuple[Image.Image, float]]],
    target_frames: List[Image.Image],
    prev_target_image: Optional[Image.Image] = None,
    prev_predicted_progress: Optional[float] = None,
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
        "Values like 3.0, 17.0, 42.0, 68.0, 91.0 are all valid. "
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


    # ---- Initial scene（target 轨迹的第一帧）----
    content.append({"type": "text", "text": "\nNow you will estimate the completion for the new episode.\n\n"})
    content.append({
        "type": "text",
        "text": (
            "In this initial robot scene, the task completion percentage is defined to be 0%.\n\n"
        ),
    })

    # ---- 上一次预测的额外上下文（若提供）----
    if (prev_target_image is not None) and (prev_predicted_progress is not None):
        content.append({
            "type": "text",
            "text": (
                "Additional context from the previous step:\n"
                f"In the previous step, the model estimated the task completion percentage "
                f"to be {prev_predicted_progress:.1f}% for the following frame:\n"
            ),
        })
        content.append({"type": "image", "image": prev_target_image})
        content.append({
            "type": "text",
            "text": (
                "Use this only for reference. Now estimate the next target frame below.\n\n"
            ),
        })

    # ---- 说明如何预测 target frame ----
    query_text = (
        f"For the task of {task_description}, we now provide one target image from the same episode.\n"
        "Estimate its task completion percentage relative to the initial scene.\n\n"
        "OUTPUT REQUIREMENT: Return ONLY the delta task completion (current minus previous) as a number between -100 and 100 WITHOUT the '%' sign.\n"
        "If previous-step context is provided above, compute delta relative to that previous percentage; otherwise compute delta relative to the initial scene which is defined as 0%.\n"
        "Do NOT output any reasoning, analysis, words, or extra text. One line, just the number.\n\n"
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
    prev_target_image: Optional[Image.Image] = None,
    prev_predicted_progress: Optional[float] = None,
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
        prev_target_image=prev_target_image,
        prev_predicted_progress=prev_predicted_progress,
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
            do_sample=False,
            temperature=0.0,
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
    model_name = "models/Qwen-VL-2B-Instruct"
    print(f"Loading model: {model_name}")
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        dtype="auto",
        device_map="auto",
        local_files_only=True
    )
    processor = AutoProcessor.from_pretrained(model_name)

    # ===== 生成带进度变化图的视频 =====
    set_global_determinism(seed=0)
    try:
        model.eval()
    except Exception:
        pass
    target_img_paths = list_image_paths(target_traj_folder)
    frame_indices = list(range(1, len(target_img_paths), 5))
    predicted_list: List[Optional[float]] = []
    expected_list: List[Optional[float]] = []
    composite_frames: List[np.ndarray] = []
    output_video_path = "outputs/progress_video.mp4"
    fps = 4

    # 维护上一帧的额外上下文（第二次及之后使用）
    prev_target_image = None
    prev_predicted_progress = None

    for frame_id in frame_indices:
        expected_progress = 100.0 * frame_id / (len(target_img_paths) - 1)
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
            prev_target_image=prev_target_image,
            prev_predicted_progress=prev_predicted_progress,
        )
        print("=== Raw model output ===")
        print(output_text)
        print("========================")
        # 解析模型输出：期望格式
        # "Frame 1: Frame Description: <short description>, Task Completion Percentages: <number>%"
        def _parse_completion_output(text: str):
            """
            从模型输出中解析出描述与百分比。返回 (percent_float, description)；
            若无法解析，返回 (None, None)。
            """
            # 优先匹配明确格式
            m = re.search(
                r"Frame\s*\d+\s*:\s*Frame\s*Description\s*:\s*(.*?),\s*Task\s*Completion\s*Percentages\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*%",
                text,
                flags=re.IGNORECASE | re.DOTALL,
            )
            if m:
                desc = m.group(1).strip()
                try:
                    pct = float(m.group(2))
                    return pct, desc
                except ValueError:
                    pass
            # 退而求其次：抓取任意 -100~100 的数字（可带或不带百分号，最后一个最可能为答案）
            candidates = re.findall(r"(-?[0-9]+(?:\.[0-9]+)?)\s*%?", text)
            if candidates:
                try:
                    pct = float(candidates[-1])
                    if -100.0 <= pct <= 100.0:
                        return pct, None
                except ValueError:
                    pass
            return None, None

        parsed_pct, parsed_desc = _parse_completion_output(output_text)
        if parsed_pct is not None:
            last_abs = prev_predicted_progress if prev_predicted_progress is not None else 0.0
            abs_pred = max(0.0, min(100.0, last_abs + parsed_pct))
            if parsed_desc:
                print(f"Parsed -> Delta: {parsed_pct:.2f}, Abs: {abs_pred:.2f}%, Description: {parsed_desc}")
            else:
                print(f"Parsed -> Delta: {parsed_pct:.2f}, Abs: {abs_pred:.2f}%")
        else:
            print("Parsed -> Progress: <unparsed>")
        print(f"Expected progress for frame {frame_id}: {expected_progress:.2f}%")
        # 聚合并渲染
        if parsed_pct is not None:
            predicted_list.append(abs_pred)
        else:
            predicted_list.append(np.nan)
        expected_list.append(expected_progress)
        cur_img = Image.open(target_img_paths[frame_id]).convert("RGB")
        comp = render_progress_frame(cur_img, predicted_list, expected_list)
        composite_frames.append(comp)

        # 更新上一帧的额外上下文（仅在成功解析出进度时启用）
        prev_target_image = cur_img
        prev_predicted_progress = abs_pred if parsed_pct is not None else None

    # 写出视频
    save_video(composite_frames, output_video_path, fps=fps)
    print(f"Video saved to: {output_video_path}")

if __name__ == "__main__":
    main()
