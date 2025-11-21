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

    # # ---- 上一次预测的额外上下文（若提供）----
    # if (prev_target_image is not None) and (prev_predicted_progress is not None):
    #     content.append({
    #         "type": "text",
    #         "text": (
    #             "Additional context from the previous step:\n"
    #             f"In the previous step, the model estimated the task completion percentage "
    #             f"to be {prev_predicted_progress:.1f}% for the following frame:\n"
    #         ),
    #     })
    #     content.append({"type": "image", "image": prev_target_image})
    #     content.append({
    #         "type": "text",
    #         "text": (
    #             "Use this only for reference. Now estimate the next target frame below.\n\n"
    #         ),
    #     })

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
# 4*. 多视角（与数据生成一致）的消息构造
# ==========================

def compute_abs_progress_from_index_int(idx: int, total: int) -> int:
    if total <= 1:
        return 0
    c = idx / float(total - 1)
    return int(round(100.0 * c))


def build_qwen_messages_multiview(
    task_desc: str,
    ref_steps_views: List[List[Image.Image]],  # [[v1_img, v2_img, v3_img], step2..., ...]
    ref_progress_ints: List[int],              # [int, int, ...] len == len(ref_steps_views)
    target_t1_views: List[Image.Image],        # [v1, v2, v3] for Image-1
    target_t2_views: List[Image.Image],        # [v1, v2, v3] for Image-2
    view_names_order: List[str],               # ["first_person_camera", "left_hand_camera", "right_hand_camera"]
) -> list:
    """
    构造与 LightwheelData/preprocessing/build_vlac_progress_json.py 一致的提示与图片顺序：
    - 若有 reference：逐步展示多步参考，每步多视角，附绝对进度整数
    - 然后展示目标 episode 的 Image-1 与 Image-2（多视角）
    - 输出要求：仅返回整数 D = round(P2 - P1)，范围 [-100, 100]
    """
    content = []

    # 开场任务说明
    content.append({"type": "text", "text": (
        "You are a robotic task progress evaluator.\n\n"
        f"Task description: {task_desc}\n\n"
        "You will first see several time steps from a reference demonstration of this task.\n"
        "Each time step contains multiple synchronized camera views and is annotated with its\n"
        "absolute task completion percentage (an integer between 0 and 100).\n"
        "Then you will see two time steps (Image-1 and Image-2) from another episode of the SAME task,\n"
        "also with multiple camera views.\n\n"
    )})

    # Reference 部分
    if ref_steps_views and ref_progress_ints:
        content.append({"type": "text", "text": (
            "Here are example time steps from a reference demonstration with their absolute completion percentages.\n"
            "For each time step, the views are given in the following fixed order:\n"
            "  - " + ", ".join(view_names_order) + "\n\n"
        )})

        assert len(ref_steps_views) == len(ref_progress_ints), "ref steps and progress length mismatch"
        for step_idx, (views_imgs, prog) in enumerate(zip(ref_steps_views, ref_progress_ints), start=1):
            content.append({"type": "text", "text": f"Reference Time Step {step_idx}:\n"})
            for v_name, v_img in zip(view_names_order, views_imgs):
                content.append({"type": "text", "text": f"- View {v_name}: "})
                content.append({"type": "image", "image": v_img})
                content.append({"type": "text", "text": "\n"})
            content.append({"type": "text", "text": f"The task completion percentage for this time step is {prog:d}%.\n\n"})
    else:
        content.append({"type": "text", "text": (
            "No reference demonstration is available for this task. Please rely on your general understanding.\n\n"
        )})

    # 目标多视角 Image-1 / Image-2
    content.append({"type": "text", "text": (
        "Now consider another episode of the SAME task.\n"
        "We will show you two time steps from this episode, each with the following camera views\n"
        "in the exact order they are provided:\n"
        "  - " + ", ".join(view_names_order) + "\n\n"
    )})

    # Image-1
    content.append({"type": "text", "text": "Image-1 (earlier or reference time step):\n"})
    for v_name, v_img in zip(view_names_order, target_t1_views):
        content.append({"type": "text", "text": f"- View {v_name}: "})
        content.append({"type": "image", "image": v_img})
        content.append({"type": "text", "text": "\n"})

    # Image-2
    content.append({"type": "text", "text": "\nImage-2 (another time step of the same episode):\n"})
    for v_name, v_img in zip(view_names_order, target_t2_views):
        content.append({"type": "text", "text": f"- View {v_name}: "})
        content.append({"type": "image", "image": v_img})
        content.append({"type": "text", "text": "\n"})

    # 输出要求（严格整数，且仅输出一个值）
    content.append({"type": "text", "text": (
        "\nLet the task completion percentages of Image-1 and Image-2 be P1 and P2 (both between 0 and 100).\n"
        "Your job is to output the integer delta progress D = round(P2 - P1),\n"
        "which must be an integer between -100 and 100 (inclusive).\n\n"
        "OUTPUT REQUIREMENT:\n"
        "- Return ONLY the integer D (with optional leading '+' or '-' sign),\n"
        "  e.g., +5, -13, 0, +42, -100, +100.\n"
        "- Do NOT output any explanation, percent sign, or extra text.\n"
    )})

    messages = [{"role": "user", "content": content}]
    return messages


# ==========================
# 4**. 多视角数据读取（与训练目录结构一致）
# ==========================

def list_subdirs(path: str) -> List[str]:
    return sorted([d for d in os.listdir(path) if (Path(path) / d).is_dir()])


def list_image_files(path: str) -> List[Path]:
    p = Path(path)
    files = [f for f in sorted(p.iterdir()) if f.is_file() and f.suffix.lower() in IMG_EXTS]
    return files


def load_multiview_frames_for_indices(
    demo_folder: str,
    view_names: List[str],
    indices: List[int],
) -> List[List[Image.Image]]:
    """
    返回形如 [[view1_img, view2_img, view3_img], ...]，每个内部列表对应一个 time step。
    """
    # 为每个视角列出帧文件
    view_to_files: Dict[str, List[Path]] = {}
    for v in view_names:
        v_path = Path(demo_folder) / v
        if not v_path.exists():
            raise FileNotFoundError(f"View folder not found: {v_path}")
        files = list_image_files(str(v_path))
        if len(files) == 0:
            raise ValueError(f"No images under view {v}: {v_path}")
        view_to_files[v] = files

    T = min(len(files) for files in view_to_files.values())
    if any(idx < 0 or idx >= T for idx in indices):
        raise IndexError(f"indices out of range for min length {T}: {indices}")

    steps: List[List[Image.Image]] = []
    for idx in indices:
        imgs_step: List[Image.Image] = []
        for v in view_names:
            img = Image.open(view_to_files[v][idx]).convert("RGB")
            imgs_step.append(img)
        steps.append(imgs_step)
    return steps


def evenly_spaced_indices_with_jitter(T: int, num_steps: int, jitter: int = 0, seed: int = 0) -> List[int]:
    rng = random.Random(seed)
    if T <= 0:
        return []
    if T <= num_steps:
        base_indices = list(range(T))
    else:
        base_indices = [int(round(i * (T - 1) / (num_steps - 1))) for i in range(num_steps)]
    indices: List[int] = []
    for idx in base_indices:
        offset = rng.randint(-jitter, jitter) if jitter > 0 else 0
        j_idx = max(0, min(T - 1, idx + offset))
        indices.append(j_idx)
    indices = sorted(set(indices))
    return indices


def prepare_multiview_reference(
    ref_demo_folder: Optional[str],
    view_names: List[str],
    num_ref_frames: int = 4,
    jitter: int = 2,
    seed: int = 0,
) -> Tuple[List[List[Image.Image]], List[int]]:
    """
    读取参考演示，多视角多时刻，返回 (ref_steps_views, ref_progress_ints)。
    若未提供 ref_demo_folder，则返回空参考（与训练脚本一致的 fallback）。
    """
    if not ref_demo_folder:
        return [], []

    # 为每个视角列出帧
    view_to_files: Dict[str, List[Path]] = {}
    for v in view_names:
        v_dir = Path(ref_demo_folder) / v
        if not v_dir.exists():
            return [], []  # 没有齐备视角则不提供 reference
        view_to_files[v] = list_image_files(str(v_dir))
        if len(view_to_files[v]) == 0:
            return [], []

    T_ref = min(len(frames) for frames in view_to_files.values())
    if T_ref == 0:
        return [], []

    indices = evenly_spaced_indices_with_jitter(T_ref, num_ref_frames, jitter=jitter, seed=seed)
    if not indices:
        return [], []

    ref_steps: List[List[Image.Image]] = []
    ref_prog_ints: List[int] = []
    for idx in indices:
        step_imgs: List[Image.Image] = []
        for v in view_names:
            img = Image.open(view_to_files[v][idx]).convert("RGB")
            step_imgs.append(img)
        ref_steps.append(step_imgs)
        ref_prog_ints.append(compute_abs_progress_from_index_int(idx, T_ref))
    return ref_steps, ref_prog_ints


# ==========================
# 6*. 多视角两时刻（Δ进度）推理流程
# ==========================

def run_multiview_delta_inference(
    model,
    processor,
    task_description: str,
    target_demo_folder: str,
    t1_index: int,
    t2_index: int,
    view_names: List[str],
    ref_demo_folder: Optional[str] = None,
    num_ref_frames: int = 4,
    ref_jitter: int = 2,
    seed: int = 0,
    max_new_tokens: int = 64,
):
    # 准备 reference（可选）
    ref_steps, ref_prog_ints = prepare_multiview_reference(
        ref_demo_folder=ref_demo_folder,
        view_names=view_names,
        num_ref_frames=num_ref_frames,
        jitter=ref_jitter,
        seed=seed,
    )

    # 准备目标两时刻（同一 demo 下多视角）
    # 为每个视角列出帧文件，取最短长度为 T，确保索引有效
    view_to_files: Dict[str, List[Path]] = {}
    for v in view_names:
        v_dir = Path(target_demo_folder) / v
        if not v_dir.exists():
            raise FileNotFoundError(f"View folder not found: {v_dir}")
        files = list_image_files(str(v_dir))
        if len(files) < 2:
            raise ValueError(f"Not enough frames under {v_dir}")
        view_to_files[v] = files
    T = min(len(files) for files in view_to_files.values())
    if not (0 <= t1_index < T and 0 <= t2_index < T):
        raise IndexError(f"t1_index/t2_index out of range for T={T}")

    t1_views: List[Image.Image] = []
    t2_views: List[Image.Image] = []
    for v in view_names:
        t1_views.append(Image.open(view_to_files[v][t1_index]).convert("RGB"))
        t2_views.append(Image.open(view_to_files[v][t2_index]).convert("RGB"))

    messages = build_qwen_messages_multiview(
        task_desc=task_description,
        ref_steps_views=ref_steps,
        ref_progress_ints=ref_prog_ints,
        target_t1_views=t1_views,
        target_t2_views=t2_views,
        view_names_order=view_names,
    )

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_texts = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return output_texts[0]

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
    # ===== 推理参数（与数据集生成保持一致的多视角与提示）=====
    task_description = "Close the microwave"
    view_names = ["first_person_camera", "left_hand_camera", "right_hand_camera"]

    # 参考 demo（可选）：若不想提供参考，设为 None
    # 可根据你的数据实际路径修改为对应 task/demo 目录（包含上述视角子目录）
    ref_demo_folder = "Example_data/L90K6CloseTheMicrowave/L90K6CloseTheMicrowave_1757730832298644"

    # 目标 demo：从同一任务的另一个 episode 里选两个时刻做 delta
    target_demo_folder = "Example_data/L90K6CloseTheMicrowave/L90K6CloseTheMicrowave_1757730544608118"
    # 选择两个帧索引（根据你目录下帧的数量与排序进行调整）
    t1_index = 29
    t2_index = 16
    print(f"expetected delta progress: {(t2_index - t1_index/30 *100)}")

    # 参考 steps 的数量与抖动，可与训练构造脚本参数风格一致
    num_ref_frames = 4
    ref_jitter = 2

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

    set_global_determinism(seed=0)
    try:
        model.eval()
    except Exception:
        pass

    # ===== 运行多视角 Δ进度推理 =====
    output_text = run_multiview_delta_inference(
        model=model,
        processor=processor,
        task_description=task_description,
        target_demo_folder=target_demo_folder,
        t1_index=t1_index,
        t2_index=t2_index,
        view_names=view_names,
        ref_demo_folder=ref_demo_folder,
        num_ref_frames=num_ref_frames,
        ref_jitter=ref_jitter,
        seed=0,
        max_new_tokens=64,
    )
    print("=== Raw model output ===")
    print(output_text)
    print("========================")

    # 解析只包含一个整数 D 的输出（允许带 + / - 号）
    m = re.search(r"([+-]?\d+)", output_text.strip())
    if m:
        try:
            d = int(m.group(1))
            if d < -100 or d > 100:
                print(f"[WARN] Parsed delta out of range: {d}")
            else:
                print(f"Parsed delta progress D = {d:+d}")
        except Exception:
            print("[WARN] Failed to parse integer delta from output.")
    else:
        print("[WARN] No integer delta found in output.")

if __name__ == "__main__":
    main()
