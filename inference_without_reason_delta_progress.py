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
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def is_video_file(path: str) -> bool:
    p = Path(path)
    return p.is_file() and p.suffix.lower() in VIDEO_EXTS


def load_video_frames(video_path: str) -> List[Image.Image]:
    frames: List[Image.Image] = []
    if cv2 is not None:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(rgb))
        cap.release()
    elif imageio is not None:
        try:
            reader = imageio.get_reader(video_path)
            for fr in reader:
                if isinstance(fr, np.ndarray):
                    img = Image.fromarray(fr).convert("RGB")
                else:
                    img = Image.fromarray(np.asarray(fr)).convert("RGB")
                frames.append(img)
            reader.close()
        except Exception as e:
            raise ValueError(f"Failed to read video via imageio: {video_path}, err={e}")
    else:
        raise RuntimeError("Neither OpenCV nor imageio is available to read videos.")
    if not frames:
        raise ValueError(f"No frames decoded from video: {video_path}")
    return frames


def load_frames_from_source(source: str) -> List[Image.Image]:
    """
    source 可以是帧文件夹或视频文件路径，返回 PIL.Image 列表（按时间顺序）。
    """
    if os.path.isdir(source):
        img_paths = list_image_paths(source)
        return [Image.open(p).convert("RGB") for p in img_paths]
    if is_video_file(source):
        return load_video_frames(source)
    raise ValueError(f"Unsupported source: {source}")


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


def build_demo_delta_examples(
    demo_source: str,
    num_example_pairs: int = 6,
    seed: int = 0,
) -> List[Tuple[Image.Image, Image.Image, float]]:
    """
    从 demo 源（视频或图像文件夹）中构造若干 (prev, curr, delta%) 示例对。
    delta 定义为 curr 相对 prev 的进度增量（基于线性绝对进度），允许为正或负。
    """
    rng = random.Random(seed)
    frames = load_frames_from_source(demo_source)
    total = len(frames)
    if total < 2:
        raise ValueError(f"Not enough frames in demo source: {demo_source}")

    examples: List[Tuple[Image.Image, Image.Image, float]] = []
    # 线性绝对进度：progress[i] = 100 * i / (total-1)
    # 随机采样若干对 (i, j)，i != j，可正可负
    indices = list(range(total))
    for _ in range(num_example_pairs):
        i = rng.choice(indices[:-1])  # 保证至少有另一个不同帧
        # 随机选择与 i 不同的 j
        j_choices = [k for k in indices if k != i]
        j = rng.choice(j_choices)
        prev_idx, curr_idx = i, j
        prev_img = frames[prev_idx]
        curr_img = frames[curr_idx]
        prev_prog = 100.0 * prev_idx / (total - 1)
        curr_prog = 100.0 * curr_idx / (total - 1)
        delta_pct = curr_prog - prev_prog  # 可正可负
        examples.append((prev_img, curr_img, float(delta_pct)))
    rng.shuffle(examples)
    return examples


# ==========================
# 3. 构造目标轨迹帧（包含 shuffle）
# ==========================

def build_target_frames(
    traj_folder: str,
    prev_frame_id: int,
    curr_frame_id: int,
    seed: int = 0,
) -> Tuple[List[Image.Image], List[int], List[int]]:
    """
    返回 [initial_image, previous_image, current_image]
    """

    # 兼容帧文件夹或视频文件
    frames_all = load_frames_from_source(traj_folder)
    indices = [0, prev_frame_id, curr_frame_id]
    frames_subsampled = [frames_all[idx] for idx in indices]
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
    ax_plot.set_title("Delta Progress (%)", fontsize=10)
    ax_plot.set_ylim(-100, 100)
    ax_plot.set_xlim(1, max(2, len(predicted_progress_list)))
    ax_plot.set_xlabel("Step Index")
    ax_plot.set_ylabel("Delta Progress (%)")
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
    demo_delta_examples: Optional[List[Tuple[Image.Image, Image.Image, float]]] = None,
) -> list:
    """
    利用 (initial_frame, previous_frame, current_frame) 构造 Qwen 的 messages，预测 current 相对 previous 的增量进度。

    target_frames[0] 视为 initial scene (0%)，
    target_frames[1] 为 previous frame，
    target_frames[2] 为 current frame（需要预测相对 previous 的 delta）。
    """
    assert len(target_frames) >= 3, "target_frames 需要包含 initial、previous、current 三帧"

    initial_frame = target_frames[0]
    previous_frame = target_frames[1]
    current_frame = target_frames[2]

    content = []

    # ---- 任务说明（强调delta为连续数值）----
    intro_text = (
        f"You are an expert roboticist tasked with estimating the DELTA in task completion percentage "
        f"for a robot performing: {task_description}.\n"
        "The delta percentage is a real-valued number between -100 and 100 (inclusive), "
        "representing how much progress the current frame has made relative to the previous frame.\n"
        "Positive values mean progress towards completion; negative values mean regression or undoing progress.\n"
        "Values like -15, -3, 0, 12, 47, 95 are all valid.\n"
        "0 means no additional progress compared to the previous frame. 100 means full completion happened within this step.\n\n"
    )
    content.append({"type": "text", "text": intro_text})

    # ---- In-context 示例部分（delta 示例对）----
    if demo_delta_examples is not None and len(demo_delta_examples) > 0:
        content.append({
            "type": "text",
            "text": "Here are example frame pairs with their delta task completion percentages:\n"
        })
        for i, (prev_img, curr_img, delta_pct) in enumerate(demo_delta_examples, start=1):
            content.append({"type": "text", "text": f"\nExample {i}:\nPrevious frame (t-1):"})
            content.append({"type": "image", "image": prev_img})
            content.append({"type": "text", "text": "Current frame (t):"})
            content.append({"type": "image", "image": curr_img})
            content.append({"type": "text", "text": f"At this step, the delta task completion is {delta_pct:.1f}%.\n"})

    # ---- Initial scene（target 轨迹的第一帧）----
    content.append({"type": "text", "text": "\nNow you will estimate the completion for the new episode.\n\n"})
    content.append({"type": "text", "text": "\nInitial robot scene (new episode):\n"})
    content.append({"type": "image", "image": initial_frame})
    content.append({
        "type": "text",
        "text": (
            "In this initial robot scene, the task completion percentage is defined to be 0%.\n\n"
        ),
    })

    # ---- 说明如何预测 delta（previous -> current）----
    query_text = (
        f"For the task of {task_description}, we now provide a previous frame and a current frame from the same episode.\n"
        "Estimate the DELTA task completion percentage of the current frame RELATIVE TO the previous frame.\n\n"
        "After your analysis, you MUST provide your numeric answer in the following EXACT format:\n"
        "Delta Progress: <number between 0 and 100>%\n\n"
        "Do not include extra text after the percent sign. Use intermediate values whenever appropriate.\n\n"
        "Here are the reference frames:\n"
    )
    content.append({"type": "text", "text": query_text})

    # ---- 插入 previous 与 current ----
    content.append({"type": "text", "text": f"\nPrevious frame (reference for delta):\n"})
    content.append({"type": "image", "image": previous_frame})
    content.append({"type": "text", "text": f"\nCurrent frame (estimate delta relative to previous):\n"})
    content.append({"type": "image", "image": current_frame})

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
    prev_frame_id: int = 15,
    curr_frame_id: int = 20,
    num_demo_frames: int = 12,
    num_demo_pairs: int = 6,
    max_new_tokens: int = 256,
):
    # 准备 demo in-context
    if demo_traj_folder is not None:
        # 使用 delta 示例对
        demo_examples = None
        demo_delta_examples = build_demo_delta_examples(
            demo_traj_folder, num_example_pairs=num_demo_pairs, seed=seed
        )
    else:
        demo_examples = None
        demo_delta_examples = None
    # 准备目标轨迹的打乱帧
    target_frames = build_target_frames(
        target_traj_folder,
        prev_frame_id=prev_frame_id,
        curr_frame_id=curr_frame_id
    )

    messages = build_qwen_messages(
        task_description=task_description,
        demo_examples=demo_examples,
        target_frames=target_frames,
        demo_delta_examples=demo_delta_examples,
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
    # 支持视频文件或帧文件夹
    demo_traj_folder = "data/1/right_shoulder"  # 可改为如 "data/1/right_shoulder.mp4"
    target_traj_folder = "data/2/right_shoulder"  # 可改为如 "data/2/right_shoulder.mp4"
    # ===== 加载模型 =====
    model_name = "Qwen/Qwen3-VL-2B-Instruct"
    print(f"Loading model: {model_name}")
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        dtype="auto",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_name)

    # ===== 生成带进度变化图的视频 =====
    set_global_determinism(seed=0)
    try:
        model.eval()
    except Exception:
        pass
    # 载入目标源的所有帧以便推理与显示
    target_frames_all = load_frames_from_source(target_traj_folder)
    frame_indices = list(range(1, len(target_frames_all), 5))
    predicted_list: List[Optional[float]] = []
    expected_list: List[Optional[float]] = []
    composite_frames: List[np.ndarray] = []
    output_video_path = "outputs/delta_progress_video.mp4"
    fps = 4

    for step_idx in range(1, len(frame_indices)):
        prev_id = frame_indices[step_idx - 1]
        curr_id = frame_indices[step_idx]
        expected_delta = 100.0 * (curr_id - prev_id) / (len(target_frames_all) - 1)
        output_text = run_gvl_qwen_for_trajectory(
            model=model,
            processor=processor,
            task_description=task_description,
            target_traj_folder=target_traj_folder,
            demo_traj_folder=demo_traj_folder,
            seed=0,
            prev_frame_id=prev_id,
            curr_frame_id=curr_id,
            num_demo_frames=12,
            num_demo_pairs=6,
            max_new_tokens=2048,
        )
        print("=== Raw model output ===")
        print(output_text)
        print("========================")
        # 解析模型输出：期望格式
        # "Delta Progress: <number>%", 允许负号
        def _parse_completion_output(text: str):
            """
            从模型输出中解析出delta百分比。返回 (percent_float, description)；
            若无法解析，返回 (None, None)。
            """
            # 优先匹配明确格式
            m = re.search(r"Delta\s*Progress\s*:\s*([+-]?[0-9]+(?:\.[0-9]+)?)\s*%", text, flags=re.IGNORECASE)
            if m:
                try:
                    pct = float(m.group(1))
                    return pct, None
                except ValueError:
                    pass
            # 退而求其次：抓取任意 0~100 的百分号数字（最后一个最可能为答案）
            candidates = re.findall(r"([+-]?[0-9]+(?:\.[0-9]+)?)\s*%", text)
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
            print(f"Parsed -> Delta Progress: {parsed_pct:.2f}%")
        else:
            print("Parsed -> Delta Progress: <unparsed>")
        print(f"Expected delta for pair ({prev_id}->{curr_id}): {expected_delta:.2f}%")
        # 聚合并渲染
        predicted_list.append(parsed_pct if parsed_pct is not None else np.nan)
        expected_list.append(expected_delta)
        cur_img = target_frames_all[curr_id]
        comp = render_progress_frame(cur_img, predicted_list, expected_list)
        composite_frames.append(comp)

    # 写出视频
    save_video(composite_frames, output_video_path, fps=fps)
    print(f"Video saved to: {output_video_path}")

if __name__ == "__main__":
    main()
