import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import imageio
import numpy as np
from PIL import Image
from tqdm import tqdm
 


VIEW_TO_FILENAME = {
    "first_person": "isaac_replay_action_state_first_person_camera.mp4",
    "left_shoulder": "isaac_replay_action_state_left_shoulder_camera.mp4",
    "right_shoulder": "isaac_replay_action_state_right_shoulder_camera.mp4",
    "eye_in_hand": "isaac_replay_action_state_eye_in_hand_camera.mp4",
    "right_hand": "isaac_replay_action_state_right_hand_camera.mp4",
    "left_hand": "isaac_replay_action_state_left_hand_camera.mp4",
}

DEFAULT_SHORT_STEPS = [1]          # 局部步长
DEFAULT_LONG_STEPS = [2, 4, 8, 16]  # 长步长（会自动裁剪到长度内）


@dataclass
class DiffThreshold:
    pixel_mean_abs: float = 0.01  # 像素级平均绝对差阈值（0~1）


def load_task_descriptions(task_json: Path) -> Dict[str, str]:
    with open(task_json, "r") as f:
        return json.load(f)


def list_episode_dirs(task_dir: Path) -> List[Path]:
    episode_dirs: List[Path] = []
    for entry in sorted(task_dir.iterdir()):
        if not entry.is_dir():
            continue
        if (entry / "replay_results").exists():
            episode_dirs.append(entry)
    return episode_dirs


def read_video_frames(video_path: Path, max_frames: int, min_frames: int = 4) -> List[np.ndarray]:
    if not video_path.exists():
        return []
    reader = imageio.get_reader(video_path)
    try:
        frames = [frame for frame in reader]
    finally:
        reader.close()
    if len(frames) < min_frames:
        return []
    # 统一抽样为 max_frames（或更少，如果视频更短）
    target_frames = min(max_frames, len(frames))
    if len(frames) > target_frames:
        indices = np.linspace(0, len(frames) - 1, target_frames).astype(int)
        frames = [frames[i] for i in indices]
    return frames


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_frames_to_dir(frames: List[np.ndarray], out_dir: Path, prefix: str) -> List[Path]:
    ensure_dir(out_dir)
    saved_paths: List[Path] = []
    for idx, frame in enumerate(frames):
        img = Image.fromarray(frame)
        img_path = out_dir / f"{prefix}_frame{idx:04d}.jpg"
        # 默认高质量 JPEG
        # img.save(img_path, format="JPEG", quality=90)
        saved_paths.append(img_path.resolve())
    return saved_paths


def pixel_mean_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    # 归一化到 [0,1]
    if a.dtype != np.float32:
        a = a.astype(np.float32)
    if b.dtype != np.float32:
        b = b.astype(np.float32)
    return float(np.mean(np.abs(a - b)) / 255.0)


def path_step_pixel_mean_diff(
    frames: List[np.ndarray],
    i: int,
    j: int,
) -> float:
    # 计算路径 i->j 每步相邻像素差分的平均
    if i == j:
        return 0.0
    step_indices = range(i, j) if j > i else range(j, i)
    px_diffs: List[float] = []
    for k in step_indices:
        a_idx, b_idx = (k, k + 1)
        if a_idx < 0 or b_idx >= len(frames):
            continue
        px_diffs.append(pixel_mean_abs_diff(frames[a_idx], frames[b_idx]))
    if len(px_diffs) == 0:
        return 0.0
    return float(np.mean(px_diffs))


def is_static_pair(
    frames: List[np.ndarray],
    i: int,
    j: int,
    thr: DiffThreshold,
) -> bool:
    px_mean = path_step_pixel_mean_diff(frames, i, j)
    return px_mean < thr.pixel_mean_abs


def progress_delta_forward(i: int, j: int, T: int) -> float:
    # c_{i,j} = (j - i) / (T - i), 限制在 [-1,1]
    if i >= T - 1:
        return 0.0
    delta = (j - i) / max(1, (T - i))
    return float(max(-1.0, min(1.0, delta)))


def build_conversation_regression(task_desc: Optional[str]) -> List[Dict[str, str]]:
    # 两张图：<image>\n<image>
    instruction = (
        (f"Task: {task_desc}\n" if task_desc else "")
        + "You are given two images from the same trajectory in the provided order (first, then second). "
          "Estimate the change in task progress (second - first) and output a single real number in the range [-1, 1]. "
          "Positive means progress toward the goal, negative means moving away, and near 0 means no change. "
          "Respond with only the number."
    )
    return [
        {"from": "human", "value": "<image>\n<image>\n" + instruction},
    ]


def make_entry_two_images(
    img_a: Path,
    img_b: Path,
    conversations_prefix: List[Dict[str, str]],
    gpt_value: str,
) -> Dict:
    return {
        "image": [str(img_a), str(img_b)],  # 直接使用绝对路径，训练时可不设置 data_path
        "conversations": conversations_prefix + [{"from": "gpt", "value": gpt_value}],
    }


def valid_index(idx: int, length: int) -> bool:
    return 0 <= idx < length


def build_pairs_for_anchor(
    i: int,
    T: int,
    short_steps: List[int],
    long_steps: List[int],
) -> List[Tuple[int, int]]:
    pairs: List[Tuple[int, int]] = []
    # local forward/backward
    if valid_index(i + 1, T):
        pairs.append((i, i + 1))      # forward
        pairs.append((i + 1, i))      # backward (取负)
    # long-range 多步长
    for step in long_steps:
        if step <= 0:
            continue
        if valid_index(i + step, T):
            pairs.append((i, i + step))       # forward
            pairs.append((i + step, i))       # backward
        if valid_index(i - step, T):
            pairs.append((i, i - step))       # backward（负进度）
            pairs.append((i - step, i))       # forward
    # 也保留短步长，避免和 long_steps 完全重复
    for step in short_steps:
        if step <= 0:
            continue
        if valid_index(i + step, T):
            pairs.append((i, i + step))
            pairs.append((i + step, i))
    # 去重
    uniq = list(dict.fromkeys(pairs).keys())
    return uniq


def main():
    parser = argparse.ArgumentParser(
        description="从 LightwheelData 自动生成 VLAC 风格的进度标签数据（QwenVL SFT JSONL）。"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="/home/erdao/Documents/LightwheelData",
        help="包含 x7s_* 任务文件夹的根目录",
    )
    parser.add_argument(
        "--task_json",
        type=str,
        default="/home/erdao/Documents/LightwheelData/task_description.json",
        help="任务名到文字描述的映射 JSON",
    )
    parser.add_argument(
        "--output_jsonl",
        type=str,
        default="/home/erdao/Documents/datasets/lightwheel_progress/train.jsonl",
        help="导出的 JSONL 文件路径",
    )
    parser.add_argument(
        "--frames_dir",
        type=str,
        default="/home/erdao/Documents/datasets/lightwheel_progress/frames",
        help="抽帧输出目录",
    )
    parser.add_argument(
        "--views",
        type=str,
        nargs="+",
        default=["first_person", "left_shoulder", "right_shoulder"],
        choices=list(VIEW_TO_FILENAME.keys()),
        help="要处理的视角列表",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=64,
        help="每段视频统一抽样的帧数",
    )
    parser.add_argument(
        "--episodes_per_task",
        type=int,
        default=1000000,
        help="每个任务采样的 episode 上限（默认基本不限制）",
    )
    parser.add_argument(
        "--num_anchors_per_episode",
        type=int,
        default=8,
        help="每个 episode 采样的 anchor 数量",
    )
    parser.add_argument(
        "--short_steps",
        type=int,
        nargs="+",
        default=DEFAULT_SHORT_STEPS,
        help="局部步长集合",
    )
    parser.add_argument(
        "--long_steps",
        type=int,
        nargs="+",
        default=DEFAULT_LONG_STEPS,
        help="长步长集合",
    )
    parser.add_argument(
        "--pixel_diff_thr",
        type=float,
        default=0.01,
        help="像素平均绝对差阈值（静态判定）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    data_root = Path(args.data_root).expanduser()
    frames_dir = Path(args.frames_dir).expanduser()
    ensure_dir(frames_dir)
    output_jsonl = Path(args.output_jsonl).expanduser()
    ensure_dir(output_jsonl.parent)

    task_desc = load_task_descriptions(Path(args.task_json))

    thr = DiffThreshold(pixel_mean_abs=args.pixel_diff_thr)

    num_written = 0
    with open(output_jsonl, "w", encoding="utf-8") as fout:
        for task_key, description in task_desc.items():
            task_dir = data_root / task_key
            if not task_dir.exists():
                print(f"[WARN] 跳过 {task_key}: 目录不存在 ({task_dir}).")
                continue

            episodes = list_episode_dirs(task_dir)
            if not episodes:
                print(f"[WARN] 未在 {task_dir} 下找到有效 episodes。")
                continue

            # 限制 episodes 数量
            episodes = episodes[: args.episodes_per_task]
            for episode_dir in tqdm(episodes, desc=f"Task {task_key}", unit="ep"):
                replay_dir = episode_dir / "replay_results"
                episode_name = episode_dir.name

                for view in args.views:
                    video_path = replay_dir / VIEW_TO_FILENAME[view]
                    frames = read_video_frames(video_path, args.max_frames)
                    if len(frames) < 4:
                        # 太短，跳过
                        continue

                    # 保存帧
                    out_dir = frames_dir / task_key / episode_name / view
                    saved_paths = save_frames_to_dir(frames, out_dir, prefix="img")
                    T = len(saved_paths)

                    # 随机采样 anchors
                    # 避免 i 接近末尾导致 T - i 过小
                    valid_anchor_max = max(2, T - 2)
                    anchors = sorted(random.sample(range(0, valid_anchor_max), k=min(args.num_anchors_per_episode, valid_anchor_max)))

                    for i in anchors:
                        pairs = build_pairs_for_anchor(i, T, args.short_steps, args.long_steps)
                        # 去除越界
                        pairs = [(a, b) for (a, b) in pairs if valid_index(a, T) and valid_index(b, T) and a != b]
                        if not pairs:
                            continue

                        # 会话前缀（提示词）仅回归
                        conv_prefix = build_conversation_regression(description)

                        for a, b in pairs:
                            # 静态段强制 0 标签
                            if is_static_pair(frames, a, b, thr):
                                label_text = "0"
                            else:
                                c = progress_delta_forward(a, b, T)
                                label_text = f"{c:.4f}"

                            entry = make_entry_two_images(
                                img_a=saved_paths[a],
                                img_b=saved_paths[b],
                                conversations_prefix=conv_prefix,
                                gpt_value=label_text,
                            )
                            fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
                            num_written += 1

    print(f"[OK] 生成完成：{output_jsonl}，共写入 {num_written} 条样本。")


if __name__ == "__main__":
    main()


