import json
import os
import subprocess
from typing import Any, Dict, Sequence

from PIL import Image

from inference.lerobot_io import LeRobotVideoSource


def frame_file_name(frame_index: int) -> str:
    return f"frame_{frame_index:06d}.png"


def resolve_frame_path(image_dir: str, frame_index: int) -> str:
    return os.path.join(image_dir, frame_file_name(frame_index))


def load_episode_image_dirs(
    video_sources: Dict[str, LeRobotVideoSource],
) -> Dict[str, str]:
    image_dirs: Dict[str, str] = {}
    for target_view, video_source in video_sources.items():
        image_dir = video_source.image_dir
        if not image_dir or not os.path.isdir(image_dir):
            raise FileNotFoundError(
                f"缺少预处理后的 images 目录: target_view={target_view}, image_dir={image_dir}"
            )
        image_dirs[target_view] = image_dir
    return image_dirs


def _load_frames_from_image_cache(
    image_dir: str,
    frame_indices: Sequence[int],
) -> Dict[int, Image.Image]:
    frame_cache: Dict[int, Image.Image] = {}
    for frame_index in frame_indices:
        frame_path = resolve_frame_path(image_dir, frame_index)
        if not os.path.exists(frame_path):
            raise FileNotFoundError(frame_path)
        with Image.open(frame_path) as image:
            frame_cache[int(frame_index)] = image.convert("RGB").copy()
    return frame_cache


def _build_select_filter(frame_indices: Sequence[int]) -> str:
    if not frame_indices:
        raise ValueError("frame_indices 不能为空")
    expr = "+".join(f"eq(n\\,{int(frame_index)})" for frame_index in frame_indices)
    return f"select={expr}"


def _decode_frames_with_ffmpeg(
    video_path: str,
    frame_indices: Sequence[int],
    width: int,
    height: int,
    ffmpeg_bin: str = "ffmpeg",
) -> Dict[int, Image.Image]:
    if not frame_indices:
        return {}

    command = [
        ffmpeg_bin,
        "-v",
        "error",
        "-i",
        video_path,
        "-vf",
        _build_select_filter(frame_indices),
        "-vsync",
        "0",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-",
    ]
    completed = subprocess.run(command, capture_output=True, check=False)
    if completed.returncode != 0:
        stderr = completed.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(f"ffmpeg 解码失败: {video_path}\n{stderr}")

    frame_size = width * height * 3
    if frame_size <= 0:
        raise ValueError(f"非法视频分辨率: width={width}, height={height}")
    if len(completed.stdout) % frame_size != 0:
        raise RuntimeError(
            f"ffmpeg 输出大小异常: video={video_path}, bytes={len(completed.stdout)}, frame_size={frame_size}"
        )

    decoded_count = len(completed.stdout) // frame_size
    if decoded_count != len(frame_indices):
        raise RuntimeError(
            f"ffmpeg 解码帧数异常: video={video_path}, expected={len(frame_indices)}, decoded={decoded_count}"
        )

    frame_cache: Dict[int, Image.Image] = {}
    for output_index, frame_index in enumerate(frame_indices):
        start = output_index * frame_size
        end = start + frame_size
        frame_bytes = completed.stdout[start:end]
        frame_cache[int(frame_index)] = Image.frombytes("RGB", (width, height), frame_bytes).copy()

    return frame_cache


def probe_video_info(video_path: str, ffprobe_bin: str = "ffprobe") -> Dict[str, Any]:
    """读取视频的帧数和分辨率，供 reference/demo 抽帧复用。"""
    command = [
        ffprobe_bin,
        "-v",
        "error",
        "-count_frames",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=nb_read_frames,nb_frames,width,height,avg_frame_rate,duration",
        "-of",
        "json",
        video_path,
    ]
    completed = subprocess.run(command, capture_output=True, check=False)
    if completed.returncode != 0:
        stderr = completed.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(f"ffprobe 读取失败: {video_path}\n{stderr}")

    payload = json.loads(completed.stdout.decode("utf-8"))
    streams = payload.get("streams", [])
    if not streams:
        raise RuntimeError(f"ffprobe 没有返回视频流信息: {video_path}")
    stream = streams[0]

    total_frames = stream.get("nb_read_frames") or stream.get("nb_frames")
    if total_frames is None:
        raise RuntimeError(f"ffprobe 无法确定总帧数: {video_path}")

    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "total_frames": int(total_frames),
        "avg_frame_rate": stream.get("avg_frame_rate"),
        "duration": float(stream.get("duration", 0.0)),
    }


def decode_video_frames(
    video_path: str,
    frame_indices: Sequence[int],
    width: int,
    height: int,
    ffmpeg_bin: str = "ffmpeg",
) -> Dict[int, Image.Image]:
    """对任意单路视频解码指定帧。"""
    return _decode_frames_with_ffmpeg(
        video_path=video_path,
        frame_indices=frame_indices,
        width=width,
        height=height,
        ffmpeg_bin=ffmpeg_bin,
    )
