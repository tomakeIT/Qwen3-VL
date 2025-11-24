from typing import List, Dict, Any
import os
from types import SimpleNamespace
from PIL import Image, ImageChops, ImageStat


def dict_to_namespace(d: Dict[str, Any]) -> SimpleNamespace:
    """递归将字典转换为 SimpleNamespace 对象，支持属性访问"""
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [dict_to_namespace(item) for item in d]
    else:
        return d

def abs_to_rel_path(root: str, abs_path: str) -> str:
    """将绝对路径转换成以数据根目录名开头的相对路径"""
    root_name = os.path.basename(os.path.abspath(root))
    rel = os.path.relpath(abs_path, root)
    return os.path.join(root_name, rel)

def rel_to_abs_path(data_root: str, rel_path: str) -> str:
    """将相对路径转换为绝对路径"""
    return os.path.normpath(os.path.join(data_root, rel_path))


def list_subdirs(path: str) -> List[str]:
    """列出目录下的所有子目录"""
    return sorted([d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))])


def list_image_files(path: str) -> List[str]:
    """列出目录下的所有图像文件"""
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and f.lower().endswith(exts)]
    return sorted(files)


def compute_mean_pixel_diff(img_path1: str, img_path2: str) -> float:
    """计算两张图像的平均像素差，归一化到 [0, 1]"""
    img1 = Image.open(img_path1).convert("RGB")
    img2 = Image.open(img_path2).convert("RGB")
    if img1.size != img2.size:
        img2 = img2.resize(img1.size)
    diff = ImageChops.difference(img1, img2)
    stat = ImageStat.Stat(diff)
    mean = sum(stat.mean) / len(stat.mean)
    return mean / 255.0

