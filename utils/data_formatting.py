import os
import re
from typing import List, Dict, Any, Optional
from utils.utils import rel_to_abs_path

################## qwen data sample 和 messages ##################

def build_qwen_data_sample(images: List[str], human_str: str, assistant_answer: str) -> Dict[str, Any]:
    return {
        "image": images,
        "conversations": [
            {"from": "human", "value": human_str},
            {"from": "gpt", "value": assistant_answer},
        ],
    }


def build_qwen_messages(human_str: str, images: List[str]) -> List[Dict[str, Any]]:
    """prompt -> messsages"""
    content = []
    parts = re.split(r'(<image>)', human_str)
    img_idx = 0
    for part in parts:
        if part == "<image>":
            if img_idx < len(images):
                content.append({"type": "image", "image": images[img_idx]})
                img_idx += 1
        elif part.strip():
            content.append({"type": "text", "text": part})
    return [{"role": "user", "content": content}]


def data_sample_to_messages(data_sample: Dict[str, Any], data_root: str) -> List[Dict[str, Any]]:
    """将data sample转换为messages格式，data_root为图片根路径"""
    images = data_sample.get("image", [])
    conversations = data_sample.get("conversations", [])

    abs_images = [rel_to_abs_path(data_root, image_path) for image_path in images]
    
    messages = []
    for conv in conversations:
        role = conv.get("from", "")
        value = conv.get("value", "")
        
        if role == "human":
            content = []
            parts = re.split(r'(<image>)', value)
            img_idx = 0
            for part in parts:
                if part == "<image>":
                    if img_idx < len(abs_images):
                        content.append({"type": "image", "image": abs_images[img_idx]})
                        img_idx += 1
                elif part.strip():
                    content.append({"type": "text", "text": part})
            messages.append({"role": "user", "content": content})
        elif role == "gpt":
            messages.append({"role": "assistant", "content": value})
    
    return messages

################## delta progress 相关函数 ##################

def compute_delta_progress_label_int(i: int, j: int, T: int) -> int:
    """计算 delta progress 标签（整数）"""
    if T <= 0:
        return 0
    c = (j - i) / float(T)
    return int(round(100.0 * c))


def compute_abs_progress_from_index_int(idx: int, total: int) -> int:
    """用帧序号给 reference demo 的绝对进度 0~100（整数）"""
    if total <= 1:
        return 0
    c = idx / float(total - 1)
    return int(round(100.0 * c))


def parse_delta_progress(output_text: str) -> Optional[int]:
    """从模型输出中解析delta_progress整数"""
    match = re.search(r'([+-]?\d+)', output_text.strip())
    if match:
        value = int(match.group(1))
        return max(-100, min(100, value))
    return None

