import os
import re
from typing import List, Dict, Any, Optional, Tuple
from utils.utils import rel_to_abs_path

################## qwen data sample 和 messages ##################

def build_qwen_data_sample(img_paths: List[str], human_str: str, answer: str) -> Dict[str, Any]:
    """
    (prompt, images, answer) -> qwen data sample
    image paths are relative paths
    """
    return {
        "image": img_paths,
        "conversations": [
            {"from": "human", "value": human_str},
            {"from": "gpt", "value": answer},
        ],
    }


def build_qwen_messages(human_str: str, img_paths: List[str]) -> List[Dict[str, Any]]:
    """
    (prompt, images) -> messsages
    image paths are absolute paths
    """
    content = []
    parts = re.split(r'(<image>)', human_str)
    img_idx = 0
    for part in parts:
        if part == "<image>":
            if img_idx < len(img_paths):
                content.append({"type": "image", "image": img_paths[img_idx]})
                img_idx += 1
        elif part.strip():
            content.append({"type": "text", "text": part})
    return [{"role": "user", "content": content}]


def data_sample_to_messages_and_answer(data_sample: Dict[str, Any], data_root: str) -> Tuple[List[Dict[str, Any]], Optional[int]]:
    """
    data_sample -> (messages, answer)
    将data sample转换为messages格式，补上data_root得到绝对路径
    """
    img_paths = data_sample.get("image", [])
    conversations = data_sample.get("conversations", [])
    delta_progress = None

    abs_img_paths = [rel_to_abs_path(data_root, img_path) for img_path in img_paths]
    
    messages = []
    for conv in conversations:
        role = conv.get("from", "")
        value = conv.get("value", "")
        
        # human -> user messages
        if role == "human":
            content = []
            parts = re.split(r'(<image>)', value)
            img_idx = 0
            for part in parts:
                if part == "<image>":
                    if img_idx < len(abs_img_paths):
                        content.append({"type": "image", "image": abs_img_paths[img_idx]})
                        img_idx += 1
                elif part.strip():
                    content.append({"type": "text", "text": part})
            messages.append({"role": "user", "content": content})
        # gpt -> assistant answer
        elif role == "gpt":
            delta_progress = parse_delta_progress_int(value)
    
    return messages, delta_progress

################## progress 相关函数 ##################

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


def parse_delta_progress_int(output_text: str) -> Optional[int]:
    """从模型输出中解析delta_progress整数"""
    match = re.search(r'([+-]?\d+)', output_text.strip())
    if match:
        value = int(match.group(1))
        return max(-100, min(100, value))
    return None

