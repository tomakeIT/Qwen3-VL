
import base64
from io import BytesIO
from typing import Any, Dict, List, Mapping, Optional

import requests

try:
    from PIL import Image
except ImportError:  # PIL 只在需要时才真正用到
    Image = None

class RewardClient:
    def __init__(self, url: str = "http://localhost:8000"):
        self.url = url.rstrip("/")
        
    def init_task(
        self, 
        task_desc: str, 
        reference_views: List[str],
        target_views: List[str],
        reference_demo_path: Optional[str] = None,
        # Config options
        avg_frames: int = 5,
    ):
        """
        Initialize the server with a task context.
        This 'prefills' the server with the reference demo and task description.
        """
        payload = {
            "task_desc": task_desc,
            "reference_views": reference_views,
            "target_views": target_views,
            "reference_demo_path": reference_demo_path,
            "avg_frames": avg_frames,
        }
        resp = requests.post(f"{self.url}/init_task", json=payload)
        resp.raise_for_status()
        return resp.json()

    # ---------------- 新版：直接发送已加载图像 ----------------
    @staticmethod
    def _encode_image_to_base64(img: Any) -> str:
        """
        将各种常见格式的图像统一转成 base64 字符串（PNG 编码）。
        支持：
        - 已加载的 PIL.Image
        - 文件路径（str）
        - torch.Tensor 或 numpy.ndarray（形状为 HWC 或 CHW，值在 [0,1] 或 [0,255]）
        """
        # 延迟导入，避免在不用图像模式时强依赖这些库
        global Image
        if Image is None:
            try:
                from PIL import Image as Image  # type: ignore
            except Exception as e:  # pragma: no cover - 环境相关
                raise RuntimeError(
                    "Pillow 未安装，无法对图像进行编码。请先 `pip install pillow`。"
                ) from e

        # 1) 如果是 PIL.Image
        if hasattr(Image, "Image") and isinstance(img, Image.Image):  # type: ignore[attr-defined]
            pil_img = img
        else:
            # 2) 如果是字符串，看作文件路径
            if isinstance(img, str):
                pil_img = Image.open(img).convert("RGB")  # type: ignore[operator]
            else:
                # 3) 尝试 torch.Tensor / numpy.ndarray
                try:
                    import torch  # type: ignore
                except Exception:
                    torch = None  # type: ignore
                try:
                    import numpy as np  # type: ignore
                except Exception:
                    np = None  # type: ignore

                pil_img = None

                if torch is not None and isinstance(img, torch.Tensor):  # type: ignore[attr-defined]
                    t = img.detach().cpu()
                    if t.ndim != 3:
                        raise ValueError("仅支持 3 维图像张量（CHW 或 HWC）")
                    # 归一化到 [0,255]
                    if t.max() <= 1.0:
                        t = t * 255.0
                    t = t.clamp(0, 255).byte()
                    if t.shape[0] in (1, 3, 4) and t.shape[0] != t.shape[1]:
                        # CHW -> HWC
                        t = t.permute(1, 2, 0)
                    arr = t.numpy()
                    pil_img = Image.fromarray(arr)  # type: ignore[attr-defined]
                elif np is not None and isinstance(img, np.ndarray):  # type: ignore[attr-defined]
                    arr = img
                    if arr.ndim != 3:
                        raise ValueError("仅支持 3 维图像数组（HWC 或 CHW）")
                    arr = arr.astype("float32")
                    if arr.max() <= 1.0:
                        arr = arr * 255.0
                    arr = arr.clip(0, 255).astype("uint8")
                    if arr.shape[0] in (1, 3, 4) and arr.shape[0] != arr.shape[1]:
                        # CHW -> HWC
                        arr = arr.transpose(1, 2, 0)
                    pil_img = Image.fromarray(arr)  # type: ignore[attr-defined]

                if pil_img is None:
                    raise TypeError(
                        f"不支持的图像类型：{type(img)}。"
                        "请传入 PIL.Image、文件路径、torch.Tensor 或 numpy.ndarray。"
                    )

        buf = BytesIO()
        pil_img.save(buf, format="PNG")
        encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
        return encoded

    def predict(
        self,
        t1: Mapping[str, Any],
        t2: Mapping[str, Any],
    ):
        """
        获取一对时刻 (t1, t2) 的奖励。

        只支持一种推荐用法：传「视角名 -> 已加载图像」字典
            - t1: Dict[view_name, image]
            - t2: Dict[view_name, image]
              其中 image 可以是 PIL.Image / torch.Tensor / numpy.ndarray / 路径字符串

        Args:
            t1: t1 时刻的图像
            t2: t2 时刻的图像

        Returns:
            Reward (integer).
        """
        t1_images: Dict[str, str] = {}
        t2_images: Dict[str, str] = {}

        if set(t1.keys()) != set(t2.keys()):
            raise ValueError("t1 和 t2 的视角 key 集合必须完全一致")

        for view_name in t1.keys():
            t1_images[view_name] = self._encode_image_to_base64(t1[view_name])
            t2_images[view_name] = self._encode_image_to_base64(t2[view_name])

        payload = {
            "t1_images": t1_images,
            "t2_images": t2_images,
        }

        resp = requests.post(f"{self.url}/predict", json=payload)
        resp.raise_for_status()
        return resp.json()["reward"]


if __name__ == "__main__":
    import argparse
    import sys
    import os

    # Example usage:
    # python rewardCS/client.py --task-desc "Put the red block on the green block" --ref-demo /path/to/demo --ref-views view1 view2 --target-views view1 view2 --test-t1 /path/to/img1_v1.jpg /path/to/img1_v2.jpg --test-t2 /path/to/img2_v1.jpg /path/to/img2_v2.jpg
    DATA_ROOT="/home/lightwheel/erdao.liang/LightwheelData/"
    TARGET_DEMO_PATH=f"{DATA_ROOT}/1W_Libero_Piper/L90L6PutTheWhiteMugOnThePlate/L90L6PutTheWhiteMugOnThePlate_1757994866950241"
    REFERENCE_DEMO_PATH=f"{DATA_ROOT}/1W_Libero_Piper/L90L6PutTheWhiteMugOnThePlate/L90L6PutTheWhiteMugOnThePlate_1757927758234053"
    parser = argparse.ArgumentParser(description="Test RewardClient")
    parser.add_argument("--url", type=str, default="http://localhost:8000", help="Server URL")
    parser.add_argument("--task-desc", type=str, default="Put the white mug on the plate", help="Task description")
    parser.add_argument("--ref-demo", type=str, help=TARGET_DEMO_PATH)
    parser.add_argument("--ref-views", nargs="+", default=["first_person_camera"], help="Reference views")
    args = parser.parse_args()

    client = RewardClient(url=args.url)

    print(f"Connecting to {args.url}...")
    # 1. Initialize Task
    print(f"Initializing task: '{args.task_desc}'")
    if args.ref_demo:
        print(f"  Reference demo: {args.ref_demo}")
    
    init_resp = client.init_task(
        task_desc=args.task_desc,
        reference_views=args.ref_views,
        # 这里直接在代码里约定 target views，调用 predict 时会用 dict 的 key 来对应视角名
        target_views=["first_person_camera", "left_hand_camera", "right_hand_camera"],
        reference_demo_path=args.ref_demo,
    )
    print("Task initialized successfully.")
    print(f"Server response: {init_resp}")
    start_frame = "frame_000001"
    end_frame = "frame_000045"
    default_t1_rel = [
        f"1W_Libero_Piper/L90L6PutTheWhiteMugOnThePlate/L90L6PutTheWhiteMugOnThePlate_1757994866950241/first_person_camera/{start_frame}.png",
        f"1W_Libero_Piper/L90L6PutTheWhiteMugOnThePlate/L90L6PutTheWhiteMugOnThePlate_1757994866950241/left_hand_camera/{start_frame}.png",
        f"1W_Libero_Piper/L90L6PutTheWhiteMugOnThePlate/L90L6PutTheWhiteMugOnThePlate_1757994866950241/right_hand_camera/{start_frame}.png"
    ]
    default_t2_rel = [
        f"1W_Libero_Piper/L90L6PutTheWhiteMugOnThePlate/L90L6PutTheWhiteMugOnThePlate_1757994866950241/first_person_camera/{end_frame}.png",
        f"1W_Libero_Piper/L90L6PutTheWhiteMugOnThePlate/L90L6PutTheWhiteMugOnThePlate_1757994866950241/left_hand_camera/{end_frame}.png",
        f"1W_Libero_Piper/L90L6PutTheWhiteMugOnThePlate/L90L6PutTheWhiteMugOnThePlate_1757994866950241/right_hand_camera/{end_frame}.png"
    ]
    t1_abs = [os.path.join(DATA_ROOT, p) for p in default_t1_rel]
    t2_abs = [os.path.join(DATA_ROOT, p) for p in default_t2_rel]

    # 使用「视角名 -> 已加载图像」的 dict 形式调用
    try:
        from PIL import Image as PILImage  # type: ignore
    except Exception as e:
        print("示例需要 Pillow 来加载图像，请先安装：pip install pillow")
        raise

    t1_images = {
        "first_person_camera": PILImage.open(t1_abs[0]).convert("RGB"),
        "left_hand_camera": PILImage.open(t1_abs[1]).convert("RGB"),
        "right_hand_camera": PILImage.open(t1_abs[2]).convert("RGB"),
    }
    t2_images = {
        "first_person_camera": PILImage.open(t2_abs[0]).convert("RGB"),
        "left_hand_camera": PILImage.open(t2_abs[1]).convert("RGB"),
        "right_hand_camera": PILImage.open(t2_abs[2]).convert("RGB"),
    }

    reward = client.predict(t1_images, t2_images)
    print(f"Prediction Result (Reward): {reward}")
