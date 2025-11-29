
import base64
from io import BytesIO
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import requests

try:
    from PIL import Image
except ImportError:  # PIL is only required when image mode is used
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

    # ---------------- New: send pre-loaded images directly ----------------
    @staticmethod
    def _encode_image_to_base64(img: Any) -> str:
        """
        Convert various common image formats into a base64 string (PNG encoded).
        Supported input types:
        - pre-loaded PIL.Image
        - file path (str)
        - torch.Tensor or numpy.ndarray (shape HWC or CHW, values in [0,1] or [0,255])
        """
        # Lazy import so that we don't hard-require these libs when image mode is unused
        global Image
        if Image is None:
            try:
                from PIL import Image as Image  # type: ignore
            except Exception as e:  # pragma: no cover - 环境相关
                raise RuntimeError(
                    "Pillow is required to encode images. Please install it via `pip install pillow`."
                ) from e

        # 1) PIL.Image instance
        if hasattr(Image, "Image") and isinstance(img, Image.Image):  # type: ignore[attr-defined]
            pil_img = img
        else:
            # 2) If it's a string, treat as a file path
            if isinstance(img, str):
                pil_img = Image.open(img).convert("RGB")  # type: ignore[operator]
            else:
                # 3) Try torch.Tensor / numpy.ndarray
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
                        raise ValueError("Only 3D image tensors (CHW or HWC) are supported")
                    # Normalize to [0, 255]
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
                        raise ValueError("Only 3D image arrays (HWC or CHW) are supported")
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
                        f"Unsupported image type: {type(img)}. "
                        "Expected PIL.Image, file path, torch.Tensor or numpy.ndarray."
                    )

        buf = BytesIO()
        pil_img.save(buf, format="PNG")
        encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
        return encoded

    def predict(
        self,
        t1: Union[Mapping[str, Any], Sequence[Mapping[str, Any]]],
        t2: Union[Mapping[str, Any], Sequence[Mapping[str, Any]]],
    ):
        """
        Get reward(s) for one or multiple (t1, t2) time steps (supports batch).

        Usage:
        1) Single query:
            - t1: Dict[view_name, image]
            - t2: Dict[view_name, image]
        2) Batch:
            - t1: List[Dict[view_name, image]]
            - t2: List[Dict[view_name, image]]
        `image` can be PIL.Image / torch.Tensor / numpy.ndarray / path string

        Args:
            t1: images or image batch at time t1
            t2: images or image batch at time t2

        Returns:
            Single query: int
            Batch: List[int]
        """
        # Normalize into batch lists
        is_single = isinstance(t1, Mapping) and isinstance(t2, Mapping)
        if is_single:
            t1_list: List[Mapping[str, Any]] = [t1]  # type: ignore[list-item]
            t2_list: List[Mapping[str, Any]] = [t2]  # type: ignore[list-item]
        else:
            if not isinstance(t1, Sequence) or not isinstance(t2, Sequence):
                raise ValueError("In batch mode, t1 and t2 must both be sequences (List[Dict[view, image]]).")
            t1_list = list(t1)  # type: ignore[arg-type]
            t2_list = list(t2)  # type: ignore[arg-type]
            if len(t1_list) != len(t2_list):
                raise ValueError("t1 和 t2 的 batch 长度必须一致")

        queries: List[Dict[str, Dict[str, str]]] = []

        for t1_item, t2_item in zip(t1_list, t2_list):
            if set(t1_item.keys()) != set(t2_item.keys()):
                raise ValueError("For each query, t1 and t2 must have the same set of view keys.")

            t1_images: Dict[str, str] = {}
            t2_images: Dict[str, str] = {}

            for view_name in t1_item.keys():
                t1_images[view_name] = self._encode_image_to_base64(t1_item[view_name])
                t2_images[view_name] = self._encode_image_to_base64(t2_item[view_name])

            queries.append(
                {
                    "t1_images": t1_images,
                    "t2_images": t2_images,
                }
            )

        payload = {
            "queries": queries,
        }

        resp = requests.post(f"{self.url}/predict", json=payload)
        resp.raise_for_status()
        rewards = resp.json()["rewards"]
        return rewards[0] if is_single else rewards


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
        # Target views are hard-coded here; predict() will use dict keys to match them
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

    # Use the dict form: "view_name -> pre-loaded image"
    try:
        from PIL import Image as PILImage  # type: ignore
    except Exception as e:
        print("This example requires Pillow to load images. Please install it via: pip install pillow")
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

    # Single prediction
    reward = client.predict(t1_images, t2_images)
    print(f"Prediction Result (Reward): {reward}")

    # Batch prediction: repeat the same query N times as a simple example
    batch_size = 4
    t1_batch = [t1_images] * batch_size
    t2_batch = [t2_images] * batch_size
    rewards = client.predict(t1_batch, t2_batch)
    print(f"Batch Prediction Result (Rewards): {rewards}")
