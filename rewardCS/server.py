
import base64
import os
import sys
import argparse
import tempfile
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Add parent directory to sys.path to allow imports from utils and inference
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from inference.inferencer import DeltaProgressInference
from utils.data_formatting import build_qwen_messages
from utils.utils import list_image_files
from utils.frame_sampling import sample_reference_frames_from_demo
from utils.prompt import build_prompt

app = FastAPI()

# Global state
service_state = {
    "inference_model": None,
    "current_task_context": None
}

class InitTaskRequest(BaseModel):
    reference_demo_path: Optional[str]
    task_desc: str
    reference_views: List[str]
    # Config for reference sampling
    avg_frames: int = 5
    frames_min: int = 3
    frames_max: int = 10
    frames_std: float = 2.0
    jitter: int = 0
    
    # Target view names (can be set here or per query, but usually fixed for task)
    target_views: List[str]

class PredictRequest(BaseModel):
    """
    推理请求（仅支持图像字典模式）：
    - t1_images / t2_images（视角名 -> base64 编码图像字典）
    """

    # 视角名 -> base64 图像
    t1_images: Dict[str, str]
    t2_images: Dict[str, str]

    # 可选地覆盖 target_views；通常可以省略，由 key 自动推导
    target_views: Optional[List[str]] = None

@app.post("/init_task")
def init_task(req: InitTaskRequest):
    if service_state["inference_model"] is None:
        raise HTTPException(status_code=500, detail="Model not loaded. Please start server with model arguments.")
    
    print(f"Initializing task: {req.task_desc}")
    
    # Prepare reference frames
    # If reference_demo_path is provided, sample frames.
    # If not, we handle it (prompt supports no reference)
    
    ref_img_paths = []
    ref_progress_ints = []
    
    if req.reference_demo_path:
        if not os.path.exists(req.reference_demo_path):
             raise HTTPException(status_code=400, detail=f"Reference path does not exist: {req.reference_demo_path}")
             
        ref_img_paths, ref_progress_ints = sample_reference_frames_from_demo(
            avg_frames=req.avg_frames,
            min_frames=req.avg_frames,
            max_frames=req.avg_frames,
            std=0,
            reference_demo_path=req.reference_demo_path,
            reference_views=req.reference_views,
            ref_jitter=0,
        )
    
    # Store context
    service_state["current_task_context"] = {
        "ref_img_paths": ref_img_paths,
        "ref_progress_ints": ref_progress_ints,
        "reference_view_names": req.reference_views,
        "task_desc": req.task_desc,
        "default_target_views": req.target_views
    }
    
    # TODO: If we were to implement true KV cache prefill, we would run the model 
    # on the prefix constructed from ref_img_paths and task_desc here, 
    # and store the past_key_values in service_state.
    
    return {"status": "initialized", "num_ref_frames": len(ref_progress_ints)}

@app.post("/predict")
def predict(req: PredictRequest):
    context = service_state["current_task_context"]
    if context is None:
        raise HTTPException(status_code=400, detail="No task initialized. Call /init_task first.")
    
    inference = service_state["inference_model"]

    # 统一构造 target_views / t1_paths / t2_paths 三个量，后续下游逻辑保持不变
    tmp_files: List[str] = []

    try:
        # -------- 图像字典模式：视角名 -> base64 图像 --------
        t1_images = req.t1_images or {}
        t2_images = req.t2_images or {}

        if set(t1_images.keys()) != set(t2_images.keys()):
            raise HTTPException(
                status_code=400,
                detail="t1_images 和 t2_images 的视角 key 集合必须完全一致",
            )

        # 视角名顺序优先使用 req.target_views，否则使用 dict 的插入顺序
        if req.target_views is not None:
            if set(req.target_views) != set(t1_images.keys()):
                raise HTTPException(
                    status_code=400,
                    detail="req.target_views 与 t*_images 的 key 集合不一致",
                )
            target_views = req.target_views
        else:
            target_views = list(t1_images.keys())

        def _save_base64_image_to_temp(b64_str: str) -> str:
            try:
                data = base64.b64decode(b64_str)
            except Exception:
                raise HTTPException(
                    status_code=400,
                    detail="图像 base64 解码失败，请检查编码是否正确",
                )
            fd, path = tempfile.mkstemp(suffix=".png")
            with os.fdopen(fd, "wb") as f:
                f.write(data)
            tmp_files.append(path)
            return path

        t1_paths: List[str] = []
        t2_paths: List[str] = []
        for view_name in target_views:
            if view_name not in t1_images or view_name not in t2_images:
                raise HTTPException(
                    status_code=400,
                    detail=f"视角 '{view_name}' 在 t1_images 或 t2_images 中缺失",
                )
            t1_paths.append(_save_base64_image_to_temp(t1_images[view_name]))
            t2_paths.append(_save_base64_image_to_temp(t2_images[view_name]))

        # Validation
        if len(t1_paths) != len(target_views) or len(t2_paths) != len(target_views):
            raise HTTPException(
                status_code=400,
                detail="Number of images does not match number of target views",
            )

        # Build prompt using shared context + specific query images
        img_paths, human_str = build_prompt(
            ref_img_paths=context["ref_img_paths"],
            ref_progress_ints=context["ref_progress_ints"],
            target_img_paths_t1=t1_paths,
            target_img_paths_t2=t2_paths,
            reference_view_names=context["reference_view_names"],
            target_view_names=target_views,
            task_desc=context["task_desc"],
        )

        # Build messages
        # Note: build_qwen_messages expects absolute paths, build_prompt returns them if input was absolute.
        # We assume client sends absolute paths or paths valid relative to server CWD.
        # Ideally client sends absolute paths.
        msgs = build_qwen_messages(human_str, img_paths)

        # Run batch inference
        rewards = inference.infer_from_messages_batch([msgs])

        return {"reward": rewards[0]}
    finally:
        # 清理本次请求生成的临时图片文件
        for p in tmp_files:
            try:
                os.remove(p)
            except OSError:
                pass


def start_server(args):
    print(f"Loading model from {args.base_model} with adapter {args.adapter}")
    service_state["inference_model"] = DeltaProgressInference(
        base_model_path=args.base_model,
        adapter_path=args.adapter,
    )
    
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=str, default="models/Qwen-VL-2B-Instruct", help="Base model path")
    parser.add_argument("--adapter", type=str, default="/home/lightwheel/erdao.liang/Qwen3-VL/qwen-vl-finetune/output/archive/checkpoint-30800-whitemugonplate-1122-final", help="LoRA adapter path")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    
    args = parser.parse_args()
    start_server(args)

