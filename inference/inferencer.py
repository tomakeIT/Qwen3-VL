"""
Delta Progress推理核心类
"""

from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import importlib.util
from pathlib import Path
import time
import torch
from typing import Any, Dict, List, Optional, Sequence, Tuple

from transformers import AutoModelForImageTextToText, AutoProcessor
from peft import PeftModel
from utils.data_formatting import parse_delta_progress_int


@lru_cache(maxsize=1)
def _resolve_process_vision_info():
    try:
        from qwen_vl_utils import process_vision_info

        return process_vision_info
    except ImportError:
        vision_process_path = (
            Path(__file__).resolve().parent.parent
            / "qwen-vl-utils"
            / "src"
            / "qwen_vl_utils"
            / "vision_process.py"
        )
        if not vision_process_path.is_file():
            raise RuntimeError(
                "未找到 qwen_vl_utils.process_vision_info；"
                "请安装 qwen-vl-utils，或确保仓库内 qwen-vl-utils/src/qwen_vl_utils/vision_process.py 存在。"
            )
        spec = importlib.util.spec_from_file_location(
            "local_qwen_vl_utils_vision_process",
            str(vision_process_path),
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(f"无法加载本地 vision_process.py: {vision_process_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        process_vision_info = getattr(module, "process_vision_info", None)
        if process_vision_info is None:
            raise RuntimeError(f"本地 vision_process.py 缺少 process_vision_info: {vision_process_path}")
        return process_vision_info


class DeltaProgressInference:
    """Delta Progress推理核心类"""
    
    def __init__(self, base_model_path: str, adapter_path: str, device: Optional[str] = None):
        """初始化模型和处理器"""
        print("[single_gpu] loading model")
        if device is not None and device.startswith("cuda:"):
            torch.cuda.set_device(int(device.split(":", 1)[1]))
        device_map = {"": device} if device is not None else "auto"
        base_model = AutoModelForImageTextToText.from_pretrained(
            base_model_path,
            dtype="auto",
            attn_implementation="flash_attention_2",
            device_map=device_map,
            trust_remote_code=True
        )
        
        self.model = PeftModel.from_pretrained(base_model, adapter_path)
        self.model.eval()
        
        self.processor = AutoProcessor.from_pretrained(base_model_path, trust_remote_code=True)
        # 对于decoder-only架构，需要设置左填充以确保生成结果正确
        if hasattr(self.processor, 'tokenizer') and self.processor.tokenizer is not None:
            self.processor.tokenizer.padding_side = 'left'
            # 确保pad_token_id已设置（如果未设置，使用eos_token_id）
            if self.processor.tokenizer.pad_token_id is None:
                self.processor.tokenizer.pad_token_id = self.processor.tokenizer.eos_token_id
        image_processor = getattr(self.processor, "image_processor", None)
        self._image_patch_size = int(getattr(image_processor, "patch_size", 14))
        self._last_batch_stats = self._empty_batch_stats()
        print("[single_gpu] ready")

    @staticmethod
    def _empty_batch_stats() -> Dict[str, float]:
        return {
            "prepare_wait_sec": 0.0,
            "prepare_cpu_sec": 0.0,
            "transfer_sec": 0.0,
            "generate_sec": 0.0,
            "decode_sec": 0.0,
            "total_sec": 0.0,
        }

    def get_last_batch_stats(self) -> Dict[str, float]:
        return dict(self._last_batch_stats)

    def _prepare_batch_inputs(
        self,
        messages_batch: Sequence[List[Dict[str, Any]]],
        padding: bool,
    ) -> Tuple[Any, float]:
        process_vision_info = _resolve_process_vision_info()
        prepare_start = time.perf_counter()
        text = self.processor.apply_chat_template(
            list(messages_batch),
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            list(messages_batch),
            return_video_kwargs=True,
            image_patch_size=self._image_patch_size,
        )
        inputs = self.processor(
            text=text,
            images=image_inputs,
            videos=video_inputs,
            padding=padding,
            return_tensors="pt",
            **video_kwargs,
        )
        return inputs, time.perf_counter() - prepare_start

    def _generate_and_decode(
        self,
        inputs,
        max_new_tokens: int,
    ) -> Tuple[List[Optional[int]], float, float]:
        generate_start = time.perf_counter()
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
            )
        generate_sec = time.perf_counter() - generate_start

        decode_start = time.perf_counter()
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_texts = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        decode_sec = time.perf_counter() - decode_start
        return [parse_delta_progress_int(text) for text in output_texts], generate_sec, decode_sec

    def infer_from_messages(self, messages: List[Dict[str, Any]], max_new_tokens: int = 128) -> Optional[int]:
        """从messages格式进行推理，返回delta_progress整数"""
        results = self.infer_from_messages_batch(
            [messages],
            max_new_tokens=max_new_tokens,
            batch_size=1,
        )
        return results[0] if results else None

    def infer_from_messages_batch(
        self, 
        messages_list: List[List[Dict[str, Any]]], 
        max_new_tokens: int = 128,
        batch_size: Optional[int] = None,
    ) -> List[Optional[int]]:
        self._last_batch_stats = self._empty_batch_stats()
        if len(messages_list) == 0:
            return []
        effective_batch_size = len(messages_list)
        if batch_size is not None and batch_size > 0:
            effective_batch_size = int(batch_size)

        message_batches = [
            messages_list[start_idx:start_idx + effective_batch_size]
            for start_idx in range(0, len(messages_list), effective_batch_size)
        ]
        results: List[Optional[int]] = []
        total_start = time.perf_counter()

        with ThreadPoolExecutor(max_workers=1) as executor:
            first_batch = message_batches[0]
            pending_future = executor.submit(
                self._prepare_batch_inputs,
                first_batch,
                len(first_batch) > 1,
            )

            for batch_idx, message_batch in enumerate(message_batches):
                wait_start = time.perf_counter()
                cpu_inputs, prepare_cpu_sec = pending_future.result()
                self._last_batch_stats["prepare_wait_sec"] += time.perf_counter() - wait_start
                self._last_batch_stats["prepare_cpu_sec"] += prepare_cpu_sec

                next_batch_idx = batch_idx + 1
                if next_batch_idx < len(message_batches):
                    next_batch = message_batches[next_batch_idx]
                    pending_future = executor.submit(
                        self._prepare_batch_inputs,
                        next_batch,
                        len(next_batch) > 1,
                    )

                transfer_start = time.perf_counter()
                inputs = cpu_inputs.to(self.model.device)
                self._last_batch_stats["transfer_sec"] += time.perf_counter() - transfer_start

                batch_results, generate_sec, decode_sec = self._generate_and_decode(
                    inputs,
                    max_new_tokens=max_new_tokens,
                )
                self._last_batch_stats["generate_sec"] += generate_sec
                self._last_batch_stats["decode_sec"] += decode_sec
                results.extend(batch_results)

        self._last_batch_stats["total_sec"] = time.perf_counter() - total_start
        return results

