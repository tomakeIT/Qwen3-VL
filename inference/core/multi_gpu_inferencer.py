"""
多 GPU 并行推理核心类
自动将任务分配到多张 GPU 并行处理
"""

import atexit
import time
import traceback
import torch
import torch.multiprocessing as mp
from typing import Any, Dict, List, Optional, Tuple
from tqdm import tqdm

from transformers import AutoModelForImageTextToText, AutoProcessor
from peft import PeftModel
from utils.data_formatting import parse_delta_progress_int


def _load_model_and_processor(gpu_id: int, base_model_path: str, adapter_path: str):
    """在指定 GPU 上加载模型和 processor。"""
    torch.cuda.set_device(gpu_id)
    device = f"cuda:{gpu_id}"

    base_model = AutoModelForImageTextToText.from_pretrained(
        base_model_path,
        dtype="auto",
        attn_implementation="flash_attention_2",
        device_map={"": device},
        trust_remote_code=True,
    )

    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()

    processor = AutoProcessor.from_pretrained(base_model_path, trust_remote_code=True)
    if hasattr(processor, "tokenizer") and processor.tokenizer is not None:
        processor.tokenizer.padding_side = "left"
        if processor.tokenizer.pad_token_id is None:
            processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

    return model, processor


def _run_inference_batches(
    model,
    processor,
    messages_list: List[List[Dict[str, Any]]],
    batch_size: int,
    max_new_tokens: int,
    gpu_id: Optional[int] = None,
) -> List[Optional[int]]:
    """对单个 worker 分到的 messages 做分批推理。"""
    results = []
    with torch.no_grad():
        for i in range(0, len(messages_list), batch_size):
            sub_batch = messages_list[i:i + batch_size]
            inputs = processor.apply_chat_template(
                sub_batch,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                padding=True,
            )
            inputs = inputs.to(model.device)

            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
            )

            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]

            output_texts = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            results.extend([parse_delta_progress_int(text) for text in output_texts])

    return results


def _gpu_worker_loop(
    gpu_id: int,
    base_model_path: str,
    adapter_path: str,
    task_queue,
    result_queue,
) -> None:
    """持久化 worker：模型只加载一次，后续复用处理多个 chunk。"""
    try:
        model, processor = _load_model_and_processor(gpu_id, base_model_path, adapter_path)
        result_queue.put(("__ready__", gpu_id, None, None))
    except Exception:
        result_queue.put(("__ready__", gpu_id, None, traceback.format_exc()))
        return

    while True:
        payload = task_queue.get()
        if payload is None:
            break

        request_id, messages_list, batch_size, max_new_tokens = payload
        try:
            results = _run_inference_batches(
                model=model,
                processor=processor,
                messages_list=messages_list,
                batch_size=batch_size,
                max_new_tokens=max_new_tokens,
                gpu_id=gpu_id,
            )
            result_queue.put((request_id, gpu_id, results, None))
        except Exception:
            result_queue.put((request_id, gpu_id, None, traceback.format_exc()))


class MultiGPUDeltaProgressInference:
    """多 GPU 并行推理器
    
    自动将批量任务分配到多张 GPU 并行处理
    """
    
    def __init__(self, base_model_path: str, adapter_path: str, num_gpus: Optional[int] = None):
        """初始化多 GPU 推理器
        
        Args:
            base_model_path: 基础模型路径
            adapter_path: LoRA 适配器路径
            num_gpus: 使用的 GPU 数量，None 或 -1 表示使用所有可用 GPU
        """
        self.base_model_path = base_model_path
        self.adapter_path = adapter_path
        self._ctx = mp.get_context("spawn")
        self._task_queues = []
        self._workers = []
        self._result_queue = None
        self._request_counter = 0
        self._single_gpu_inference = None
        self._closed = False
        
        available_gpus = torch.cuda.device_count()
        if available_gpus <= 0:
            raise RuntimeError("未检测到可用 CUDA GPU，无法进行多 GPU 推理")

        if num_gpus is None or num_gpus == -1:
            self.num_gpus = available_gpus
        else:
            self.num_gpus = min(num_gpus, available_gpus)

        self.gpu_ids = list(range(self.num_gpus))
        print(f"[multi_gpu] using {self.num_gpus}/{available_gpus} GPUs")
        if self.num_gpus > 1:
            self._start_workers()
        atexit.register(self.close)

    def _start_workers(self) -> None:
        """启动常驻 worker，让 chunked 推理避免重复加载模型。"""
        self._result_queue = self._ctx.Queue()

        for gpu_id in self.gpu_ids:
            task_queue = self._ctx.Queue()
            worker = self._ctx.Process(
                target=_gpu_worker_loop,
                args=(gpu_id, self.base_model_path, self.adapter_path, task_queue, self._result_queue),
                daemon=True,
            )
            worker.start()
            self._task_queues.append(task_queue)
            self._workers.append(worker)

        ready_gpus = set()
        while len(ready_gpus) < len(self._workers):
            tag, gpu_id, _, error = self._result_queue.get()
            if tag != "__ready__":
                continue
            if error:
                self.close()
                raise RuntimeError(f"[GPU {gpu_id}] worker 初始化失败:\n{error}")
            ready_gpus.add(gpu_id)
    
    def infer_from_messages_batch(
        self, 
        messages_list: List[List[Dict[str, Any]]], 
        batch_size: int = 8,
        max_new_tokens: int = 128,
        desc: str = "Multi-GPU inference"
    ) -> List[Optional[int]]:
        """批量推理，自动分配到多张 GPU
        
        Args:
            messages_list: messages 列表
            batch_size: 每个 batch 的大小
            max_new_tokens: 最大生成 token 数
            desc: 进度条描述
            
        Returns:
            List[Optional[int]]: 所有预测结果（保持输入顺序）
        """
        if len(messages_list) == 0:
            return []

        if self._closed:
            raise RuntimeError("MultiGPUDeltaProgressInference 已关闭，不能继续推理")

        if self.num_gpus == 1:
            return self._infer_single_gpu(messages_list, batch_size, max_new_tokens, desc)

        total_start = time.perf_counter()
        split_start = total_start
        active_gpu_count = min(self.num_gpus, len(messages_list))
        active_gpu_ids = self.gpu_ids[:active_gpu_count]

        # 将数据分配到各个 GPU
        tasks_per_gpu = len(messages_list) // active_gpu_count
        remainder = len(messages_list) % active_gpu_count

        gpu_tasks = []
        start_idx = 0
        for gpu_id in active_gpu_ids:
            # 分配任务，多余的给前面的 GPU
            end_idx = start_idx + tasks_per_gpu + (1 if gpu_id < remainder else 0)
            sublist = messages_list[start_idx:end_idx]
            if len(sublist) > 0:
                gpu_tasks.append((gpu_id, sublist))
            start_idx = end_idx
        split_elapsed = time.perf_counter() - split_start

        enqueue_start = time.perf_counter()
        request_id = self._request_counter
        self._request_counter += 1

        for gpu_id, sublist in gpu_tasks:
            self._task_queues[gpu_id].put((request_id, sublist, batch_size, max_new_tokens))
        enqueue_elapsed = time.perf_counter() - enqueue_start

        results_by_gpu: Dict[int, List[Optional[int]]] = {}
        wait_start = time.perf_counter()
        iterator = tqdm(
            total=len(gpu_tasks),
            desc=f"{desc} (across {len(gpu_tasks)} GPUs)",
        )
        while len(results_by_gpu) < len(gpu_tasks):
            recv_request_id, gpu_id, results, error = self._result_queue.get()
            if recv_request_id != request_id:
                continue
            if error:
                iterator.close()
                self.close()
                raise RuntimeError(f"[GPU {gpu_id}] 推理失败:\n{error}")
            results_by_gpu[gpu_id] = results
            iterator.update(1)
        iterator.close()
        wait_elapsed = time.perf_counter() - wait_start

        # 合并结果（保持原始顺序）
        merge_start = time.perf_counter()
        all_results = []
        for gpu_id, _ in gpu_tasks:
            all_results.extend(results_by_gpu[gpu_id])
        merge_elapsed = time.perf_counter() - merge_start

        total_elapsed = time.perf_counter() - total_start
        assignment_summary = ", ".join(
            f"gpu{gpu_id}={len(sublist)}"
            for gpu_id, sublist in gpu_tasks
        )
        tqdm.write(
            "[multi_gpu_timing] "
            f"desc={desc}, messages={len(messages_list)}, batch_size={batch_size}, "
            f"assignments=[{assignment_summary}], "
            f"split={split_elapsed:.3f}s, enqueue={enqueue_elapsed:.3f}s, "
            f"wait={wait_elapsed:.3f}s, merge={merge_elapsed:.3f}s, "
            f"total={total_elapsed:.3f}s"
        )

        return all_results
    
    def _infer_single_gpu(
        self, 
        messages_list: List[List[Dict[str, Any]]], 
        batch_size: int,
        max_new_tokens: int,
        desc: str
    ) -> List[Optional[int]]:
        """单 GPU 推理（退化为原来的 DeltaProgressInference）"""
        from inference.core.inferencer import DeltaProgressInference

        if self._single_gpu_inference is None:
            self._single_gpu_inference = DeltaProgressInference(self.base_model_path, self.adapter_path)

        results = []
        for i in tqdm(range(0, len(messages_list), batch_size), desc=desc):
            batch = messages_list[i:i + batch_size]
            batch_results = self._single_gpu_inference.infer_from_messages_batch(batch, max_new_tokens)
            results.extend(batch_results)

        return results
    
    def infer_from_messages(self, messages: List[Dict[str, Any]], max_new_tokens: int = 128) -> Optional[int]:
        """单条推理（兼容单卡接口）"""
        results = self.infer_from_messages_batch([messages], batch_size=1, max_new_tokens=max_new_tokens)
        return results[0] if results else None

    def close(self) -> None:
        """关闭常驻 worker 进程。"""
        if self._closed:
            return

        self._closed = True
        for task_queue in self._task_queues:
            try:
                task_queue.put(None)
            except Exception:
                pass

        for worker in self._workers:
            worker.join(timeout=5)
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=5)

        for task_queue in self._task_queues:
            try:
                task_queue.close()
            except Exception:
                pass

        if self._result_queue is not None:
            try:
                self._result_queue.close()
            except Exception:
                pass

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
