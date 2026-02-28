"""
多 GPU 并行推理核心类
自动将任务分配到多张 GPU 并行处理
"""

import os
import torch
import torch.multiprocessing as mp
from typing import List, Dict, Any, Optional, Tuple, Callable
from tqdm import tqdm
import numpy as np

from transformers import AutoModelForImageTextToText, AutoProcessor
from peft import PeftModel
from utils.data_formatting import parse_delta_progress_int


def _init_worker(gpu_id, base_model_path, adapter_path):
    """在每个 worker 进程中初始化模型（绑定到指定 GPU）"""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    print(f"[GPU {gpu_id}] 正在加载模型...")
    base_model = AutoModelForImageTextToText.from_pretrained(
        base_model_path,
        dtype="auto",
        attn_implementation="flash_attention_2",
        device_map="auto",
        trust_remote_code=True
    )
    
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    
    processor = AutoProcessor.from_pretrained(base_model_path, trust_remote_code=True)
    if hasattr(processor, 'tokenizer') and processor.tokenizer is not None:
        processor.tokenizer.padding_side = 'left'
        if processor.tokenizer.pad_token_id is None:
            processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id
    
    print(f"[GPU {gpu_id}] 模型加载完成！")
    return model, processor


def _infer_batch_on_gpu(args):
    """在指定 GPU 上批量推理
    
    Args:
        args: (gpu_id, base_model_path, adapter_path, messages_list, max_new_tokens)
    
    Returns:
        List[Optional[int]]: 预测结果列表
    """
    gpu_id, base_model_path, adapter_path, messages_list, batch_size, max_new_tokens = args
    print(f"[GPU {gpu_id}] worker 启动，样本数={len(messages_list)}")

    # 显式绑定到目标 GPU，避免多进程下 device_map="auto" 误分配到同一卡
    torch.cuda.set_device(gpu_id)
    device = f"cuda:{gpu_id}"

    # 显式指定设备映射到当前 GPU
    base_model = AutoModelForImageTextToText.from_pretrained(
        base_model_path,
        dtype="auto",
        attn_implementation="flash_attention_2",
        device_map={"": device},
        trust_remote_code=True
    )
    
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    
    processor = AutoProcessor.from_pretrained(base_model_path, trust_remote_code=True)
    if hasattr(processor, 'tokenizer') and processor.tokenizer is not None:
        processor.tokenizer.padding_side = 'left'
        if processor.tokenizer.pad_token_id is None:
            processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id
    
    results = []
    with torch.no_grad():
        # 在每个 GPU worker 内继续按 batch_size 切分，避免显存峰值过高
        for i in range(0, len(messages_list), batch_size):
            sub_batch = messages_list[i:i + batch_size]
            inputs = processor.apply_chat_template(
                sub_batch,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                padding=True
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
                clean_up_tokenization_spaces=False
            )

            results.extend([parse_delta_progress_int(text) for text in output_texts])
    
    return results


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
        
        available_gpus = torch.cuda.device_count()
        if available_gpus <= 0:
            raise RuntimeError("未检测到可用 CUDA GPU，无法进行多 GPU 推理")

        if num_gpus is None or num_gpus == -1:
            self.num_gpus = available_gpus
        else:
            self.num_gpus = min(num_gpus, available_gpus)

        self.gpu_ids = list(range(self.num_gpus))
        print(
            f"MultiGPU Inference 初始化完成，将使用 {self.num_gpus}/{available_gpus} 张 GPU，"
            f"gpu_ids={self.gpu_ids}"
        )
    
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
        print(f"messages_list length: {len(messages_list)}")
        print(f"batch_size: {batch_size}")
        if len(messages_list) == 0:
            return []
        
        # 如果只有一条数据或少于 num_gpus，退化为单卡模式
        if len(messages_list) < self.num_gpus or self.num_gpus == 1:
            return self._infer_single_gpu(messages_list, batch_size, max_new_tokens, desc)
        
        # 将数据分配到各个 GPU
        tasks_per_gpu = len(messages_list) // self.num_gpus
        remainder = len(messages_list) % self.num_gpus
        
        gpu_tasks = []
        start_idx = 0
        for gpu_id in self.gpu_ids:
            # 分配任务，多余的给前面的 GPU
            end_idx = start_idx + tasks_per_gpu + (1 if gpu_id < remainder else 0)
            sublist = messages_list[start_idx:end_idx]
            if len(sublist) > 0:
                gpu_tasks.append((gpu_id, self.base_model_path, self.adapter_path, sublist, batch_size, max_new_tokens))
            start_idx = end_idx
        
        # 使用进程池并行处理
        ctx = mp.get_context('spawn')
        with ctx.Pool(processes=len(gpu_tasks)) as pool:
            results = list(tqdm(
                pool.imap(_infer_batch_on_gpu, gpu_tasks),
                total=len(gpu_tasks),
                desc=f"{desc} (across {len(gpu_tasks)} GPUs)"
            ))
        
        # 合并结果（保持原始顺序）
        all_results = []
        for r in results:
            all_results.extend(r)
        
        return all_results
    
    def _infer_single_gpu(
        self, 
        messages_list: List[List[Dict[str, Any]]], 
        batch_size: int,
        max_new_tokens: int,
        desc: str
    ) -> List[Optional[int]]:
        """单 GPU 推理（退化为原来的 DeltaProgressInference）"""
        from inferencer import DeltaProgressInference
        
        inference = DeltaProgressInference(self.base_model_path, self.adapter_path)
        
        results = []
        for i in tqdm(range(0, len(messages_list), batch_size), desc=desc):
            batch = messages_list[i:i + batch_size]
            batch_results = inference.infer_from_messages_batch(batch, max_new_tokens)
            results.extend(batch_results)
        
        return results
    
    def infer_from_messages(self, messages: List[Dict[str, Any]], max_new_tokens: int = 128) -> Optional[int]:
        """单条推理（兼容单卡接口）"""
        results = self.infer_from_messages_batch([messages], batch_size=1, max_new_tokens=max_new_tokens)
        return results[0] if results else None
