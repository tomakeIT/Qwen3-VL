"""
Delta Progress推理核心类
"""

import torch
from typing import List, Dict, Any, Optional

from transformers import AutoModelForImageTextToText, AutoProcessor
from peft import PeftModel
from utils.data_formatting import parse_delta_progress_int


class DeltaProgressInference:
    """Delta Progress推理核心类"""
    
    def __init__(self, base_model_path: str, adapter_path: str):
        """初始化模型和处理器"""
        print("正在加载基础模型...")
        base_model = AutoModelForImageTextToText.from_pretrained(
            base_model_path,
            dtype="auto",
            attn_implementation="flash_attention_2",
            device_map="auto",  # 自动将模型分配到可用的设备GPU上
            trust_remote_code=True
        )
        
        print("正在加载LoRA适配器...")
        self.model = PeftModel.from_pretrained(base_model, adapter_path)
        self.model.eval()
        
        print("正在加载处理器...")
        self.processor = AutoProcessor.from_pretrained(base_model_path, trust_remote_code=True)
        # 对于decoder-only架构，需要设置左填充以确保生成结果正确
        if hasattr(self.processor, 'tokenizer') and self.processor.tokenizer is not None:
            self.processor.tokenizer.padding_side = 'left'
            # 确保pad_token_id已设置（如果未设置，使用eos_token_id）
            if self.processor.tokenizer.pad_token_id is None:
                self.processor.tokenizer.pad_token_id = self.processor.tokenizer.eos_token_id
        print("模型加载完成！")
    
    
    def infer_from_messages(self, messages: List[Dict[str, Any]], max_new_tokens: int = 128) -> Optional[int]:
        """从messages格式进行推理，返回delta_progress整数"""
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        inputs = inputs.to(self.model.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
            )
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] 
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        
        return parse_delta_progress_int(output_text)
    
    
    def infer_from_messages_batch(
        self, 
        messages_list: List[List[Dict[str, Any]]], 
        max_new_tokens: int = 128
    ) -> List[Optional[int]]:

        # 批量 apply chat template
        inputs = self.processor.apply_chat_template(
            messages_list,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            padding=True # 批量处理时不同样本的序列长度不一致，需要在 apply_chat_template 中启用 padding。
        )
        inputs = inputs.to(self.model.device)
        
        # 批量generate
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
            )
        
        # 批量trim和decode
        generated_ids_trimmed = [
            out_ids[len(in_ids):] 
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output_texts = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        
        return [parse_delta_progress_int(text) for text in output_texts]

