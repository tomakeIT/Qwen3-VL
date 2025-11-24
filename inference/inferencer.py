"""
data sample -> inference result
"""

import json
import os
import sys
import torch
import argparse
from typing import List, Dict, Any, Optional

from transformers import AutoModelForImageTextToText, AutoProcessor
from peft import PeftModel
from utils.data_formatting import data_sample_to_messages_and_answer, parse_delta_progress_int


class DeltaProgressInference:
    """Delta Progress推理核心类"""
    
    def __init__(self, base_model_path: str, adapter_path: str, use_flash_attention: bool = True):
        """初始化模型和处理器"""
        print("正在加载基础模型...")
        base_model = AutoModelForImageTextToText.from_pretrained(
            base_model_path,
            torch_dtype="auto",
            attn_implementation="flash_attention_2",
            device_map="auto",
            trust_remote_code=True
        )
        
        print("正在加载LoRA适配器...")
        self.model = PeftModel.from_pretrained(base_model, adapter_path)
        self.model.eval()
        
        print("正在加载处理器...")
        self.processor = AutoProcessor.from_pretrained(base_model_path, trust_remote_code=True)
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
    

def main():

    parser = argparse.ArgumentParser(description="Level 1: 从data sample推理")
    parser.add_argument("--base-model", type=str, required=True, help="基础模型路径")
    parser.add_argument("--adapter", type=str, required=True, help="LoRA适配器路径")
    parser.add_argument("--data-sample", type=str, required=True, help="data sample文件路径")
    parser.add_argument("--data-root", type=str, default="", help="data sample里所有图片路径的根目录")
    
    args = parser.parse_args()
    
    inference = DeltaProgressInference(
        base_model_path=args.base_model,
        adapter_path=args.adapter,
        use_flash_attention=not args.no_flash_attn
    )
    
    with open(args.data_sample, "r", encoding="utf-8") as f:
        data_sample = json.load(f)
    
    messages = data_sample_to_messages_and_answer(data_sample, data_root=args.data_root)
    result = inference.infer_from_messages(messages)
    print(f"Delta Progress: {result}")


if __name__ == "__main__":
    main()

