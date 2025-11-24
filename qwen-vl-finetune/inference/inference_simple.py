"""
简单的推理示例 - 使用微调后的LoRA适配器
"""
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from peft import PeftModel

# 1. 加载基础模型
print("加载基础模型...")
base_model = AutoModelForImageTextToText.from_pretrained(
    "/home/lightwheel/erdao.liang/Qwen3-VL/models/Qwen-VL-2B-Instruct",
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True
)

# 2. 加载LoRA适配器
print("加载LoRA适配器...")
model = PeftModel.from_pretrained(base_model, "./output")
model.eval()

# 3. 加载处理器
print("加载处理器...")
processor = AutoProcessor.from_pretrained(
    "/home/lightwheel/erdao.liang/Qwen3-VL/models/Qwen-VL-2B-Instruct",
    trust_remote_code=True
)

# 4. 准备输入（替换为你的图片路径）
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "path/to/your/image.jpg"},  # 修改这里！
            {"type": "text", "text": "描述这张图片。"}  # 修改这里！
        ]
    }
]

# 5. 处理输入
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
).to(model.device)

# 6. 生成输出
print("正在推理...")
with torch.no_grad():
    generated_ids = model.generate(**inputs, max_new_tokens=128)

# 7. 解码输出
generated_ids_trimmed = [
    out_ids[len(in_ids):] 
    for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False
)

print(f"\n模型输出: {output_text[0]}")

