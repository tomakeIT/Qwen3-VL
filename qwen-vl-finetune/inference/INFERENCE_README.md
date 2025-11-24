# 微调模型推理指南

本指南说明如何使用微调后的LoRA适配器进行推理。

## 快速开始

### 方法1：使用简单脚本（推荐新手）

```bash
# 编辑 inference_simple.py，修改图片路径和问题
# 然后运行：
python inference_simple.py
```

### 方法2：使用完整脚本（推荐）

```bash
python inference_finetuned.py
```

## 详细说明

### 1. 加载微调后的模型

微调完成后，模型包含两部分：
- **基础模型**：`/home/lightwheel/erdao.liang/Qwen3-VL/models/Qwen-VL-2B-Instruct`
- **LoRA适配器**：`./output` 目录（包含 `adapter_model.safetensors` 和 `adapter_config.json`）

加载步骤：
```python
from transformers import AutoModelForImageTextToText, AutoProcessor
from peft import PeftModel

# 1. 加载基础模型
base_model = AutoModelForImageTextToText.from_pretrained(
    BASE_MODEL_PATH,
    torch_dtype="auto",
    device_map="auto"
)

# 2. 加载LoRA适配器
model = PeftModel.from_pretrained(base_model, "./output")
model.eval()

# 3. 加载处理器
processor = AutoProcessor.from_pretrained(BASE_MODEL_PATH)
```

### 2. 准备输入

#### 单张图片示例
```python
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "path/to/image.jpg"},
            {"type": "text", "text": "这张图片中有什么？"}
        ]
    }
]
```

#### 多张图片示例
```python
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "path/to/image1.jpg"},
            {"type": "image", "image": "path/to/image2.jpg"},
            {"type": "text", "text": "比较这两张图片的差异。"}
        ]
    }
]
```

#### 视频示例
```python
messages = [
    {
        "role": "user",
        "content": [
            {"type": "video", "video": "path/to/video.mp4"},
            {"type": "text", "text": "这个视频中发生了什么？"}
        ]
    }
]
```

### 3. 进行推理

```python
# 处理输入
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
).to(model.device)

# 生成输出
with torch.no_grad():
    generated_ids = model.generate(**inputs, max_new_tokens=128)

# 解码输出
generated_ids_trimmed = [
    out_ids[len(in_ids):] 
    for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False
)
print(output_text[0])
```

## 完整示例

### 示例1：单张图片问答

```python
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from peft import PeftModel

# 加载模型
base_model = AutoModelForImageTextToText.from_pretrained(
    "/home/lightwheel/erdao.liang/Qwen3-VL/models/Qwen-VL-2B-Instruct",
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True
)
model = PeftModel.from_pretrained(base_model, "./output")
model.eval()
processor = AutoProcessor.from_pretrained(
    "/home/lightwheel/erdao.liang/Qwen3-VL/models/Qwen-VL-2B-Instruct",
    trust_remote_code=True
)

# 准备消息
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "your_image.jpg"},
            {"type": "text", "text": "描述这张图片的主要内容。"}
        ]
    }
]

# 推理
inputs = processor.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=True,
    return_dict=True, return_tensors="pt"
).to(model.device)

with torch.no_grad():
    generated_ids = model.generate(**inputs, max_new_tokens=128)

generated_ids_trimmed = [
    out_ids[len(in_ids):] 
    for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True
)
print(output_text[0])
```

### 示例2：机器人操作进度判断（根据你的微调任务）

```python
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "frame1.jpg"},
            {"type": "image", "image": "frame2.jpg"},
            {
                "type": "text",
                "text": (
                    "You are a progress judge for robot manipulation.\n"
                    "The current task is: put the white mug on the plate.\n"
                    "\n"
                    "Two images are provided in temporal order: first, then second.\n"
                    "Your job is to estimate how much the second image has advanced the task relative to the first.\n"
                    "\n"
                    "Define delta_progress = progress(second) - progress(first).\n"
                    "Output a real number strictly in [-1, 1].\n"
                    "Respond with ONLY the number. No explanation, no units."
                )
            }
        ]
    }
]
# ... 然后使用相同的推理代码
```

## 注意事项

1. **路径设置**：
   - 确保图片/视频路径正确
   - 可以使用绝对路径或相对路径

2. **内存优化**：
   - 如果显存不足，可以设置 `device_map="cpu"` 使用CPU（会很慢）
   - 或者使用 `torch_dtype=torch.float16` 减少显存占用

3. **Flash Attention**：
   - 如果安装了flash-attn，可以在加载模型时启用：
   ```python
   base_model = AutoModelForImageTextToText.from_pretrained(
       BASE_MODEL_PATH,
       attn_implementation="flash_attention_2",
       ...
   )
   ```

4. **生成参数**：
   - `max_new_tokens`: 控制最大生成长度
   - `do_sample`: True使用采样，False使用贪婪解码
   - `temperature`: 控制随机性（仅在do_sample=True时有效）

## 常见问题

**Q: 如何批量处理多张图片？**
A: 使用循环，每次处理一张图片或一组图片。

**Q: 如何提高推理速度？**
A: 
- 启用flash attention
- 使用更大的batch size（如果显存允许）
- 减少max_new_tokens

**Q: 模型输出不符合预期？**
A: 
- 检查是否正确加载了适配器
- 确认输入格式正确
- 尝试调整生成参数（temperature, top_p等）

## 相关文件

- `inference_simple.py`: 最简单的推理示例
- `inference_finetuned.py`: 完整的推理脚本，包含多个示例
- `output/adapter_config.json`: LoRA适配器配置
- `output/adapter_model.safetensors`: LoRA权重文件

