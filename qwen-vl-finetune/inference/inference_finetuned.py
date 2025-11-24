"""
使用微调后的LoRA适配器进行推理的脚本
支持单张图片、多张图片和视频输入
"""
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from peft import PeftModel

# ======================
# 配置参数
# ======================
# 基础模型路径（从adapter_config.json中获取）
BASE_MODEL_PATH = "/home/lightwheel/erdao.liang/Qwen3-VL/models/Qwen-VL-2B-Instruct"

# LoRA适配器路径（微调后的输出目录）
ADAPTER_PATH = "./output"

# 是否使用flash attention（推荐，可以加速并节省内存）
USE_FLASH_ATTENTION = True

# 设备配置
DEVICE_MAP = "auto"  # 自动分配到可用GPU
DTYPE = "auto"  # 自动选择数据类型

# ======================
# 加载模型和处理器
# ======================
print("正在加载基础模型...")
if USE_FLASH_ATTENTION:
    base_model = AutoModelForImageTextToText.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=DTYPE,
        attn_implementation="flash_attention_2",
        device_map=DEVICE_MAP,
        trust_remote_code=True
    )
else:
    base_model = AutoModelForImageTextToText.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=DTYPE,
        device_map=DEVICE_MAP,
        trust_remote_code=True
    )

print("正在加载LoRA适配器...")
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval()  # 设置为评估模式

print("正在加载处理器...")
processor = AutoProcessor.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)

print("模型加载完成！")


# ======================
# 推理函数
# ======================
def inference_with_images(messages, max_new_tokens=128):
    """
    使用微调后的模型进行推理
    
    Args:
        messages: 消息列表，格式参考下面的示例
        max_new_tokens: 最大生成token数
    
    Returns:
        生成的文本
    """
    # 准备输入
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )
    inputs = inputs.to(model.device)
    
    # 推理生成
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # 使用贪婪解码
            temperature=None,
        )
    
    # 截取新生成的部分（去掉输入部分）
    generated_ids_trimmed = [
        out_ids[len(in_ids):] 
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    
    # 解码输出
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )
    
    return output_text[0]


# ======================
# 使用示例
# ======================

def example_single_image():
    """单张图片示例"""
    print("\n=== 单张图片推理示例 ===")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "path/to/your/image.jpg"},  # 替换为你的图片路径
                {"type": "text", "text": "这张图片中有什么？"}
            ]
        }
    ]
    result = inference_with_images(messages)
    print(f"模型输出: {result}")


def example_multiple_images():
    """多张图片示例"""
    print("\n=== 多张图片推理示例 ===")
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
    result = inference_with_images(messages)
    print(f"模型输出: {result}")


def example_video():
    """视频示例"""
    print("\n=== 视频推理示例 ===")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": "path/to/your/video.mp4"},  # 替换为你的视频路径
                {"type": "text", "text": "这个视频中发生了什么？"}
            ]
        }
    ]
    result = inference_with_images(messages)
    print(f"模型输出: {result}")


def example_custom_task():
    """自定义任务示例（根据你的微调任务）"""
    print("\n=== 自定义任务推理示例 ===")
    # 根据你的微调任务调整这个示例
    # 例如：机器人操作进度判断任务
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "path/to/frame1.jpg"},
                {"type": "image", "image": "path/to/frame2.jpg"},
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
    result = inference_with_images(messages, max_new_tokens=128)
    print(f"模型输出: {result}")


if __name__ == "__main__":
    # 运行示例（取消注释你想要运行的示例）
    
    # example_single_image()
    # example_multiple_images()
    # example_video()
    example_custom_task()
    
    # 或者直接在这里编写你的推理代码
    # messages = [
    #     {
    #         "role": "user",
    #         "content": [
    #             {"type": "image", "image": "your_image_path.jpg"},
    #             {"type": "text", "text": "你的问题"}
    #         ]
    #     }
    # ]
    # result = inference_with_images(messages)
    # print(result)

