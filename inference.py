from transformers import AutoModelForImageTextToText, AutoProcessor

# default: Load the model on the available device(s)
model = AutoModelForImageTextToText.from_pretrained(
    "Qwen/Qwen3-VL-2B-Instruct", dtype="auto", device_map="auto"
)

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
# model = AutoModelForImageTextToText.from_pretrained(
#     "Qwen/Qwen3-VL-235B-A22B-Instruct",
#     dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     device_map="auto",
# )
root_path = "/home/erdao/Documents/datasets/lightwheel_progress/frames/x7s_1/L90L6PutTheWhiteMugOnThePlate_1762235411920142/left_shoulder/"
processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": f"{root_path}img_frame0056.jpg"},
            {"type": "image", "image": f"{root_path}img_frame0001.jpg"},
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
                    "For example, 0.8 means the task almost fully completed at the second image.\n"
                    "0.2 means the task slightly regression at the second image.\n"
                    "0 means no meaningful progress.\n"
                    "Respond with ONLY the number. No explanation, no units."
                )
            }
        ],
    }
]




# Preparation for inference
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
)
inputs = inputs.to(model.device)

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)