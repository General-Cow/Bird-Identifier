from transformers import pipeline
from PIL import Image
import torch

pipe = pipeline(
    "image-text-to-text",
    model="Qwen/Qwen3-VL-4B-Instruct",
    device="cuda"
)

print("Model device:", pipe.model.device)
# print("CUDA available:", torch.cuda.is_available())
# print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")

image = Image.open("data/IMG_0374.jpg").convert("RGB")
image.thumbnail((1280, 1280))

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": image
            },
            {
                "type": "text",
                "text": "Try to identify the species of birds in this image. Photo taken in Virginia, USA."
            }
        ]
    }
]

result = pipe(messages)

print(result)