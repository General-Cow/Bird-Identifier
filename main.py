from transformers import pipeline
from PIL import Image
import torch
import pprint

pipe = pipeline(
    "image-text-to-text",
    model="Qwen/Qwen3-VL-4B-Instruct",
    device="cuda" if torch.cuda.is_available() else "cpu"
)

print("Model device:", pipe.model.device)
# print("CUDA available:", torch.cuda.is_available())
# print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")

image = Image.open("data/IMG_0315.jpg").convert("RGB")
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
                "text":
                """You are identifying a bird from a photograph. 

                Location: Virginia, USA
                Date: 11 May
                Habitat: Lake

                Identify the bird in the photograph.

                Consider geographic distribution and seasonal occurrence,
                but do not assume that a species is present merely because
                it is common in this location.
                Return ONLY valid JSON. Do not include Markdown, code fences, or any
                text outside the JSON object.

                Use exactly this structure:

                {
                    "species": "Common name of the most likely species",
                    "scientific_name": "Scientific name",
                    "confidence": "high/medium/low",
                    "reasoning": "Reasoning for why this is the most likely species",
                    "alternatives": [
                        {
                            "species": "Alternative species",
                            "reason": "Why this is less likely"
                        }
                    ]
                }""" # play with this. consider just a reasoning string instead of a list of strings
            }
        ]
    }
]

result = pipe(messages, max_new_tokens=512)#, temperature=0.7, top_p=0.9)
response = result[0]["generated_text"][1]["content"]
# pprint.pprint(result[0]["generated_text"][1]["content"])
print(response) # keep reworking on this. I want to get the model to return a JSON object with the bird identification and reasoning. The current output is not in the desired format.