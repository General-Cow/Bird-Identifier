import torch
from transformers import pipeline


class BirdVLM:
    def __init__(self, model_name="Qwen/Qwen3-VL-4B-Instruct"):
        self.model_name = model_name
        self.pipe = pipeline(
            "image-text-to-text",
            model=self.model_name,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )

    def identify_bird(self, image, location, date, habitat):
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
                        "text": f"""You are identifying a bird from a photograph. 

                        Location: {location}
                        Date: {date}
                        Habitat: {habitat}

                        Identify the bird in the photograph.

                        Consider geographic distribution and seasonal occurrence,
                        but do not assume that a species is present merely because
                        it is common in this location.
                        Return ONLY valid JSON. Do not include Markdown, code fences, or any
                        text outside the JSON object.

                        Use exactly this structure:

                        {{
                            "species": "Common name of the most likely species",
                            "scientific_name": "Scientific name",
                            "confidence": "high/medium/low",
                            "reasoning": "Reasoning for why this is the most likely species",
                            "alternatives": [
                                {{
                                    "species": "Alternative species",
                                    "reason": "Why this is less likely"
                                }}
                            ]
                        }}"""
                    }
                ]
            }
        ]

        result = self.pipe(messages, max_new_tokens=512)
        return result[0]["generated_text"][1]["content"]

# Subclasses for specific bird identification models. Need to be implemented for each model if they have different behavior or parameters.
class QwenBirdVLM(BirdVLM):
    def __init__(self):
        super().__init__(model_name="Qwen/Qwen3-VL-4B-Instruct")
