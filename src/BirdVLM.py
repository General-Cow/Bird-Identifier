import json
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

    @staticmethod
    def _parse_model_json(payload):
        if isinstance(payload, dict):
            return payload

        if not isinstance(payload, str):
            raise TypeError(f"Expected JSON string or dict, got {type(payload).__name__}")

        cleaned = payload.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:].lstrip()

        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            cleaned = cleaned[start:end + 1]

        return json.loads(cleaned)

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
        generated = result[0]["generated_text"][1]["content"]
        return self._parse_model_json(generated)

# Subclasses for specific bird identification models. Need to be implemented for each model if they have different behavior or parameters.
class QwenBirdVLM(BirdVLM):
    def __init__(self):
        super().__init__(model_name="Qwen/Qwen3-VL-4B-Instruct")
