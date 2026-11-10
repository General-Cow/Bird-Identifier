
from pathlib import Path

from inference.base import InferenceBackend


class BirdIdentifier:
    """
    Handles bird identification requests.

    Responsible for:
    - Building identification prompts.
    - Incorporating observation context.
    - Submitting image and text requests through an inference backend.

    Image manipulation and API-specific image encoding are handled
    by other components.
    """

    def __init__(self, backend: InferenceBackend):
        self.backend = backend

    def build_prompt(
        self,
        location: str | None = None,
        observation_date: str | None = None,
    ) -> str:
        """Construct the bird identification prompt."""

#         prompt = """
# Identify the bird shown in the photograph.

# Analyze the visible characteristics carefully, including:

# - Plumage colors and patterns
# - Head markings
# - Beak shape and size
# - Body shape and proportions
# - Wing and tail characteristics
# - Distinguishing field marks

# Provide your identification using the following sections:

# 1. Most likely species, including its scientific name.
# 2. Visual evidence supporting the identification.
# 3. Other plausible species, if there is meaningful ambiguity.
# 4. Qualitative confidence level: High, Medium, or Low.

# Important guidelines:

# - Do not invent field marks that are not visible.
# - Distinguish observed characteristics from assumptions.
# - Do not give a numerical confidence percentage.
# - If the photograph does not support identification to
#   species level, state that explicitly.
# - Consider similar-looking species before settling
#   on an identification.
# """

        prompt = """
Identify the bird in the provided photograph.

Examine visible characteristics:
- Plumage colors and patterns
- Head markings
- Beak shape and size
- Body proportions
- Wing and tail characteristics
- Distinguishing field marks

Identification guidelines:
- Only report characteristics clearly visible in the image.
- Do not invent or infer unobservable field marks.
- Consider plausible lookalike species before selecting
  the leading candidate.
- Include only genuinely plausible alternatives.
- Explain which visible characteristics support or
  distinguish each candidate.
- Do not use speculative or non-diagnostic characteristics
  to justify an identification.
- If distinguishing features cannot be assessed, acknowledge
  uncertainty rather than favoring a species without evidence.
- Consider geographic and seasonal plausibility when
  observation information is provided.
- Always provide the best plausible candidate when one
  exists, even with low confidence. Use null only when no
  meaningful candidate can be suggested.
- Confidence must be exactly "high", "medium", or "low".
- Do not assign high confidence when important distinguishing
  characteristics cannot be assessed.

Return valid JSON only. No Markdown or additional text.
Include every field and use empty arrays when no entries exist.

Required JSON structure:

{
    "species": {
        "common_name": "Common name",
        "scientific_name": "Scientific name"
    },
    "confidence": "high/medium/low",
    "visual_evidence": [],
    "alternative_species": [
        {
            "common_name": "Common name",
            "scientific_name": "Scientific name",
            "reason": "Explanation"
        }
    ],
    "limitations": []
}

The species field may also be null if no meaningful
candidate can be suggested.
 """

        if location:
            prompt += (
                f"\nObservation location: {location}\n"
                "Use this geographic context when considering "
                "species distributions and similar species."
            )

        if observation_date:
            prompt += (
                f"\nObservation date: {observation_date}\n"
                "Consider seasonal occurrence and migration patterns."
            )

        return prompt

    def identify(
        self,
        image_path: str | Path,
        *,
        location: str | None = None,
        observation_date: str | None = None,
        max_tokens: int = 500,
        temperature: float = 0.2,
    ) -> str:
        """
        Identify a bird in an already-processed image.

        Args:
            image_path: Path to a prepared image.
            location: Optional observation location.
            observation_date: Optional observation date.
            max_tokens: Maximum response length.
            temperature: Model generation temperature.

        Returns:
            The model's identification response.
        """

        image_path = Path(image_path)

        if not image_path.is_file():
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        prompt = self.build_prompt(
            location=location,
            observation_date=observation_date,
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    },
                    {
                        "type": "image_path",
                        "path": str(image_path.resolve()),
                    },
                ],
            }
        ]

        return self.backend.generate(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
