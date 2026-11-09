
import base64
from io import BytesIO
from pathlib import Path

from PIL import Image

from inference.vllm_backend import VLLMBackend


def encode_image(
    image_path: Path,
    max_dimension: int = 1280,
) -> tuple[str, tuple[int, int]]:
    """Resize an image and return its Base64 encoding and dimensions."""

    with Image.open(image_path) as image:
        image.thumbnail(
            (max_dimension, max_dimension),
            Image.Resampling.LANCZOS,
        )

        image = image.convert("RGB")
        dimensions = image.size

        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=90)

        encoded = base64.b64encode(
            buffer.getvalue()
        ).decode("utf-8")

    return encoded, dimensions


def main():

    project_root = Path(__file__).resolve().parents[1]

    image_path = (
        project_root
        / "data"
        / "processed"
        / "IMG_0374_processed.jpg"
    )

    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    print("\n" + "=" * 60)
    print("VLLM VISION BACKEND TEST")
    print("=" * 60)

    print(f"Image: {image_path.name}")

    # Initialize inference backend.
    backend = VLLMBackend(
        model_name="bird-vl",
        base_url="http://localhost:8000/v1",
    )

    # Resize and encode image for inference.
    encoded_image, dimensions = encode_image(image_path)

    print(f"Image dimensions sent: {dimensions}")

    # Construct multimodal request.
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Identify the bird in this photograph. "
                        "Provide the most likely species, "
                        "the visual evidence supporting your identification, "
                        "and a qualitative confidence level."
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": (
                            f"data:image/jpeg;base64,{encoded_image}"
                        )
                    },
                },
            ],
        }
    ]

    print("\nSending image to vLLM...")

    result = backend.generate(
        messages=messages,
        max_tokens=400,
        temperature=0.2,
    )

    print("\n" + "=" * 60)
    print("MODEL RESPONSE")
    print("=" * 60)

    print(result)

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
