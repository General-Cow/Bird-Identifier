
import base64
import mimetypes
from pathlib import Path
from typing import Any

from openai import OpenAI

from .base import InferenceBackend


class VLLMBackend(InferenceBackend):
    """
    Inference backend that communicates with a vLLM server
    through its OpenAI-compatible API.

    Responsible for converting local image paths into API-compatible
    image data URLs.
    """

    def __init__(
        self,
        model_name: str,
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "not-needed",
        timeout: float = 120.0,
    ):
        self.model_name = model_name

        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
        )

    def _prepare_messages(
        self,
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Convert internal image_path messages into OpenAI-compatible
        image_url messages.

        Text messages and other content are passed through unchanged.
        """

        prepared_messages = []

        for message in messages:

            prepared_message = message.copy()
            content = message.get("content")

            if not isinstance(content, list):
                prepared_messages.append(prepared_message)
                continue

            prepared_content = []

            for item in content:

                if item.get("type") != "image_path":
                    prepared_content.append(item)
                    continue

                image_path = Path(item["path"])

                if not image_path.is_file():
                    raise FileNotFoundError(
                        f"Image not found: {image_path}"
                    )

                mime_type, _ = mimetypes.guess_type(image_path.name)

                if mime_type is None or not mime_type.startswith("image/"):
                    raise ValueError(
                        f"Could not determine image MIME type: {image_path}"
                    )

                encoded_image = base64.b64encode(
                    image_path.read_bytes()
                ).decode("utf-8")

                prepared_content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": (
                                f"data:{mime_type};base64,{encoded_image}"
                            )
                        },
                    }
                )

            prepared_message["content"] = prepared_content
            prepared_messages.append(prepared_message)

        return prepared_messages

    def generate(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int = 512,
        temperature: float = 0.2,
    ) -> str:
        """Submit an inference request to the vLLM server."""

        prepared_messages = self._prepare_messages(messages)

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=prepared_messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        content = response.choices[0].message.content

        if content is None:
            return ""

        return content
