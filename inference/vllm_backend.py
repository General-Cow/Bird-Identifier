from typing import Any

from openai import OpenAI

from .base import InferenceBackend


class VLLMBackend(InferenceBackend):
    """
    Inference backend that communicates with a vLLM server
    through its OpenAI-compatible API.
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

    def generate(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int = 512,
        temperature: float = 0.2,
    ) -> str:
        """
        Submit an inference request to the vLLM server.
        """

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        content = response.choices[0].message.content

        if content is None:
            return ""

        return content