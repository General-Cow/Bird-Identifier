from abc import ABC, abstractmethod
from typing import Any


class InferenceBackend(ABC):
    """
    Abstract interface for model inference backends.

    All inference backends should implement this interface
    so the application can use them interchangeably.
    """

    @abstractmethod
    def generate(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int = 512,
        temperature: float = 0.2,
    ) -> str:
        """
        Generate a response from the model.

        Args:
            messages: Conversation messages, including text and images.
            max_tokens: Maximum number of tokens to generate.
            temperature: Controls generation randomness.

        Returns:
            Generated model response as a string.
        """
        pass