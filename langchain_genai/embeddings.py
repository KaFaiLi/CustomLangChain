"""GenAIEmbeddings — LangChain Embeddings backed by GenAiApiClient.

This module provides ``GenAIEmbeddings``, a thin wrapper that delegates
embedding requests to :class:`genai_api_client.GenAiApiClient` while
exposing the standard LangChain ``Embeddings`` interface.
"""

from __future__ import annotations

from typing import Any, Optional

from langchain_core.embeddings import Embeddings
from pydantic import BaseModel, ConfigDict

from genai_api_client import GenAiApiClient


class GenAIEmbeddings(BaseModel, Embeddings):
    """LangChain embeddings model backed by :class:`GenAiApiClient`.

    All network calls are synchronous and delegate to
    ``GenAiApiClient.get_genai_embeddings``.  Async support is provided
    by the default ``Embeddings`` base class implementation which uses
    ``run_in_executor``.

    Args:
        model: Azure OpenAI embedding model name (required).
        dimensions: Embedding dimensionality; ``None`` uses model default.
        encoding_format: Encoding format (``"float"`` or ``"base64"``);
            ``None`` uses model default.
        config_path: Path to ``config.yaml``.  ``None`` uses the client's
            built-in default.

    Example:
        .. code-block:: python

            from langchain_genai import GenAIEmbeddings

            embeddings = GenAIEmbeddings(model="text-embedding-3-small")
            vectors = embeddings.embed_documents(["Hello", "World"])
    """

    model: str
    dimensions: Optional[int] = None
    encoding_format: Optional[str] = None
    config_path: Optional[str] = None

    # --- internal state ---
    _client: GenAiApiClient = None  # type: ignore[assignment]

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context: Any) -> None:
        """Instantiate the underlying GenAiApiClient."""
        super().model_post_init(__context)
        self._client = (
            GenAiApiClient(config_path=self.config_path)
            if self.config_path
            else GenAiApiClient()
        )

    # ------------------------------------------------------------------
    # Embeddings interface
    # ------------------------------------------------------------------

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of documents.

        Args:
            texts: The texts to embed.

        Returns:
            A list of embedding vectors (one per input text).
            Returns ``[]`` for empty input without making a network call.
        """
        if not texts:
            return []

        params: dict[str, Any] = {}
        if self.dimensions is not None:
            params["dimensions"] = self.dimensions
        if self.encoding_format is not None:
            params["encoding_format"] = self.encoding_format

        return self._client.get_genai_embeddings(
            input=texts,
            model=self.model,
            embeddings_only=True,
            **params,
        )

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query text.

        Args:
            text: The text to embed.

        Returns:
            A single embedding vector.
        """
        params: dict[str, Any] = {}
        if self.dimensions is not None:
            params["dimensions"] = self.dimensions
        if self.encoding_format is not None:
            params["encoding_format"] = self.encoding_format

        return self._client.get_genai_embeddings(
            input=text,
            model=self.model,
            embeddings_only=True,
            **params,
        )
