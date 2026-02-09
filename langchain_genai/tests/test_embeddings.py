"""Unit tests for langchain_genai.embeddings.GenAIEmbeddings."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from langchain_genai.embeddings import GenAIEmbeddings


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_client() -> MagicMock:
    """Return a mocked GenAiApiClient for embeddings."""
    client = MagicMock()
    # Default: return a list of 3 vectors
    client.get_genai_embeddings.return_value = [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
        [0.7, 0.8, 0.9],
    ]
    return client


@pytest.fixture()
def embeddings(mock_client: MagicMock) -> GenAIEmbeddings:
    """Return a GenAIEmbeddings instance with a mocked client."""
    with patch(
        "langchain_genai.embeddings.GenAiApiClient", return_value=mock_client
    ):
        emb = GenAIEmbeddings(model="text-embedding-3-small")
    return emb


# ---------------------------------------------------------------------------
# T015 — Tests for GenAIEmbeddings
# ---------------------------------------------------------------------------


class TestGenAIEmbeddings:
    """Tests for GenAIEmbeddings embed_documents and embed_query."""

    def test_embed_documents_multiple(
        self, embeddings: GenAIEmbeddings, mock_client: MagicMock
    ) -> None:
        texts = ["hello", "world", "test"]
        result = embeddings.embed_documents(texts)
        assert len(result) == 3
        assert result[0] == [0.1, 0.2, 0.3]
        assert result[2] == [0.7, 0.8, 0.9]

        mock_client.get_genai_embeddings.assert_called_once_with(
            input=texts,
            model="text-embedding-3-small",
            embeddings_only=True,
        )

    def test_embed_query(
        self, embeddings: GenAIEmbeddings, mock_client: MagicMock
    ) -> None:
        mock_client.get_genai_embeddings.return_value = [0.1, 0.2, 0.3]
        result = embeddings.embed_query("hello")
        assert result == [0.1, 0.2, 0.3]

        mock_client.get_genai_embeddings.assert_called_once_with(
            input="hello",
            model="text-embedding-3-small",
            embeddings_only=True,
        )

    def test_embed_documents_empty_list(
        self, embeddings: GenAIEmbeddings, mock_client: MagicMock
    ) -> None:
        """embed_documents([]) returns [] without calling the client."""
        result = embeddings.embed_documents([])
        assert result == []
        mock_client.get_genai_embeddings.assert_not_called()

    def test_dimensions_forwarding(self, mock_client: MagicMock) -> None:
        with patch(
            "langchain_genai.embeddings.GenAiApiClient", return_value=mock_client
        ):
            emb = GenAIEmbeddings(
                model="text-embedding-3-small", dimensions=256
            )
        emb.embed_documents(["test"])
        call_kwargs = mock_client.get_genai_embeddings.call_args
        assert call_kwargs.kwargs["dimensions"] == 256

    def test_encoding_format_forwarding(self, mock_client: MagicMock) -> None:
        with patch(
            "langchain_genai.embeddings.GenAiApiClient", return_value=mock_client
        ):
            emb = GenAIEmbeddings(
                model="text-embedding-3-small", encoding_format="float"
            )
        emb.embed_documents(["test"])
        call_kwargs = mock_client.get_genai_embeddings.call_args
        assert call_kwargs.kwargs["encoding_format"] == "float"

    def test_dimensions_and_format_on_query(self, mock_client: MagicMock) -> None:
        mock_client.get_genai_embeddings.return_value = [0.1, 0.2]
        with patch(
            "langchain_genai.embeddings.GenAiApiClient", return_value=mock_client
        ):
            emb = GenAIEmbeddings(
                model="text-embedding-3-small",
                dimensions=256,
                encoding_format="float",
            )
        emb.embed_query("test")
        call_kwargs = mock_client.get_genai_embeddings.call_args
        assert call_kwargs.kwargs["dimensions"] == 256
        assert call_kwargs.kwargs["encoding_format"] == "float"
