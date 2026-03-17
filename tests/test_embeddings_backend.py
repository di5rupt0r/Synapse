"""TDD RED Phase: Tests for Embedding Backend Interface."""

import pytest
from unittest.mock import Mock, patch


def test_embedding_backend_interface():
    """Test abstract embedding backend interface."""
    # This will fail - embedding backend doesn't exist yet (RED phase)
    from synapse.embeddings.backend import EmbeddingBackend

    # Test that it's an abstract class
    with pytest.raises(TypeError) as exc:
        EmbeddingBackend()

    assert "abstract" in str(exc.value).lower()


def test_sentence_transformer_backend():
    """Test SentenceTransformer backend implementation."""
    from synapse.embeddings.sentence_transformer import SentenceTransformerBackend

    mock_model = Mock()
    mock_model.get_sentence_embedding_dimension.return_value = 768

    with patch(
        "synapse.embeddings.sentence_transformer.SentenceTransformer",
        return_value=mock_model,
    ):
        backend = SentenceTransformerBackend("test-model")

        assert backend.model == mock_model
        assert backend.dimension == 768


def test_single_embedding():
    """Test single text embedding generation."""
    from synapse.embeddings.sentence_transformer import SentenceTransformerBackend

    mock_model = Mock()
    mock_model.get_sentence_embedding_dimension.return_value = 768
    mock_model.encode.return_value = [0.1] * 768

    with patch(
        "synapse.embeddings.sentence_transformer.SentenceTransformer",
        return_value=mock_model,
    ):
        backend = SentenceTransformerBackend("test-model")

        result = backend.embed("test text")

        assert result == [0.1] * 768
        mock_model.encode.assert_called_once_with("test text", convert_to_numpy=True)


def test_batch_embedding():
    """Test batch text embedding generation."""
    from synapse.embeddings.sentence_transformer import SentenceTransformerBackend

    mock_model = Mock()
    mock_model.get_sentence_embedding_dimension.return_value = 768
    mock_model.encode.return_value = [[0.1] * 768, [0.2] * 768]

    with patch(
        "synapse.embeddings.sentence_transformer.SentenceTransformer",
        return_value=mock_model,
    ):
        backend = SentenceTransformerBackend("test-model")

        texts = ["text1", "text2"]
        result = backend.embed_batch(texts)

        assert result == [[0.1] * 768, [0.2] * 768]
        mock_model.encode.assert_called_once_with(texts, convert_to_numpy=True)


def test_embedding_dimension_validation():
    """Test embedding dimension validation."""
    from synapse.embeddings.sentence_transformer import SentenceTransformerBackend

    mock_model = Mock()
    mock_model.get_sentence_embedding_dimension.return_value = 384  # Wrong dimension

    with patch(
        "synapse.embeddings.sentence_transformer.SentenceTransformer",
        return_value=mock_model,
    ):
        with pytest.raises(ValueError) as exc:
            SentenceTransformerBackend("test-model")

        assert "768" in str(exc.value)
