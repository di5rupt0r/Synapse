"""Tests for embeddings/sentence_transformer.py - 100% Coverage."""

from unittest.mock import Mock, patch

import numpy as np
import pytest


class TestSentenceTransformerBackend:
    """Test SentenceTransformer backend functionality."""

    @patch("synapse.embeddings.sentence_transformer.SentenceTransformer")
    def test_initialization_default_model(self, mock_sentence_transformer):
        """Test initialization with default model."""
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_sentence_transformer.return_value = mock_model

        from synapse.embeddings.sentence_transformer import SentenceTransformerBackend

        backend = SentenceTransformerBackend()

        assert backend.model_name == "all-MiniLM-L6-v2"
        assert backend.dimension == 384
        mock_sentence_transformer.assert_called_once_with("all-MiniLM-L6-v2")

    @patch("synapse.embeddings.sentence_transformer.SentenceTransformer")
    def test_initialization_custom_model(self, mock_sentence_transformer):
        """Test initialization with custom model."""
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 768
        mock_sentence_transformer.return_value = mock_model

        from synapse.embeddings.sentence_transformer import SentenceTransformerBackend

        backend = SentenceTransformerBackend("custom-model")

        assert backend.model_name == "custom-model"
        assert backend.dimension == 768
        mock_sentence_transformer.assert_called_once_with("custom-model")

    @patch("synapse.embeddings.sentence_transformer.SentenceTransformer")
    def test_initialization_dimension_validation_failure(
        self, mock_sentence_transformer
    ):
        """Test initialization fails when dimension doesn't match ADR-001."""
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = (
            512  # Wrong dimension
        )
        mock_sentence_transformer.return_value = mock_model

        from synapse.embeddings.sentence_transformer import SentenceTransformerBackend

        with pytest.raises(
            ValueError,
            match="Model dimension 512 does not match ADR-001 requirement of 768",
        ):
            SentenceTransformerBackend()

    @patch("synapse.embeddings.sentence_transformer.SentenceTransformer")
    def test_get_dimension(self, mock_sentence_transformer):
        """Test _get_dimension method."""
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 768
        mock_sentence_transformer.return_value = mock_model

        from synapse.embeddings.sentence_transformer import SentenceTransformerBackend

        backend = SentenceTransformerBackend()

        # Mock the validation to avoid error
        with patch.object(backend, "_validate_dimension"):
            backend._validate_dimension()

        dimension = backend._get_dimension()
        assert dimension == 768

    @patch("synapse.embeddings.sentence_transformer.SentenceTransformer")
    def test_embed_numpy_array_return(self, mock_sentence_transformer):
        """Test embed with numpy array return from model."""
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 768
        mock_sentence_transformer.return_value = mock_model

        # Mock numpy array return
        mock_embedding = np.array([0.1] * 768)
        mock_model.encode.return_value = mock_embedding

        from synapse.embeddings.sentence_transformer import SentenceTransformerBackend

        backend = SentenceTransformerBackend()

        # Mock validation
        with patch.object(backend, "_validate_dimension"):
            backend._validate_dimension()

        result = backend.embed("test text")

        assert isinstance(result, list)
        assert len(result) == 768
        assert all(x == 0.1 for x in result)
        mock_model.encode.assert_called_once_with("test text", convert_to_numpy=True)

    @patch("synapse.embeddings.sentence_transformer.SentenceTransformer")
    def test_embed_list_return(self, mock_sentence_transformer):
        """Test embed with list return from model."""
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 768
        mock_sentence_transformer.return_value = mock_model

        # Mock list return
        mock_embedding = [0.2] * 768
        mock_model.encode.return_value = mock_embedding

        from synapse.embeddings.sentence_transformer import SentenceTransformerBackend

        backend = SentenceTransformerBackend()

        # Mock validation
        with patch.object(backend, "_validate_dimension"):
            backend._validate_dimension()

        result = backend.embed("test text")

        assert isinstance(result, list)
        assert len(result) == 768
        assert all(x == 0.2 for x in result)

    @patch("synapse.embeddings.sentence_transformer.SentenceTransformer")
    def test_embed_batch_numpy_arrays_return(self, mock_sentence_transformer):
        """Test embed_batch with numpy arrays return from model."""
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 768
        mock_sentence_transformer.return_value = mock_model

        # Mock numpy array return for batch
        mock_embeddings = np.array([[0.1] * 768, [0.2] * 768])
        mock_model.encode.return_value = mock_embeddings

        from synapse.embeddings.sentence_transformer import SentenceTransformerBackend

        backend = SentenceTransformerBackend()

        # Mock validation
        with patch.object(backend, "_validate_dimension"):
            backend._validate_dimension()

        result = backend.embed_batch(["text1", "text2"])

        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(emb, list) for emb in result)
        assert all(len(emb) == 768 for emb in result)
        assert result[0][0] == 0.1
        assert result[1][0] == 0.2
        mock_model.encode.assert_called_once_with(
            ["text1", "text2"], convert_to_numpy=True
        )

    @patch("synapse.embeddings.sentence_transformer.SentenceTransformer")
    def test_embed_batch_lists_return(self, mock_sentence_transformer):
        """Test embed_batch with lists return from model."""
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 768
        mock_sentence_transformer.return_value = mock_model

        # Mock list of lists return
        mock_embeddings = [[0.1] * 768, [0.2] * 768]
        mock_model.encode.return_value = mock_embeddings

        from synapse.embeddings.sentence_transformer import SentenceTransformerBackend

        backend = SentenceTransformerBackend()

        # Mock validation
        with patch.object(backend, "_validate_dimension"):
            backend._validate_dimension()

        result = backend.embed_batch(["text1", "text2"])

        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(emb, list) for emb in result)
        assert all(len(emb) == 768 for emb in result)

    @patch("synapse.embeddings.sentence_transformer.SentenceTransformer")
    def test_embed_batch_single_embedding_return(self, mock_sentence_transformer):
        """Test embed_batch when model returns single embedding instead of list."""
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 768
        mock_sentence_transformer.return_value = mock_model

        # Mock single embedding return (edge case)
        mock_embeddings = [0.1] * 768
        mock_model.encode.return_value = mock_embeddings

        from synapse.embeddings.sentence_transformer import SentenceTransformerBackend

        backend = SentenceTransformerBackend()

        # Mock validation
        with patch.object(backend, "_validate_dimension"):
            backend._validate_dimension()

        result = backend.embed_batch(["text1"])

        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], list)
        assert len(result[0]) == 768

    @patch("synapse.embeddings.sentence_transformer.SentenceTransformer")
    def test_validate_dimension_success(self, mock_sentence_transformer):
        """Test _validate_dimension succeeds with correct dimension."""
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 768
        mock_sentence_transformer.return_value = mock_model

        from synapse.embeddings.sentence_transformer import SentenceTransformerBackend

        backend = SentenceTransformerBackend()

        # Should not raise any exception
        backend._validate_dimension()

    @patch("synapse.embeddings.sentence_transformer.SentenceTransformer")
    def test_validate_dimension_failure(self, mock_sentence_transformer):
        """Test _validate_dimension fails with wrong dimension."""
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 512
        mock_sentence_transformer.return_value = mock_model

        from synapse.embeddings.sentence_transformer import SentenceTransformerBackend

        backend = SentenceTransformerBackend()

        with pytest.raises(
            ValueError,
            match="Model dimension 512 does not match ADR-001 requirement of 768",
        ):
            backend._validate_dimension()

    @patch("synapse.embeddings.sentence_transformer.SentenceTransformer")
    def test_inheritance_from_embedding_backend(self, mock_sentence_transformer):
        """Test that SentenceTransformerBackend inherits from EmbeddingBackend."""
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 768
        mock_sentence_transformer.return_value = mock_model

        from synapse.embeddings.backend import EmbeddingBackend
        from synapse.embeddings.sentence_transformer import SentenceTransformerBackend

        backend = SentenceTransformerBackend()

        # Mock validation
        with patch.object(backend, "_validate_dimension"):
            backend._validate_dimension()

        assert isinstance(backend, EmbeddingBackend)
        assert hasattr(backend, "embed")
        assert hasattr(backend, "embed_batch")
        assert hasattr(backend, "dimension")

    @patch("synapse.embeddings.sentence_transformer.SentenceTransformer")
    def test_embed_empty_text(self, mock_sentence_transformer):
        """Test embedding empty text."""
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 768
        mock_sentence_transformer.return_value = mock_model

        mock_embedding = np.array([0.1] * 768)
        mock_model.encode.return_value = mock_embedding

        from synapse.embeddings.sentence_transformer import SentenceTransformerBackend

        backend = SentenceTransformerBackend()

        # Mock validation
        with patch.object(backend, "_validate_dimension"):
            backend._validate_dimension()

        result = backend.embed("")

        assert isinstance(result, list)
        assert len(result) == 768
        mock_model.encode.assert_called_once_with("", convert_to_numpy=True)

    @patch("synapse.embeddings.sentence_transformer.SentenceTransformer")
    def test_embed_batch_empty_list(self, mock_sentence_transformer):
        """Test embedding empty list."""
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 768
        mock_sentence_transformer.return_value = mock_model

        mock_embeddings = np.array([]).reshape(0, 768)
        mock_model.encode.return_value = mock_embeddings

        from synapse.embeddings.sentence_transformer import SentenceTransformerBackend

        backend = SentenceTransformerBackend()

        # Mock validation
        with patch.object(backend, "_validate_dimension"):
            backend._validate_dimension()

        result = backend.embed_batch([])

        assert isinstance(result, list)
        assert len(result) == 0
        mock_model.encode.assert_called_once_with([], convert_to_numpy=True)
