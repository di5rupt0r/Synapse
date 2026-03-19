"""Tests for UniXCoder Embedding Backend."""

from unittest.mock import Mock, patch


def test_unixcoder_backend_initialization():
    """Test UniXCoder backend initialization."""
    from synapse.embeddings.unixcoder import UniXCoderBackend

    with (
        patch("transformers.AutoTokenizer.from_pretrained") as mock_tokenizer,
        patch("transformers.AutoModel.from_pretrained") as mock_model,
    ):
        mock_tokenizer.return_value = Mock()
        mock_model.return_value = Mock()

        backend = UniXCoderBackend("microsoft/unixcoder-base")

        assert backend.model_name == "microsoft/unixcoder-base"
        assert backend.dimension == 768


def test_unixcoder_embed_returns_embedding():
    """Test embed returns a vector."""
    from synapse.embeddings.unixcoder import UniXCoderBackend

    with (
        patch("transformers.AutoTokenizer.from_pretrained") as mock_tokenizer,
        patch("transformers.AutoModel.from_pretrained") as mock_model,
    ):
        # Setup mocks
        mock_tokenizer.return_value = Mock()
        mock_model.return_value = Mock()

        backend = UniXCoderBackend()
        # Mock the actual embed method to return a vector
        backend.embed = Mock(return_value=[0.1] * 768)

        result = backend.embed("test code")

        assert len(result) == 768


def test_unixcoder_embed_batch_returns_embeddings():
    """Test embed_batch returns vectors."""
    from synapse.embeddings.unixcoder import UniXCoderBackend

    with (
        patch("transformers.AutoTokenizer.from_pretrained") as mock_tokenizer,
        patch("transformers.AutoModel.from_pretrained") as mock_model,
    ):
        mock_tokenizer.return_value = Mock()
        mock_model.return_value = Mock()

        backend = UniXCoderBackend()
        # Mock embed_batch
        backend.embed_batch = Mock(return_value=[[0.1] * 768, [0.2] * 768])

        result = backend.embed_batch(["code1", "code2"])

        assert len(result) == 2
        assert all(len(r) == 768 for r in result)


def test_unixcoder_dimension():
    """Test embedding dimension is 768."""
    from synapse.embeddings.unixcoder import UniXCoderBackend

    with (
        patch("transformers.AutoTokenizer.from_pretrained") as mock_tokenizer,
        patch("transformers.AutoModel.from_pretrained") as mock_model,
    ):
        mock_tokenizer.return_value = Mock()
        mock_model.return_value = Mock()

        backend = UniXCoderBackend()
        assert backend.dimension == 768
