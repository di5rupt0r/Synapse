"""TDD RED Phase: Tests for UniXCoder Embedding Backend."""

from unittest.mock import Mock, patch

import pytest


def test_unixcoder_backend_initialization():
    """Test UniXCoder backend initialization."""
    # This will fail - UniXCoder backend doesn't exist yet (RED phase)
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


def test_unixcoder_single_embedding():
    """Test single text embedding generation with mean pooling."""
    from synapse.embeddings.unixcoder import UniXCoderBackend

    mock_tokenizer = Mock()
    mock_model = Mock()

    # Mock tokenizer output
    mock_tokenizer.return_value = Mock()
    mock_tokenizer.return_value.to.return_value = {
        "input_ids": [[1, 2, 3, 4, 5]],
        "attention_mask": Mock(),
    }
    # Make attention mask behave like a tensor
    mock_attention_mask = mock_tokenizer.return_value.to.return_value["attention_mask"]
    mock_attention_mask.unsqueeze.return_value = Mock()
    mock_attention_mask.unsqueeze.return_value.expand.return_value = Mock()
    mock_attention_mask.unsqueeze.return_value.expand.return_value.float.return_value = Mock()

    # Mock model output - return the expected embedding directly
    mock_model.return_value = Mock()
    mock_model.return_value.last_hidden_state = Mock()
    mock_cpu_mock = Mock()
    mock_cpu_mock.numpy.return_value = [0.1] * 768  # Return flat list directly
    mock_model.return_value.last_hidden_state.cpu.return_value = mock_cpu_mock

    with (
        patch(
            "transformers.AutoTokenizer.from_pretrained", return_value=mock_tokenizer
        ),
        patch("transformers.AutoModel.from_pretrained", return_value=mock_model),
    ):
        backend = UniXCoderBackend("microsoft/unixcoder-base")

        result = backend.embed("def hello(): pass")

        assert result == [0.1] * 768
        mock_tokenizer.assert_called_once()
        mock_model.assert_called_once()


def test_unixcoder_batch_embedding():
    """Test batch text embedding generation."""
    from synapse.embeddings.unixcoder import UniXCoderBackend

    mock_tokenizer = Mock()
    mock_model = Mock()

    # Mock tokenizer for batch
    mock_tokenizer.return_value = {
        "input_ids": [[1, 2, 3], [4, 5, 6]],
        "attention_mask": [[1, 1, 1], [1, 1, 1]],
    }

    # Mock model output for batch
    mock_model.return_value = Mock()
    mock_model.return_value.last_hidden_state = Mock()
    mock_cpu_mock = Mock()
    mock_cpu_mock.numpy.return_value = [[0.1] * 768, [0.2] * 768]
    mock_model.return_value.last_hidden_state.cpu.return_value = mock_cpu_mock

    with (
        patch(
            "transformers.AutoTokenizer.from_pretrained", return_value=mock_tokenizer
        ),
        patch("transformers.AutoModel.from_pretrained", return_value=mock_model),
    ):
        backend = UniXCoderBackend("microsoft/unixcoder-base")

        texts = ["def func1():", "def func2():"]
        result = backend.embed_batch(texts)

        assert result == [[0.1] * 768, [0.2] * 768]


def test_unixcoder_mean_pooling():
    """Test mean pooling implementation matches codebase-rag."""
    from synapse.embeddings.unixcoder import UniXCoderBackend

    mock_tokenizer = Mock()
    mock_model = Mock()

    # Create mock attention mask for mean pooling
    mock_tokenizer.return_value = {
        "input_ids": [[1, 2, 3, 0, 0]],  # 3 real tokens, 2 padding
        "attention_mask": [[1, 1, 1, 0, 0]],  # 3 real, 2 padding
    }

    # Mock model output
    mock_model.return_value = Mock()
    mock_model.return_value.last_hidden_state = Mock()
    mock_cpu_mock = Mock()
    mock_cpu_mock.numpy.return_value = [0.1] * 768
    mock_model.return_value.last_hidden_state.cpu.return_value = mock_cpu_mock

    with (
        patch(
            "transformers.AutoTokenizer.from_pretrained", return_value=mock_tokenizer
        ),
        patch("transformers.AutoModel.from_pretrained", return_value=mock_model),
    ):
        backend = UniXCoderBackend("microsoft/unixcoder-base")

        result = backend.embed("test code")

        assert result == [0.1] * 768


def test_unixcoder_dimension_validation():
    """Test embedding dimension validation."""
    from synapse.embeddings.unixcoder import UniXCoderBackend

    mock_tokenizer = Mock()
    mock_model = Mock()

    # Mock model with wrong dimension
    mock_tokenizer.return_value = {
        "input_ids": [[1, 2, 3]],
        "attention_mask": [[1, 1, 1]],
    }

    mock_model.return_value = Mock()
    mock_model.return_value.last_hidden_state = Mock()
    mock_cpu_mock = Mock()
    mock_cpu_mock.numpy.return_value = [0.1] * 384  # Wrong dimension
    mock_model.return_value.last_hidden_state.cpu.return_value = mock_cpu_mock

    with (
        patch(
            "transformers.AutoTokenizer.from_pretrained", return_value=mock_tokenizer
        ),
        patch("transformers.AutoModel.from_pretrained", return_value=mock_model),
    ):
        backend = UniXCoderBackend("microsoft/unixcoder-base")

        # Validation happens during embedding, not initialization
        with pytest.raises(ValueError) as exc:
            backend.embed("test code")

        assert "768" in str(exc.value)
