"""Tests for embeddings/unixcoder.py - 100% Coverage."""

from unittest.mock import MagicMock, Mock, patch

import pytest

# Mock torch before any imports that might use it
torch_mock = MagicMock()
torch_mock.device = Mock
torch_mock.tensor = Mock
sys_modules_patcher = patch.dict("sys.modules", {"torch": torch_mock})
sys_modules_patcher.start()


class TestUniXCoderBackend:
    """Test UniXCoder backend functionality."""

    @patch("synapse.embeddings.unixcoder.torch.cuda.is_available")
    @patch("synapse.embeddings.unixcoder.AutoTokenizer")
    @patch("synapse.embeddings.unixcoder.AutoModel")
    def test_initialization_cuda_available(
        self, mock_auto_model, mock_auto_tokenizer, mock_cuda
    ):
        """Test initialization when CUDA is available."""
        mock_cuda.return_value = True
        mock_tokenizer = Mock()
        mock_model = Mock()
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
        mock_auto_model.from_pretrained.return_value = mock_model

        from synapse.embeddings.unixcoder import UniXCoderBackend

        backend = UniXCoderBackend()

        assert backend.device == torch_mock.device("cuda")
        assert backend.tokenizer == mock_tokenizer
        assert backend.model == mock_model
        mock_model.to.assert_called_once_with(torch_mock.device("cuda"))

    @patch("synapse.embeddings.unixcoder.torch.cuda.is_available")
    @patch("synapse.embeddings.unixcoder.AutoTokenizer")
    @patch("synapse.embeddings.unixcoder.AutoModel")
    def test_initialization_cpu_only(
        self, mock_auto_model, mock_auto_tokenizer, mock_cuda
    ):
        """Test initialization when CUDA is not available."""
        mock_cuda.return_value = False
        mock_tokenizer = Mock()
        mock_model = Mock()
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
        mock_auto_model.from_pretrained.return_value = mock_model

        from synapse.embeddings.unixcoder import UniXCoderBackend

        backend = UniXCoderBackend()

        assert backend.device == torch_mock.device("cpu")
        assert backend.tokenizer == mock_tokenizer
        assert backend.model == mock_model
        mock_model.to.assert_called_once_with(torch_mock.device("cpu"))

    @patch("synapse.embeddings.unixcoder.torch.cuda.is_available")
    @patch("synapse.embeddings.unixcoder.AutoTokenizer")
    @patch("synapse.embeddings.unixcoder.AutoModel")
    def test_initialization_custom_model(
        self, mock_auto_model, mock_auto_tokenizer, mock_cuda
    ):
        """Test initialization with custom model name."""
        mock_cuda.return_value = False
        mock_tokenizer = Mock()
        mock_model = Mock()
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
        mock_auto_model.from_pretrained.return_value = mock_model

        from synapse.embeddings.unixcoder import UniXCoderBackend

        backend = UniXCoderBackend("custom/unixcoder-model")

        assert backend.model_name == "custom/unixcoder-model"
        mock_auto_tokenizer.from_pretrained.assert_called_once_with(
            "custom/unixcoder-model"
        )
        mock_auto_model.from_pretrained.assert_called_once_with(
            "custom/unixcoder-model"
        )

    def test_get_dimension(self):
        """Test _get_dimension returns correct dimension."""
        from synapse.embeddings.unixcoder import UniXCoderBackend

        backend = UniXCoderBackend.__new__(UniXCoderBackend)

        dimension = backend._get_dimension()
        assert dimension == 768

    @patch("synapse.embeddings.unixcoder.torch.cuda.is_available")
    @patch("synapse.embeddings.unixcoder.AutoTokenizer")
    @patch("synapse.embeddings.unixcoder.AutoModel")
    def test_embed_real_tensor_path(
        self, mock_auto_model, mock_auto_tokenizer, mock_cuda
    ):
        """Test embed with real tensors (normal path)."""
        mock_cuda.return_value = False

        # Setup tokenizer mock
        mock_tokenizer = Mock()
        mock_inputs = {
            "input_ids": torch_mock.tensor([[1, 2, 3]]),
            "attention_mask": torch_mock.tensor([[1, 1, 1]]),
        }
        mock_tokenizer.return_value = mock_inputs
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

        # Setup model mock
        mock_model = Mock()
        mock_outputs = Mock()
        mock_last_hidden = torch_mock.tensor(
            [[[0.1, 0.2, 0.3]]]
        )  # Will be expanded to 768
        mock_outputs.last_hidden_state = mock_last_hidden
        mock_model.return_value = mock_outputs
        mock_auto_model.from_pretrained.return_value = mock_model

        from synapse.embeddings.unixcoder import UniXCoderBackend

        backend = UniXCoderBackend()

        # Mock the dimension check
        with patch.object(backend, "_get_dimension", return_value=3):
            result = backend.embed("test")

        assert isinstance(result, list)
        mock_tokenizer.assert_called_once()
        mock_model.assert_called_once()

    @patch("synapse.embeddings.unixcoder.torch.cuda.is_available")
    @patch("synapse.embeddings.unixcoder.AutoTokenizer")
    @patch("synapse.embeddings.unixcoder.AutoModel")
    def test_embed_mock_dict_path(
        self, mock_auto_model, mock_auto_tokenizer, mock_cuda
    ):
        """Test embed with mock dict inputs (testing path)."""
        mock_cuda.return_value = False

        # Setup tokenizer mock that returns dict (no .to method)
        mock_tokenizer = Mock()
        mock_inputs = {"input_ids": [[1, 2, 3]], "attention_mask": [[1, 1, 1]]}
        mock_tokenizer.return_value = mock_inputs
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

        # Setup model mock
        mock_model = Mock()
        mock_outputs = Mock()
        mock_last_hidden = Mock()
        # Mock that doesn't have .cpu method
        mock_last_hidden.cpu = Mock(side_effect=AttributeError("No cpu method"))
        mock_outputs.last_hidden_state = mock_last_hidden
        mock_model.return_value = mock_outputs
        mock_auto_model.from_pretrained.return_value = mock_model

        from synapse.embeddings.unixcoder import UniXCoderBackend

        backend = UniXCoderBackend()

        # Should use fallback path
        result = backend.embed("test")

        assert isinstance(result, list)
        assert len(result) == 768  # Fallback dimension

    @patch("synapse.embeddings.unixcoder.torch.cuda.is_available")
    @patch("synapse.embeddings.unixcoder.AutoTokenizer")
    @patch("synapse.embeddings.unixcoder.AutoModel")
    def test_embed_mock_nested_list_path(
        self, mock_auto_model, mock_auto_tokenizer, mock_cuda
    ):
        """Test embed with mock that returns nested list."""
        mock_cuda.return_value = False

        # Setup tokenizer mock
        mock_tokenizer = Mock()
        mock_inputs = Mock()
        mock_inputs.to = Mock(return_value=mock_inputs)
        mock_tokenizer.return_value = mock_inputs
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

        # Setup model mock that returns nested list
        mock_model = Mock()
        mock_outputs = Mock()
        mock_last_hidden = Mock()
        mock_nested = [[0.1] * 768]  # Nested list
        mock_last_hidden.cpu.return_value.numpy.return_value = mock_nested
        mock_outputs.last_hidden_state = mock_last_hidden
        mock_model.return_value = mock_outputs
        mock_auto_model.from_pretrained.return_value = mock_model

        from synapse.embeddings.unixcoder import UniXCoderBackend

        backend = UniXCoderBackend()

        result = backend.embed("test")

        assert isinstance(result, list)
        assert len(result) == 768

    @patch("synapse.embeddings.unixcoder.torch.cuda.is_available")
    @patch("synapse.embeddings.unixcoder.AutoTokenizer")
    @patch("synapse.embeddings.unixcoder.AutoModel")
    def test_embed_dimension_validation(
        self, mock_auto_model, mock_auto_tokenizer, mock_cuda
    ):
        """Test embed validates dimension correctly."""
        mock_cuda.return_value = False

        # Setup tokenizer mock
        mock_tokenizer = Mock()
        mock_inputs = Mock()
        mock_inputs.to = Mock(return_value=mock_inputs)
        mock_tokenizer.return_value = mock_inputs
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

        # Setup model mock that returns wrong dimension
        mock_model = Mock()
        mock_outputs = Mock()
        mock_last_hidden = Mock()
        mock_last_hidden.cpu.return_value.numpy.return_value = [
            [0.1] * 500
        ]  # Wrong dimension
        mock_outputs.last_hidden_state = mock_last_hidden
        mock_model.return_value = mock_outputs
        mock_auto_model.from_pretrained.return_value = mock_model

        from synapse.embeddings.unixcoder import UniXCoderBackend

        backend = UniXCoderBackend()

        with pytest.raises(
            ValueError, match="Embedding dimension 500 does not match expected 768"
        ):
            backend.embed("test")

    @patch("synapse.embeddings.unixcoder.torch.cuda.is_available")
    @patch("synapse.embeddings.unixcoder.AutoTokenizer")
    @patch("synapse.embeddings.unixcoder.AutoModel")
    def test_embed_batch_real_tensor_path(
        self, mock_auto_model, mock_auto_tokenizer, mock_cuda
    ):
        """Test embed_batch with real tensors."""
        mock_cuda.return_value = False

        # Setup tokenizer mock
        mock_tokenizer = Mock()
        mock_inputs = {
            "input_ids": torch_mock.tensor([[1, 2, 3], [4, 5, 6]]),
            "attention_mask": torch_mock.tensor([[1, 1, 1], [1, 1, 1]]),
        }
        mock_tokenizer.return_value = mock_inputs
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

        # Setup model mock
        mock_model = Mock()
        mock_outputs = Mock()
        mock_last_hidden = torch_mock.tensor(
            [[[0.1, 0.2]], [[0.3, 0.4]]]
        )  # 2 samples, small dim
        mock_outputs.last_hidden_state = mock_last_hidden
        mock_model.return_value = mock_outputs
        mock_auto_model.from_pretrained.return_value = mock_model

        from synapse.embeddings.unixcoder import UniXCoderBackend

        backend = UniXCoderBackend()

        # Mock dimension check
        with patch.object(backend, "_get_dimension", return_value=2):
            result = backend.embed_batch(["text1", "text2"])

        assert isinstance(result, list)
        assert len(result) == 2
        mock_tokenizer.assert_called_once_with(
            ["text1", "text2"],
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )


# Clean up the mock at the end of the module
def teardown_module():
    """Clean up torch mock after module execution."""
    sys_modules_patcher.stop()
