"""Tests for UniXCoder Embedding Backend - Complete coverage."""

from unittest.mock import Mock, patch
from types import SimpleNamespace
import numpy as np
import pytest
import torch


class TestUniXCoderBackend:
    """UniXCoder backend test coverage."""

    def test_initialization_cpu_only(self):
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("synapse.embeddings.unixcoder.AutoTokenizer") as mock_tokenizer,
            patch("synapse.embeddings.unixcoder.AutoModel") as mock_model,
        ):
            mock_tokenizer.from_pretrained.return_value = Mock()
            mock_model.from_pretrained.return_value = Mock()

            from synapse.embeddings.unixcoder import UniXCoderBackend
            backend = UniXCoderBackend()
            
            assert backend.model_name == "microsoft/unixcoder-base"
            assert backend.device.type == "cpu"
            assert backend.dimension == 768

    def test_initialization_cuda_available(self):
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("synapse.embeddings.unixcoder.AutoTokenizer") as mock_tokenizer,
            patch("synapse.embeddings.unixcoder.AutoModel") as mock_model,
        ):
            mock_tokenizer.from_pretrained.return_value = Mock()
            mock_model.from_pretrained.return_value = Mock()

            from synapse.embeddings.unixcoder import UniXCoderBackend
            backend = UniXCoderBackend()
            assert backend.device.type == "cuda"

    def test_initialization_custom_model(self):
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("synapse.embeddings.unixcoder.AutoTokenizer") as mock_tokenizer,
            patch("synapse.embeddings.unixcoder.AutoModel") as mock_model,
        ):
            mock_tokenizer.from_pretrained.return_value = Mock()
            mock_model.from_pretrained.return_value = Mock()

            from synapse.embeddings.unixcoder import UniXCoderBackend
            backend = UniXCoderBackend("custom-model")
            assert backend.model_name == "custom-model"

    def test_get_dimension(self):
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("synapse.embeddings.unixcoder.AutoTokenizer") as mock_tokenizer,
            patch("synapse.embeddings.unixcoder.AutoModel") as mock_model,
        ):
            from synapse.embeddings.unixcoder import UniXCoderBackend
            backend = UniXCoderBackend()
            assert backend._get_dimension() == 768

    def test_embed_real_tensor_path(self):
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("synapse.embeddings.unixcoder.AutoTokenizer") as mock_tokenizer,
            patch("synapse.embeddings.unixcoder.AutoModel") as mock_model,
        ):
            mock_tokenizer_instance = Mock()
            mock_tokenizer_instance.return_value = {
                "input_ids": torch.tensor([[1, 2, 3]]), 
                "attention_mask": torch.tensor([[1, 1, 1]])
            }
            mock_tokenizer_instance.return_value["to"] = lambda x: mock_tokenizer_instance.return_value
            mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

            mock_model_instance = Mock()
            mock_model_instance.to.return_value = mock_model_instance
            
            # 1 batch, 3 tokens, 768 dim
            mock_outputs = SimpleNamespace(last_hidden_state=torch.ones(1, 3, 768) * 0.1)
            mock_model_instance.return_value = mock_outputs
            mock_model.from_pretrained.return_value = mock_model_instance

            from synapse.embeddings.unixcoder import UniXCoderBackend
            backend = UniXCoderBackend()
            result = backend.embed("test code")

            assert isinstance(result, list)
            assert len(result) == 768
            assert isinstance(result[0], float)

    def test_embed_mock_dict_path(self):
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("synapse.embeddings.unixcoder.AutoTokenizer") as mock_tokenizer,
            patch("synapse.embeddings.unixcoder.AutoModel") as mock_model,
        ):
            # No "to" method, just a dict mock
            mock_tokenizer_instance = Mock()
            # To trigger the mock path where attention_mask has no unsqueeze
            mock_tokenizer_instance.return_value = {
                "input_ids": [1, 2, 3],
                "attention_mask": [1, 1, 1] 
            }
            mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

            mock_model_instance = Mock()
            mock_model_instance.to.return_value = mock_model_instance
            
            # last_hidden_state returns something with cpu
            class MockTensor:
                def cpu(self):
                    return self
                def numpy(self):
                    return [0.1] * 768
            
            mock_outputs = SimpleNamespace(last_hidden_state=MockTensor())
            mock_model_instance.return_value = mock_outputs
            mock_model.from_pretrained.return_value = mock_model_instance

            from synapse.embeddings.unixcoder import UniXCoderBackend
            backend = UniXCoderBackend()
            result = backend.embed("test code")

            assert isinstance(result, list)
            assert len(result) == 768

    def test_embed_mock_nested_list_path(self):
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("synapse.embeddings.unixcoder.AutoTokenizer") as mock_tokenizer,
            patch("synapse.embeddings.unixcoder.AutoModel") as mock_model,
        ):
            mock_tokenizer_instance = Mock()
            mock_tokenizer_instance.return_value = {
                "input_ids": [1, 2, 3],
                "attention_mask": [1, 1, 1] 
            }
            mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

            mock_model_instance = Mock()
            mock_model_instance.to.return_value = mock_model_instance
            
            class MockTensor:
                def cpu(self):
                    return self
                def numpy(self):
                    return [[0.1] * 768]
            
            mock_outputs = SimpleNamespace(last_hidden_state=MockTensor())
            mock_model_instance.return_value = mock_outputs
            mock_model.from_pretrained.return_value = mock_model_instance

            from synapse.embeddings.unixcoder import UniXCoderBackend
            backend = UniXCoderBackend()
            result = backend.embed("test code")

            assert isinstance(result, list)
            assert len(result) == 768

    def test_embed_dimension_validation(self):
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("synapse.embeddings.unixcoder.AutoTokenizer") as mock_tokenizer,
            patch("synapse.embeddings.unixcoder.AutoModel") as mock_model,
        ):
            mock_tokenizer_instance = Mock()
            mock_tokenizer_instance.return_value = {
                "input_ids": [1, 2, 3],
                "attention_mask": [1, 1, 1] 
            }
            mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

            mock_model_instance = Mock()
            mock_model_instance.to.return_value = mock_model_instance
            
            class MockTensor:
                def cpu(self):
                    return self
                def numpy(self):
                    return [0.1] * 500  # Wrong dimension
            
            mock_outputs = SimpleNamespace(last_hidden_state=MockTensor())
            mock_model_instance.return_value = mock_outputs
            mock_model.from_pretrained.return_value = mock_model_instance

            from synapse.embeddings.unixcoder import UniXCoderBackend
            backend = UniXCoderBackend()
            
            with pytest.raises(ValueError, match="Embedding dimension 500 does not match expected 768"):
                backend.embed("test code")

    def test_embed_batch_real_tensor_path(self):
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("synapse.embeddings.unixcoder.AutoTokenizer") as mock_tokenizer,
            patch("synapse.embeddings.unixcoder.AutoModel") as mock_model,
        ):
            mock_tokenizer_instance = Mock()
            mock_tokenizer_instance.return_value = {
                "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]), 
                "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 1]])
            }
            mock_tokenizer_instance.return_value["to"] = lambda x: mock_tokenizer_instance.return_value
            mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

            mock_model_instance = Mock()
            mock_model_instance.to.return_value = mock_model_instance
            
            # 2 batch, 3 tokens, 768 dim
            mock_outputs = SimpleNamespace(last_hidden_state=torch.ones(2, 3, 768) * 0.1)
            mock_model_instance.return_value = mock_outputs
            mock_model.from_pretrained.return_value = mock_model_instance

            from synapse.embeddings.unixcoder import UniXCoderBackend
            backend = UniXCoderBackend()
            result = backend.embed_batch(["test code 1", "test code 2"])

            assert isinstance(result, list)
            assert len(result) == 2
            assert len(result[0]) == 768
            assert len(result[1]) == 768
