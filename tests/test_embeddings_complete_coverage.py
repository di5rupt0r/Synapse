"""Tests for embedding backends - Complete coverage."""

from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import pytest


class TestEmbeddingBackendComplete:
    """Complete embedding backend coverage."""

    def test_embedding_backend_abstract_methods(self):
        """Test EmbeddingBackend abstract methods."""
        from synapse.embeddings.backend import EmbeddingBackend

        # Cannot instantiate abstract class
        with pytest.raises(TypeError):
            EmbeddingBackend()

    def test_sentence_transformer_backend_complete(self):
        """Test SentenceTransformer backend with all paths."""
        with patch(
            "synapse.embeddings.sentence_transformer.SentenceTransformer"
        ) as mock_st:
            mock_model = Mock()
            mock_model.get_sentence_embedding_dimension.return_value = 768
            mock_st.return_value = mock_model

            from synapse.embeddings.sentence_transformer import (
                SentenceTransformerBackend,
            )

            backend = SentenceTransformerBackend()
            assert backend.model_name == "all-MiniLM-L6-v2"
            assert backend.dimension == 768

    def test_sentence_transformer_backend_custom_model(self):
        """Test SentenceTransformer with custom model."""
        with patch(
            "synapse.embeddings.sentence_transformer.SentenceTransformer"
        ) as mock_st:
            mock_model = Mock()
            mock_model.get_sentence_embedding_dimension.return_value = 768
            mock_st.return_value = mock_model

            from synapse.embeddings.sentence_transformer import (
                SentenceTransformerBackend,
            )

            backend = SentenceTransformerBackend("custom-model")
            assert backend.model_name == "custom-model"
            mock_st.assert_called_once_with("custom-model")

    def test_sentence_transformer_dimension_validation_failure(self):
        """Test SentenceTransformer dimension validation failure."""
        with patch(
            "synapse.embeddings.sentence_transformer.SentenceTransformer"
        ) as mock_st:
            mock_model = Mock()
            mock_model.get_sentence_embedding_dimension.return_value = (
                512  # Wrong dimension
            )
            mock_st.return_value = mock_model

            from synapse.embeddings.sentence_transformer import (
                SentenceTransformerBackend,
            )

            with pytest.raises(
                ValueError,
                match="Model dimension 512 does not match ADR-001 requirement of 768",
            ):
                SentenceTransformerBackend()

    def test_sentence_transformer_embed_numpy_array(self):
        """Test SentenceTransformer embed with numpy array return."""
        with patch(
            "synapse.embeddings.sentence_transformer.SentenceTransformer"
        ) as mock_st:
            mock_model = Mock()
            mock_model.get_sentence_embedding_dimension.return_value = 768
            mock_model.encode.return_value = np.array([0.1] * 768)
            mock_st.return_value = mock_model

            from synapse.embeddings.sentence_transformer import (
                SentenceTransformerBackend,
            )

            backend = SentenceTransformerBackend()
            result = backend.embed("test text")

            assert isinstance(result, list)
            assert len(result) == 768
            assert all(x == 0.1 for x in result)

    def test_sentence_transformer_embed_list_return(self):
        """Test SentenceTransformer embed with list return."""
        with patch(
            "synapse.embeddings.sentence_transformer.SentenceTransformer"
        ) as mock_st:
            mock_model = Mock()
            mock_model.get_sentence_embedding_dimension.return_value = 768
            mock_model.encode.return_value = [0.2] * 768
            mock_st.return_value = mock_model

            from synapse.embeddings.sentence_transformer import (
                SentenceTransformerBackend,
            )

            backend = SentenceTransformerBackend()
            result = backend.embed("test text")

            assert isinstance(result, list)
            assert len(result) == 768

    def test_sentence_transformer_embed_batch_numpy_arrays(self):
        """Test SentenceTransformer embed_batch with numpy arrays."""
        with patch(
            "synapse.embeddings.sentence_transformer.SentenceTransformer"
        ) as mock_st:
            mock_model = Mock()
            mock_model.get_sentence_embedding_dimension.return_value = 768
            mock_embeddings = np.array([[0.1] * 768, [0.2] * 768])
            mock_model.encode.return_value = mock_embeddings
            mock_st.return_value = mock_model

            from synapse.embeddings.sentence_transformer import (
                SentenceTransformerBackend,
            )

            backend = SentenceTransformerBackend()
            result = backend.embed_batch(["text1", "text2"])

            assert isinstance(result, list)
            assert len(result) == 2
            assert all(isinstance(emb, list) for emb in result)
            assert result[0][0] == 0.1
            assert result[1][0] == 0.2

    def test_sentence_transformer_embed_batch_lists(self):
        """Test SentenceTransformer embed_batch with lists."""
        with patch(
            "synapse.embeddings.sentence_transformer.SentenceTransformer"
        ) as mock_st:
            mock_model = Mock()
            mock_model.get_sentence_embedding_dimension.return_value = 768
            mock_embeddings = [[0.1] * 768, [0.2] * 768]
            mock_model.encode.return_value = mock_embeddings
            mock_st.return_value = mock_model

            from synapse.embeddings.sentence_transformer import (
                SentenceTransformerBackend,
            )

            backend = SentenceTransformerBackend()
            result = backend.embed_batch(["text1", "text2"])

            assert isinstance(result, list)
            assert len(result) == 2

    def test_sentence_transformer_embed_batch_single_embedding(self):
        """Test SentenceTransformer embed_batch with single embedding."""
        with patch(
            "synapse.embeddings.sentence_transformer.SentenceTransformer"
        ) as mock_st:
            mock_model = Mock()
            mock_model.get_sentence_embedding_dimension.return_value = 768
            mock_model.encode.return_value = [0.1] * 768
            mock_st.return_value = mock_model

            from synapse.embeddings.sentence_transformer import (
                SentenceTransformerBackend,
            )

            backend = SentenceTransformerBackend()
            result = backend.embed_batch(["text1"])

            assert isinstance(result, list)
            assert len(result) == 1

    def test_sentence_transformer_validate_dimension_success(self):
        """Test SentenceTransformer dimension validation success."""
        with patch(
            "synapse.embeddings.sentence_transformer.SentenceTransformer"
        ) as mock_st:
            mock_model = Mock()
            mock_model.get_sentence_embedding_dimension.return_value = 768
            mock_st.return_value = mock_model

            from synapse.embeddings.sentence_transformer import (
                SentenceTransformerBackend,
            )

            backend = SentenceTransformerBackend()
            backend._validate_dimension()  # Should not raise

    def test_sentence_transformer_validate_dimension_failure(self):
        """Test SentenceTransformer dimension validation failure."""
        with patch(
            "synapse.embeddings.sentence_transformer.SentenceTransformer"
        ) as mock_st:
            mock_model = Mock()
            mock_model.get_sentence_embedding_dimension.return_value = 512
            mock_st.return_value = mock_model

            from synapse.embeddings.sentence_transformer import (
                SentenceTransformerBackend,
            )

            backend = SentenceTransformerBackend()
            with pytest.raises(
                ValueError,
                match="Model dimension 512 does not match ADR-001 requirement of 768",
            ):
                backend._validate_dimension()

    def test_unixcoder_backend_complete(self):
        """Test UniXCoder backend with all paths."""
        with (
            patch(
                "synapse.embeddings.unixcoder.torch.cuda.is_available",
                return_value=False,
            ),
            patch("synapse.embeddings.unixcoder.AutoTokenizer") as mock_tokenizer,
            patch("synapse.embeddings.unixcoder.AutoModel") as mock_model,
        ):
            mock_tokenizer_instance = Mock()
            mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

            mock_model_instance = Mock()
            mock_model.from_pretrained.return_value = mock_model_instance

            from synapse.embeddings.unixcoder import UniXCoderBackend

            backend = UniXCoderBackend()
            assert backend.model_name == "microsoft/unixcoder-base"
            assert backend.device.type == "cpu"

    def test_unixcoder_backend_cuda_available(self):
        """Test UniXCoder backend with CUDA available."""
        with (
            patch(
                "synapse.embeddings.unixcoder.torch.cuda.is_available",
                return_value=True,
            ),
            patch("synapse.embeddings.unixcoder.AutoTokenizer") as mock_tokenizer,
            patch("synapse.embeddings.unixcoder.AutoModel") as mock_model,
        ):
            mock_tokenizer_instance = Mock()
            mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

            mock_model_instance = Mock()
            mock_model.from_pretrained.return_value = mock_model_instance

            from synapse.embeddings.unixcoder import UniXCoderBackend

            backend = UniXCoderBackend()
            assert backend.device.type == "cuda"

    def test_unixcoder_backend_custom_model(self):
        """Test UniXCoder backend with custom model."""
        with (
            patch(
                "synapse.embeddings.unixcoder.torch.cuda.is_available",
                return_value=False,
            ),
            patch("synapse.embeddings.unixcoder.AutoTokenizer") as mock_tokenizer,
            patch("synapse.embeddings.unixcoder.AutoModel") as mock_model,
        ):
            mock_tokenizer_instance = Mock()
            mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

            mock_model_instance = Mock()
            mock_model.from_pretrained.return_value = mock_model_instance

            from synapse.embeddings.unixcoder import UniXCoderBackend

            backend = UniXCoderBackend("custom-model")
            assert backend.model_name == "custom-model"

    def test_unixcoder_get_dimension(self):
        """Test UniXCoder get_dimension method."""
        from synapse.embeddings.unixcoder import UniXCoderBackend

        backend = UniXCoderBackend.__new__(UniXCoderBackend)
        dimension = backend._get_dimension()
        assert dimension == 768

    def test_unixcoder_embed_complete_flow(self):
        """Test UniXCoder embed complete flow."""
        with (
            patch(
                "synapse.embeddings.unixcoder.torch.cuda.is_available",
                return_value=False,
            ),
            patch("synapse.embeddings.unixcoder.AutoTokenizer") as mock_tokenizer,
            patch("synapse.embeddings.unixcoder.AutoModel") as mock_model,
            patch("torch.tensor") as mock_tensor,
        ):
            # Setup tokenizer mock
            mock_tokenizer_instance = Mock()
            mock_inputs = {
                "input_ids": mock_tensor([[1, 2, 3]]),
                "attention_mask": mock_tensor([[1, 1, 1]]),
            }
            mock_tokenizer_instance.return_value = mock_inputs
            mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

            # Setup model mock
            mock_model_instance = Mock()
            mock_outputs = Mock()
            mock_last_hidden = mock_tensor([[[0.1] * 768]])
            mock_outputs.last_hidden_state = mock_last_hidden
            mock_model_instance.return_value = mock_outputs
            mock_model.from_pretrained.return_value = mock_model_instance

            # Setup tensor mock for mean calculation
            mock_mean_tensor = Mock()
            mock_mean_tensor.cpu.return_value.numpy.return_value = [0.1] * 768
            mock_last_hidden.mean.return_value = mock_mean_tensor

            from synapse.embeddings.unixcoder import UniXCoderBackend

            backend = UniXCoderBackend()
            result = backend.embed("test code")

            assert isinstance(result, list)
            assert len(result) == 768

    def test_unixcoder_embed_dimension_validation(self):
        """Test UniXCoder embed dimension validation."""
        with (
            patch(
                "synapse.embeddings.unixcoder.torch.cuda.is_available",
                return_value=False,
            ),
            patch("synapse.embeddings.unixcoder.AutoTokenizer") as mock_tokenizer,
            patch("synapse.embeddings.unixcoder.AutoModel") as mock_model,
            patch("torch.tensor") as mock_tensor,
        ):
            # Setup mocks
            mock_tokenizer_instance = Mock()
            mock_inputs = {"input_ids": mock_tensor([[1, 2, 3]])}
            mock_inputs["to"] = Mock(return_value=mock_inputs)
            mock_tokenizer_instance.return_value = mock_inputs
            mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

            mock_model_instance = Mock()
            mock_outputs = Mock()
            mock_last_hidden = Mock()
            mock_last_hidden.cpu.return_value.numpy.return_value = [
                [0.1] * 500
            ]  # Wrong dimension
            mock_outputs.last_hidden_state = mock_last_hidden
            mock_model_instance.return_value = mock_outputs
            mock_model.from_pretrained.return_value = mock_model_instance

            from synapse.embeddings.unixcoder import UniXCoderBackend

            backend = UniXCoderBackend()
            with pytest.raises(
                ValueError, match="Embedding dimension 500 does not match expected 768"
            ):
                backend.embed("test")

    def test_unixcoder_embed_batch_complete_flow(self):
        """Test UniXCoder embed_batch complete flow."""
        with (
            patch(
                "synapse.embeddings.unixcoder.torch.cuda.is_available",
                return_value=False,
            ),
            patch("synapse.embeddings.unixcoder.AutoTokenizer") as mock_tokenizer,
            patch("synapse.embeddings.unixcoder.AutoModel") as mock_model,
            patch("torch.tensor") as mock_tensor,
        ):
            # Setup tokenizer mock
            mock_tokenizer_instance = Mock()
            mock_inputs = {
                "input_ids": mock_tensor([[1, 2, 3], [4, 5, 6]]),
                "attention_mask": mock_tensor([[1, 1, 1], [1, 1, 1]]),
            }
            mock_tokenizer_instance.return_value = mock_inputs
            mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

            # Setup model mock
            mock_model_instance = Mock()
            mock_outputs = Mock()
            mock_last_hidden = mock_tensor([[[0.1] * 768], [[0.2] * 768]])
            mock_outputs.last_hidden_state = mock_last_hidden
            mock_model_instance.return_value = mock_outputs
            mock_model.from_pretrained.return_value = mock_model_instance

            # Setup tensor mock for mean calculation
            mock_mean_tensor = Mock()
            mock_mean_tensor.cpu.return_value.numpy.return_value = [
                [0.1] * 768,
                [0.2] * 768,
            ]
            mock_last_hidden.mean.return_value = mock_mean_tensor

            from synapse.embeddings.unixcoder import UniXCoderBackend

            backend = UniXCoderBackend()
            result = backend.embed_batch(["code1", "code2"])

            assert isinstance(result, list)
            assert len(result) == 2
            assert all(len(emb) == 768 for emb in result)

    def test_unixcoder_embed_batch_dimension_validation(self):
        """Test UniXCoder embed_batch dimension validation."""
        with (
            patch(
                "synapse.embeddings.unixcoder.torch.cuda.is_available",
                return_value=False,
            ),
            patch("synapse.embeddings.unixcoder.AutoTokenizer") as mock_tokenizer,
            patch("synapse.embeddings.unixcoder.AutoModel") as mock_model,
            patch("torch.tensor") as mock_tensor,
        ):
            # Setup mocks
            mock_tokenizer_instance = Mock()
            mock_inputs = {"input_ids": mock_tensor([[1, 2, 3]])}
            mock_inputs["to"] = Mock(return_value=mock_inputs)
            mock_tokenizer_instance.return_value = mock_inputs
            mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

            mock_model_instance = Mock()
            mock_outputs = Mock()
            mock_last_hidden = Mock()
            mock_last_hidden.cpu.return_value.numpy.return_value = [
                [0.1] * 500
            ]  # Wrong dimension
            mock_outputs.last_hidden_state = mock_last_hidden
            mock_model_instance.return_value = mock_outputs
            mock_model.from_pretrained.return_value = mock_model_instance

            from synapse.embeddings.unixcoder import UniXCoderBackend

            backend = UniXCoderBackend()
            with pytest.raises(
                ValueError, match="Embedding dimension 500 does not match expected 768"
            ):
                backend.embed_batch(["test"])

    def test_unixcoder_error_handling(self):
        """Test UniXCoder error handling."""
        with (
            patch(
                "synapse.embeddings.unixcoder.torch.cuda.is_available",
                return_value=False,
            ),
            patch("synapse.embeddings.unixcoder.AutoTokenizer") as mock_tokenizer,
            patch("synapse.embeddings.unixcoder.AutoModel"),
        ):
            mock_tokenizer_instance = Mock()
            mock_tokenizer_instance.side_effect = Exception("Tokenizer error")
            mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

            from synapse.embeddings.unixcoder import UniXCoderBackend

            with pytest.raises(Exception, match="Tokenizer error"):
                UniXCoderBackend()

    def test_embedding_cache_complete_coverage(self):
        """Test embedding cache complete coverage."""
        with patch("synapse.embeddings.cache.SynapseRedis") as mock_redis:
            mock_redis_instance = Mock()
            mock_redis.return_value = mock_redis_instance

            from synapse.embeddings.cache import EmbeddingCache

            cache = EmbeddingCache(max_size=100)
            assert cache.max_size == 100

            # Test cache miss
            mock_redis_instance.get.return_value = None
            mock_redis_instance.set = AsyncMock()

            # Test embedding and caching
            result = cache.embed("test text")
            assert isinstance(result, list)

            # Test cache hit
            mock_redis_instance.get.return_value = "[0.1, 0.2]"
            result = cache.embed("cached text")
            assert isinstance(result, list)

    def test_embedding_cache_batch_operations(self):
        """Test embedding cache batch operations."""
        with patch("synapse.embeddings.cache.SynapseRedis") as mock_redis:
            mock_redis_instance = Mock()
            mock_redis.return_value = mock_redis_instance

            from synapse.embeddings.cache import EmbeddingCache

            cache = EmbeddingCache()

            # Test batch embedding
            mock_redis_instance.mget = AsyncMock(return_value=[None, None])
            mock_redis_instance.mset = AsyncMock()

            result = cache.embed_batch(["text1", "text2"])
            assert isinstance(result, list)
            assert len(result) == 2

    def test_embedding_cache_stats(self):
        """Test embedding cache statistics."""
        with patch("synapse.embeddings.cache.SynapseRedis") as mock_redis:
            mock_redis_instance = Mock()
            mock_redis.return_value = mock_redis_instance

            from synapse.embeddings.cache import EmbeddingCache

            cache = EmbeddingCache()

            # Mock cache stats
            mock_redis_instance.info = AsyncMock(
                return_value={"keyspace_hits": 100, "keyspace_misses": 50}
            )

            stats = cache.get_stats()
            assert "hits" in stats
            assert "misses" in stats
