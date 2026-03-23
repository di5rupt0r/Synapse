"""Tests for embedding backends - Complete coverage."""

from unittest.mock import Mock, patch
from types import SimpleNamespace
import numpy as np
import pytest
import torch


class TestEmbeddingBackendComplete:
    """Complete embedding backend coverage."""

    def test_embedding_backend_abstract_methods(self):
        """Test EmbeddingBackend abstract methods."""
        from synapse.embeddings.backend import EmbeddingBackend

        with pytest.raises(TypeError):
            EmbeddingBackend()

    def test_sentence_transformer_backend_complete(self):
        with patch("synapse.embeddings.sentence_transformer.SentenceTransformer") as mock_st:
            mock_model = Mock()
            mock_model.get_sentence_embedding_dimension.return_value = 768
            mock_st.return_value = mock_model

            from synapse.embeddings.sentence_transformer import SentenceTransformerBackend
            backend = SentenceTransformerBackend()
            
            assert backend.model_name == "all-MiniLM-L6-v2"
            assert backend.dimension == 768

    def test_sentence_transformer_dimension_validation_failure(self):
        with patch("synapse.embeddings.sentence_transformer.SentenceTransformer") as mock_st:
            mock_model = Mock()
            mock_model.get_sentence_embedding_dimension.return_value = 512
            mock_st.return_value = mock_model

            from synapse.embeddings.sentence_transformer import SentenceTransformerBackend
            
            with pytest.raises(ValueError):
                SentenceTransformerBackend()

    def test_sentence_transformer_embed(self):
        with patch("synapse.embeddings.sentence_transformer.SentenceTransformer") as mock_st:
            mock_model = Mock()
            mock_model.get_sentence_embedding_dimension.return_value = 768
            mock_model.encode.return_value = np.array([0.1] * 768)
            mock_st.return_value = mock_model

            from synapse.embeddings.sentence_transformer import SentenceTransformerBackend
            backend = SentenceTransformerBackend()
            result = backend.embed("test text")

            assert isinstance(result, list)
            assert len(result) == 768

    def test_sentence_transformer_embed_batch(self):
        with patch("synapse.embeddings.sentence_transformer.SentenceTransformer") as mock_st:
            mock_model = Mock()
            mock_model.get_sentence_embedding_dimension.return_value = 768
            mock_model.encode.return_value = np.array([[0.1] * 768, [0.2] * 768])
            mock_st.return_value = mock_model

            from synapse.embeddings.sentence_transformer import SentenceTransformerBackend
            backend = SentenceTransformerBackend()
            result = backend.embed_batch(["text1", "text2"])

            assert len(result) == 2
            assert result[0][0] == 0.1

    def test_unixcoder_backend_complete(self):
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

    def test_unixcoder_embed(self):
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

    def test_unixcoder_error_handling(self):
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("synapse.embeddings.unixcoder.AutoTokenizer") as mock_tokenizer,
            patch("synapse.embeddings.unixcoder.AutoModel") as mock_model,
        ):
            mock_tokenizer.from_pretrained.side_effect = Exception("Tokenizer error")

            from synapse.embeddings.unixcoder import UniXCoderBackend
            with pytest.raises(Exception, match="Tokenizer error"):
                UniXCoderBackend()

    def test_embedding_cache_complete_coverage(self):
        from synapse.embeddings.cache import EmbeddingCache
        from synapse.embeddings.backend import EmbeddingBackend

        class DummyBackend(EmbeddingBackend):
            def __init__(self):
                self.dimension = 768
                self.model_name = "dummy"
            def _get_dimension(self) -> int:
                return self.dimension
            def embed(self, text: str):
                return [0.1] * self.dimension
            def embed_batch(self, texts: list[str]):
                return [[0.1] * self.dimension for _ in texts]

        dummy = DummyBackend()
        cache = EmbeddingCache(backend=dummy, max_size=2)
        
        assert cache.max_size == 2
        
        # Test Cache miss
        res1 = cache.embed("text1")
        assert len(res1) == 768
        assert cache.misses == 1
        
        # Test Cache hit
        res2 = cache.embed("text1")
        assert cache.hits == 1
        
        # Test eviction
        cache.embed("text2")
        cache.embed("text3") # Should evict text1 (since text2 was more recently added)
        assert "text1" not in cache.cache

    def test_embedding_cache_batch_operations(self):
        from synapse.embeddings.cache import EmbeddingCache
        from synapse.embeddings.backend import EmbeddingBackend

        class DummyBackend(EmbeddingBackend):
            def __init__(self):
                self.dimension = 768
                self.model_name = "dummy"
            def _get_dimension(self) -> int:
                return self.dimension
            def embed(self, text: str):
                return [0.1] * self.dimension
            def embed_batch(self, texts: list[str]):
                return [[0.1] * self.dimension for _ in texts]

        dummy = DummyBackend()
        cache = EmbeddingCache(backend=dummy)

        # Mixed batch operation
        cache.embed("text1") # miss -> caches text1
        
        res = cache.embed_batch(["text1", "text2"])
        assert len(res) == 2
        assert cache.hits == 1 # text1
        assert cache.misses == 2 # initial text1 + batch text2

    def test_embedding_cache_stats(self):
        from synapse.embeddings.cache import EmbeddingCache
        from synapse.embeddings.backend import EmbeddingBackend

        class DummyBackend(EmbeddingBackend):
            def __init__(self):
                self.dimension = 768
                self.model_name = "dummy"
            def _get_dimension(self) -> int:
                return self.dimension
            def embed(self, text: str):
                return [0.1] * self.dimension
            def embed_batch(self, texts: list[str]):
                return [[0.1] * self.dimension for _ in texts]

        dummy = DummyBackend()
        cache = EmbeddingCache(backend=dummy)
        
        cache.embed("text1")
        cache.embed_batch(["text1", "text2"])
        
        stats = cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 2
        assert stats["size"] == 2
