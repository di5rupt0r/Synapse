"""TDD RED Phase: Tests for Embedding LRU Cache."""

from unittest.mock import Mock


def test_lru_cache_basic():
    """Test basic LRU cache functionality."""
    # This will fail - cache doesn't exist yet (RED phase)
    from synapse.embeddings.cache import EmbeddingCache

    mock_backend = Mock()
    mock_backend.embed.return_value = [0.1] * 768

    cache = EmbeddingCache(mock_backend, max_size=2)

    # First call should hit backend
    result1 = cache.embed("test text")
    assert result1 == [0.1] * 768
    assert mock_backend.embed.call_count == 1

    # Second call with same text should use cache
    result2 = cache.embed("test text")
    assert result2 == [0.1] * 768
    assert mock_backend.embed.call_count == 1  # Still only 1 call


def test_lru_cache_eviction():
    """Test LRU eviction when cache is full."""
    from synapse.embeddings.cache import EmbeddingCache

    mock_backend = Mock()
    mock_backend.embed.return_value = [0.1] * 768

    cache = EmbeddingCache(mock_backend, max_size=2)

    # Fill cache
    cache.embed("text1")
    cache.embed("text2")

    # Add third item, should evict first
    cache.embed("text3")

    # First item should be evicted, call backend again
    cache.embed("text1")
    assert mock_backend.embed.call_count == 4  # text1, text2, text3, text1 again


def test_lru_cache_batch():
    """Test batch embedding with cache."""
    from synapse.embeddings.cache import EmbeddingCache

    mock_backend = Mock()
    mock_backend.embed.return_value = [0.1] * 768
    # Configure mock to return different values for different calls
    mock_backend.embed_batch.side_effect = [
        [[0.1] * 768, [0.2] * 768],  # First call: text1, text2
        [[0.3] * 768],  # Second call: text3 only
    ]

    cache = EmbeddingCache(mock_backend, max_size=10)

    # First batch call
    texts = ["text1", "text2"]
    result1 = cache.embed_batch(texts)
    assert result1 == [[0.1] * 768, [0.2] * 768]
    assert mock_backend.embed_batch.call_count == 1

    # Second batch with one cached, one new
    texts = ["text1", "text3"]
    result2 = cache.embed_batch(texts)
    assert result2 == [[0.1] * 768, [0.3] * 768]  # text3 gets new embedding
    assert mock_backend.embed_batch.call_count == 2  # Only for text3


def test_lru_cache_stats():
    """Test cache statistics tracking."""
    from synapse.embeddings.cache import EmbeddingCache

    mock_backend = Mock()
    mock_backend.embed.return_value = [0.1] * 768

    cache = EmbeddingCache(mock_backend, max_size=2)

    # Cache miss
    cache.embed("text1")
    stats = cache.get_stats()
    assert stats["hits"] == 0
    assert stats["misses"] == 1

    # Cache hit
    cache.embed("text1")
    stats = cache.get_stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 1


def test_lru_cache_clear():
    """Test cache clearing functionality."""
    from synapse.embeddings.cache import EmbeddingCache

    mock_backend = Mock()
    mock_backend.embed.return_value = [0.1] * 768

    cache = EmbeddingCache(mock_backend, max_size=2)

    cache.embed("text1")
    cache.embed("text2")

    # Clear cache
    cache.clear()

    # Should hit backend again
    cache.embed("text1")
    assert mock_backend.embed.call_count == 3  # text1, text2, text1 after clear
