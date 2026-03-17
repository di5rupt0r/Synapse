"""LRU Cache for embedding backend."""

from collections import OrderedDict
from typing import Dict, List

from .backend import EmbeddingBackend


class EmbeddingCache(EmbeddingBackend):
    """LRU cache wrapper for embedding backends."""

    def __init__(self, backend: EmbeddingBackend, max_size: int = 1000) -> None:
        """Initialize cache with backend and max size."""
        self.backend = backend
        self.max_size = max_size
        self.cache: OrderedDict[str, List[float]] = OrderedDict()
        self.hits = 0
        self.misses = 0

        # Initialize parent with backend's model name
        super().__init__(backend.model_name)

    def _get_dimension(self) -> int:
        """Get embedding dimension from backend."""
        return self.backend.dimension

    def embed(self, text: str) -> List[float]:
        """Generate embedding with LRU cache."""
        if text in self.cache:
            # Cache hit - move to end (most recently used)
            self.cache.move_to_end(text)
            self.hits += 1
            return self.cache[text]

        # Cache miss - generate and store
        embedding = self.backend.embed(text)
        self._store_in_cache(text, embedding)
        self.misses += 1
        return embedding

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate batch embeddings with cache optimization."""
        results = []
        uncached_texts = []
        uncached_indices = []

        # Check cache for each text
        for i, text in enumerate(texts):
            if text in self.cache:
                # Cache hit
                self.cache.move_to_end(text)
                self.hits += 1
                results.append(self.cache[text])
            else:
                # Cache miss - placeholder for now
                self.misses += 1
                results.append(None)  # Placeholder
                uncached_texts.append(text)
                uncached_indices.append(i)

        # Generate embeddings for uncached texts
        if uncached_texts:
            uncached_embeddings = self.backend.embed_batch(uncached_texts)

            # Store in cache and update results
            for text, embedding, index in zip(
                uncached_texts, uncached_embeddings, uncached_indices
            ):
                self._store_in_cache(text, embedding)
                results[index] = embedding

        return results

    def _store_in_cache(self, text: str, embedding: List[float]) -> None:
        """Store embedding in cache with LRU eviction."""
        # Remove oldest if cache is full
        if len(self.cache) >= self.max_size and text not in self.cache:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]

        # Store new item
        self.cache[text] = embedding
        self.cache.move_to_end(text)

    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0

        return {
            "hits": self.hits,
            "misses": self.misses,
            "size": len(self.cache),
            "max_size": self.max_size,
            "hit_rate": hit_rate,
        }

    def clear(self) -> None:
        """Clear cache and reset statistics."""
        self.cache.clear()
        self.hits = 0
        self.misses = 0
