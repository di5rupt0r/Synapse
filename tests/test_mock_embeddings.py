"""Mock embedding backend for testing."""

from typing import List
from synapse.embeddings.backend import EmbeddingBackend


class MockEmbeddingBackend(EmbeddingBackend):
    """Mock embedding backend for testing."""
    
    def __init__(self, model_name: str = "mock-model") -> None:
        super().__init__(model_name)
    
    def _get_dimension(self) -> int:
        """Return fixed dimension for testing."""
        return 768
    
    def embed(self, text: str) -> List[float]:
        """Generate mock embedding based on text hash."""
        # Generate consistent but different embeddings based on text
        import hashlib
        hash_obj = hashlib.md5(text.encode())
        hash_bytes = hash_obj.digest()
        
        # Convert to 768-dim float vector
        embedding = []
        for i in range(768):
            byte_val = hash_bytes[i % len(hash_bytes)]
            # Convert byte to float between -1 and 1
            embedding.append((byte_val - 128) / 128.0)
        
        return embedding
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate mock embeddings for multiple texts."""
        return [self.embed(text) for text in texts]
