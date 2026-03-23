"""SentenceTransformer embedding backend."""

from typing import List

from sentence_transformers import SentenceTransformer

from .backend import EmbeddingBackend


class SentenceTransformerBackend(EmbeddingBackend):
    """SentenceTransformer implementation of embedding backend."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        """Initialize SentenceTransformer backend."""
        self.model = SentenceTransformer(model_name)
        super().__init__(model_name)
        self._validate_dimension()

    def _get_dimension(self) -> int:
        """Get embedding dimension for the model."""
        return self.model.get_sentence_embedding_dimension()

    def embed(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        embedding = self.model.encode(text, convert_to_numpy=True)
        # Handle both numpy arrays and lists
        if hasattr(embedding, "tolist"):
            return embedding.tolist()
        return list(embedding)

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        # Handle both numpy arrays and lists
        if hasattr(embeddings, "tolist"):
            res = embeddings.tolist()
            if res and not isinstance(res[0], list):
                return [res]
            return res
        
        # In case it's a 1D list of floats
        if embeddings and isinstance(embeddings[0], float):
            return [embeddings]
            
        return [list(emb) for emb in embeddings]

    def _validate_dimension(self) -> None:
        """Validate embedding dimension matches ADR-001 requirements."""
        if self.dimension != 768:
            raise ValueError(
                f"Model dimension {self.dimension} does not match ADR-001 requirement of 768"
            )
