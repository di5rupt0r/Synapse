"""BM25 sparse search implementation for Synapse AKG."""

from typing import List, Dict, Any, Tuple
from rank_bm25 import BM25Okapi
import re
from synapse.schema.node import Chunk


class BM25Index:
    """In-memory BM25 index for fast sparse search."""
    
    def __init__(self, chunks: List[Chunk]) -> None:
        """Initialize BM25 index with chunks.
        
        Args:
            chunks: List of chunks to index
        """
        self.chunks = chunks
        self._chunk_id_map = {i: chunk.id for i, chunk in enumerate(chunks)}
        self.bm25 = self._create_bm25_index(chunks)
    
    def _create_bm25_index(self, chunks: List[Chunk]) -> BM25Okapi:
        """Create BM25 index from chunks."""
        # Tokenize chunk texts
        tokenized_docs = []
        for chunk in chunks:
            tokens = self._tokenize(chunk.text)
            tokenized_docs.append(tokens)
        
        return BM25Okapi(tokenized_docs)
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text for BM25 indexing.
        
        Handles:
        - Case normalization
        - Underscore/splitting (hello_world -> hello, world)
        - Code-specific patterns
        """
        # Convert to lowercase
        text = text.lower()
        
        # Split on common delimiters
        tokens = re.split(r'[\s\(\)\[\]\{\},;:.]+', text)
        
        # Further split camelCase and snake_case
        expanded_tokens = []
        for token in tokens:
            if token:
                # Split snake_case
                snake_parts = token.split('_')
                for part in snake_parts:
                    if part:
                        # Split camelCase
                        camel_parts = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?![a-z])', part)
                        expanded_tokens.extend(camel_parts)
        
        # Filter empty tokens and short tokens
        return [token for token in expanded_tokens if len(token) >= 2]
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Search for chunks using BM25.
        
        Args:
            query: Search query
            top_k: Number of top results to return
            
        Returns:
            List of (chunk_id, score) tuples, sorted by score descending
        """
        if not query.strip():
            return []
        
        # Tokenize query
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []
        
        # Get BM25 scores
        scores = self.bm25.get_scores(query_tokens)
        
        # Create (chunk_id, score) pairs
        results = []
        for doc_idx, score in enumerate(scores):
            # BM25 can return negative scores, treat non-zero as valid matches
            if score != 0:  # Filter out zero scores
                chunk_id = self._chunk_id_map[doc_idx]
                results.append((chunk_id, float(score)))
        
        # Sort by score descending and limit to top_k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def get_chunk_by_id(self, chunk_id: str) -> Chunk:
        """Get chunk by ID."""
        for chunk in self.chunks:
            if chunk.id == chunk_id:
                return chunk
        raise ValueError(f"Chunk with ID {chunk_id} not found")
    
    def update_index(self, new_chunks: List[Chunk]) -> None:
        """Update the BM25 index with new chunks.
        
        Args:
            new_chunks: New chunks to add to the index
        """
        # Combine existing and new chunks
        all_chunks = self.chunks + new_chunks
        
        # Rebuild index
        self.chunks = all_chunks
        self._chunk_id_map = {i: chunk.id for i, chunk in enumerate(all_chunks)}
        self.bm25 = self._create_bm25_index(all_chunks)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        return {
            "total_chunks": len(self.chunks),
            "languages": list(set(chunk.language for chunk in self.chunks)),
            "node_types": list(set(chunk.node_type for chunk in self.chunks)),
            "avg_chunk_length": sum(len(chunk.text) for chunk in self.chunks) / len(self.chunks) if self.chunks else 0
        }
