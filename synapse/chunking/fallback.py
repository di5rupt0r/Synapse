"""Fallback line-based code chunking."""

from typing import List, Dict, Any
import uuid


def fallback_chunk_by_lines(content: str, chunk_size: int = 50, overlap: int = 5) -> List[Dict[str, Any]]:
    """Fallback line-based chunking when tree-sitter fails.
    
    Args:
        content: The code content to chunk
        chunk_size: Maximum number of lines per chunk
        overlap: Number of overlapping lines between chunks
        
    Returns:
        List of chunk dictionaries
    """
    if not content.strip():
        return []
    
    lines = content.split('\n')
    chunks = []
    
    for i in range(0, len(lines), chunk_size - overlap):
        chunk_lines = lines[i:i + chunk_size]
        if not chunk_lines:
            continue
            
        chunk_text = '\n'.join(chunk_lines)
        
        chunks.append({
            "id": f"chunk:{uuid.uuid4()}",
            "text": chunk_text,
            "language": "unknown",
            "node_type": "line_chunk",
            "line_start": i + 1,
            "line_end": min(i + chunk_size, len(lines)),
        })
    
    return chunks
