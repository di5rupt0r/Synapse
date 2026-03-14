"""TDD RED Phase: Tests for MCP Memorize Handler."""

import pytest
from unittest.mock import Mock, patch
import uuid


def test_memorize_pipeline():
    """Test complete memorize pipeline: validate → embed → store → respond."""
    # This will fail - memorize handler doesn't exist yet (RED phase)
    from synapse.mcp.memorize import MCPMemorize
    
    mock_redis = Mock()
    mock_embeddings = Mock()
    
    handler = MCPMemorize(mock_redis, mock_embeddings)
    
    params = {
        "domain": "test",
        "type": "entity",
        "content": "Test content"
    }
    
    response = handler.handle_memorize(params)
    
    assert response["status"] == "success"
    assert "id" in response
    assert response["id"].startswith("node:test:")
    assert len(response["id"].split(":")) == 3  # node:domain:uuid


def test_memorize_validation():
    """Test input validation for memorize requests."""
    from synapse.mcp.memorize import MCPMemorize
    
    mock_redis = Mock()
    mock_embeddings = Mock()
    
    handler = MCPMemorize(mock_redis, mock_embeddings)
    
    # Test missing required fields
    response = handler.handle_memorize({})
    
    assert response["status"] == "error"
    assert "Missing required field" in response["error"]


def test_memorize_embedding_generation():
    """Test embedding generation during memorize."""
    from synapse.mcp.memorize import MCPMemorize
    
    mock_redis = Mock()
    mock_redis.store_node.return_value = "node:test:123"
    
    mock_embeddings = Mock()
    mock_embeddings.embed.return_value = [0.0] * 768
    
    handler = MCPMemorize(mock_redis, mock_embeddings)
    
    params = {
        "domain": "test",
        "type": "entity", 
        "content": "Test content"
    }
    
    handler.handle_memorize(params)
    
    # Verify embedding was generated
    mock_embeddings.embed.assert_called_once_with("Test content")
    
    # Verify node was stored with embedding
    mock_redis.store_node.assert_called_once()
    call_args = mock_redis.store_node.call_args
    assert len(call_args[1]["embedding"]) == 768


def test_memorize_with_optional_fields():
    """Test memorize with optional metadata and links."""
    from synapse.mcp.memorize import MCPMemorize
    
    mock_redis = Mock()
    mock_redis.store_node.return_value = "node:test:123"
    
    mock_embeddings = Mock()
    mock_embeddings.embed.return_value = [0.0] * 768
    
    handler = MCPMemorize(mock_redis, mock_embeddings)
    
    params = {
        "domain": "test",
        "type": "entity",
        "content": "Test content",
        "metadata": {"source": "test"},
        "links": {"inbound": ["node:other:123"], "outbound": []}
    }
    
    response = handler.handle_memorize(params)
    
    assert response["status"] == "success"
    
    # Verify optional fields were passed through
    call_args = mock_redis.store_node.call_args
    assert call_args[1]["metadata"] == {"source": "test"}
    assert call_args[1]["links"] == {"inbound": ["node:other:123"], "outbound": []}
