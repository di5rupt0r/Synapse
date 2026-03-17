"""TDD RED Phase: Tests for MCP Recall Handler."""

from unittest.mock import Mock


def test_recall_pipeline():
    """Test complete recall pipeline: embed → search → resolve → compress."""
    from synapse.mcp.recall import MCPRecall

    mock_redis = Mock()
    # Mock search results with expected structure
    mock_redis.search_hybrid.return_value = [
        {
            "json": {
                "id": "node:test:123",
                "domain": "test",
                "type": "entity",
                "content": "test content",
            }
        }
    ]
    mock_redis.get_linked_nodes.return_value = []

    mock_embeddings = Mock()
    mock_embeddings.embed.return_value = [0.1] * 768

    handler = MCPRecall(mock_redis, mock_embeddings)

    params = {"query": "test query", "domain_filter": ["test"], "limit": 5}

    response = handler.handle_recall(params)

    assert response["format"] == "compressed_yaml"
    assert "content" in response
    assert isinstance(response["content"], str)


def test_recall_embedding_and_search():
    """Test embedding generation and search during recall."""
    from synapse.mcp.recall import MCPRecall

    mock_redis = Mock()
    mock_redis.search_hybrid.return_value = [
        {
            "id": "node:test:123",
            "domain": "test",
            "type": "entity",
            "content": "test content",
        }
    ]
    mock_redis.get_linked_nodes.return_value = []

    mock_embeddings = Mock()
    mock_embeddings.embed.return_value = [0.1] * 768

    handler = MCPRecall(mock_redis, mock_embeddings)

    params = {"query": "test query"}

    handler.handle_recall(params)

    # Verify embedding was generated
    mock_embeddings.embed.assert_called_once_with("test query")

    # Verify search was called
    mock_redis.search_hybrid.assert_called_once()


def test_recall_domain_filtering():
    """Test domain filtering in recall queries."""
    from synapse.mcp.recall import MCPRecall

    mock_redis = Mock()
    mock_redis.search_hybrid.return_value = []
    mock_redis.get_linked_nodes.return_value = []

    mock_embeddings = Mock()
    mock_embeddings.embed.return_value = [0.1] * 768

    handler = MCPRecall(mock_redis, mock_embeddings)

    params = {"query": "test query", "domain_filter": ["test", "docs"]}

    handler.handle_recall(params)

    # Verify search was called with domain filter
    call_args = mock_redis.search_hybrid.call_args
    assert call_args[1]["domain_filter"] == ["test", "docs"]


def test_recall_yaml_compression():
    """Test YAML compression output format."""
    from synapse.mcp.recall import MCPRecall

    mock_redis = Mock()
    mock_redis.search_hybrid.return_value = [
        {
            "id": "node:test:123",
            "domain": "test",
            "type": "entity",
            "content": "test content",
        }
    ]
    mock_redis.get_linked_nodes.return_value = [
        {
            "id": "node:test:456",
            "domain": "test",
            "type": "relation",
            "content": "related content",
        }
    ]

    mock_embeddings = Mock()
    mock_embeddings.embed.return_value = [0.1] * 768

    handler = MCPRecall(mock_redis, mock_embeddings)

    params = {"query": "test query"}

    response = handler.handle_recall(params)

    assert response["format"] == "compressed_yaml"
    assert isinstance(response["content"], str)
    # Verify YAML contains expected data
    assert "matched_nodes" in response["content"]
    assert "resolved_edges" in response["content"]
