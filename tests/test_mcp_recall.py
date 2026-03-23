"""Tests for MCP Recall - 100% Coverage."""

from unittest.mock import Mock

import pytest


class TestMCPRecall:
    """Test MCPRecall handler."""

    def test_init(self):
        """Test initialization."""
        from synapse.mcp.recall import MCPRecall

        mock_redis = Mock()
        mock_embeddings = Mock()
        recall = MCPRecall(mock_redis, mock_embeddings)
        assert recall.redis == mock_redis
        assert recall.embeddings == mock_embeddings

    def test_validate_params_success(self):
        """Test validate_params with valid params."""
        from synapse.mcp.recall import MCPRecall

        mock_redis = Mock()
        mock_embeddings = Mock()
        recall = MCPRecall(mock_redis, mock_embeddings)
        recall._validate_params({"query": "test"})

    def test_validate_params_missing_query(self):
        """Test validate_params with missing query."""
        from synapse.mcp.recall import MCPRecall

        mock_redis = Mock()
        mock_embeddings = Mock()
        recall = MCPRecall(mock_redis, mock_embeddings)
        with pytest.raises(ValueError, match="Missing required field: query"):
            recall._validate_params({})

    def test_validate_params_empty_query(self):
        """Test validate_params with empty query."""
        from synapse.mcp.recall import MCPRecall

        mock_redis = Mock()
        mock_embeddings = Mock()
        recall = MCPRecall(mock_redis, mock_embeddings)
        with pytest.raises(ValueError, match="Query must be a non-empty string"):
            recall._validate_params({"query": ""})

    def test_handle_recall_success(self):
        """Test handle_recall success."""
        from synapse.mcp.recall import MCPRecall

        mock_redis = Mock()
        mock_embeddings = Mock()
        mock_embeddings.embed.return_value = [0.1] * 768
        mock_redis.search_hybrid.return_value = [
            {
                "id": "node:test:123",
                "domain": "test",
                "type": "entity",
                "content": "test content",
            }
        ]
        mock_redis.get_linked_nodes.return_value = []
        recall = MCPRecall(mock_redis, mock_embeddings)
        result = recall.handle_recall({"query": "test"})
        assert "results" in result
        assert result["total"] == 1

    def test_handle_recall_validation_error(self):
        """Test handle_recall validation error."""
        from synapse.mcp.recall import MCPRecall

        mock_redis = Mock()
        mock_embeddings = Mock()
        recall = MCPRecall(mock_redis, mock_embeddings)
        result = recall.handle_recall({})
        assert "format" in result
        assert result["format"] == "error"

    def test_handle_recall_exception(self):
        """Test handle_recall exception handling."""
        from synapse.mcp.recall import MCPRecall

        mock_redis = Mock()
        mock_embeddings = Mock()
        mock_embeddings.embed.side_effect = Exception("Embedding error")
        recall = MCPRecall(mock_redis, mock_embeddings)
        result = recall.handle_recall({"query": "test"})
        assert "format" in result
        assert result["format"] == "error"
