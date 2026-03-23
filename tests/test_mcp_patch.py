"""Tests for MCP Patch - 100% Coverage."""

from unittest.mock import Mock

import pytest


class TestMCPPatch:
    """Test MCPPatch handler."""

    def test_init(self):
        """Test initialization."""
        from synapse.mcp.patch import MCPPatch

        mock_redis = Mock()
        patch = MCPPatch(mock_redis)
        assert patch.redis == mock_redis

    def test_validate_params_success(self):
        """Test validate_params with valid params."""
        from synapse.mcp.patch import MCPPatch

        mock_redis = Mock()
        patch = MCPPatch(mock_redis)
        patch._validate_params(
            {
                "node_id": "node:test:123",
                "operations": [{"op": "set", "path": "$.field", "value": "test"}],
            }
        )

    def test_validate_params_missing_node_id(self):
        """Test validate_params with missing node_id."""
        from synapse.mcp.patch import MCPPatch

        mock_redis = Mock()
        patch = MCPPatch(mock_redis)
        with pytest.raises(ValueError, match="Missing required field: node_id"):
            patch._validate_params({"operations": []})

    def test_validate_params_missing_operations(self):
        """Test validate_params with missing operations."""
        from synapse.mcp.patch import MCPPatch

        mock_redis = Mock()
        patch = MCPPatch(mock_redis)
        with pytest.raises(ValueError, match="Missing required field: operations"):
            patch._validate_params({"node_id": "node:test:123"})

    def test_validate_operations_success(self):
        """Test validate_operations with valid operations."""
        from synapse.mcp.patch import MCPPatch

        mock_redis = Mock()
        patch = MCPPatch(mock_redis)
        patch._validate_operations([{"op": "set", "path": "$.field", "value": "test"}])

    def test_validate_operations_missing_op(self):
        """Test validate_operations with missing op."""
        from synapse.mcp.patch import MCPPatch

        mock_redis = Mock()
        patch = MCPPatch(mock_redis)
        with pytest.raises(ValueError, match="missing required field: op"):
            patch._validate_operations([{"path": "$.field"}])

    def test_validate_operations_missing_path(self):
        """Test validate_operations with missing path."""
        from synapse.mcp.patch import MCPPatch

        mock_redis = Mock()
        patch = MCPPatch(mock_redis)
        with pytest.raises(ValueError, match="missing required field: path"):
            patch._validate_operations([{"op": "set"}])

    def test_validate_operations_invalid_op(self):
        """Test validate_operations with invalid op."""
        from synapse.mcp.patch import MCPPatch

        mock_redis = Mock()
        patch = MCPPatch(mock_redis)
        with pytest.raises(ValueError, match="invalid op"):
            patch._validate_operations([{"op": "invalid", "path": "$.field"}])

    def test_handle_patch_success(self):
        """Test handle_patch success."""
        from synapse.mcp.patch import MCPPatch

        mock_redis = Mock()
        mock_redis.get_node = Mock(return_value={"id": "node:test:123"})
        mock_redis.update_node = Mock(return_value=True)
        patch = MCPPatch(mock_redis)
        result = patch.handle_patch(
            {
                "node_id": "node:test:123",
                "operations": [{"op": "set", "path": "$.field", "value": "test"}],
            }
        )
        assert result["status"] == "success"
        assert result["updated"] is True

    def test_handle_patch_node_not_found(self):
        """Test handle_patch node not found."""
        from synapse.mcp.patch import MCPPatch

        mock_redis = Mock()
        mock_redis.get_node = Mock(return_value=None)
        patch = MCPPatch(mock_redis)
        result = patch.handle_patch(
            {
                "node_id": "node:test:123",
                "operations": [{"op": "delete", "path": "$.field"}],
            }
        )
        assert result["status"] == "error"

    def test_handle_patch_update_failure(self):
        """Test handle_patch update failure."""
        from synapse.mcp.patch import MCPPatch

        mock_redis = Mock()
        mock_redis.get_node = Mock(return_value={"id": "node:test:123"})
        mock_redis.update_node = Mock(return_value=False)
        patch = MCPPatch(mock_redis)
        result = patch.handle_patch(
            {
                "node_id": "node:test:123",
                "operations": [{"op": "set", "path": "$.field", "value": "test"}],
            }
        )
        assert result["status"] == "error"

    def test_handle_patch_validation_error(self):
        """Test handle_patch validation error."""
        from synapse.mcp.patch import MCPPatch

        mock_redis = Mock()
        patch = MCPPatch(mock_redis)
        result = patch.handle_patch({})
        assert result["status"] == "error"

    def test_handle_patch_exception(self):
        """Test handle_patch exception handling."""
        from synapse.mcp.patch import MCPPatch

        mock_redis = Mock()
        mock_redis.get_node = Mock(side_effect=Exception("Redis error"))
        patch = MCPPatch(mock_redis)
        result = patch.handle_patch(
            {
                "node_id": "node:test:123",
                "operations": [{"op": "set", "path": "$.field", "value": "test"}],
            }
        )
        assert result["status"] == "error"
