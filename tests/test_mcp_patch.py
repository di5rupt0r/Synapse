"""TDD RED Phase: Tests for MCP Patch Handler."""

from unittest.mock import Mock


def test_patch_pipeline():
    """Test complete patch pipeline: validate → MULTI/EXEC → update timestamp."""
    # This will fail - patch handler doesn't exist yet (RED phase)
    from synapse.mcp.patch import MCPPatch

    mock_redis = Mock()

    handler = MCPPatch(mock_redis)

    params = {
        "node_id": "node:test:12345678-1234-1234-1234-123456789abc",
        "operations": [
            {"path": "$.content", "op": "set", "value": "updated content"},
            {"path": "$.metadata.version", "op": "set", "value": "2.0"},
        ],
    }

    response = handler.handle_patch(params)

    assert response["status"] == "success"
    assert "updated_fields" in response
    assert len(response["updated_fields"]) == 2


def test_patch_validation():
    """Test input validation for patch requests."""
    from synapse.mcp.patch import MCPPatch

    mock_redis = Mock()

    handler = MCPPatch(mock_redis)

    # Test missing required fields
    response = handler.handle_patch({})

    assert response["status"] == "error"
    assert "Missing required field" in response["error"]


def test_patch_node_not_found():
    """Test patch behavior when node doesn't exist."""
    from synapse.mcp.patch import MCPPatch

    mock_redis = Mock()
    mock_redis.get_node.return_value = None

    handler = MCPPatch(mock_redis)

    params = {
        "node_id": "node:test:12345678-1234-1234-1234-123456789abc",
        "operations": [{"path": "$.content", "op": "set", "value": "new content"}],
    }

    response = handler.handle_patch(params)

    assert response["status"] == "error"
    assert "not found" in response["error"]


def test_patch_atomic_operations():
    """Test atomic patch operations with different operation types."""
    from synapse.mcp.patch import MCPPatch

    mock_redis = Mock()
    mock_redis.get_node.return_value = {"id": "node:test:123"}
    mock_redis.update_node.return_value = True

    handler = MCPPatch(mock_redis)

    params = {
        "node_id": "node:test:12345678-1234-1234-1234-123456789abc",
        "operations": [
            {"path": "$.content", "op": "set", "value": "new content"},
            {"path": "$.metadata.old_field", "op": "delete"},
            {"path": "$.links.outbound", "op": "append", "value": "node:other:123"},
        ],
    }

    response = handler.handle_patch(params)

    assert response["status"] == "success"
    assert len(response["updated_fields"]) == 3

    # Verify update_node was called with all operations
    mock_redis.update_node.assert_called_once()
    call_args = mock_redis.update_node.call_args
    # call_args[0] is positional args, call_args[1] is kwargs
    operations = call_args[0][1] if call_args[0] else call_args[1].get("operations", [])
    assert len(operations) == 3  # Three operations
