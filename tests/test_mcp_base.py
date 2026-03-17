"""TDD RED Phase: Tests for MCP Base Handler."""


def test_json_rpc_request_parsing():
    """Test JSON-RPC 2.0 request parsing."""
    # This will fail - MCP base doesn't exist yet (RED phase)
    from synapse.mcp.base import MCPBase

    handler = MCPBase()

    raw_request = {
        "jsonrpc": "2.0",
        "id": "test-123",
        "method": "test_method",
        "params": {"arg1": "value1"},
    }

    parsed = handler.parse_request(raw_request)

    assert parsed.id == "test-123"
    assert parsed.method == "test_method"
    assert parsed.params == {"arg1": "value1"}


def test_json_rpc_response_format():
    """Test JSON-RPC 2.0 response formatting."""
    from synapse.mcp.base import MCPBase

    handler = MCPBase()

    response = handler.format_response(id="test-123", result={"status": "success"})

    assert response["jsonrpc"] == "2.0"
    assert response["id"] == "test-123"
    assert response["result"] == {"status": "success"}
    assert "error" not in response


def test_json_rpc_error_format():
    """Test JSON-RPC 2.0 error formatting."""
    from synapse.mcp.base import MCPBase

    handler = MCPBase()

    error_response = handler.format_error(
        id="test-123", code=-32601, message="Method not found"
    )

    assert error_response["jsonrpc"] == "2.0"
    assert error_response["id"] == "test-123"
    assert error_response["error"]["code"] == -32601
    assert error_response["error"]["message"] == "Method not found"


def test_method_registration():
    """Test MCP method registration."""
    from synapse.mcp.base import MCPBase

    handler = MCPBase()

    @handler.register("test_method")
    def test_handler(params):
        return {"result": "ok"}

    assert "test_method" in handler.methods
    assert handler.methods["test_method"] == test_handler
