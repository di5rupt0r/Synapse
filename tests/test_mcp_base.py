"""Tests for MCP Base - 100% Coverage."""



class TestMCPRequest:
    """Test MCPRequest dataclass."""

    def test_mcp_request_creation(self):
        """Test MCPRequest creation."""
        from synapse.mcp.base import MCPRequest
        request = MCPRequest(
            jsonrpc="2.0",
            id="test-123",
            method="test_method",
            params={"key": "value"}
        )
        assert request.jsonrpc == "2.0"
        assert request.id == "test-123"
        assert request.method == "test_method"
        assert request.params == {"key": "value"}

    def test_mcp_request_defaults(self):
        """Test MCPRequest with default params."""
        from synapse.mcp.base import MCPRequest
        request = MCPRequest(
            jsonrpc="2.0",
            id="test-123",
            method="test_method"
        )
        assert request.params is None


class TestMCPBase:
    """Test MCPBase class."""

    def test_mcp_base_init(self):
        """Test MCPBase initialization."""
        from synapse.mcp.base import MCPBase
        base = MCPBase()
        assert base.methods == {}

    def test_parse_request(self):
        """Test parse_request method."""
        from synapse.mcp.base import MCPBase
        base = MCPBase()
        raw_request = {
            "jsonrpc": "2.0",
            "id": "test-123",
            "method": "test_method",
            "params": {"key": "value"}
        }
        request = base.parse_request(raw_request)
        assert request.jsonrpc == "2.0"
        assert request.id == "test-123"
        assert request.method == "test_method"
        assert request.params == {"key": "value"}

    def test_parse_request_no_params(self):
        """Test parse_request without params."""
        from synapse.mcp.base import MCPBase
        base = MCPBase()
        raw_request = {
            "jsonrpc": "2.0",
            "id": "test-123",
            "method": "test_method"
        }
        request = base.parse_request(raw_request)
        assert request.params is None

    def test_format_response(self):
        """Test format_response method."""
        from synapse.mcp.base import MCPBase
        base = MCPBase()
        response = base.format_response("test-123", {"status": "success"})
        assert response["jsonrpc"] == "2.0"
        assert response["id"] == "test-123"
        assert response["result"] == {"status": "success"}

    def test_format_error(self):
        """Test format_error method."""
        from synapse.mcp.base import MCPBase
        base = MCPBase()
        response = base.format_error("test-123", -32600, "Invalid request")
        assert response["jsonrpc"] == "2.0"
        assert response["id"] == "test-123"
        assert response["error"]["code"] == -32600
        assert response["error"]["message"] == "Invalid request"

    def test_register_decorator(self):
        """Test register decorator."""
        from synapse.mcp.base import MCPBase
        base = MCPBase()

        @base.register("test_method")
        def test_handler(params):
            return {"status": "ok"}

        assert "test_method" in base.methods
        assert base.methods["test_method"] == test_handler

    def test_handle_request_success(self):
        """Test handle_request with registered method."""
        from synapse.mcp.base import MCPBase
        base = MCPBase()

        @base.register("test_method")
        def test_handler(params):
            return {"status": "ok"}

        raw_request = {
            "jsonrpc": "2.0",
            "id": "test-123",
            "method": "test_method",
            "params": {"key": "value"}
        }
        response = base.handle_request(raw_request)
        assert response["jsonrpc"] == "2.0"
        assert response["id"] == "test-123"
        assert response["result"] == {"status": "ok"}

    def test_handle_request_method_not_found(self):
        """Test handle_request with unregistered method."""
        from synapse.mcp.base import MCPBase
        base = MCPBase()

        raw_request = {
            "jsonrpc": "2.0",
            "id": "test-123",
            "method": "unknown_method"
        }
        response = base.handle_request(raw_request)
        assert response["jsonrpc"] == "2.0"
        assert response["id"] == "test-123"
        assert "error" in response
        assert response["error"]["code"] == -32601

    def test_handle_request_exception(self):
        """Test handle_request when handler raises exception."""
        from synapse.mcp.base import MCPBase
        base = MCPBase()

        @base.register("error_method")
        def error_handler(params):
            raise ValueError("Test error")

        raw_request = {
            "jsonrpc": "2.0",
            "id": "test-123",
            "method": "error_method"
        }
        response = base.handle_request(raw_request)
        assert response["jsonrpc"] == "2.0"
        assert response["id"] == "test-123"
        assert "error" in response
        assert response["error"]["code"] == -32603
