"""MCP Base Handler for JSON-RPC 2.0."""

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional


@dataclass
class MCPRequest:
    """JSON-RPC 2.0 request structure."""

    jsonrpc: str
    id: str
    method: str
    params: Optional[Dict[str, Any]] = None


class MCPBase:
    """Base MCP handler with JSON-RPC 2.0 support."""

    def __init__(self) -> None:
        """Initialize MCP handler."""
        self.methods: Dict[str, Callable] = {}

    def parse_request(self, raw_request: Dict[str, Any]) -> MCPRequest:
        """Parse JSON-RPC 2.0 request."""
        return MCPRequest(
            jsonrpc=raw_request["jsonrpc"],
            id=raw_request["id"],
            method=raw_request["method"],
            params=raw_request.get("params"),
        )

    def format_response(self, id: str, result: Dict[str, Any]) -> Dict[str, Any]:
        """Format JSON-RPC 2.0 success response."""
        return {"jsonrpc": "2.0", "id": id, "result": result}

    def format_error(self, id: str, code: int, message: str) -> Dict[str, Any]:
        """Format JSON-RPC 2.0 error response."""
        return {"jsonrpc": "2.0", "id": id, "error": {"code": code, "message": message}}

    def register(self, method_name: str) -> Callable:
        """Decorator to register MCP methods."""

        def decorator(func: Callable) -> Callable:
            self.methods[method_name] = func
            return func

        return decorator

    def handle_request(self, raw_request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle incoming MCP request."""
        try:
            request = self.parse_request(raw_request)

            if request.method not in self.methods:
                return self.format_error(
                    request.id,
                    -32601,
                    "Method not found",  # Method not found
                )

            result = self.methods[request.method](request.params or {})
            return self.format_response(request.id, result)

        except Exception as e:
            return self.format_error(
                raw_request.get("id", ""),
                -32603,
                str(e),  # Internal error
            )
