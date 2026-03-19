"""MCP Discovery Service for Server Registration and Tool Listing."""

import time
from typing import Any, Dict, List, Optional

from .redis.client import SynapseRedis


class MCPDiscovery:
    """MCP Server Discovery and Registration Service."""

    SERVER_PREFIX = "mcp:server:"
    HEALTH_PREFIX = "mcp:health:"

    def __init__(self, redis_client: SynapseRedis) -> None:
        """Initialize MCP Discovery with Redis client."""
        self.redis = redis_client

    def register_server(self, name: str, info: Dict[str, Any]) -> bool:
        """Register MCP server with its capabilities and endpoints."""
        try:
            server_key = f"{self.SERVER_PREFIX}{name}"
            server_data = {
                **info,
                "registered_at": time.time(),
                "status": "active"
            }
            return self.redis.json_set(server_key, ".", server_data)
        except Exception:
            return False

    def list_servers(self) -> List[Dict[str, Any]]:
        """List all registered MCP servers."""
        try:
            pattern = f"{self.SERVER_PREFIX}*"
            keys = self.redis.keys(pattern)
            servers = []

            for key in keys:
                server_data = self.redis.json_get(key)
                if server_data:
                    servers.append(server_data)

            return servers
        except Exception:
            return []

    def get_server_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific server."""
        try:
            server_key = f"{self.SERVER_PREFIX}{name}"
            server_data = self.redis.json_get(server_key)

            if not server_data:
                return None

            # Include health data if available
            health_key = f"{self.HEALTH_PREFIX}{name}"
            health_data = self.redis.json_get(health_key)

            result = {"server": server_data}
            if health_data:
                result["health"] = health_data

            return result
        except Exception:
            return None

    def get_server_tools(self, name: str) -> List[Dict[str, Any]]:
        """Get available tools for a specific server."""
        try:
            server_key = f"{self.SERVER_PREFIX}{name}"
            server_data = self.redis.json_get(server_key)

            if not server_data:
                return []

            # Extract tools from server capabilities
            capabilities = server_data.get("capabilities", [])
            tools = []

            # Map capabilities to tool definitions
            tool_mapping = {
                "memorize": {
                    "name": "memorize",
                    "description": "Store knowledge with metadata tracking",
                    "endpoint": "/mcp/memorize"
                },
                "recall": {
                    "name": "recall",
                    "description": "Hybrid search with latency tracking",
                    "endpoint": "/mcp/recall"
                },
                "patch": {
                    "name": "patch",
                    "description": "Atomic mutations with request tracking",
                    "endpoint": "/mcp/patch"
                },
                "hybrid-search": {
                    "name": "hybrid_search",
                    "description": "Combined semantic and sparse search",
                    "endpoint": "/mcp/recall"
                }
            }

            for capability in capabilities:
                if capability in tool_mapping:
                    tools.append(tool_mapping[capability])

            return tools
        except Exception:
            return []

    def update_health(self, name: str, health_data: Dict[str, Any]) -> bool:
        """Update health status for a specific server."""
        try:
            health_key = f"{self.HEALTH_PREFIX}{name}"
            health_with_timestamp = {
                **health_data,
                "updated_at": time.time()
            }
            return self.redis.json_set(health_key, ".", health_with_timestamp)
        except Exception:
            return False
