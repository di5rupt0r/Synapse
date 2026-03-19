"""Tests for MCP Discovery - 100% Coverage."""

import time
from unittest.mock import Mock


class TestMCPDiscoveryInit:
    """Test MCPDiscovery initialization."""

    def test_init_sets_redis_client(self):
        """Test __init__ sets redis client."""
        from synapse.mcp_discovery import MCPDiscovery
        mock_redis = Mock()
        discovery = MCPDiscovery(mock_redis)
        assert discovery.redis == mock_redis


class TestMCPDiscoveryRegisterServer:
    """Test register_server method."""

    def test_register_server_success(self):
        """Test successful server registration."""
        from synapse.mcp_discovery import MCPDiscovery
        mock_redis = Mock()
        mock_redis.json_set.return_value = True

        discovery = MCPDiscovery(mock_redis)
        result = discovery.register_server("test-server", {
            "name": "test-server",
            "version": "1.0.0",
            "capabilities": ["memorize"]
        })

        assert result is True
        mock_redis.json_set.assert_called_once()

    def test_register_server_failure(self):
        """Test server registration failure."""
        from synapse.mcp_discovery import MCPDiscovery
        mock_redis = Mock()
        mock_redis.json_set.side_effect = Exception("Redis error")

        discovery = MCPDiscovery(mock_redis)
        result = discovery.register_server("test-server", {})

        assert result is False

    def test_register_server_sets_timestamp_and_status(self):
        """Test register_server adds timestamp and status."""
        from synapse.mcp_discovery import MCPDiscovery
        mock_redis = Mock()
        mock_redis.json_set.return_value = True

        discovery = MCPDiscovery(mock_redis)
        discovery.register_server("test-server", {"name": "test"})

        call_args = mock_redis.json_set.call_args
        server_data = call_args[0][2]
        assert "registered_at" in server_data
        assert server_data["status"] == "active"


class TestMCPDiscoveryListServers:
    """Test list_servers method."""

    def test_list_servers_success(self):
        """Test successful server listing."""
        from synapse.mcp_discovery import MCPDiscovery
        mock_redis = Mock()
        mock_redis.keys.return_value = ["mcp:server:server1", "mcp:server:server2"]
        mock_redis.json_get.side_effect = [
            {"name": "server1"},
            {"name": "server2"}
        ]

        discovery = MCPDiscovery(mock_redis)
        result = discovery.list_servers()

        assert len(result) == 2
        assert result[0]["name"] == "server1"

    def test_list_servers_empty(self):
        """Test listing with no servers."""
        from synapse.mcp_discovery import MCPDiscovery
        mock_redis = Mock()
        mock_redis.keys.return_value = []

        discovery = MCPDiscovery(mock_redis)
        result = discovery.list_servers()

        assert result == []

    def test_list_servers_error(self):
        """Test list_servers with Redis error."""
        from synapse.mcp_discovery import MCPDiscovery
        mock_redis = Mock()
        mock_redis.keys.side_effect = Exception("Redis error")

        discovery = MCPDiscovery(mock_redis)
        result = discovery.list_servers()

        assert result == []


class TestMCPDiscoveryGetServerInfo:
    """Test get_server_info method."""

    def test_get_server_info_success(self):
        """Test successful server info retrieval."""
        from synapse.mcp_discovery import MCPDiscovery
        mock_redis = Mock()
        mock_redis.json_get.side_effect = [
            {"name": "test-server", "capabilities": ["memorize"]},
            {"status": "healthy"}
        ]

        discovery = MCPDiscovery(mock_redis)
        result = discovery.get_server_info("test-server")

        assert "server" in result
        assert "health" in result
        assert result["server"]["name"] == "test-server"

    def test_get_server_info_not_found(self):
        """Test get_server_info for non-existent server."""
        from synapse.mcp_discovery import MCPDiscovery
        mock_redis = Mock()
        mock_redis.json_get.return_value = None

        discovery = MCPDiscovery(mock_redis)
        result = discovery.get_server_info("nonexistent")

        assert result is None

    def test_get_server_info_no_health_data(self):
        """Test get_server_info without health data."""
        from synapse.mcp_discovery import MCPDiscovery
        mock_redis = Mock()
        mock_redis.json_get.side_effect = [
            {"name": "test-server"},
            None  # No health data
        ]

        discovery = MCPDiscovery(mock_redis)
        result = discovery.get_server_info("test-server")

        assert "server" in result
        assert "health" not in result

    def test_get_server_info_error(self):
        """Test get_server_info with Redis error."""
        from synapse.mcp_discovery import MCPDiscovery
        mock_redis = Mock()
        mock_redis.json_get.side_effect = Exception("Redis error")

        discovery = MCPDiscovery(mock_redis)
        result = discovery.get_server_info("test-server")

        assert result is None


class TestMCPDiscoveryGetServerTools:
    """Test get_server_tools method."""

    def test_get_server_tools_success(self):
        """Test successful tools retrieval."""
        from synapse.mcp_discovery import MCPDiscovery
        mock_redis = Mock()
        mock_redis.json_get.return_value = {
            "name": "synapse",
            "capabilities": ["memorize", "recall", "patch", "hybrid-search"]
        }

        discovery = MCPDiscovery(mock_redis)
        result = discovery.get_server_tools("synapse")

        assert len(result) == 4
        tool_names = [t["name"] for t in result]
        assert "memorize" in tool_names
        assert "recall" in tool_names
        assert "patch" in tool_names

    def test_get_server_tools_server_not_found(self):
        """Test get_server_tools for non-existent server."""
        from synapse.mcp_discovery import MCPDiscovery
        mock_redis = Mock()
        mock_redis.json_get.return_value = None

        discovery = MCPDiscovery(mock_redis)
        result = discovery.get_server_tools("nonexistent")

        assert result == []

    def test_get_server_tools_unknown_capability(self):
        """Test get_server_tools filters unknown capabilities."""
        from synapse.mcp_discovery import MCPDiscovery
        mock_redis = Mock()
        mock_redis.json_get.return_value = {
            "capabilities": ["memorize", "unknown-capability"]
        }

        discovery = MCPDiscovery(mock_redis)
        result = discovery.get_server_tools("test")

        assert len(result) == 1
        assert result[0]["name"] == "memorize"

    def test_get_server_tools_error(self):
        """Test get_server_tools with Redis error."""
        from synapse.mcp_discovery import MCPDiscovery
        mock_redis = Mock()
        mock_redis.json_get.side_effect = Exception("Redis error")

        discovery = MCPDiscovery(mock_redis)
        result = discovery.get_server_tools("test-server")

        assert result == []


class TestMCPDiscoveryUpdateHealth:
    """Test update_health method."""

    def test_update_health_success(self):
        """Test successful health update."""
        from synapse.mcp_discovery import MCPDiscovery
        mock_redis = Mock()
        mock_redis.json_set.return_value = True

        discovery = MCPDiscovery(mock_redis)
        result = discovery.update_health("test-server", {
            "status": "healthy",
            "timestamp": time.time()
        })

        assert result is True
        call_args = mock_redis.json_set.call_args
        health_data = call_args[0][2]
        assert "updated_at" in health_data

    def test_update_health_failure(self):
        """Test health update failure."""
        from synapse.mcp_discovery import MCPDiscovery
        mock_redis = Mock()
        mock_redis.json_set.side_effect = Exception("Redis error")

        discovery = MCPDiscovery(mock_redis)
        result = discovery.update_health("test-server", {})

        assert result is False
