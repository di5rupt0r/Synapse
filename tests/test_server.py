"""Tests for FastAPI Server - MCP Standard Refactor."""

from unittest.mock import AsyncMock, Mock, patch

from fastapi.testclient import TestClient


def test_server_health_endpoint():
    """Test health check endpoint."""
    from synapse.server import app

    client = TestClient(app)

    with (
        patch("synapse.server.synapse_redis") as mock_synapse_redis,
        patch("synapse.server.embedding_cache") as mock_cache,
        patch("synapse.server.get_settings") as mock_get_settings,
        patch("synapse.server.mcp_discovery") as mock_mcp_discovery,
    ):
        mock_synapse_redis.ping = AsyncMock(return_value=True)
        mock_cache.embed.return_value = [0.1] * 768
        mock_cache.get_stats.return_value = {"hits": 10, "misses": 2}

        mock_settings = Mock()
        mock_settings.embedding_model = "microsoft/unixcoder-base"
        mock_get_settings.return_value = mock_settings

        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "services" in data
        assert data["services"]["redis"] == "connected"
        assert data["services"]["embedding"] == "available"


def test_server_mcp_discovery_endpoints():
    """Test MCP discovery endpoints."""
    from synapse.server import app

    client = TestClient(app)

    with patch("synapse.server.mcp_discovery") as mock_discovery:
        mock_discovery.list_servers.return_value = [
            {"name": "synapse", "version": "0.1.0"}
        ]

        response = client.get("/mcp/servers")
        assert response.status_code == 200
        data = response.json()
        assert "servers" in data


def test_server_metrics_endpoint():
    """Test metrics endpoint."""
    from synapse.server import app

    client = TestClient(app)

    with (
        patch("synapse.server.synapse_redis") as mock_redis,
        patch("synapse.server.embedding_cache") as mock_cache,
    ):
        mock_client = Mock()
        mock_client.info = AsyncMock(return_value={
            "connected_clients": 1,
            "used_memory_human": "10M",
            "total_commands_processed": 100
        })
        mock_redis._client = mock_client
        mock_cache.get_stats.return_value = {"hits": 5, "misses": 1}

        response = client.get("/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "timestamp" in data
        assert "redis" in data
