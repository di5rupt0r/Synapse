"""Server coverage tests - post MCP standard refactor."""

from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client():
    from synapse.server import app

    return TestClient(app)


class TestServerHealth:
    """Health check endpoint tests."""

    def test_health_success(self, client):
        """Health endpoint returns 200 with correct structure."""
        with (
            patch("synapse.server.synapse_redis") as mock_redis,
            patch("synapse.server.embedding_cache") as mock_cache,
            patch("synapse.server.get_settings") as mock_settings,
        ):
            mock_redis.ping.return_value = True
            mock_cache.embed.return_value = [0.1] * 768
            mock_cache.get_stats.return_value = {"hits": 0, "misses": 0}
            s = Mock()
            s.embedding_model = "microsoft/unixcoder-base"
            mock_settings.return_value = s

            resp = client.get("/health")
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "healthy"
            assert data["services"]["redis"] == "connected"
            assert data["services"]["embedding"] == "available"
            assert data["services"]["mcp"] == "running"

    def test_health_redis_failure(self, client):
        """Health endpoint returns 503 when Redis throws."""
        with (
            patch("synapse.server.synapse_redis") as mock_redis,
            patch("synapse.server.embedding_cache"),
        ):
            mock_redis.ping.side_effect = ConnectionError("Redis down")

            resp = client.get("/health")
            assert resp.status_code == 503
            data = resp.json()
            assert data["status"] == "unhealthy"


class TestServerMetrics:
    """Metrics endpoint tests."""

    def test_metrics_success(self, client):
        """Metrics endpoint returns 200 with expected keys."""
        with (
            patch("synapse.server.synapse_redis") as mock_redis,
            patch("synapse.server.embedding_cache") as mock_cache,
        ):
            mock_client = Mock()
            mock_client.info = AsyncMock(
                return_value={
                    "connected_clients": 2,
                    "used_memory_human": "5M",
                    "total_commands_processed": 50,
                }
            )
            mock_client.ft.return_value.info = AsyncMock(
                return_value={
                    "num_docs": 10,
                    "max_doc_id": 10,
                    "num_terms": 100,
                    "num_records": 100,
                }
            )
            mock_redis._client = mock_client
            mock_cache.get_stats.return_value = {}

            resp = client.get("/metrics")
            assert resp.status_code == 200
            data = resp.json()
            assert "redis" in data
            assert "timestamp" in data


class TestServerMCPMount:
    """Assert the /mcp mount is present (structural, not protocol test)."""

    def test_mcp_mount_in_routes(self):
        """FastMCP is mounted at /mcp in the app route table."""
        from synapse.server import app

        paths = [getattr(r, "path", "") for r in app.routes]
        assert "/mcp" in paths, f"/mcp mount not found. Routes: {paths}"

    def test_no_mcp_discovery_routes(self):
        """Non-standard /mcp/servers route does not exist as a direct route."""
        from synapse.server import app

        # Only top-level routes — sub-app routes don't appear here
        direct_paths = [getattr(r, "path", "") for r in app.routes]
        # /mcp/servers was a direct FastAPI route; it should be gone
        assert "/mcp/servers" not in direct_paths, (
            "Old MCPDiscovery route still registered"
        )
