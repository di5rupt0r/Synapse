"""Tests for Server Redis Refactor - MCP Standard."""

from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient


def test_server_health_check():
    """Test health check endpoint with mocked Redis."""
    from synapse.server import app

    client = TestClient(app)

    with (
        patch("synapse.server.synapse_redis") as mock_redis,
        patch("synapse.server.embedding_cache") as mock_cache,
    ):
        mock_redis.ping = AsyncMock(return_value=True)
        mock_cache.embed.return_value = [0.1] * 384
        mock_cache.get_stats.return_value = {"hits": 5, "misses": 1}

        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
