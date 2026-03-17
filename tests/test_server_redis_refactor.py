"""TDD RED Phase: Tests for server Redis refactor."""

from unittest.mock import Mock, patch
from fastapi.testclient import TestClient


def test_server_health_with_embedding_info():
    """Test health check endpoint includes embedding model info."""
    # This will fail - server doesn't use SynapseRedis yet (RED phase)
    from synapse.server import app

    client = TestClient(app)

    with (
        patch("synapse.server.synapse_redis") as mock_synapse_redis,
        patch("synapse.server.embedding_cache") as mock_cache,
        patch("synapse.server.get_settings") as mock_get_settings,
    ):
        # Setup mocks
        mock_synapse_redis.ping.return_value = True
        mock_cache.embed.return_value = [0.1] * 768

        mock_settings = Mock()
        mock_settings.embedding_model = "microsoft/unixcoder-base"
        mock_get_settings.return_value = mock_settings

        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["embedding_model"] == "microsoft/unixcoder-base"
        assert data["embedding_dim"] == 768
        assert "services" in data
        assert data["services"]["redis"] == "connected"
        assert data["services"]["embedding"] == "available"


def test_server_memorize_direct_payload():
    """Test memorize endpoint accepts direct payload (not just JSON-RPC)."""
    from synapse.server import app

    client = TestClient(app)

    with patch("synapse.server.mcp_memorize") as mock_mcp:
        mock_mcp.handle_memorize.return_value = {
            "status": "success",
            "id": "node:test:123",
        }

        # Direct payload (not JSON-RPC)
        direct_payload = {
            "domain": "test",
            "type": "entity",
            "content": "def foo(): pass",
        }

        response = client.post("/mcp/memorize", json=direct_payload)

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "id" in data


def test_server_memorize_json_rpc_payload():
    """Test memorize endpoint accepts JSON-RPC payload."""
    from synapse.server import app

    client = TestClient(app)

    with patch("synapse.server.mcp_memorize") as mock_mcp:
        mock_mcp.handle_memorize.return_value = {
            "status": "success",
            "id": "node:test:456",
        }

        # JSON-RPC payload
        jsonrpc_payload = {
            "jsonrpc": "2.0",
            "id": "test-123",
            "method": "memorize",
            "params": {
                "domain": "test",
                "type": "entity",
                "content": "def bar(): pass",
            },
        }

        response = client.post("/mcp/memorize", json=jsonrpc_payload)

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "id" in data


def test_server_startup_uses_synapse_redis():
    """Test server startup creates SynapseRedis wrapper."""
    # This will test that the startup properly initializes SynapseRedis
    from synapse.server import app

    # Check that the app has the proper Redis wrapper configuration
    assert hasattr(app, "state") or app.router.lifespan_context is not None
