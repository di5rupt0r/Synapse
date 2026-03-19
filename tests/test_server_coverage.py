"""Tests for server.py - 100% Coverage."""

from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi.testclient import TestClient


class TestServerHealth:
    """Test health endpoint."""

    def test_health_success(self):
        """Test health check success."""
        from synapse.server import app
        client = TestClient(app)

        with (
            patch("synapse.server.synapse_redis") as mock_redis,
            patch("synapse.server.embedding_cache") as mock_cache,
            patch("synapse.server.mcp_discovery"),
            patch("synapse.server.get_settings") as mock_settings,
        ):
            mock_redis.ping = AsyncMock(return_value=True)
            mock_cache.embed.return_value = [0.1] * 768
            mock_cache.get_stats.return_value = {"hits": 10}
            mock_settings.return_value.embedding_model = "test-model"

            response = client.get("/health")
            assert response.status_code == 200
            assert response.json()["status"] == "healthy"

    def test_health_redis_failure(self):
        """Test health check with Redis failure."""
        from synapse.server import app
        client = TestClient(app)

        with patch("synapse.server.synapse_redis") as mock_redis:
            mock_redis.ping = AsyncMock(side_effect=Exception("Redis down"))

            response = client.get("/health")
            assert response.status_code == 503
            assert response.json()["status"] == "unhealthy"

    def test_health_with_mcp_discovery(self):
        """Test health check updates MCP discovery."""
        from synapse.server import app
        client = TestClient(app)

        with (
            patch("synapse.server.synapse_redis") as mock_redis,
            patch("synapse.server.embedding_cache") as mock_cache,
            patch("synapse.server.mcp_discovery") as mock_discovery,
            patch("synapse.server.get_settings") as mock_settings,
        ):
            mock_redis.ping = AsyncMock(return_value=True)
            mock_cache.embed.return_value = [0.1] * 768
            mock_cache.get_stats.return_value = {"hits": 10}
            mock_settings.return_value.embedding_model = "test-model"

            response = client.get("/health")
            assert response.status_code == 200
            mock_discovery.update_health.assert_called_once()


class TestServerMetrics:
    """Test metrics endpoint."""

    def test_metrics_success(self):
        """Test metrics endpoint success."""
        from synapse.server import app
        client = TestClient(app)

        with (
            patch("synapse.server.synapse_redis") as mock_redis,
            patch("synapse.server.embedding_cache") as mock_cache,
        ):
            mock_client = Mock()
            mock_client.info = AsyncMock(return_value={
                "connected_clients": 5,
                "used_memory_human": "100M",
                "total_commands_processed": 1000
            })
            mock_redis._client = mock_client
            mock_cache.get_stats.return_value = {"hits": 50}

            response = client.get("/metrics")
            assert response.status_code == 200
            data = response.json()
            assert "redis" in data
            assert "cache" in data

    def test_metrics_index_error(self):
        """Test metrics when index info fails."""
        from synapse.server import app
        client = TestClient(app)

        with (
            patch("synapse.server.synapse_redis") as mock_redis,
            patch("synapse.server.embedding_cache") as mock_cache,
        ):
            mock_client = Mock()
            mock_client.info = AsyncMock(return_value={
                "connected_clients": 5,
                "used_memory_human": "100M",
                "total_commands_processed": 1000
            })
            mock_client.ft = Mock(side_effect=Exception("Index error"))
            mock_redis._client = mock_client
            mock_cache.get_stats.return_value = {"hits": 50}

            response = client.get("/metrics")
            assert response.status_code == 200
            assert response.json()["index"]["error"] == "Index not available"

    def test_metrics_redis_error(self):
        """Test metrics when Redis fails."""
        from synapse.server import app
        client = TestClient(app)

        with patch("synapse.server.synapse_redis") as mock_redis:
            mock_redis._client = Mock()
            mock_redis._client.info = AsyncMock(side_effect=Exception("Redis error"))

            response = client.get("/metrics")
            assert response.status_code == 500


class TestMCPDiscoveryEndpoints:
    """Test MCP discovery endpoints."""

    def test_list_servers_success(self):
        """Test list servers success."""
        from synapse.server import app
        client = TestClient(app)

        with patch("synapse.server.mcp_discovery") as mock_discovery:
            mock_discovery.list_servers.return_value = [
                {"name": "synapse", "version": "0.1.0"}
            ]

            response = client.get("/mcp/servers")
            assert response.status_code == 200
            assert response.json()["count"] == 1

    def test_list_servers_error(self):
        """Test list servers error."""
        from synapse.server import app
        client = TestClient(app)

        with patch("synapse.server.mcp_discovery") as mock_discovery:
            mock_discovery.list_servers.side_effect = Exception("Redis error")

            response = client.get("/mcp/servers")
            assert response.status_code == 500

    def test_get_server_info_success(self):
        """Test get server info success."""
        from synapse.server import app
        client = TestClient(app)

        with patch("synapse.server.mcp_discovery") as mock_discovery:
            mock_discovery.get_server_info.return_value = {
                "server": {"name": "synapse"}
            }

            response = client.get("/mcp/server/synapse")
            assert response.status_code == 200

    def test_get_server_info_not_found(self):
        """Test get server info not found."""
        from synapse.server import app
        client = TestClient(app)

        with patch("synapse.server.mcp_discovery") as mock_discovery:
            mock_discovery.get_server_info.return_value = None

            response = client.get("/mcp/server/nonexistent")
            assert response.status_code == 404

    def test_get_server_info_error(self):
        """Test get server info error."""
        from synapse.server import app
        client = TestClient(app)

        with patch("synapse.server.mcp_discovery") as mock_discovery:
            mock_discovery.get_server_info.side_effect = Exception("Redis error")

            response = client.get("/mcp/server/synapse")
            assert response.status_code == 500

    def test_get_server_tools_success(self):
        """Test get server tools success."""
        from synapse.server import app
        client = TestClient(app)

        with patch("synapse.server.mcp_discovery") as mock_discovery:
            mock_discovery.get_server_tools.return_value = [
                {"name": "memorize"}
            ]

            response = client.get("/mcp/server/synapse/tools")
            assert response.status_code == 200
            assert response.json()["count"] == 1

    def test_get_server_tools_error(self):
        """Test get server tools error."""
        from synapse.server import app
        client = TestClient(app)

        with patch("synapse.server.mcp_discovery") as mock_discovery:
            mock_discovery.get_server_tools.side_effect = Exception("Redis error")

            response = client.get("/mcp/server/synapse/tools")
            assert response.status_code == 500

    def test_register_server_success(self):
        """Test register server success."""
        from synapse.server import app
        client = TestClient(app)

        with patch("synapse.server.mcp_discovery") as mock_discovery:
            mock_discovery.register_server.return_value = True

            response = client.post("/mcp/server/new-server/register", json={
                "name": "new-server"
            })
            assert response.status_code == 200
            assert response.json()["status"] == "registered"

    def test_register_server_failure(self):
        """Test register server failure."""
        from synapse.server import app
        client = TestClient(app)

        with patch("synapse.server.mcp_discovery") as mock_discovery:
            mock_discovery.register_server.return_value = False

            response = client.post("/mcp/server/new-server/register", json={})
            assert response.status_code == 500

    def test_register_server_error(self):
        """Test register server error."""
        from synapse.server import app
        client = TestClient(app)

        with patch("synapse.server.mcp_discovery") as mock_discovery:
            mock_discovery.register_server.side_effect = Exception("Redis error")

            response = client.post("/mcp/server/new-server/register", json={})
            assert response.status_code == 500

    def test_update_health_success(self):
        """Test update health success."""
        from synapse.server import app
        client = TestClient(app)

        with patch("synapse.server.mcp_discovery") as mock_discovery:
            mock_discovery.update_health.return_value = True

            response = client.post("/mcp/server/synapse/health", json={
                "status": "healthy"
            })
            assert response.status_code == 200
            assert response.json()["status"] == "updated"

    def test_update_health_failure(self):
        """Test update health failure."""
        from synapse.server import app
        client = TestClient(app)

        with patch("synapse.server.mcp_discovery") as mock_discovery:
            mock_discovery.update_health.return_value = False

            response = client.post("/mcp/server/synapse/health", json={})
            assert response.status_code == 500

    def test_update_health_error(self):
        """Test update health error."""
        from synapse.server import app
        client = TestClient(app)

        with patch("synapse.server.mcp_discovery") as mock_discovery:
            mock_discovery.update_health.side_effect = Exception("Redis error")

            response = client.post("/mcp/server/synapse/health", json={})
            assert response.status_code == 500


class TestGlobalExceptionHandler:
    """Test global exception handler."""

    def test_exception_handler(self):
        """Test global exception handler returns 500."""
        from synapse.server import app
        client = TestClient(app)

        with patch("synapse.server.mcp_discovery") as mock_discovery:
            mock_discovery.list_servers.side_effect = Exception("Unexpected error")

            response = client.get("/mcp/servers")
            assert response.status_code == 500
            assert "error" in response.json()


class TestCreateTestClient:
    """Test create_test_client function."""

    def test_create_test_client(self):
        """Test create_test_client returns TestClient."""
        from synapse.server import create_test_client
        client = create_test_client()
        assert isinstance(client, TestClient)


class TestServerMain:
    """Test main block execution."""

    def test_uvicorn_import(self):
        """Test that uvicorn can be imported."""
        import importlib.util
        spec = importlib.util.find_spec("uvicorn")
        assert spec is not None, "uvicorn not installed"


class TestLifespanContextManager:
    """Test lifespan context manager."""

    @pytest.mark.asyncio
    async def test_lifespan_startup_shutdown(self):
        """Test lifespan startup and shutdown."""
        from fastapi import FastAPI

        from synapse.server import lifespan

        app = FastAPI()

        with (
            patch("synapse.server.get_settings") as mock_settings,
            patch("redis.asyncio.from_url") as mock_redis_from_url,
            patch("synapse.server.SynapseRedis") as mock_synapse_redis,
            patch("synapse.server.UniXCoderBackend") as mock_unixcoder,
            patch("synapse.server.EmbeddingCache") as mock_cache,
            patch("synapse.server.IndexManager") as mock_index_manager,
            patch("synapse.server.init_mcp") as mock_init_mcp,
            patch("synapse.server.MCPDiscovery") as mock_mcp_discovery,
            patch("asyncio.create_task") as mock_create_task,
        ):
            # Setup mocks
            mock_settings_instance = Mock()
            mock_settings_instance.redis_host = "localhost"
            mock_settings_instance.redis_port = 6379
            mock_settings_instance.cache_size = 1000
            mock_settings.return_value = mock_settings_instance

            mock_redis_client = Mock()
            mock_redis_client.ping = AsyncMock()
            mock_redis_from_url.return_value = mock_redis_client

            mock_synapse_redis_instance = Mock()
            mock_synapse_redis_instance.close = AsyncMock()
            mock_synapse_redis.return_value = mock_synapse_redis_instance

            mock_unixcoder_instance = Mock()
            mock_unixcoder.return_value = mock_unixcoder_instance

            mock_cache_instance = Mock()
            mock_cache.return_value = mock_cache_instance

            mock_index_manager_instance = Mock()
            mock_index_manager_instance.ensure_index = Mock()
            mock_index_manager.return_value = mock_index_manager_instance

            mock_mcp_discovery_instance = Mock()
            mock_mcp_discovery_instance.register_server = Mock()
            mock_mcp_discovery.return_value = mock_mcp_discovery_instance

            mock_task = Mock()
            mock_task.cancel = Mock()
            mock_create_task.return_value = mock_task

            # Test lifespan
            async with lifespan(app):
                # Verify startup calls
                mock_redis_from_url.assert_called_once()
                mock_redis_client.ping.assert_called_once()
                mock_synapse_redis.assert_called_once()
                mock_unixcoder.assert_called_once()
                mock_cache.assert_called_once()
                mock_index_manager_instance.ensure_index.assert_called_once()
                mock_init_mcp.assert_called_once()
                mock_mcp_discovery.assert_called_once()
                mock_mcp_discovery_instance.register_server.assert_called_once()
                mock_create_task.assert_called_once()

            # Verify shutdown calls
            mock_task.cancel.assert_called_once()

    @pytest.mark.asyncio
    async def test_lifespan_exception_handling(self):
        """Test lifespan handles exceptions gracefully."""
        from fastapi import FastAPI

        from synapse.server import lifespan

        app = FastAPI()

        with (
            patch("synapse.server.get_settings") as mock_settings,
            patch("redis.asyncio.from_url") as mock_redis_from_url,
        ):
            mock_settings_instance = Mock()
            mock_settings_instance.redis_host = "localhost"
            mock_settings_instance.redis_port = 6379
            mock_settings.return_value = mock_settings_instance

            mock_redis_client = Mock()
            mock_redis_client.ping = AsyncMock(side_effect=Exception("Redis connection failed"))
            mock_redis_from_url.return_value = mock_redis_client

            # Should raise exception during startup
            with pytest.raises(Exception, match="Redis connection failed"):
                async with lifespan(app):
                    pass
