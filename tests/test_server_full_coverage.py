"""Tests for server.py - Complete coverage including ML dependencies."""

import asyncio
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi.testclient import TestClient


class TestServerCompleteCoverage:
    """Complete server coverage tests."""

    def test_app_creation_and_configuration(self):
        """Test FastAPI app creation and configuration."""
        from synapse.server import app

        assert app.title == "Synapse AKG"
        assert app.version == "0.1.0"
        assert len(app.routes) > 0

    def test_health_endpoint_success(self):
        """Test health endpoint with all components healthy."""
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
            data = response.json()
            assert data["status"] == "healthy"
            assert "timestamp" in data
            assert data["components"]["redis"] == "healthy"
            assert data["components"]["embedding_cache"] == "healthy"

    def test_health_endpoint_embedding_cache_failure(self):
        """Test health endpoint when embedding cache fails."""
        from synapse.server import app
        client = TestClient(app)

        with (
            patch("synapse.server.synapse_redis") as mock_redis,
            patch("synapse.server.embedding_cache") as mock_cache,
        ):
            mock_redis.ping = AsyncMock(return_value=True)
            mock_cache.embed.side_effect = Exception("Cache error")

            response = client.get("/health")
            assert response.status_code == 503
            data = response.json()
            assert data["status"] == "unhealthy"
            assert "Cache error" in data["error"]

    def test_metrics_endpoint_detailed(self):
        """Test metrics endpoint with detailed information."""
        from synapse.server import app
        client = TestClient(app)

        with (
            patch("synapse.server.synapse_redis") as mock_redis,
            patch("synapse.server.embedding_cache") as mock_cache,
            patch("synapse.server.index_manager") as mock_index,
        ):
            mock_client = Mock()
            mock_client.info = AsyncMock(return_value={
                "connected_clients": 5,
                "used_memory_human": "100M",
                "total_commands_processed": 1000,
                "uptime_in_seconds": 3600
            })
            mock_redis._client = mock_client
            mock_cache.get_stats.return_value = {"hits": 50, "misses": 10}

            mock_ft = Mock()
            mock_ft.info = AsyncMock(return_value={
                "num_docs": 100,
                "max_doc_id": "100",
                "num_terms": 1000,
                "num_records": 5000
            })
            mock_client.ft.return_value = mock_ft
            mock_index._index = "test_index"

            response = client.get("/metrics")
            assert response.status_code == 200
            data = response.json()
            assert "redis" in data
            assert "cache" in data
            assert "index" in data
            assert data["redis"]["connected_clients"] == 5
            assert data["cache"]["hits"] == 50

    def test_metrics_endpoint_no_index(self):
        """Test metrics endpoint when no index is available."""
        from synapse.server import app
        client = TestClient(app)

        with (
            patch("synapse.server.synapse_redis") as mock_redis,
            patch("synapse.server.embedding_cache") as mock_cache,
            patch("synapse.server.index_manager") as mock_index,
        ):
            mock_client = Mock()
            mock_client.info = AsyncMock(return_value={
                "connected_clients": 5,
                "used_memory_human": "100M"
            })
            mock_redis._client = mock_client
            mock_cache.get_stats.return_value = {"hits": 50}
            mock_index._index = None

            response = client.get("/metrics")
            assert response.status_code == 200
            data = response.json()
            assert data["index"]["error"] == "No index configured"

    @pytest.mark.asyncio
    async def test_lifespan_startup_complete(self):
        """Test complete lifespan startup process."""
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
            mock_settings_instance.redis_db = 0
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
                # Verify all startup calls
                mock_redis_from_url.assert_called_once_with(
                    "redis://localhost:6379/0",
                    decode_responses=True
                )
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
            mock_synapse_redis_instance.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_lifespan_startup_redis_connection_failure(self):
        """Test lifespan startup with Redis connection failure."""
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
            mock_settings_instance.redis_db = 0
            mock_settings.return_value = mock_settings_instance

            mock_redis_client = Mock()
            mock_redis_client.ping = AsyncMock(side_effect=Exception("Connection failed"))
            mock_redis_from_url.return_value = mock_redis_client

            # Should raise exception during startup
            with pytest.raises(Exception, match="Connection failed"):
                async with lifespan(app):
                    pass

    @pytest.mark.asyncio
    async def test_lifespan_startup_embedding_backend_failure(self):
        """Test lifespan startup with embedding backend failure."""
        from fastapi import FastAPI

        from synapse.server import lifespan

        app = FastAPI()

        with (
            patch("synapse.server.get_settings") as mock_settings,
            patch("redis.asyncio.from_url") as mock_redis_from_url,
            patch("synapse.server.SynapseRedis") as mock_synapse_redis,
            patch("synapse.server.UniXCoderBackend") as mock_unixcoder,
        ):
            mock_settings_instance = Mock()
            mock_settings_instance.redis_host = "localhost"
            mock_settings_instance.redis_port = 6379
            mock_settings_instance.redis_db = 0
            mock_settings.return_value = mock_settings_instance

            mock_redis_client = Mock()
            mock_redis_client.ping = AsyncMock()
            mock_redis_from_url.return_value = mock_redis_client

            mock_synapse_redis_instance = Mock()
            mock_synapse_redis.return_value = mock_synapse_redis_instance

            mock_unixcoder.side_effect = Exception("Backend failed")

            # Should raise exception during startup
            with pytest.raises(Exception, match="Backend failed"):
                async with lifespan(app):
                    pass

    def test_mcp_discovery_endpoints_complete(self):
        """Test all MCP discovery endpoints."""
        from synapse.server import app
        client = TestClient(app)

        with patch("synapse.server.mcp_discovery") as mock_discovery:
            # List servers
            mock_discovery.list_servers.return_value = [
                {"name": "synapse", "version": "0.1.0", "status": "healthy"}
            ]
            response = client.get("/mcp/servers")
            assert response.status_code == 200
            assert response.json()["count"] == 1

            # Get server info
            mock_discovery.get_server_info.return_value = {
                "server": {"name": "synapse", "version": "0.1.0"},
                "tools": ["memorize", "recall", "patch"]
            }
            response = client.get("/mcp/server/synapse")
            assert response.status_code == 200
            assert "server" in response.json()

            # Get server tools
            mock_discovery.get_server_tools.return_value = [
                {"name": "memorize", "description": "Store information"}
            ]
            response = client.get("/mcp/server/synapse/tools")
            assert response.status_code == 200
            assert response.json()["count"] == 1

            # Register server
            mock_discovery.register_server.return_value = True
            response = client.post("/mcp/server/new-server/register", json={
                "name": "new-server",
                "version": "1.0.0"
            })
            assert response.status_code == 200
            assert response.json()["status"] == "registered"

            # Update health
            mock_discovery.update_health.return_value = True
            response = client.post("/mcp/server/synapse/health", json={
                "status": "healthy"
            })
            assert response.status_code == 200
            assert response.json()["status"] == "updated"

    def test_mcp_discovery_error_handling(self):
        """Test MCP discovery error handling."""
        from synapse.server import app
        client = TestClient(app)

        with patch("synapse.server.mcp_discovery") as mock_discovery:
            # List servers error
            mock_discovery.list_servers.side_effect = Exception("Redis error")
            response = client.get("/mcp/servers")
            assert response.status_code == 500

            # Get server info not found
            mock_discovery.get_server_info.return_value = None
            response = client.get("/mcp/server/nonexistent")
            assert response.status_code == 404

            # Register server failure
            mock_discovery.register_server.return_value = False
            response = client.post("/mcp/server/new-server/register", json={})
            assert response.status_code == 500

    def test_global_exception_handler(self):
        """Test global exception handler."""
        from synapse.server import app
        client = TestClient(app)

        with patch("synapse.server.mcp_discovery") as mock_discovery:
            mock_discovery.list_servers.side_effect = Exception("Unexpected error")

            response = client.get("/mcp/servers")
            assert response.status_code == 500
            assert "error" in response.json()

    def test_create_test_client_function(self):
        """Test create_test_client utility function."""
        from synapse.server import create_test_client

        client = create_test_client()
        assert isinstance(client, TestClient)
        assert client.app is not None

    def test_background_task_function(self):
        """Test background task function."""
        from synapse.server import background_task

        mock_mcp_discovery = Mock()
        mock_mcp_discovery.update_health = AsyncMock()

        # Run the background task
        asyncio.run(background_task(mock_mcp_discovery))
        mock_mcp_discovery.update_health.assert_called_once()

    def test_imports_and_dependencies(self):
        """Test all imports and dependencies are available."""
        from synapse.server import (
            app,
            background_task,
            create_test_client,
            embedding_cache,
            index_manager,
            lifespan,
            mcp_discovery,
            synapse_redis,
        )

        # Verify all components are importable
        assert app is not None
        assert lifespan is not None
        assert create_test_client is not None
        assert background_task is not None

        # Global variables should be None initially
        assert synapse_redis is None
        assert embedding_cache is None
        assert index_manager is None
        assert mcp_discovery is None
