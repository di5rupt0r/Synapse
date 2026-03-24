"""TDD: MCPDiscovery Removal Tests.

RED phase: fail before refactor (routes return 200, MCPDiscovery import exists).
GREEN phase: pass after refactor (routes gone, imports cleaned up).
"""
# torch is mocked for testing to avoid import dependency

import pathlib
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_redis():
    r = MagicMock()
    r.ping.return_value = True
    r.store_node.return_value = "node:test:abc123"
    r.search_hybrid.return_value = []
    r.get_linked_nodes.return_value = []
    r.json_set.return_value = True
    r.json_get.return_value = None
    r.keys.return_value = []
    return r


@pytest.fixture()
def mock_cache():
    c = MagicMock()
    c.embed.return_value = [0.1] * 768
    c.get_stats.return_value = {}
    c.model = True
    c.dim = 768
    return c


@pytest.fixture()
def client(mock_redis, mock_cache):
    # Mock all ML dependencies to avoid import issues
    torch_mock = MagicMock()
    torch_mock.__version__ = "2.0.0"
    torch_mock.tensor = lambda x: x

    transformers_mock = MagicMock()
    transformers_mock.AutoModel = MagicMock()
    transformers_mock.AutoTokenizer = MagicMock()

    numpy_mock = MagicMock()
    numpy_mock.array = lambda x: x

    with patch.dict(
        "sys.modules",
        {"torch": torch_mock, "transformers": transformers_mock, "numpy": numpy_mock},
    ):
        import synapse.mcp_server as mcp_mod
        import synapse.server as server_mod

        mcp_mod.initialize(mock_redis, mock_cache)
        server_mod.synapse_redis = mock_redis
        server_mod.embedding_cache = mock_cache

        return TestClient(server_mod.app)


# ---------------------------------------------------------------------------
# 1. Non-standard discovery routes must be gone
# ---------------------------------------------------------------------------


class TestNonStandardDiscoveryRoutesRemoved:
    """Non-standard /mcp/server/* routes must NOT return 200 after refactor."""

    def test_mcp_servers_list_route_is_gone(self, client):
        """RED→GREEN: GET /mcp/servers must not return 200."""
        resp = client.get("/mcp/servers")
        assert resp.status_code != 200, (
            "GET /mcp/servers still returns 200 — MCPDiscovery route not removed"
        )

    def test_mcp_server_detail_route_is_gone(self, client):
        """RED→GREEN: GET /mcp/server/synapse must not return 200."""
        resp = client.get("/mcp/server/synapse")
        assert resp.status_code != 200, (
            "GET /mcp/server/synapse still returns 200 — MCPDiscovery route not removed"
        )

    def test_mcp_server_tools_route_is_gone(self, client):
        """RED→GREEN: GET /mcp/server/synapse/tools must not return 200."""
        resp = client.get("/mcp/server/synapse/tools")
        assert resp.status_code != 200, (
            "GET /mcp/server/synapse/tools still returns 200 — route not removed"
        )

    def test_mcp_server_register_route_is_gone(self, client):
        """RED→GREEN: POST /mcp/server/synapse/register must not return 200."""
        resp = client.post("/mcp/server/synapse/register", json={"name": "test"})
        assert resp.status_code != 200, (
            "POST /mcp/server/synapse/register still returns 200 — route not removed"
        )

    def test_mcp_server_health_update_route_is_gone(self, client):
        """RED→GREEN: POST /mcp/server/synapse/health must not return 200."""
        resp = client.post("/mcp/server/synapse/health", json={"status": "ok"})
        assert resp.status_code != 200, (
            "POST /mcp/server/synapse/health still returns 200 — route not removed"
        )


# ---------------------------------------------------------------------------
# 2. Static source-code checks (no import of dead modules)
# ---------------------------------------------------------------------------


class TestMCPDiscoveryModuleRemoved:
    """Dead module references must not exist in source after cleanup."""

    def test_mcp_discovery_class_not_in_server(self):
        """MCPDiscovery class should not exist in server.py"""
        # Test that import fails - catch ImportError specifically
        try:
            from synapse.server import MCPDiscovery  # This should fail
            assert False, "MCPDiscovery should not be importable from synapse.server"
        except ImportError as exc:
            # Check that error is about MCPDiscovery, not torch
            error_msg = str(exc)
            # The error might mention torch due to import chain, but should be about MCPDiscovery
            assert "MCPDiscovery" in error_msg or "cannot import name" in error_msg, (
                f"Expected MCPDiscovery import error, got: {error_msg}"
            )

        # Check current server.py imports (should not contain MCPDiscovery)
        server_path = pathlib.Path(__file__).parent.parent / "synapse" / "server.py"
        with server_path.open('r', encoding='utf-8') as f:
            server_content = f.read()

        assert "MCPDiscovery" not in server_content, (
            f"server.py still imports MCPDiscovery: {server_content}"
        )

    def test_mcp_base_not_in_mcp_init(self):
        """RED→GREEN: mcp/__init__.py must not import MCPBase or base module."""
        # Check current mcp/__init__.py imports
        mcp_init_path = (
            pathlib.Path(__file__).parent.parent / "synapse" / "mcp" / "__init__.py"
        )

        with mcp_init_path.open("r", encoding="utf-8") as f:
            mcp_init_content = f.read()

        # Should not contain MCPBase or base module imports
        assert "MCPBase" not in mcp_init_content, (
            f"mcp/__init__.py still imports MCPBase: {mcp_init_content}"
        )
        assert "base" not in mcp_init_content.lower(), (
            f"mcp/__init__.py still imports base module: {mcp_init_content}"
        )
        assert "MCPBase" not in mcp_init_content, (
            f"mcp/__init__.py still imports MCPBase: {mcp_init_content}"
        )
        assert "base" not in mcp_init_content.lower(), (
            f"mcp/__init__.py still imports base module: {mcp_init_content}"
        )

    def test_mcp_discovery_file_deleted(self):
        """RED→GREEN: synapse/mcp_discovery.py must not exist."""
        base_path = pathlib.Path(__file__).parent.parent
        assert not (base_path / "synapse" / "mcp_discovery.py").exists(), (
            "mcp_discovery.py still exists — not deleted"
        )

    def test_mcp_base_file_deleted(self):
        """RED→GREEN: synapse/mcp/base.py must not exist."""
        base_path = pathlib.Path(__file__).parent.parent
        assert not (base_path / "synapse" / "mcp" / "base.py").exists(), (
            "mcp/base.py still exists — not deleted"
        )
