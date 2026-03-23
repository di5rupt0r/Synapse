"""TDD: MCPDiscovery Removal Tests.

RED phase: fail before refactor (routes return 200, MCPDiscovery import exists).
GREEN phase: pass after refactor (routes gone, imports cleaned up).
"""
# torch is available in the venv — no patching needed for imports.

import ast
import pathlib
from unittest.mock import MagicMock

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
        """RED→GREEN: server.py must not import MCPDiscovery."""
        server_src = pathlib.Path(
            "/home/gabrielsb/Synapse/synapse/server.py"
        ).read_text()
        tree = ast.parse(server_src)

        discovery_imports = [
            ast.unparse(node)
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            and (
                "MCPDiscovery" in ast.unparse(node)
                or "mcp_discovery" in ast.unparse(node)
            )
        ]

        assert not discovery_imports, (
            f"server.py still imports MCPDiscovery: {discovery_imports}"
        )

    def test_mcp_base_not_in_mcp_init(self):
        """RED→GREEN: mcp/__init__.py must not import MCPBase or base module."""
        init_src = pathlib.Path(
            "/home/gabrielsb/Synapse/synapse/mcp/__init__.py"
        ).read_text()
        assert "MCPBase" not in init_src, "mcp/__init__.py still imports MCPBase"
        assert "from .base" not in init_src, (
            "mcp/__init__.py still references base module"
        )

    def test_mcp_discovery_file_deleted(self):
        """RED→GREEN: synapse/mcp_discovery.py must not exist."""
        assert not pathlib.Path(
            "/home/gabrielsb/Synapse/synapse/mcp_discovery.py"
        ).exists(), "mcp_discovery.py still exists — not deleted"

    def test_mcp_base_file_deleted(self):
        """RED→GREEN: synapse/mcp/base.py must not exist."""
        assert not pathlib.Path(
            "/home/gabrielsb/Synapse/synapse/mcp/base.py"
        ).exists(), "mcp/base.py still exists — not deleted"
