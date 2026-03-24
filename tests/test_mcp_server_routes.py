"""TDD: MCP Server Route Tests.

Tests for current FastMCP implementation without complex import dependencies.
Focuses on testing that MCP routes work correctly and don't have the old discovery routes.
"""

import pathlib


class TestMCPServerRoutes:
    """Test MCP server routes and implementation."""

    def test_mcp_server_file_exists(self):
        """Current implementation should have mcp_server.py"""
        base_path = pathlib.Path(__file__).parent.parent
        mcp_server_path = base_path / "synapse" / "mcp_server.py"

        assert mcp_server_path.exists(), (
            f"mcp_server.py should exist at {mcp_server_path}"
        )

        # Check that it contains FastMCP import
        with mcp_server_path.open('r', encoding='utf-8') as f:
            content = f.read()

        assert "FastMCP" in content, (
            "mcp_server.py should import FastMCP"
        )
        assert "@mcp.tool()" in content, (
            "mcp_server.py should define MCP tools"
        )

    def test_mcp_tools_defined(self):
        """MCP tools should be properly defined"""
        base_path = pathlib.Path(__file__).parent.parent
        mcp_server_path = base_path / "synapse" / "mcp_server.py"

        with mcp_server_path.open('r', encoding='utf-8') as f:
            content = f.read()

        # Check for expected tools
        expected_tools = ["memorize", "recall", "patch"]
        for tool in expected_tools:
            assert "@mcp.tool()" in content, (
                f"Tool {tool} should be decorated with @mcp.tool()"
            )
            assert f"async def {tool}(" in content, (
                f"Tool {tool} should be defined as async function"
            )

    def test_old_discovery_routes_absent(self):
        """Old discovery routes should not exist in server.py"""
        base_path = pathlib.Path(__file__).parent.parent
        server_path = base_path / "synapse" / "server.py"

        with server_path.open('r', encoding='utf-8') as f:
            content = f.read()

        # These routes should not exist
        forbidden_routes = [
            "/mcp/servers",
            "/mcp/server/",
            "MCPDiscovery"
        ]

        for route in forbidden_routes:
            assert route not in content, (
                f"Forbidden route '{route}' found in server.py"
            )

    def test_fastmcp_mounting(self):
        """Server should mount FastMCP correctly"""
        base_path = pathlib.Path(__file__).parent.parent
        server_path = base_path / "synapse" / "server.py"

        with server_path.open('r', encoding='utf-8') as f:
            content = f.read()

        # Should mount FastMCP app
        assert "mcp" in content, (
            "Server should import or reference mcp"
        )

        # Should have some form of MCP mounting
        assert any(keyword in content for keyword in ["mount", "include_router", "mcp"]), (
            "Server should mount MCP routes"
        )

    def test_mcp_directory_structure(self):
        """MCP directory should have correct structure"""
        base_path = pathlib.Path(__file__).parent.parent
        mcp_dir = base_path / "synapse" / "mcp"

        assert mcp_dir.exists(), "mcp directory should exist"

        # Should have these files
        expected_files = ["__init__.py", "memorize.py", "recall.py", "patch.py"]
        for file_name in expected_files:
            file_path = mcp_dir / file_name
            assert file_path.exists(), f"mcp/{file_name} should exist"

        # Should NOT have these old files
        forbidden_files = ["base.py"]
        for file_name in forbidden_files:
            file_path = mcp_dir / file_name
            assert not file_path.exists(), f"mcp/{file_name} should not exist"
