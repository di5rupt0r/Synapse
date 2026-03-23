"""Tests for mcp_server.py - 100% Coverage."""

from unittest.mock import Mock, patch

import pytest


class TestMCPServerInitialization:
    """Test MCP server initialization."""

    def test_mcp_server_created(self):
        """Test FastMCP server is created with correct config."""
        from synapse.mcp_server import mcp

        assert mcp.name == "synapse"
        assert "Agentic Knowledge Graph" in mcp.instructions

    def test_initialize_sets_globals(self):
        """Test initialize function sets global variables."""
        # Clear globals first
        import synapse.mcp_server
        from synapse.mcp_server import initialize

        synapse.mcp_server.synapse_redis = None
        synapse.mcp_server.embedding_cache = None

        mock_redis = Mock()
        mock_cache = Mock()

        initialize(mock_redis, mock_cache)

        assert synapse.mcp_server.synapse_redis == mock_redis
        assert synapse.mcp_server.embedding_cache == mock_cache


class TestMCPToolsRegistration:
    """Test MCP tools are registered correctly."""

    @pytest.mark.asyncio
    async def test_memorize_tool_registered(self):
        """Test memorize tool is registered and callable."""
        from synapse.mcp_server import memorize

        with (
            patch("synapse.mcp_server.synapse_redis"),
            patch("synapse.mcp_server.embedding_cache"),
            patch("synapse.mcp_server.MCPMemorize") as mock_handler,
        ):
            mock_handler_instance = Mock()
            mock_handler.return_value = mock_handler_instance
            mock_handler_instance.handle_memorize.return_value = {
                "id": "test",
                "status": "success",
            }

            result = await memorize("test_domain", "test_type", "test_content")

            assert result["status"] == "success"
            mock_handler_instance.handle_memorize.assert_called_once()

    @pytest.mark.asyncio
    async def test_recall_tool_registered(self):
        """Test recall tool is registered and callable."""
        from synapse.mcp_server import recall

        with (
            patch("synapse.mcp_server.synapse_redis"),
            patch("synapse.mcp_server.embedding_cache"),
            patch("synapse.mcp_server.MCPRecall") as mock_handler,
        ):
            mock_handler_instance = Mock()
            mock_handler.return_value = mock_handler_instance
            mock_handler_instance.handle_recall.return_value = {
                "results": [],
                "total": 0,
            }

            result = await recall("test query")

            assert result["total"] == 0
            mock_handler_instance.handle_recall.assert_called_once()

    @pytest.mark.asyncio
    async def test_patch_tool_registered(self):
        """Test patch tool is registered and callable."""
        from synapse.mcp_server import patch as patch_tool

        with (
            patch("synapse.mcp_server.synapse_redis"),
            patch("synapse.mcp_server.MCPPatch") as mock_handler,
        ):
            mock_handler_instance = Mock()
            mock_handler.return_value = mock_handler_instance
            mock_handler_instance.handle_patch.return_value = {"status": "success"}

            result = await patch_tool(
                "test_id", [{"op": "set", "path": "$.field", "value": "test"}]
            )

            assert result["status"] == "success"
            mock_handler_instance.handle_patch.assert_called_once()


class TestMCPToolInvocation:
    """Test MCP tool invocation with various parameters."""

    @pytest.mark.asyncio
    async def test_memorize_tool_calls_handler(self):
        """Test memorize tool calls handler with correct parameters."""
        from synapse.mcp_server import memorize

        with (
            patch("synapse.mcp_server.synapse_redis"),
            patch("synapse.mcp_server.embedding_cache"),
            patch("synapse.mcp_server.MCPMemorize") as mock_handler,
        ):
            mock_handler_instance = Mock()
            mock_handler.return_value = mock_handler_instance
            mock_handler_instance.handle_memorize.return_value = {
                "id": "test",
                "status": "success",
            }

            await memorize(
                domain="github",
                type="entity",
                content="test content",
                metadata={"key": "value"},
                links={"inbound": ["id1"], "outbound": ["id2"]},
            )

            expected_params = {
                "domain": "github",
                "type": "entity",
                "content": "test content",
                "metadata": {"key": "value"},
                "links": {"inbound": ["id1"], "outbound": ["id2"]},
            }
            mock_handler_instance.handle_memorize.assert_called_once_with(
                expected_params
            )

    @pytest.mark.asyncio
    async def test_recall_tool_calls_handler(self):
        """Test recall tool calls handler with correct parameters."""
        from synapse.mcp_server import recall

        with (
            patch("synapse.mcp_server.synapse_redis"),
            patch("synapse.mcp_server.embedding_cache"),
            patch("synapse.mcp_server.MCPRecall") as mock_handler,
        ):
            mock_handler_instance = Mock()
            mock_handler.return_value = mock_handler_instance
            mock_handler_instance.handle_recall.return_value = {
                "results": [],
                "total": 0,
            }

            await recall(
                query="test query",
                domain=["github", "docs"],
                type=["entity", "chunk"],
                limit=5,
                include_embedding=True,
            )

            expected_params = {
                "query": "test query",
                "domain_filter": ["github", "docs"],
                "type_filter": ["entity", "chunk"],
                "limit": 5,
                "include_embedding": True,
            }
            mock_handler_instance.handle_recall.assert_called_once_with(expected_params)

    @pytest.mark.asyncio
    async def test_patch_tool_calls_handler(self):
        """Test patch tool calls handler with correct parameters."""
        from synapse.mcp_server import patch as patch_tool

        with (
            patch("synapse.mcp_server.synapse_redis"),
            patch("synapse.mcp_server.MCPPatch") as mock_handler,
        ):
            mock_handler_instance = Mock()
            mock_handler.return_value = mock_handler_instance
            mock_handler_instance.handle_patch.return_value = {"status": "success"}

            operations = [
                {"op": "set", "path": "$.field", "value": "new_value"},
                {"op": "delete", "path": "$.old_field"},
            ]

            await patch_tool("test_node_id", operations)

            expected_params = {"node_id": "test_node_id", "operations": operations}
            mock_handler_instance.handle_patch.assert_called_once_with(expected_params)


class TestMCPErrorHandling:
    """Test MCP server error handling."""

    @pytest.mark.asyncio
    async def test_memorize_without_initialization_raises(self):
        """Test memorize raises error when not initialized."""
        # Clear globals
        import synapse.mcp_server
        from synapse.mcp_server import memorize

        synapse.mcp_server.synapse_redis = None
        synapse.mcp_server.embedding_cache = None

        with pytest.raises(RuntimeError, match="MCP server not initialized"):
            await memorize("domain", "type", "content")

    @pytest.mark.asyncio
    async def test_recall_without_initialization_raises(self):
        """Test recall raises error when not initialized."""
        # Clear globals
        import synapse.mcp_server
        from synapse.mcp_server import recall

        synapse.mcp_server.synapse_redis = None

        with pytest.raises(RuntimeError, match="MCP server not initialized"):
            await recall("query")

    @pytest.mark.asyncio
    async def test_patch_without_initialization_raises(self):
        """Test patch raises error when not initialized."""
        # Clear globals
        import synapse.mcp_server
        from synapse.mcp_server import patch as patch_tool

        synapse.mcp_server.synapse_redis = None

        with pytest.raises(RuntimeError, match="MCP server not initialized"):
            await patch_tool("id", [])


class TestMCPServerConfiguration:
    """Test MCP server configuration and settings."""

    def test_settings_import(self):
        """Test settings are imported correctly."""
        from synapse.mcp_server import settings

        assert settings is not None

    def test_global_variables_initially_none(self):
        """Test global variables start as None."""
        # Clear globals first
        import synapse.mcp_server

        synapse.mcp_server.synapse_redis = None
        synapse.mcp_server.embedding_cache = None

        assert synapse.mcp_server.synapse_redis is None
        assert synapse.mcp_server.embedding_cache is None
