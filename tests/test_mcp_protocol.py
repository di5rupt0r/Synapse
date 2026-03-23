"""MCP Protocol Tests - FastMCP integration."""

from unittest.mock import MagicMock

import pytest


class TestMCPInitialization:
    """Test MCP server initialization."""

    def test_mcp_server_created(self):
        """GREEN: FastMCP server is created with correct name."""
        from synapse.mcp_server import mcp

        assert mcp is not None
        assert mcp.name == "synapse"

    def test_initialize_sets_globals(self):
        """GREEN: initialize() sets redis and cache globals."""
        import synapse.mcp_server as mcp_mod
        from synapse.embeddings.cache import EmbeddingCache
        from synapse.redis.client import SynapseRedis

        mock_redis = MagicMock(spec=SynapseRedis)
        mock_cache = MagicMock(spec=EmbeddingCache)

        mcp_mod.initialize(mock_redis, mock_cache)

        assert mcp_mod.synapse_redis is mock_redis
        assert mcp_mod.embedding_cache is mock_cache


class TestMCPToolsRegistration:
    """Test tools are registered in FastMCP."""

    def test_memorize_tool_registered(self):
        """GREEN: memorize tool is registered."""
        from synapse.mcp_server import mcp

        tools = mcp._tool_manager.list_tools()
        tool_names = [t.name for t in tools]
        assert "memorize" in tool_names

    def test_recall_tool_registered(self):
        """GREEN: recall tool is registered."""
        from synapse.mcp_server import mcp

        tools = mcp._tool_manager.list_tools()
        tool_names = [t.name for t in tools]
        assert "recall" in tool_names

    def test_patch_tool_registered(self):
        """GREEN: patch tool is registered."""
        from synapse.mcp_server import mcp

        tools = mcp._tool_manager.list_tools()
        tool_names = [t.name for t in tools]
        assert "patch" in tool_names


class TestMCPToolInvocation:
    """Test tool invocation through handlers."""

    @pytest.mark.asyncio
    async def test_memorize_tool_calls_handler(self):
        """GREEN: memorize tool calls MCPMemorize handler."""
        import synapse.mcp_server as mcp_mod

        mock_redis = MagicMock()
        mock_cache = MagicMock()
        mock_cache.embed.return_value = [0.1] * 384
        mock_redis.store_node.return_value = "node:test:123"

        mcp_mod.initialize(mock_redis, mock_cache)

        result = await mcp_mod.memorize(
            domain="test", type="entity", content="test content"
        )

        assert result["status"] == "success"
        assert "id" in result

    @pytest.mark.asyncio
    async def test_recall_tool_calls_handler(self):
        """GREEN: recall tool calls MCPRecall handler."""
        import synapse.mcp_server as mcp_mod

        mock_redis = MagicMock()
        mock_redis.search_hybrid.return_value = []
        mock_cache = MagicMock()
        mock_cache.embed.return_value = [0.1] * 384

        mcp_mod.initialize(mock_redis, mock_cache)

        result = await mcp_mod.recall(query="test query", limit=5)

        assert "results" in result
        assert "total" in result

    @pytest.mark.asyncio
    async def test_patch_tool_calls_handler(self):
        """GREEN: patch tool calls MCPPatch handler."""
        import synapse.mcp_server as mcp_mod

        mock_redis = MagicMock()
        mock_redis.get_node.return_value = {"id": "node:test:123"}
        mock_redis.update_node.return_value = True

        mcp_mod.initialize(mock_redis, None)

        result = await mcp_mod.patch(
            node_id="node:test:123",
            operations=[{"op": "set", "path": "$.metadata.foo", "value": "bar"}],
        )

        assert result["status"] == "success"
        assert result["updated"] is True


class TestMCPErrorHandling:
    """Test error handling."""

    @pytest.mark.asyncio
    async def test_memorize_without_initialization_raises(self):
        """GREEN: memorize raises if not initialized."""
        import synapse.mcp_server as mcp_mod

        mcp_mod.synapse_redis = None
        mcp_mod.embedding_cache = None

        with pytest.raises(RuntimeError, match="MCP server not initialized"):
            await mcp_mod.memorize(domain="test", type="entity", content="test")

    @pytest.mark.asyncio
    async def test_recall_without_initialization_raises(self):
        """GREEN: recall raises if not initialized."""
        import synapse.mcp_server as mcp_mod

        mcp_mod.synapse_redis = None

        with pytest.raises(RuntimeError, match="MCP server not initialized"):
            await mcp_mod.recall(query="test")

    @pytest.mark.asyncio
    async def test_patch_without_initialization_raises(self):
        """GREEN: patch raises if not initialized."""
        import synapse.mcp_server as mcp_mod

        mcp_mod.synapse_redis = None

        with pytest.raises(RuntimeError, match="MCP server not initialized"):
            await mcp_mod.patch(node_id="node:test:123", operations=[])
