"""Tests for remaining modules - Complete coverage."""

from unittest.mock import Mock, patch

import pytest


class TestIndexSetupComplete:
    """Complete index setup coverage."""

    def test_create_index_success(self):
        """Test successful index creation."""
        with patch("synapse.index.setup.redis.asyncio.from_url") as mock_redis:
            mock_client = Mock()
            mock_redis.return_value = mock_client

            mock_ft = Mock()
            mock_client.ft.return_value = mock_ft

            from synapse.index.setup import create_index

            result = create_index("redis://localhost:6379", "test_index")

            assert result is True
            mock_ft.create_index.assert_called_once()

    def test_create_index_already_exists(self):
        """Test index creation when index already exists."""
        with patch("synapse.index.setup.redis.asyncio.from_url") as mock_redis:
            mock_client = Mock()
            mock_redis.return_value = mock_client

            mock_ft = Mock()
            mock_ft.create_index.side_effect = Exception("Index already exists")
            mock_client.ft.return_value = mock_ft

            from synapse.index.setup import create_index

            result = create_index("redis://localhost:6379", "test_index")

            assert result is True  # Should handle gracefully

    def test_create_index_connection_failure(self):
        """Test index creation with connection failure."""
        with patch("synapse.index.setup.redis.asyncio.from_url") as mock_redis:
            mock_redis.side_effect = Exception("Connection failed")

            from synapse.index.setup import create_index

            with pytest.raises(Exception, match="Connection failed"):
                create_index("redis://localhost:6379", "test_index")

    def test_ensure_index_exists_success(self):
        """Test ensure_index_exists when index exists."""
        with patch("synapse.index.setup.redis.asyncio.from_url") as mock_redis:
            mock_client = Mock()
            mock_redis.return_value = mock_client

            mock_ft = Mock()
            mock_ft.info.return_value = {"index_name": "test_index"}
            mock_client.ft.return_value = mock_ft

            from synapse.index.setup import ensure_index_exists

            result = ensure_index_exists("redis://localhost:6379", "test_index")

            assert result is True

    def test_ensure_index_exists_not_found(self):
        """Test ensure_index_exists when index doesn't exist."""
        with patch("synapse.index.setup.redis.asyncio.from_url") as mock_redis:
            mock_client = Mock()
            mock_redis.return_value = mock_client

            mock_ft = Mock()
            mock_ft.info.side_effect = Exception("Index not found")
            mock_client.ft.return_value = mock_ft

            from synapse.index.setup import ensure_index_exists

            result = ensure_index_exists("redis://localhost:6379", "test_index")

            assert result is False

    def test_delete_index_success(self):
        """Test successful index deletion."""
        with patch("synapse.index.setup.redis.asyncio.from_url") as mock_redis:
            mock_client = Mock()
            mock_redis.return_value = mock_client

            mock_ft = Mock()
            mock_client.ft.return_value = mock_ft

            from synapse.index.setup import delete_index

            result = delete_index("redis://localhost:6379", "test_index")

            assert result is True
            mock_ft.dropindex.assert_called_once()

    def test_delete_index_not_exists(self):
        """Test index deletion when index doesn't exist."""
        with patch("synapse.index.setup.redis.asyncio.from_url") as mock_redis:
            mock_client = Mock()
            mock_redis.return_value = mock_client

            mock_ft = Mock()
            mock_ft.dropindex.side_effect = Exception("Index not found")
            mock_client.ft.return_value = mock_ft

            from synapse.index.setup import delete_index

            result = delete_index("redis://localhost:6379", "test_index")

            assert result is True  # Should handle gracefully


class TestMCPBaseComplete:
    """Complete MCP base coverage."""

    def test_mcp_base_initialization(self):
        """Test MCP base initialization."""
        from synapse.mcp.base import MCPBase

        mcp = MCPBase("test-server", "Test description")

        assert mcp.name == "test-server"
        assert mcp.description == "Test description"

    def test_mcp_base_register_tool(self):
        """Test MCP base tool registration."""
        from synapse.mcp.base import MCPBase

        mcp = MCPBase("test-server", "Test description")

        @mcp.tool
        def test_tool(param: str) -> str:
            return f"Hello {param}"

        assert "test_tool" in mcp.tools
        assert mcp.tools["test_tool"]["function"] == test_tool

    def test_mcp_base_register_resource(self):
        """Test MCP base resource registration."""
        from synapse.mcp.base import MCPBase

        mcp = MCPBase("test-server", "Test description")

        @mcp.resource("test://data")
        def test_resource():
            return {"data": "test"}

        assert "test://data" in mcp.resources

    def test_mcp_base_get_tool_schema(self):
        """Test MCP base tool schema generation."""
        from synapse.mcp.base import MCPBase

        mcp = MCPBase("test-server", "Test description")

        @mcp.tool
        def test_tool(param: str, optional_param: int = 42) -> str:
            return f"Hello {param}"

        schema = mcp.get_tool_schema("test_tool")

        assert schema["name"] == "test_tool"
        assert "parameters" in schema
        assert "param" in schema["parameters"]["properties"]
        assert "optional_param" in schema["parameters"]["properties"]

    def test_mcp_base_execute_tool(self):
        """Test MCP base tool execution."""
        from synapse.mcp.base import MCPBase

        mcp = MCPBase("test-server", "Test description")

        @mcp.tool
        def test_tool(param: str) -> str:
            return f"Hello {param}"

        result = mcp.execute_tool("test_tool", {"param": "World"})

        assert result == "Hello World"

    def test_mcp_base_execute_tool_not_found(self):
        """Test MCP base tool execution with non-existent tool."""
        from synapse.mcp.base import MCPBase

        mcp = MCPBase("test-server", "Test description")

        with pytest.raises(ValueError, match="Tool not_found not found"):
            mcp.execute_tool("not_found", {})

    def test_mcp_base_execute_tool_invalid_params(self):
        """Test MCP base tool execution with invalid parameters."""
        from synapse.mcp.base import MCPBase

        mcp = MCPBase("test-server", "Test description")

        @mcp.tool
        def test_tool(param: str) -> str:
            return f"Hello {param}"

        with pytest.raises(TypeError):
            mcp.execute_tool("test_tool", {"wrong_param": "value"})


class TestGraphCompressorComplete:
    """Complete graph compressor coverage."""

    def test_compress_graph_empty(self):
        """Test compressing empty graph."""
        from synapse.graph.compressor import compress_graph

        result = compress_graph([])

        assert result == []

    def test_compress_graph_single_node(self):
        """Test compressing single node graph."""
        from synapse.graph.compressor import compress_graph

        nodes = [{"id": "1", "type": "entity", "content": "test"}]

        result = compress_graph(nodes)

        assert len(result) == 1
        assert result[0]["id"] == "1"

    def test_compress_graph_with_relationships(self):
        """Test compressing graph with relationships."""
        from synapse.graph.compressor import compress_graph

        nodes = [
            {
                "id": "1",
                "type": "entity",
                "content": "test1",
                "links": {"outbound": ["2"]},
            },
            {
                "id": "2",
                "type": "entity",
                "content": "test2",
                "links": {"inbound": ["1"]},
            },
        ]

        result = compress_graph(nodes)

        assert len(result) == 2
        assert "compressed" in result[0] or "compressed" in result[1]

    def test_compress_graph_with_duplicates(self):
        """Test compressing graph with duplicate nodes."""
        from synapse.graph.compressor import compress_graph

        nodes = [
            {"id": "1", "type": "entity", "content": "duplicate"},
            {"id": "2", "type": "entity", "content": "duplicate"},
        ]

        result = compress_graph(nodes)

        # Should merge duplicates
        assert len(result) <= 2

    def test_decompress_graph_empty(self):
        """Test decompressing empty graph."""
        from synapse.graph.compressor import decompress_graph

        result = decompress_graph([])

        assert result == []

    def test_decompress_graph_compressed(self):
        """Test decompressing compressed graph."""
        from synapse.graph.compressor import decompress_graph

        compressed = [
            {
                "id": "1",
                "type": "entity",
                "content": "test",
                "compressed": True,
                "original_ids": ["1", "2"],
            }
        ]

        result = decompress_graph(compressed)

        assert len(result) >= 1
        assert result[0]["id"] == "1"

    def test_get_compression_stats(self):
        """Test getting compression statistics."""
        from synapse.graph.compressor import get_compression_stats

        original = [{"id": "1"}, {"id": "2"}, {"id": "3"}]
        compressed = [{"id": "1", "compressed": True, "original_ids": ["1", "2", "3"]}]

        stats = get_compression_stats(original, compressed)

        assert "original_size" in stats
        assert "compressed_size" in stats
        assert "compression_ratio" in stats
        assert stats["original_size"] == 3
        assert stats["compressed_size"] == 1


class TestGraphResolverComplete:
    """Complete graph resolver coverage."""

    def test_resolve_node_exists(self):
        """Test resolving existing node."""
        from synapse.graph.resolver import resolve_node

        nodes = {"1": {"id": "1", "type": "entity", "content": "test"}}

        result = resolve_node("1", nodes)

        assert result["id"] == "1"
        assert result["content"] == "test"

    def test_resolve_node_not_exists(self):
        """Test resolving non-existent node."""
        from synapse.graph.resolver import resolve_node

        nodes = {"1": {"id": "1", "type": "entity", "content": "test"}}

        result = resolve_node("2", nodes)

        assert result is None

    def test_resolve_path_simple(self):
        """Test resolving simple path."""
        from synapse.graph.resolver import resolve_path

        nodes = {
            "1": {"id": "1", "type": "entity", "links": {"outbound": ["2"]}},
            "2": {"id": "2", "type": "entity", "links": {"outbound": ["3"]}},
            "3": {"id": "3", "type": "entity"},
        }

        result = resolve_path("1", "3", nodes)

        assert len(result) == 3
        assert result[0]["id"] == "1"
        assert result[2]["id"] == "3"

    def test_resolve_path_no_path(self):
        """Test resolving path when no path exists."""
        from synapse.graph.resolver import resolve_path

        nodes = {"1": {"id": "1", "type": "entity"}, "2": {"id": "2", "type": "entity"}}

        result = resolve_path("1", "2", nodes)

        assert result == []

    def test_resolve_path_circular(self):
        """Test resolving path with circular references."""
        from synapse.graph.resolver import resolve_path

        nodes = {
            "1": {"id": "1", "type": "entity", "links": {"outbound": ["2"]}},
            "2": {"id": "2", "type": "entity", "links": {"outbound": ["1"]}},
        }

        result = resolve_path("1", "2", nodes)

        assert len(result) == 2
        assert result[0]["id"] == "1"
        assert result[1]["id"] == "2"

    def test_get_connected_components(self):
        """Test getting connected components."""
        from synapse.graph.resolver import get_connected_components

        nodes = {
            "1": {"id": "1", "type": "entity", "links": {"outbound": ["2"]}},
            "2": {"id": "2", "type": "entity"},
            "3": {"id": "3", "type": "entity"},
            "4": {"id": "4", "type": "entity", "links": {"outbound": ["3"]}},
        }

        result = get_connected_components(nodes)

        assert len(result) == 2  # Two separate components
        assert all(isinstance(comp, list) for comp in result)

    def test_find_shortest_path(self):
        """Test finding shortest path."""
        from synapse.graph.resolver import find_shortest_path

        nodes = {
            "1": {"id": "1", "type": "entity", "links": {"outbound": ["2", "3"]}},
            "2": {"id": "2", "type": "entity", "links": {"outbound": ["4"]}},
            "3": {"id": "3", "type": "entity", "links": {"outbound": ["4"]}},
            "4": {"id": "4", "type": "entity"},
        }

        result = find_shortest_path("1", "4", nodes)

        assert len(result) == 3  # 1 -> 2 -> 4 or 1 -> 3 -> 4
        assert result[0]["id"] == "1"
        assert result[-1]["id"] == "4"


class TestChunkingFallbackComplete:
    """Complete chunking fallback coverage."""

    def test_chunk_by_lines_empty(self):
        """Test chunking empty content by lines."""
        from synapse.chunking.fallback import chunk_by_lines

        result = chunk_by_lines("")

        assert result == []

    def test_chunk_by_lines_single_line(self):
        """Test chunking single line."""
        from synapse.chunking.fallback import chunk_by_lines

        result = chunk_by_lines("single line")

        assert len(result) == 1
        assert result[0]["content"] == "single line"

    def test_chunk_by_lines_multiple_lines(self):
        """Test chunking multiple lines."""
        from synapse.chunking.fallback import chunk_by_lines

        content = "line1\nline2\nline3\nline4\nline5"

        result = chunk_by_lines(content, chunk_size=2)

        assert len(result) == 3
        assert result[0]["content"] == "line1\nline2"
        assert result[1]["content"] == "line3\nline4"
        assert result[2]["content"] == "line5"

    def test_chunk_by_lines_with_overlap(self):
        """Test chunking with overlap."""
        from synapse.chunking.fallback import chunk_by_lines

        content = "line1\nline2\nline3\nline4"

        result = chunk_by_lines(content, chunk_size=2, overlap=1)

        assert len(result) == 3
        assert result[0]["content"] == "line1\nline2"
        assert result[1]["content"] == "line2\nline3"
        assert result[2]["content"] == "line3\nline4"

    def test_chunk_by_lines_preserve_metadata(self):
        """Test chunking preserves metadata."""
        from synapse.chunking.fallback import chunk_by_lines

        content = "line1\nline2\nline3"
        metadata = {"source": "test.py", "language": "python"}

        result = chunk_by_lines(content, chunk_size=2, metadata=metadata)

        assert len(result) == 2
        assert result[0]["metadata"]["source"] == "test.py"
        assert result[0]["metadata"]["language"] == "python"

    def test_chunk_by_lines_line_numbers(self):
        """Test chunking includes line numbers."""
        from synapse.chunking.fallback import chunk_by_lines

        content = "line1\nline2\nline3"

        result = chunk_by_lines(content, chunk_size=2)

        assert result[0]["line_start"] == 1
        assert result[0]["line_end"] == 2
        assert result[1]["line_start"] == 3
        assert result[1]["line_end"] == 3
