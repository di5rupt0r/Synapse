"""Tests for remaining modules - Complete coverage."""

from unittest.mock import Mock, patch
import pytest

class TestIndexSetupComplete:
    """Complete index setup coverage."""

    def test_ensure_index_creates_new(self):
        from synapse.index.setup import IndexManager
        mock_redis = Mock()
        mock_ft = Mock()
        mock_redis.ft.return_value = mock_ft
        mock_ft.info.side_effect = Exception("Unknown Index name")
        
        manager = IndexManager(mock_redis)
        manager.ensure_index()
        
        mock_ft.create_index.assert_called_once()
        mock_ft.dropindex.assert_not_called()

    def test_ensure_index_recreates_existing(self):
        from synapse.index.setup import IndexManager
        mock_redis = Mock()
        mock_ft = Mock()
        mock_redis.ft.return_value = mock_ft
        mock_ft.info.return_value = {"index_name": "synapse_idx"}
        
        manager = IndexManager(mock_redis)
        manager.ensure_index()
        
        mock_ft.dropindex.assert_called_once()
        mock_ft.create_index.assert_called_once()

class TestGraphCompressorComplete:
    """Complete graph compressor coverage."""

    def test_compress_yaml(self):
        from synapse.graph.compressor import GraphCompressor
        graph_data = {
            "matched_nodes": [
                {
                    "id": "1",
                    "domain": "test",
                    "type": "entity",
                    "content": "This is a very long text that should be compressed down completely by the compressor.",
                }
            ],
            "resolved_edges": [
                {
                    "source": "1",
                    "target": "2",
                    "relation_type": "linked"
                }
            ]
        }
        compressor = GraphCompressor(max_content_length=15)
        result = compressor.compress_yaml(graph_data)
        
        assert "long text" in result
        assert "compressed down completely" not in result
        assert "linked" in result

class TestGraphResolverComplete:
    """Complete graph resolver coverage."""

    def test_resolve_1_degree_success(self):
        from synapse.graph.resolver import GraphResolver
        mock_redis = Mock()
        mock_redis.get_node.return_value = {
            "id": "1",
            "type": "entity",
            "content": "root",
            "links": {"outbound": ["2"], "inbound": ["3"]}
        }
        mock_redis.get_linked_nodes.return_value = [
            {"id": "2", "content": "child"},
            {"id": "3", "content": "parent"}
        ]
        
        resolver = GraphResolver(mock_redis)
        result = resolver.resolve_1_degree("1")
        
        assert "nodes" in result
        assert len(result["nodes"]) == 3
        # root, plus 2 linked
        ids = {node["id"] for node in result["nodes"]}
        assert ids == {"1", "2", "3"}
        assert len(result["edges"]) == 2

    def test_resolve_1_degree_node_not_found(self):
        from synapse.graph.resolver import GraphResolver
        mock_redis = Mock()
        mock_redis.get_node.return_value = None
        
        resolver = GraphResolver(mock_redis)
        result = resolver.resolve_1_degree("1")
        assert result == {"nodes": [], "edges": []}

class TestChunkingFallbackComplete:
    """Complete chunking fallback coverage."""

    def test_fallback_chunk_by_lines(self):
        from synapse.chunking.fallback import fallback_chunk_by_lines
        content = "line1\\nline2\\nline3\\nline4"
        result = fallback_chunk_by_lines(content, chunk_size=2, overlap=1)
        
        assert len(result) > 0
        assert "text" in result[0]
        assert "line_start" in result[0]
        assert "line_end" in result[0]

    def test_fallback_chunk_by_lines_empty(self):
        from synapse.chunking.fallback import fallback_chunk_by_lines
        assert fallback_chunk_by_lines("", chunk_size=2) == []
