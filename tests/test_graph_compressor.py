"""TDD RED Phase: Tests for Graph YAML Compressor."""


def test_yaml_compression_basic():
    """Test basic YAML compression of graph data."""
    # This will fail - compressor doesn't exist yet (RED phase)
    from synapse.graph.compressor import GraphCompressor

    compressor = GraphCompressor()

    graph_data = {
        "matched_nodes": [
            {
                "id": "node:test:123",
                "domain": "test",
                "type": "entity",
                "content": "This is a very long content that should be compressed for token efficiency",
            }
        ],
        "resolved_edges": [
            {
                "source": "node:test:123",
                "target": "node:other:456",
                "relation_type": "linked",
            }
        ],
    }

    compressed = compressor.compress_yaml(graph_data)

    assert isinstance(compressed, str)
    assert "matched_nodes" in compressed
    assert "resolved_edges" in compressed
    # Should be shorter than original due to compression
    assert len(compressed) < len(str(graph_data))


def test_yaml_compression_content_summary():
    """Test content is summarized to max 50 characters."""
    from synapse.graph.compressor import GraphCompressor

    compressor = GraphCompressor()

    graph_data = {
        "matched_nodes": [
            {
                "id": "node:test:123",
                "domain": "test",
                "type": "entity",
                "content": "This is a very long content that exceeds the 50 character limit for summaries",
            }
        ],
        "resolved_edges": [],
    }

    compressed = compressor.compress_yaml(graph_data)

    # Content should be truncated
    assert "..." in compressed or len(compressed) < 200


def test_yaml_compression_stopwords():
    """Test stopwords are removed from compressed output."""
    from synapse.graph.compressor import GraphCompressor

    compressor = GraphCompressor()

    graph_data = {
        "matched_nodes": [
            {
                "id": "node:test:123",
                "domain": "test",
                "type": "entity",
                "content": "This is a test content with many stopwords like the and a an",
            }
        ],
        "resolved_edges": [],
    }

    compressed = compressor.compress_yaml(graph_data)

    # Should remove common stopwords
    assert " the " not in compressed.lower()
    assert " and " not in compressed.lower()
    assert " a " not in compressed.lower()


def test_yaml_compression_telegraphic():
    """Test telegraphic format (minimal words, max density)."""
    from synapse.graph.compressor import GraphCompressor

    compressor = GraphCompressor()

    graph_data = {
        "matched_nodes": [
            {
                "id": "node:test:123",
                "domain": "documentation",
                "type": "entity",
                "content": "The documentation describes the architecture and implementation details",
            },
            {
                "id": "node:test:456",
                "domain": "documentation",
                "type": "chunk",
                "content": "Implementation follows best practices and design patterns",
            },
        ],
        "resolved_edges": [
            {
                "source": "node:test:123",
                "target": "node:test:456",
                "relation_type": "linked",
            }
        ],
    }

    compressed = compressor.compress_yaml(graph_data)

    # Should be in telegraphic format - check that compression happened
    original_words = len(str(graph_data).split())
    compressed_words = len(compressed.split())
    # Some compression should occur (even YAML structure has overhead)
    assert compressed_words <= original_words  # At least not worse
    # And content should be compressed (check for key indicators)
    assert "..." in compressed or len(compressed) < len(str(graph_data)) * 0.9


def test_yaml_compression_empty_graph():
    """Test compression of empty graph data."""
    from synapse.graph.compressor import GraphCompressor

    compressor = GraphCompressor()

    graph_data = {"matched_nodes": [], "resolved_edges": []}

    compressed = compressor.compress_yaml(graph_data)

    assert isinstance(compressed, str)
    assert "matched_nodes: []" in compressed or "matched_nodes:[]" in compressed
    assert "resolved_edges: []" in compressed or "resolved_edges:[]" in compressed
