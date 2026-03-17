"""TDD RED Phase: Tests for Graph Resolution Engine."""

from unittest.mock import Mock


def test_graph_resolver_1_degree_traversal():
    """Test 1-degree graph traversal from a node."""
    # This will fail - graph resolver doesn't exist yet (RED phase)
    from synapse.graph.resolver import GraphResolver

    mock_redis = Mock()

    # Mock node with links
    mock_redis.get_node.return_value = {
        "id": "node:test:123",
        "links": {
            "inbound": ["node:other:456", "node:other:789"],
            "outbound": ["node:target:111"],
        },
    }

    # Mock linked nodes
    mock_redis.get_linked_nodes.return_value = [
        {"id": "node:other:456", "content": "related content 1"},
        {"id": "node:other:789", "content": "related content 2"},
        {"id": "node:target:111", "content": "target content"},
    ]

    resolver = GraphResolver(mock_redis)

    result = resolver.resolve_1_degree("node:test:123")

    assert len(result["nodes"]) == 4  # Original + 3 linked
    assert len(result["edges"]) == 3  # 3 connections
    assert any(node["id"] == "node:test:123" for node in result["nodes"])
    assert any(edge["source"] == "node:test:123" for edge in result["edges"])


def test_graph_resolver_no_links():
    """Test graph resolution for node with no links."""
    from synapse.graph.resolver import GraphResolver

    mock_redis = Mock()

    # Mock node with no links
    mock_redis.get_node.return_value = {
        "id": "node:test:123",
        "links": {"inbound": [], "outbound": []},
    }

    mock_redis.get_linked_nodes.return_value = []

    resolver = GraphResolver(mock_redis)

    result = resolver.resolve_1_degree("node:test:123")

    assert len(result["nodes"]) == 1  # Only original node
    assert len(result["edges"]) == 0  # No connections


def test_graph_resolver_node_not_found():
    """Test graph resolution for non-existent node."""
    from synapse.graph.resolver import GraphResolver

    mock_redis = Mock()
    mock_redis.get_node.return_value = None

    resolver = GraphResolver(mock_redis)

    result = resolver.resolve_1_degree("node:nonexistent:123")

    assert len(result["nodes"]) == 0
    assert len(result["edges"]) == 0


def test_graph_resolver_duplicate_links():
    """Test graph resolution with duplicate links."""
    from synapse.graph.resolver import GraphResolver

    mock_redis = Mock()

    # Mock node with duplicate links
    mock_redis.get_node.return_value = {
        "id": "node:test:123",
        "links": {
            "inbound": ["node:other:456", "node:other:456"],  # Duplicate
            "outbound": ["node:target:111"],
        },
    }

    # Mock linked nodes (should only be called once for duplicate)
    mock_redis.get_linked_nodes.return_value = [
        {"id": "node:other:456", "content": "related content"},
        {"id": "node:target:111", "content": "target content"},
    ]

    resolver = GraphResolver(mock_redis)

    result = resolver.resolve_1_degree("node:test:123")

    assert len(result["nodes"]) == 3  # Original + 2 unique linked
    assert len(result["edges"]) == 2  # 2 unique connections


def test_graph_resolver_edge_directions():
    """Test graph resolution preserves edge directions."""
    from synapse.graph.resolver import GraphResolver

    mock_redis = Mock()

    mock_redis.get_node.return_value = {
        "id": "node:test:123",
        "links": {"inbound": ["node:source:456"], "outbound": ["node:target:789"]},
    }

    mock_redis.get_linked_nodes.return_value = [
        {"id": "node:source:456", "content": "source content"},
        {"id": "node:target:789", "content": "target content"},
    ]

    resolver = GraphResolver(mock_redis)

    result = resolver.resolve_1_degree("node:test:123")

    # Check edge directions
    inbound_edges = [e for e in result["edges"] if e["target"] == "node:test:123"]
    outbound_edges = [e for e in result["edges"] if e["source"] == "node:test:123"]

    assert len(inbound_edges) == 1
    assert len(outbound_edges) == 1
    assert inbound_edges[0]["source"] == "node:source:456"
    assert outbound_edges[0]["target"] == "node:target:789"
