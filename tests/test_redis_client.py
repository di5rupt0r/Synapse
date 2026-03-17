"""TDD RED Phase: Tests for SynapseRedis wrapper."""

from unittest.mock import Mock
from synapse.redis.client import SynapseRedis


def test_redis_store_node():
    """Test storing a node in Redis."""
    # This will fail - SynapseRedis doesn't exist yet (RED phase)

    mock_redis = Mock()
    synapse_redis = SynapseRedis(mock_redis)

    node_id = "node:test:123"
    domain = "test"
    node_type = "entity"
    content = "test content"
    embedding = [0.1] * 768
    metadata = {"key": "value"}
    links = {"inbound": [], "outbound": []}

    result = synapse_redis.store_node(
        node_id=node_id,
        domain=domain,
        node_type=node_type,
        content=content,
        embedding=embedding,
        metadata=metadata,
        links=links,
    )

    assert result == node_id
    mock_redis.json().set.assert_called_once()

    # Verify the stored node structure
    call_args = mock_redis.json().set.call_args
    stored_node = call_args[0][
        2
    ]  # Third argument is the node data (path, "$", node_data)

    assert stored_node["id"] == node_id
    assert stored_node["domain"] == domain
    assert stored_node["type"] == node_type
    assert stored_node["content"] == content
    assert stored_node["embedding"] == embedding
    assert stored_node["metadata"] == metadata
    assert stored_node["links"] == links
    assert "created_at" in stored_node


def test_redis_get_node():
    """Test retrieving a node by ID."""

    mock_redis = Mock()
    synapse_redis = SynapseRedis(mock_redis)

    node_id = "node:test:123"
    expected_node = {
        "id": node_id,
        "domain": "test",
        "type": "entity",
        "content": "test content",
    }

    mock_redis.json().get.return_value = expected_node

    result = synapse_redis.get_node(node_id)

    assert result == expected_node
    mock_redis.json().get.assert_called_once_with(node_id)


def test_redis_get_node_not_found():
    """Test retrieving a non-existent node."""

    mock_redis = Mock()
    synapse_redis = SynapseRedis(mock_redis)

    mock_redis.json().get.return_value = None

    result = synapse_redis.get_node("node:nonexistent")

    assert result is None


def test_redis_update_node():
    """Test updating a node with patch operations."""

    mock_redis = Mock()
    synapse_redis = SynapseRedis(mock_redis)

    node_id = "node:test:123"
    existing_node = {
        "id": node_id,
        "domain": "test",
        "type": "entity",
        "content": "original content",
        "metadata": {"key": "old_value"},
    }

    operations = [
        {"op": "set", "path": "$.content", "value": "updated content"},
        {"op": "set", "path": "$.metadata.key", "value": "new_value"},
    ]

    mock_redis.json().get.return_value = existing_node

    result = synapse_redis.update_node(node_id, operations)

    assert result is True
    assert mock_redis.json().set.call_count == 2


def test_redis_update_node_not_found():
    """Test updating a non-existent node."""

    mock_redis = Mock()
    synapse_redis = SynapseRedis(mock_redis)

    mock_redis.json().get.return_value = None

    result = synapse_redis.update_node("node:nonexistent", [])

    assert result is False


def test_redis_search_hybrid_with_embedding():
    """Test hybrid search with embedding (KNN + BM25)."""

    mock_redis = Mock()
    mock_ft = Mock()
    mock_redis.ft.return_value = mock_ft

    synapse_redis = SynapseRedis(mock_redis)

    query = "test query"
    embedding = [0.1] * 768
    domain_filter = ["test"]
    type_filter = ["entity"]
    limit = 10

    # Mock search results
    mock_doc = Mock()
    mock_doc.__dict__ = {
        "id": "node:test:123",
        "domain": "test",
        "type": "entity",
        "content": "test content",
        "score": 0.95,
    }

    mock_results = Mock()
    mock_results.docs = [mock_doc]
    mock_ft.search.return_value = mock_results

    result = synapse_redis.search_hybrid(
        query=query,
        embedding=embedding,
        domain_filter=domain_filter,
        type_filter=type_filter,
        limit=limit,
    )

    assert len(result) == 1
    assert result[0]["id"] == "node:test:123"
    assert result[0]["domain"] == "test"
    assert result[0]["type"] == "entity"


def test_redis_search_hybrid_bm25_only():
    """Test hybrid search with BM25 only (no embedding)."""

    mock_redis = Mock()
    mock_ft = Mock()
    mock_redis.ft.return_value = mock_ft

    synapse_redis = SynapseRedis(mock_redis)

    query = "test query"
    limit = 5

    # Mock search results
    mock_doc = Mock()
    mock_doc.__dict__ = {
        "id": "node:test:456",
        "domain": "docs",
        "type": "chunk",
        "content": "test content",
    }

    mock_results = Mock()
    mock_results.docs = [mock_doc]
    mock_ft.search.return_value = mock_results

    result = synapse_redis.search_hybrid(query=query, limit=limit)

    assert len(result) == 1
    assert result[0]["id"] == "node:test:456"


def test_redis_get_linked_nodes():
    """Test getting linked nodes via graph traversal."""

    mock_redis = Mock()
    synapse_redis = SynapseRedis(mock_redis)

    node_id = "node:test:123"
    linked_node_id = "node:test:456"

    # Mock main node with links
    main_node = {
        "id": node_id,
        "domain": "test",
        "type": "entity",
        "content": "main content",
        "links": {"inbound": [], "outbound": [linked_node_id]},
    }

    # Mock linked node
    linked_node = {
        "id": linked_node_id,
        "domain": "test",
        "type": "relation",
        "content": "linked content",
    }

    mock_redis.json().get.side_effect = [main_node, linked_node]

    result = synapse_redis.get_linked_nodes(node_id, direction="outbound")

    assert len(result) == 1
    assert result[0]["id"] == linked_node_id


def test_redis_ping():
    """Test Redis ping functionality."""

    mock_redis = Mock()
    mock_redis.ping.return_value = True

    synapse_redis = SynapseRedis(mock_redis)

    result = synapse_redis.ping()

    assert result is True
    mock_redis.ping.assert_called_once()


def test_redis_close():
    """Test Redis close functionality."""

    mock_redis = Mock()

    synapse_redis = SynapseRedis(mock_redis)
    synapse_redis.close()

    mock_redis.close.assert_called_once()


def test_redis_float_to_bytes():
    """Test float to bytes conversion for KNN."""

    mock_redis = Mock()
    synapse_redis = SynapseRedis(mock_redis)

    vec = [0.1, 0.2, 0.3]
    result = synapse_redis._float_to_bytes(vec)

    assert isinstance(result, bytes)
    assert len(result) == 12  # 3 floats * 4 bytes each


def test_redis_doc_to_dict():
    """Test RediSearch document to dict conversion."""

    mock_redis = Mock()
    synapse_redis = SynapseRedis(mock_redis)

    mock_doc = Mock()
    mock_doc.__dict__ = {
        "id": "node:test:123",
        "json": '{"domain": "test", "type": "entity"}',
        "score": 0.95,
    }

    result = synapse_redis._doc_to_dict(mock_doc)

    assert result["id"] == "node:test:123"
    assert result["score"] == 0.95
    assert isinstance(result["json"], dict)
    assert result["json"]["domain"] == "test"
