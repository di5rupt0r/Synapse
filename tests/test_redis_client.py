"""Tests for Redis Client - Coverage improvement."""

from unittest.mock import Mock


class TestSynapseRedis:
    """Test SynapseRedis class."""

    def test_init(self):
        """Test initialization."""
        from synapse.redis.client import SynapseRedis

        mock_client = Mock()
        redis = SynapseRedis(mock_client)
        assert redis._client == mock_client

    def test_ping(self):
        """Test ping."""
        from synapse.redis.client import SynapseRedis

        mock_client = Mock()
        mock_client.ping = Mock(return_value=True)
        redis = SynapseRedis(mock_client)
        result = redis.ping()
        assert result is True

    def test_close(self):
        """Test close."""
        from synapse.redis.client import SynapseRedis

        mock_client = Mock()
        mock_client.close = Mock()
        redis = SynapseRedis(mock_client)
        redis.close()
        mock_client.close.assert_called_once()

    def test_store_node(self):
        """Test store_node."""
        from synapse.redis.client import SynapseRedis

        mock_client = Mock()
        mock_json = Mock()
        mock_json.set = Mock(return_value=True)
        mock_client.json = Mock(return_value=mock_json)
        redis = SynapseRedis(mock_client)
        result = redis.store_node(
            node_id="node:test:123",
            domain="test",
            node_type="entity",
            content="test content",
            embedding=[0.1] * 768,
        )
        assert result == "node:test:123"

    def test_get_node_success(self):
        """Test get_node success."""
        from synapse.redis.client import SynapseRedis

        mock_client = Mock()
        mock_json = Mock()
        mock_json.get = Mock(return_value={"id": "node:test:123"})
        mock_client.json = Mock(return_value=mock_json)
        redis = SynapseRedis(mock_client)
        result = redis.get_node("node:test:123")
        assert result["id"] == "node:test:123"

    def test_get_node_list_result(self):
        """Test get_node with list result."""
        from synapse.redis.client import SynapseRedis

        mock_client = Mock()
        mock_json = Mock()
        mock_json.get = Mock(return_value=[{"id": "node:test:123"}])
        mock_client.json = Mock(return_value=mock_json)
        redis = SynapseRedis(mock_client)
        result = redis.get_node("node:test:123")
        assert result["id"] == "node:test:123"

    def test_get_node_not_found(self):
        """Test get_node not found."""
        from synapse.redis.client import SynapseRedis

        mock_client = Mock()
        mock_json = Mock()
        mock_json.get = Mock(side_effect=Exception("Not found"))
        mock_client.json = Mock(return_value=mock_json)
        redis = SynapseRedis(mock_client)
        result = redis.get_node("node:test:123")
        assert result is None

    def test_update_node_success(self):
        """Test update_node success."""
        from synapse.redis.client import SynapseRedis

        mock_client = Mock()
        mock_json = Mock()
        mock_json.get = Mock(return_value={"id": "node:test:123", "metadata": {}})
        mock_json.set = Mock(return_value=True)
        mock_client.json = Mock(return_value=mock_json)
        redis = SynapseRedis(mock_client)
        result = redis.update_node(
            "node:test:123", [{"op": "set", "path": "$.metadata.foo", "value": "bar"}]
        )
        assert result is True

    def test_update_node_not_found(self):
        """Test update_node not found."""
        from synapse.redis.client import SynapseRedis

        mock_client = Mock()
        mock_json = Mock()
        mock_json.get = Mock(return_value=None)
        mock_client.json = Mock(return_value=mock_json)
        redis = SynapseRedis(mock_client)
        result = redis.update_node("node:test:123", [])
        assert result is False

    def test_update_node_delete_op(self):
        """Test update_node delete operation."""
        from synapse.redis.client import SynapseRedis

        mock_client = Mock()
        mock_json = Mock()
        mock_json.get = Mock(
            return_value={"id": "node:test:123", "metadata": {"foo": "bar"}}
        )
        mock_json.delete = Mock(return_value=True)
        mock_client.json = Mock(return_value=mock_json)
        redis = SynapseRedis(mock_client)
        result = redis.update_node(
            "node:test:123", [{"op": "delete", "path": "$.metadata.foo"}]
        )
        assert result is True

    def test_update_node_append_op(self):
        """Test update_node append operation."""
        from synapse.redis.client import SynapseRedis

        mock_client = Mock()
        mock_json = Mock()
        mock_json.get = Mock(side_effect=[{"id": "node:test:123", "list": []}, []])
        mock_json.set = Mock(return_value=True)
        mock_client.json = Mock(return_value=mock_json)
        redis = SynapseRedis(mock_client)
        result = redis.update_node(
            "node:test:123", [{"op": "append", "path": "$.list", "value": "item"}]
        )
        assert result is True

    def test_search_hybrid_with_embedding(self):
        """Test search_hybrid with embedding."""
        from synapse.redis.client import SynapseRedis

        mock_client = Mock()
        mock_ft = Mock()
        mock_doc = Mock()
        mock_doc.__dict__ = {"id": "node:test:123", "json": '{"id": "node:test:123"}'}
        mock_results = Mock()
        mock_results.docs = [mock_doc]
        mock_ft.search = Mock(return_value=mock_results)
        mock_client.ft = Mock(return_value=mock_ft)
        redis = SynapseRedis(mock_client)
        result = redis.search_hybrid("query", embedding=[0.1] * 768)
        assert len(result) == 1

    def test_search_hybrid_fallback_bm25(self):
        """Test search_hybrid fallback to BM25."""
        from synapse.redis.client import SynapseRedis

        mock_client = Mock()
        mock_ft = Mock()
        mock_ft.search = Mock(side_effect=[Exception("KNN failed"), Mock(docs=[])])
        mock_client.ft = Mock(return_value=mock_ft)
        redis = SynapseRedis(mock_client)
        result = redis.search_hybrid("query", embedding=[0.1] * 768)
        assert result == []

    def test_search_hybrid_error(self):
        """Test search_hybrid error."""
        from synapse.redis.client import SynapseRedis

        mock_client = Mock()
        mock_client.ft = Mock(side_effect=Exception("Redis error"))
        redis = SynapseRedis(mock_client)
        result = redis.search_hybrid("query")
        assert result == []

    def test_get_linked_nodes(self):
        """Test get_linked_nodes."""
        from synapse.redis.client import SynapseRedis

        mock_client = Mock()
        mock_json = Mock()
        mock_json.get = Mock(
            side_effect=[
                {"id": "node:test:1", "links": {"outbound": ["node:test:2"]}},
                {"id": "node:test:2"},
            ]
        )
        mock_client.json = Mock(return_value=mock_json)
        redis = SynapseRedis(mock_client)
        result = redis.get_linked_nodes("node:test:1")
        assert len(result) == 1

    def test_get_linked_nodes_not_found(self):
        """Test get_linked_nodes node not found."""
        from synapse.redis.client import SynapseRedis

        mock_client = Mock()
        mock_json = Mock()
        mock_json.get = Mock(return_value=None)
        mock_client.json = Mock(return_value=mock_json)
        redis = SynapseRedis(mock_client)
        result = redis.get_linked_nodes("node:test:1")
        assert result == []

    def test_float_to_bytes(self):
        """Test _float_to_bytes."""
        from synapse.redis.client import SynapseRedis

        mock_client = Mock()
        redis = SynapseRedis(mock_client)
        result = redis._float_to_bytes([1.0, 2.0, 3.0])
        assert isinstance(result, bytes)
        assert len(result) == 12  # 3 floats * 4 bytes

    def test_doc_to_dict(self):
        """Test _doc_to_dict."""
        from synapse.redis.client import SynapseRedis

        mock_client = Mock()
        redis = SynapseRedis(mock_client)
        mock_doc = Mock()
        mock_doc.__dict__ = {"id": "doc:1", "json": '{"key": "value"}'}
        result = redis._doc_to_dict(mock_doc)
        assert result["id"] == "doc:1"
        assert result["json"] == {"key": "value"}

    def test_doc_to_dict_invalid_json(self):
        """Test _doc_to_dict with invalid JSON."""
        from synapse.redis.client import SynapseRedis

        mock_client = Mock()
        redis = SynapseRedis(mock_client)
        mock_doc = Mock()
        mock_doc.__dict__ = {"id": "doc:1", "json": "invalid json"}
        result = redis._doc_to_dict(mock_doc)
        assert result["json"] == "invalid json"
