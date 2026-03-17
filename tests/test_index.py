"""TDD RED Phase: Tests for RediSearch Index Setup."""

from unittest.mock import Mock


def test_index_creation():
    """Test FT.CREATE command generation."""
    # This will fail - index setup doesn't exist yet (RED phase)
    from synapse.index.setup import IndexManager

    mock_redis = Mock()
    manager = IndexManager(mock_redis)

    create_command = manager.get_create_command()

    assert "FT.CREATE" in create_command
    assert "synapse_idx" in create_command
    assert "ON JSON" in create_command
    assert 'PREFIX 1 "node:"' in create_command
    assert "domain TAG" in create_command
    assert "type TAG" in create_command
    assert "content TEXT" in create_command
    assert "embedding VECTOR" in create_command
    assert "768" in create_command
    assert "COSINE" in create_command


def test_index_idempotent():
    """Test index creation is idempotent (drop if exists, then recreate)."""
    from synapse.index.setup import IndexManager

    mock_redis = Mock()
    mock_ft = Mock()
    mock_redis.ft.return_value = mock_ft

    # Simulate index exists
    mock_ft.info.return_value = {"num_docs": 10}

    manager = IndexManager(mock_redis)
    manager.ensure_index()

    # Should drop existing then create new
    mock_ft.dropindex.assert_called_once()
    mock_ft.create_index.assert_called_once()


def test_index_not_exists():
    """Test index creation when index doesn't exist."""
    from synapse.index.setup import IndexManager

    mock_redis = Mock()
    mock_ft = Mock()
    mock_redis.ft.return_value = mock_ft

    # Simulate index doesn't exist
    mock_ft.info.side_effect = Exception("Index not found")

    manager = IndexManager(mock_redis)
    manager.ensure_index()

    # Should only create, not drop
    mock_ft.dropindex.assert_not_called()
    mock_ft.create_index.assert_called_once()
