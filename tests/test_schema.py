"""TDD RED Phase: Tests for Synapse Node Schema."""

import uuid
from datetime import datetime

import pytest
from pydantic import ValidationError


def test_node_creation_minimal():
    """Test creating a valid node with minimal fields."""
    # This will fail - schema doesn't exist yet (RED phase)
    from synapse.schema.node import SynapseNode

    node_id = f"node:test:{uuid.uuid4()}"
    node = SynapseNode(
        id=node_id,
        domain="test",
        type="entity",
        content="Test content",
        embedding=[0.0] * 768,  # ADR-001 specifies 768-dim
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )

    assert node.id == node_id
    assert node.domain == "test"
    assert node.type == "entity"
    assert len(node.embedding) == 768


def test_node_invalid_embedding_dimension():
    """Test validation fails for wrong embedding dimension."""
    from synapse.schema.node import SynapseNode

    with pytest.raises(ValidationError) as exc:
        SynapseNode(
            id=f"node:test:{uuid.uuid4()}",
            domain="test",
            type="entity",
            content="Test content",
            embedding=[0.0] * 384,  # Wrong dimension
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )

    assert "768" in str(exc.value)


def test_node_invalid_type():
    """Test validation fails for invalid node type."""
    from synapse.schema.node import SynapseNode

    with pytest.raises(ValidationError) as exc:
        SynapseNode(
            id=f"node:test:{uuid.uuid4()}",
            domain="test",
            type="invalid_type",
            content="Test content",
            embedding=[0.0] * 768,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )

    assert "entity" in str(exc.value) or "observation" in str(exc.value)


def test_node_invalid_id_format():
    """Test validation fails for wrong ID format."""
    from synapse.schema.node import SynapseNode

    with pytest.raises(ValidationError) as exc:
        SynapseNode(
            id="invalid_id",
            domain="test",
            type="entity",
            content="Test content",
            embedding=[0.0] * 768,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )

    assert "node:" in str(exc.value)
