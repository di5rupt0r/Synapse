"""TDD RED Phase: Tests for Updated Schema with Chunk support."""

import pytest
from pydantic import ValidationError
from datetime import datetime


def test_chunk_model_basic():
    """Test basic Chunk model validation."""
    # This will fail - Chunk model doesn't exist yet (RED phase)
    from synapse.schema.node import Chunk

    chunk = Chunk(
        id="chunk:123e4567-e89b-12d3-a456-426614174000",
        text="def hello(): pass",
        language="python",
        node_type="function_definition",
        line_start=1,
        line_end=1,
        embedding=[0.1] * 768,
    )

    assert chunk.id == "chunk:123e4567-e89b-12d3-a456-426614174000"
    assert chunk.text == "def hello(): pass"
    assert chunk.language == "python"
    assert chunk.node_type == "function_definition"
    assert chunk.line_start == 1
    assert chunk.line_end == 1
    assert len(chunk.embedding) == 768


def test_chunk_model_validation():
    """Test Chunk model validation."""
    from synapse.schema.node import Chunk

    # Test invalid embedding dimension
    with pytest.raises(ValidationError) as exc:
        Chunk(
            id="chunk:test",
            text="test",
            language="python",
            node_type="function_definition",
            line_start=1,
            line_end=1,
            embedding=[0.1] * 500,  # Wrong dimension
        )

    assert "768" in str(exc.value)

    # Test invalid line numbers
    with pytest.raises(ValidationError) as exc:
        Chunk(
            id="chunk:test",
            text="test",
            language="python",
            node_type="function_definition",
            line_start=5,  # Start > end
            line_end=1,
            embedding=[0.1] * 768,
        )

    assert "line_end" in str(exc.value).lower()


def test_chunk_model_optional_fields():
    """Test Chunk model with optional fields."""
    from synapse.schema.node import Chunk

    chunk = Chunk(
        id="chunk:123e4567-e89b-12d3-a456-426614174000",
        text="test code",
        language="python",
        node_type="function_definition",
        line_start=1,
        line_end=3,
        embedding=[0.1] * 768,
    )

    # Should work without optional fields
    assert chunk.id == "chunk:123e4567-e89b-12d3-a456-426614174000"
    assert chunk.embedding == [0.1] * 768


def test_updated_node_model_with_chunks():
    """Test updated Node model with chunks support."""
    from synapse.schema.node import SynapseNode, Chunk

    chunks = [
        Chunk(
            id="chunk:123e4567-e89b-12d3-a456-426614174001",
            text="def func1(): pass",
            language="python",
            node_type="function_definition",
            line_start=1,
            line_end=1,
            embedding=[0.1] * 768,
        ),
        Chunk(
            id="chunk:123e4567-e89b-12d3-a456-426614174002",
            text="def func2(): pass",
            language="python",
            node_type="function_definition",
            line_start=2,
            line_end=2,
            embedding=[0.2] * 768,
        ),
    ]

    node = SynapseNode(
        id="node:test:123e4567-e89b-12d3-a456-426614174000",
        domain="test",
        type="entity",
        content="class Test:\n    def func1(): pass\n    def func2(): pass",
        chunks=chunks,
        embedding=[0.3] * 768,
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )

    assert len(node.chunks) == 2
    assert node.chunks[0].id == "chunk:123e4567-e89b-12d3-a456-426614174001"
    assert node.chunks[1].id == "chunk:123e4567-e89b-12d3-a456-426614174002"
    assert node.chunks[0].language == "python"
    assert node.chunks[1].language == "python"


def test_chunk_model_serialization():
    """Test Chunk model serialization/deserialization."""
    from synapse.schema.node import Chunk

    chunk = Chunk(
        id="chunk:123e4567-e89b-12d3-a456-426614174003",
        text="def hello(): pass",
        language="python",
        node_type="function_definition",
        line_start=1,
        line_end=1,
        embedding=[0.1] * 768,
    )

    # Test model_dump
    data = chunk.model_dump()
    assert data["id"] == "chunk:123e4567-e89b-12d3-a456-426614174003"
    assert data["text"] == "def hello(): pass"
    assert data["language"] == "python"
    assert data["node_type"] == "function_definition"
    assert data["line_start"] == 1
    assert data["line_end"] == 1
    assert len(data["embedding"]) == 768

    # Test model_validate
    chunk2 = Chunk.model_validate(data)
    assert chunk2.id == chunk.id
    assert chunk2.text == chunk.text


def test_chunk_model_with_metadata():
    """Test Chunk model with metadata field."""
    from synapse.schema.node import Chunk

    chunk = Chunk(
        id="chunk:123e4567-e89b-12d3-a456-426614174004",
        text="def hello(): pass",
        language="python",
        node_type="function_definition",
        line_start=1,
        line_end=1,
        embedding=[0.1] * 768,
        metadata={"complexity": "low", "calls": 0},
    )

    assert chunk.metadata == {"complexity": "low", "calls": 0}


def test_chunk_model_line_validation():
    """Test line number validation."""
    from synapse.schema.node import Chunk

    # Valid: start == end
    chunk1 = Chunk(
        id="chunk:123e4567-e89b-12d3-a456-426614174005",
        text="single line",
        language="python",
        node_type="function_definition",
        line_start=5,
        line_end=5,
        embedding=[0.1] * 768,
    )
    assert chunk1.line_start == 5
    assert chunk1.line_end == 5

    # Valid: start < end
    chunk2 = Chunk(
        id="chunk:123e4567-e89b-12d3-a456-426614174006",
        text="multi\nline\ncode",
        language="python",
        node_type="class_definition",
        line_start=1,
        line_end=3,
        embedding=[0.2] * 768,
    )
    assert chunk2.line_start == 1
    assert chunk2.line_end == 3

    # Invalid: negative line numbers
    with pytest.raises(ValidationError):
        Chunk(
            id="chunk:123e4567-e89b-12d3-a456-426614174007",
            text="test",
            language="python",
            node_type="function_definition",
            line_start=-1,  # Invalid
            line_end=1,
            embedding=[0.1] * 768,
        )
