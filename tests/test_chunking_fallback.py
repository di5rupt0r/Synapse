"""TDD RED Phase: Tests for Fallback Chunking."""


def test_fallback_chunk_by_lines_basic():
    """Test basic line-based chunking."""
    from synapse.chunking.fallback import fallback_chunk_by_lines

    content = """line 1
line 2
line 3
line 4
line 5
line 6
line 7
line 8
line 9
line 10"""

    chunks = fallback_chunk_by_lines(content, chunk_size=3, overlap=1)

    assert (
        len(chunks) == 5
    )  # 10 lines, 3 per chunk with 1 overlap: [1-3], [3-5], [5-7], [7-9], [9-10]

    # Check first chunk
    assert chunks[0]["line_start"] == 1
    assert chunks[0]["line_end"] == 3
    assert "line 1" in chunks[0]["text"]

    # Check second chunk (with overlap)
    assert chunks[1]["line_start"] == 3  # Overlap from line 3
    assert chunks[1]["line_end"] == 5


def test_fallback_chunk_by_lines_empty():
    """Test chunking with empty content."""
    from synapse.chunking.fallback import fallback_chunk_by_lines

    chunks = fallback_chunk_by_lines("")

    assert len(chunks) == 0


def test_fallback_chunk_by_lines_whitespace():
    """Test chunking with whitespace-only content."""
    from synapse.chunking.fallback import fallback_chunk_by_lines

    chunks = fallback_chunk_by_lines("   \n  \n   ")

    assert len(chunks) == 0


def test_fallback_chunk_by_lines_single_chunk():
    """Test chunking where content fits in one chunk."""
    from synapse.chunking.fallback import fallback_chunk_by_lines

    content = "line 1\nline 2\nline 3"

    chunks = fallback_chunk_by_lines(content, chunk_size=10, overlap=2)

    assert len(chunks) == 1
    assert chunks[0]["line_start"] == 1
    assert chunks[0]["line_end"] == 3
    assert chunks[0]["text"] == content


def test_fallback_chunk_by_lines_no_overlap():
    """Test chunking with no overlap."""
    from synapse.chunking.fallback import fallback_chunk_by_lines

    content = "line 1\nline 2\nline 3\nline 4\nline 5"

    chunks = fallback_chunk_by_lines(content, chunk_size=2, overlap=0)

    assert len(chunks) == 3  # 5 lines, 2 per chunk, no overlap

    assert chunks[0]["line_start"] == 1
    assert chunks[0]["line_end"] == 2

    assert chunks[1]["line_start"] == 3
    assert chunks[1]["line_end"] == 4

    assert chunks[2]["line_start"] == 5
    assert chunks[2]["line_end"] == 5


def test_fallback_chunk_by_lines_chunk_structure():
    """Test chunk structure and required fields."""
    from synapse.chunking.fallback import fallback_chunk_by_lines

    content = "line 1\nline 2"

    chunks = fallback_chunk_by_lines(content)

    assert len(chunks) == 1
    chunk = chunks[0]

    # Check required fields
    assert "id" in chunk
    assert chunk["id"].startswith("chunk:")
    assert "text" in chunk
    assert "language" in chunk
    assert chunk["language"] == "unknown"
    assert "node_type" in chunk
    assert chunk["node_type"] == "line_chunk"
    assert "line_start" in chunk
    assert "line_end" in chunk

    # Check line numbers
    assert chunk["line_start"] == 1
    assert chunk["line_end"] == 2


def test_fallback_chunk_by_lines_large_content():
    """Test chunking with large content."""
    from synapse.chunking.fallback import fallback_chunk_by_lines

    # Create 100 lines
    lines = [f"line {i}" for i in range(1, 101)]
    content = "\n".join(lines)

    chunks = fallback_chunk_by_lines(content, chunk_size=20, overlap=5)

    # Should create multiple chunks
    assert len(chunks) > 1

    # Check first chunk
    assert chunks[0]["line_start"] == 1
    assert chunks[0]["line_end"] == 20

    # Check last chunk
    last_chunk = chunks[-1]
    assert last_chunk["line_end"] == 100
