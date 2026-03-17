"""TDD RED Phase: Tests for BM25 Sparse Search."""


def test_bm25_index_initialization():
    """Test BM25 index initialization with chunks."""
    # This will fail - BM25Index doesn't exist yet (RED phase)
    from synapse.schema.node import Chunk
    from synapse.search.bm25 import BM25Index

    chunks = [
        Chunk(
            id="chunk:123e4567-e89b-12d3-a456-426614174001",
            text="def hello_world():",
            language="python",
            node_type="function_definition",
            line_start=1,
            line_end=1,
            embedding=[0.1] * 768,
        ),
        Chunk(
            id="chunk:123e4567-e89b-12d3-a456-426614174002",
            text='print("Hello, World!")',
            language="python",
            node_type="expression_statement",
            line_start=2,
            line_end=2,
            embedding=[0.2] * 768,
        ),
    ]

    index = BM25Index(chunks)

    assert index.chunks == chunks
    assert hasattr(index, "bm25")


def test_bm25_search_basic():
    """Test basic BM25 search functionality."""
    from synapse.schema.node import Chunk
    from synapse.search.bm25 import BM25Index

    chunks = [
        Chunk(
            id="chunk:123e4567-e89b-12d3-a456-426614174001",
            text="def hello_world():",
            language="python",
            node_type="function_definition",
            line_start=1,
            line_end=1,
            embedding=[0.1] * 768,
        ),
        Chunk(
            id="chunk:123e4567-e89b-12d3-a456-426614174002",
            text='print("Hello, World!")',
            language="python",
            node_type="expression_statement",
            line_start=2,
            line_end=2,
            embedding=[0.2] * 768,
        ),
        Chunk(
            id="chunk:123e4567-e89b-12d3-a456-426614174003",
            text="def goodbye_world():",
            language="python",
            node_type="function_definition",
            line_start=3,
            line_end=3,
            embedding=[0.3] * 768,
        ),
    ]

    index = BM25Index(chunks)

    # Search for "hello"
    results = index.search("hello", top_k=3)

    assert len(results) == 2
    assert all(isinstance(result, tuple) for result in results)
    assert all(len(result) == 2 for result in results)  # (chunk_id, score)

    # Should return chunks containing "hello"
    chunk_ids = [result[0] for result in results]
    assert (
        "chunk:123e4567-e89b-12d3-a456-426614174001" in chunk_ids
    )  # hello_world function
    assert (
        "chunk:123e4567-e89b-12d3-a456-426614174002" in chunk_ids
    )  # Hello, World print


def test_bm25_search_empty_query():
    """Test BM25 search with empty query."""
    from synapse.schema.node import Chunk
    from synapse.search.bm25 import BM25Index

    chunks = [
        Chunk(
            id="chunk:123e4567-e89b-12d3-a456-426614174001",
            text="some content",
            language="python",
            node_type="function_definition",
            line_start=1,
            line_end=1,
            embedding=[0.1] * 768,
        )
    ]

    index = BM25Index(chunks)

    results = index.search("", top_k=5)

    assert len(results) == 0


def test_bm25_search_no_results():
    """Test BM25 search with no matching results."""
    from synapse.schema.node import Chunk
    from synapse.search.bm25 import BM25Index

    chunks = [
        Chunk(
            id="chunk:123e4567-e89b-12d3-a456-426614174001",
            text="function definition",
            language="python",
            node_type="function_definition",
            line_start=1,
            line_end=1,
            embedding=[0.1] * 768,
        )
    ]

    index = BM25Index(chunks)

    results = index.search("nonexistent_term", top_k=5)

    assert len(results) == 0


def test_bm25_search_top_k_limiting():
    """Test BM25 search respects top_k parameter."""
    from synapse.schema.node import Chunk
    from synapse.search.bm25 import BM25Index

    chunks = [
        Chunk(
            id=f"chunk:123e4567-e89b-12d3-a456-42661417{i:04d}",
            text=f"def function_{i}():",
            language="python",
            node_type="function_definition",
            line_start=i,
            line_end=i,
            embedding=[0.1] * 768,
        )
        for i in range(1, 6)  # 5 chunks
    ]

    index = BM25Index(chunks)

    results = index.search("function", top_k=3)

    assert len(results) == 3  # Should be limited to top_k


def test_bm25_tokenization():
    """Test BM25 tokenization behavior."""
    from synapse.schema.node import Chunk
    from synapse.search.bm25 import BM25Index

    chunks = [
        Chunk(
            id="chunk:123e4567-e89b-12d3-a456-426614174001",
            text="def hello_world():",
            language="python",
            node_type="function_definition",
            line_start=1,
            line_end=1,
            embedding=[0.1] * 768,
        )
    ]

    index = BM25Index(chunks)

    # Test with different cases
    results1 = index.search("Hello", top_k=5)  # Capital H
    results2 = index.search("hello", top_k=5)  # Lowercase h

    assert len(results1) == len(results2) == 1
    assert results1[0][0] == results2[0][0]  # Same chunk ID

    # Test with underscores
    results3 = index.search("hello_world", top_k=5)
    assert len(results3) == 1


def test_bm25_performance_target():
    """Test BM25 search performance meets target (<10ms for 10k chunks)."""
    import time

    # Create 1000 chunks (scaled down for testing)
    import uuid

    from synapse.schema.node import Chunk
    from synapse.search.bm25 import BM25Index

    chunks = [
        Chunk(
            id=f"chunk:{uuid.uuid4()}",
            text=f"def function_{i}(): return {i}",
            language="python",
            node_type="function_definition",
            line_start=i + 1,  # Start from 1
            line_end=i + 1,  # End from 1
            embedding=[0.1] * 768,
        )
        for i in range(1000)
    ]

    index = BM25Index(chunks)

    start_time = time.time()
    results = index.search("function", top_k=10)
    end_time = time.time()

    search_time_ms = (end_time - start_time) * 1000

    # Should be very fast for 1000 chunks (target is <10ms for 10k)
    assert search_time_ms < 5.0  # Conservative target for 1000 chunks
    assert len(results) == 10
