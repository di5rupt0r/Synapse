"""TDD RED Phase: Tests for Tree-sitter Chunking."""

import pytest


def test_extension_mapping():
    """Test file extension to language mapping."""
    # This will fail - chunking doesn't exist yet (RED phase)
    from synapse.chunking.treesitter import EXTENSION_MAP

    assert EXTENSION_MAP[".py"] == "python"
    assert EXTENSION_MAP[".js"] == "javascript"
    assert EXTENSION_MAP[".ts"] == "typescript"
    assert EXTENSION_MAP[".java"] == "java"
    assert EXTENSION_MAP[".cpp"] == "cpp"
    assert EXTENSION_MAP[".c"] == "c"


def test_chunk_node_types():
    """Test chunk node type definitions."""
    from synapse.chunking.treesitter import CHUNK_NODE_TYPES

    assert "function_definition" in CHUNK_NODE_TYPES
    assert "class_definition" in CHUNK_NODE_TYPES
    assert "method_definition" in CHUNK_NODE_TYPES
    assert "block" in CHUNK_NODE_TYPES


def test_chunk_by_treesitter_python():
    """Test Python code chunking with tree-sitter."""
    from synapse.chunking.treesitter import chunk_by_treesitter

    code = """
def hello_world():
    print("Hello, World!")

class MyClass:
    def method1(self):
        return "method1"

    def method2(self):
        return "method2"
"""

    chunks = chunk_by_treesitter(code, ".py")

    assert len(chunks) >= 2  # Should find function and class

    # Check function chunk
    func_chunk = next(
        (c for c in chunks if c["node_type"] == "function_definition"), None
    )
    assert func_chunk is not None
    assert func_chunk["language"] == "python"
    assert "hello_world" in func_chunk["text"]
    assert func_chunk["line_start"] >= 1
    assert func_chunk["line_end"] >= func_chunk["line_start"]

    # Check class chunk
    class_chunk = next(
        (c for c in chunks if c["node_type"] == "class_definition"), None
    )
    assert class_chunk is not None
    assert "MyClass" in class_chunk["text"]


def test_chunk_by_treesitter_javascript():
    """Test JavaScript code chunking with tree-sitter."""
    from synapse.chunking.treesitter import chunk_by_treesitter

    code = """
function greet(name) {
    return `Hello, ${name}!`;
}

class Person {
    constructor(name) {
        this.name = name;
    }

    sayHello() {
        return greet(this.name);
    }
}
"""

    chunks = chunk_by_treesitter(code, ".js")

    assert len(chunks) >= 2  # Should find function and class

    # Check function chunk
    func_chunk = next(
        (
            c
            for c in chunks
            if c["node_type"] in ["function_definition", "function_declaration"]
        ),
        None,
    )
    assert func_chunk is not None
    assert func_chunk["language"] == "javascript"
    assert "greet" in func_chunk["text"]

    # Check class chunk
    class_chunk = next(
        (
            c
            for c in chunks
            if c["node_type"] in ["class_definition", "class_declaration"]
        ),
        None,
    )
    assert class_chunk is not None
    assert "Person" in class_chunk["text"]


def test_chunk_by_treesitter_empty_code():
    """Test chunking with empty code."""
    from synapse.chunking.treesitter import chunk_by_treesitter

    chunks = chunk_by_treesitter("", ".py")

    assert len(chunks) == 0


def test_chunk_by_treesitter_unsupported_extension():
    """Test chunking with unsupported file extension."""
    from synapse.chunking.treesitter import chunk_by_treesitter

    code = "some random content"

    with pytest.raises(ValueError) as exc:
        chunk_by_treesitter(code, ".xyz")

    assert "Unsupported file extension" in str(exc.value)


def test_chunk_by_treesitter_line_numbers():
    """Test accurate line number calculation."""
    from synapse.chunking.treesitter import chunk_by_treesitter

    code = """line 1
line 2
def func():
    line 4
    line 5
"""

    chunks = chunk_by_treesitter(code, ".py")

    func_chunk = next(
        (c for c in chunks if c["node_type"] == "function_definition"), None
    )
    assert func_chunk is not None
    assert func_chunk["line_start"] == 3  # Function starts at line 3
    assert func_chunk["line_end"] == 5  # Function ends at line 5
