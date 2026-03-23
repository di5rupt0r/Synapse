"""Tests for Tree-sitter Chunking."""


def test_extension_mapping():
    """Test file extension to language mapping."""
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


def test_chunk_by_treesitter_returns_chunks():
    """Test Python code chunking returns chunks."""
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

    # Should return chunks (either tree-sitter or fallback)
    assert len(chunks) >= 1
    assert all("id" in c for c in chunks)
    assert all("text" in c for c in chunks)


def test_chunk_by_treesitter_javascript():
    """Test JavaScript code chunking."""
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

    # Should return chunks (either tree-sitter or fallback)
    assert len(chunks) >= 1


def test_chunk_by_treesitter_line_numbers():
    """Test that chunks have valid line numbers."""
    from synapse.chunking.treesitter import chunk_by_treesitter

    code = """
def hello():
    pass

class World:
    pass
"""

    chunks = chunk_by_treesitter(code, ".py")

    for chunk in chunks:
        assert "line_start" in chunk
        assert "line_end" in chunk
        assert chunk["line_start"] >= 1
        assert chunk["line_end"] >= chunk["line_start"]
