"""Tests for tree-sitter based code chunking - Complete coverage."""

from unittest.mock import Mock, patch
import pytest

# Test the core chunking module
from synapse.chunking.treesitter import (
    EXTENSION_MAP,
    CHUNK_NODE_TYPES,
    chunk_by_treesitter,
    get_parser,
    extract_chunk,
    fallback_chunk_by_lines,
)


class TestTreeSitterChunkingComplete:
    """Complete tree-sitter chunking coverage."""

    def test_extension_mapping(self):
        """Test extension to language mapping."""
        assert EXTENSION_MAP[".py"] == "python"
        assert EXTENSION_MAP[".js"] == "javascript"
        assert EXTENSION_MAP[".cpp"] == "cpp"

    def test_chunk_by_treesitter_empty(self):
        """Test chunking empty content."""
        assert chunk_by_treesitter("", ".py") == []

    def test_chunk_by_treesitter_unsupported_extension(self):
        """Test unsupported extension."""
        with pytest.raises(ValueError, match="Unsupported file extension"):
            chunk_by_treesitter("def test(): pass", ".unknown")

    def test_get_parser_success(self):
        """Test get_parser when language imports successfully."""
        mock_lang_module = Mock()
        mock_lang_module.language.return_value = 123456
        
        mock_Language = Mock()
        mock_Language.return_value = "mock_Lang_obj"
        
        mock_Parser = Mock()
        mock_Parser.return_value = "mock_Parser_obj"

        with (
            patch.dict("sys.modules", {"tree_sitter_python": mock_lang_module}),
            patch("synapse.chunking.treesitter.Language", mock_Language),
            patch("synapse.chunking.treesitter.Parser", mock_Parser)
        ):
            parser = get_parser("python")
            assert parser == "mock_Parser_obj"
            mock_Language.assert_called_with(123456)
            mock_Parser.assert_called_with("mock_Lang_obj")

    def test_get_parser_fallback_to_python(self):
        """Test getting parser falls back to Python if language not found."""
        mock_py_lang_module = Mock()
        mock_py_lang_module.language.return_value = 123456
        
        mock_Language = Mock()
        mock_Language.return_value = "mock_Lang_obj"
        
        mock_Parser = Mock()
        mock_Parser.return_value = "mock_Parser_obj"
        
        import builtins
        original_import = builtins.__import__
        def safe_mock_import(name, *args, **kwargs):
            if name == "tree_sitter_unknown":
                raise ImportError("mock error")
            if name == "tree_sitter_python":
                return mock_py_lang_module
            return original_import(name, *args, **kwargs)
            
        with (
            patch("builtins.__import__", side_effect=safe_mock_import),
            patch("synapse.chunking.treesitter.Language", mock_Language),
            patch("synapse.chunking.treesitter.Parser", mock_Parser)
        ):
            parser = get_parser("unknown")
            assert parser == "mock_Parser_obj"

    def test_get_parser_complete_failure(self):
        """Test get_parser when tree-sitter completely fails to import python."""
        import builtins
        original_import = builtins.__import__
        def failing_mock_import(name, *args, **kwargs):
            if name.startswith("tree_sitter_"):
                raise ImportError("mock error")
            return original_import(name, *args, **kwargs)

        with (
            patch("builtins.__import__", side_effect=failing_mock_import),
            patch("synapse.chunking.treesitter.Language", Mock()),
            patch("synapse.chunking.treesitter.Parser", Mock())
        ):
            with pytest.raises(ImportError, match="Tree-sitter language"):
                get_parser("unknown")

    def test_chunk_by_treesitter_fallback(self):
        """Test chunk_by_treesitter falls back to lines if parsing fails."""
        with patch("synapse.chunking.treesitter.get_parser", side_effect=Exception("parse error")):
            result = chunk_by_treesitter("def auth():\n    pass\n", ".py")
            # Should have fallen back to fallback_chunk_by_lines
            assert len(result) == 1
            assert result[0]["node_type"] == "line_chunk"

    def test_extract_chunk_complete(self):
        """Test extracting a chunk from a node."""
        content = "def test():\n    pass\n"
        b_content = content.encode("utf8")
        extracted_text = "def test():\n    pass"
        start_byte = b_content.find(extracted_text.encode("utf8"))
        end_byte = start_byte + len(extracted_text.encode("utf8"))

        mock_node = Mock()
        mock_node.start_byte = start_byte
        mock_node.end_byte = end_byte
        mock_node.type = "function_definition"

        with patch("uuid.uuid4", return_value="1234"):
            chunk = extract_chunk(mock_node, content, "python")

        assert chunk["id"] == "chunk:1234"
        assert chunk["text"] == "def test():\n    pass"
        assert chunk["language"] == "python"
        assert chunk["node_type"] == "function_definition"
        assert chunk["line_start"] == 1
        assert chunk["line_end"] == 2

    def test_chunk_by_treesitter_traverse(self):
        """Test the AST traversal extraction."""
        content = "def test():\n    pass"
        
        mock_parser = Mock()
        mock_tree = Mock()
        mock_root = Mock()
        
        # Root node
        mock_root.type = "module"
        
        # Child node (a function)
        mock_child = Mock()
        mock_child.type = "function_definition"
        mock_child.children = []
        mock_child.start_byte = 0
        mock_child.end_byte = len(content.encode("utf8"))
        
        mock_root.children = [mock_child]
        mock_tree.root_node = mock_root
        mock_parser.parse.return_value = mock_tree

        with patch("synapse.chunking.treesitter.get_parser", return_value=mock_parser):
            chunks = chunk_by_treesitter(content, ".py")

        assert len(chunks) == 1
        assert chunks[0]["node_type"] == "function_definition"
        assert chunks[0]["language"] == "python"

    def test_fallback_chunk_by_lines_with_overlap(self):
        """Test fallback_chunk_by_lines overlap math."""
        content = "1\n2\n3\n4\n5\n6"
        chunks = fallback_chunk_by_lines(content, chunk_size=3, overlap=1)
        
        assert len(chunks) == 3
        assert chunks[0]["text"] == "1\n2\n3"
        assert chunks[1]["text"] == "3\n4\n5"
        assert chunks[2]["text"] == "5\n6"
        
        assert chunks[0]["line_start"] == 1
        assert chunks[0]["line_end"] == 3
        
        assert chunks[1]["line_start"] == 3
        assert chunks[1]["line_end"] == 5
        
        assert chunks[2]["line_start"] == 5
        assert chunks[2]["line_end"] == 6

    def test_fallback_chunk_by_lines_empty(self):
        """Test fallback line chunker with empty input."""
        chunks = fallback_chunk_by_lines("")
        # According to its implementation, split("\n") on "" returns [""]
        # which creates 1 chunk if not stripped beforehand
        assert len(chunks) == 1
        assert chunks[0]["text"] == ""
