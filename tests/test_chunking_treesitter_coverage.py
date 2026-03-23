"""Tests for chunking/treesitter.py - 100% Coverage."""

from unittest.mock import Mock, patch

import pytest


class TestTreeSitterChunking:
    """Test tree-sitter chunking functionality."""

    def test_chunk_by_treesitter_empty_content(self):
        """Test chunking empty content returns empty list."""
        from synapse.chunking.treesitter import chunk_by_treesitter

        result = chunk_by_treesitter("", ".py")
        assert result == []

        result = chunk_by_treesitter("   \n  \t  ", ".py")
        assert result == []

    def test_chunk_by_treesitter_unsupported_extension(self):
        """Test chunking with unsupported extension raises error."""
        from synapse.chunking.treesitter import chunk_by_treesitter

        with pytest.raises(ValueError, match="Unsupported file extension: .xyz"):
            chunk_by_treesitter("content", ".xyz")

    def test_chunk_by_treesitter_parsing_failure_fallback(self):
        """Test parsing failure falls back to line chunking."""
        from synapse.chunking.treesitter import chunk_by_treesitter

        content = "line1\nline2\nline3\nline4\nline5\nline6"

        with patch("synapse.chunking.treesitter.get_parser") as mock_get_parser:
            mock_get_parser.side_effect = Exception("Parsing failed")

            result = chunk_by_treesitter(content, ".py")

            # Should return line-based chunks
            assert len(result) > 0
            assert result[0]["language"] == "unknown"
            assert result[0]["node_type"] == "line_chunk"

    def test_chunk_by_treesitter_success(self):
        """Test successful chunking with tree-sitter."""
        from synapse.chunking.treesitter import chunk_by_treesitter

        content = "def test(): pass"

        with (
            patch("synapse.chunking.treesitter.get_parser") as mock_get_parser,
            patch("synapse.chunking.treesitter.extract_chunk") as mock_extract,
        ):
            mock_parser = Mock()
            mock_tree = Mock()
            mock_root = Mock()
            mock_node = Mock()

            mock_parser.parse.return_value = mock_tree
            mock_tree.root_node = mock_root
            mock_root.children = [mock_node]
            mock_node.type = "function_definition"
            mock_node.children = []

            mock_get_parser.return_value = mock_parser
            mock_extract.return_value = {"id": "test", "text": "test"}

            result = chunk_by_treesitter(content, ".py")

            assert len(result) == 1
            mock_extract.assert_called_once()

    def test_get_parser_tree_sitter_not_available(self):
        """Test get_parser when tree-sitter is not available."""
        from synapse.chunking.treesitter import get_parser

        with patch("synapse.chunking.treesitter.Language", None):
            with pytest.raises(ImportError, match="tree-sitter not available"):
                get_parser("python")

    def test_get_parser_language_not_available(self):
        """Test get_parser when specific language is not available."""
        from synapse.chunking.treesitter import get_parser

        with (
            patch("synapse.chunking.treesitter.Language"),
            patch("synapse.chunking.treesitter.Parser") as mock_parser,
            patch("builtins.__import__") as mock_import,
        ):
            mock_import.side_effect = ImportError("Language not found")

            with patch("synapse.chunking.treesitter.tree_sitter_python") as mock_python:
                mock_python.language.return_value = "python_lang"

                result = get_parser("unknown_lang")

                assert result is not None
                mock_parser.assert_called_once()

    def test_get_parser_python_fallback(self):
        """Test get_parser falls back to Python when language unavailable."""
        from synapse.chunking.treesitter import get_parser

        with (
            patch("synapse.chunking.treesitter.Language"),
            patch("synapse.chunking.treesitter.Parser"),
            patch("builtins.__import__") as mock_import,
        ):
            # First import fails, Python fallback succeeds
            mock_import.side_effect = [ImportError("Language not found"), None]

            with patch("synapse.chunking.treesitter.tree_sitter_python") as mock_python:
                mock_python.language.return_value = "python_lang"

                result = get_parser("unknown_lang")

                assert result is not None

    def test_get_parser_complete_failure(self):
        """Test get_parser when no language is available."""
        from synapse.chunking.treesitter import get_parser

        with (
            patch("synapse.chunking.treesitter.Language"),
            patch("builtins.__import__") as mock_import,
        ):
            mock_import.side_effect = ImportError("Language not found")

            with patch("synapse.chunking.treesitter.tree_sitter_python") as mock_python:
                mock_python.side_effect = ImportError("Python not available")

                with pytest.raises(
                    ImportError, match="Tree-sitter language unknown_lang not available"
                ):
                    get_parser("unknown_lang")

    def test_get_parser_success(self):
        """Test successful parser creation."""
        from synapse.chunking.treesitter import get_parser

        with (
            patch("synapse.chunking.treesitter.Language"),
            patch("synapse.chunking.treesitter.Parser") as mock_parser,
            patch("builtins.__import__") as mock_import,
        ):
            mock_lang_module = Mock()
            mock_lang_module.language.return_value = "test_lang"
            mock_import.return_value = mock_lang_module

            result = get_parser("python")

            assert result is not None
            mock_parser.assert_called_once()

    def test_extract_chunk(self):
        """Test chunk extraction from node."""
        from synapse.chunking.treesitter import extract_chunk

        content = "line1\nline2\nline3\ndef test():\n    pass"
        mock_node = Mock()
        mock_node.start_byte = 20
        mock_node.end_byte = 35
        mock_node.type = "function_definition"

        result = extract_chunk(mock_node, content, "python")

        assert result["id"].startswith("chunk:")
        assert result["text"] == "def test():\n    "
        assert result["language"] == "python"
        assert result["node_type"] == "function_definition"
        assert result["line_start"] == 4
        assert result["line_end"] == 4

    def test_fallback_chunk_by_lines_basic(self):
        """Test basic line-based fallback chunking."""
        from synapse.chunking.treesitter import fallback_chunk_by_lines

        content = "line1\nline2\nline3\nline4\nline5"

        result = fallback_chunk_by_lines(content, chunk_size=2, overlap=0)

        assert len(result) == 3  # 5 lines / 2 per chunk = 3 chunks
        assert result[0]["text"] == "line1\nline2"
        assert result[1]["text"] == "line3\nline4"
        assert result[2]["text"] == "line5"
        assert result[0]["line_start"] == 1
        assert result[0]["line_end"] == 2

    def test_fallback_chunk_by_lines_with_overlap(self):
        """Test line-based chunking with overlap."""
        from synapse.chunking.treesitter import fallback_chunk_by_lines

        content = "line1\nline2\nline3\nline4\nline5\nline6"

        result = fallback_chunk_by_lines(content, chunk_size=3, overlap=1)

        assert len(result) == 2
        assert result[0]["text"] == "line1\nline2\nline3"
        assert result[1]["text"] == "line3\nline4\nline5\nline6"  # Overlap line3

    def test_fallback_chunk_by_lines_empty_chunk(self):
        """Test fallback chunking handles empty chunks gracefully."""
        from synapse.chunking.treesitter import fallback_chunk_by_lines

        content = "line1"

        # Large chunk size that would create empty chunks
        result = fallback_chunk_by_lines(content, chunk_size=10, overlap=5)

        assert len(result) == 1
        assert result[0]["text"] == "line1"

    def test_extension_mapping(self):
        """Test file extension mapping."""
        from synapse.chunking.treesitter import EXTENSION_MAP

        # Test common extensions
        assert EXTENSION_MAP[".py"] == "python"
        assert EXTENSION_MAP[".js"] == "javascript"
        assert EXTENSION_MAP[".ts"] == "typescript"
        assert EXTENSION_MAP[".java"] == "java"
        assert EXTENSION_MAP[".cpp"] == "cpp"
        assert EXTENSION_MAP[".go"] == "go"
        assert EXTENSION_MAP[".rs"] == "rust"

        # Test case insensitivity
        assert EXTENSION_MAP[".PY"] == "python"  # Will be lowercased in usage

    def test_chunk_node_types(self):
        """Test chunk node types set."""
        from synapse.chunking.treesitter import CHUNK_NODE_TYPES

        # Test important node types are included
        assert "function_definition" in CHUNK_NODE_TYPES
        assert "class_definition" in CHUNK_NODE_TYPES
        assert "method_definition" in CHUNK_NODE_TYPES
        assert "if_statement" in CHUNK_NODE_TYPES
        assert "for_statement" in CHUNK_NODE_TYPES

    def test_chunk_by_treesitter_traverse_function(self):
        """Test the traverse function in chunk_by_treesitter."""
        from synapse.chunking.treesitter import chunk_by_treesitter

        content = "def outer():\n    def inner():\n        pass"

        with (
            patch("synapse.chunking.treesitter.get_parser") as mock_get_parser,
            patch("synapse.chunking.treesitter.extract_chunk") as mock_extract,
        ):
            mock_parser = Mock()
            mock_tree = Mock()
            mock_root = Mock()
            mock_outer = Mock()
            mock_inner = Mock()

            # Set up nested structure
            mock_parser.parse.return_value = mock_tree
            mock_tree.root_node = mock_root
            mock_root.children = [mock_outer]
            mock_outer.type = "function_definition"
            mock_outer.children = [mock_inner]
            mock_inner.type = "function_definition"
            mock_inner.children = []

            mock_get_parser.return_value = mock_parser
            mock_extract.return_value = {"id": "test", "text": "test"}

            result = chunk_by_treesitter(content, ".py")

            # Should find both functions
            assert len(result) == 2
            assert mock_extract.call_count == 2
