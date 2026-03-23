"""Tests for chunking/treesitter.py - Complete coverage."""

from unittest.mock import Mock, patch

import pytest


class TestTreeSitterChunkingComplete:
    """Complete tree-sitter chunking coverage."""

    def test_chunk_by_treesitter_empty_content(self):
        """Test chunking empty content."""
        from synapse.chunking.treesitter import chunk_by_treesitter

        result = chunk_by_treesitter("", ".py")
        assert result == []

        result = chunk_by_treesitter("   \n  \t  ", ".py")
        assert result == []

    def test_chunk_by_treesitter_unsupported_extension(self):
        """Test chunking with unsupported extension."""
        from synapse.chunking.treesitter import chunk_by_treesitter

        with pytest.raises(ValueError, match="Unsupported file extension: .xyz"):
            chunk_by_treesitter("content", ".xyz")

    def test_chunk_by_treesitter_parsing_failure(self):
        """Test chunking when parsing fails."""
        from synapse.chunking.treesitter import chunk_by_treesitter

        content = "line1\nline2\nline3"

        with patch("synapse.chunking.treesitter.get_parser") as mock_get_parser:
            mock_get_parser.side_effect = Exception("Parsing failed")

            result = chunk_by_treesitter(content, ".py")

            # Should return line-based chunks
            assert len(result) > 0
            assert result[0]["language"] == "unknown"
            assert result[0]["node_type"] == "line_chunk"

    def test_chunk_by_treesitter_success_with_chunks(self):
        """Test successful chunking with found chunks."""
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

    def test_chunk_by_treesitter_no_chunks_found(self):
        """Test chunking when no chunks are found."""
        from synapse.chunking.treesitter import chunk_by_treesitter

        content = "just a comment"

        with patch("synapse.chunking.treesitter.get_parser") as mock_get_parser:
            mock_parser = Mock()
            mock_tree = Mock()
            mock_root = Mock()

            mock_parser.parse.return_value = mock_tree
            mock_tree.root_node = mock_root
            mock_root.children = []
            mock_root.type = "comment"

            mock_get_parser.return_value = mock_parser

            result = chunk_by_treesitter(content, ".py")

            assert result == []

    def test_get_parser_tree_sitter_unavailable(self):
        """Test get_parser when tree-sitter is not available."""
        from synapse.chunking.treesitter import get_parser

        with patch("synapse.chunking.treesitter.Language", None):
            with pytest.raises(ImportError, match="tree-sitter not available"):
                get_parser("python")

    def test_get_parser_language_import_success(self):
        """Test get_parser with successful language import."""
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

    def test_get_parser_language_import_failure_python_fallback_success(self):
        """Test get_parser with language import failure but Python fallback success."""
        from synapse.chunking.treesitter import get_parser

        with (
            patch("synapse.chunking.treesitter.Language"),
            patch("synapse.chunking.treesitter.Parser"),
            patch("builtins.__import__") as mock_import,
        ):
            # First import fails
            mock_import.side_effect = ImportError("Language not found")

            # Mock tree_sitter_python import
            with patch("synapse.chunking.treesitter.tree_sitter_python") as mock_python:
                mock_python.language.return_value = "python_lang"

                result = get_parser("python")

                assert result is not None

    def test_get_parser_complete_failure(self):
        """Test get_parser complete failure."""
        from synapse.chunking.treesitter import get_parser

        with (
            patch("synapse.chunking.treesitter.Language"),
            patch("builtins.__import__") as mock_import,
        ):
            mock_import.side_effect = ImportError("Language not found")

            with patch("synapse.chunking.treesitter.tree_sitter_python") as mock_python:
                mock_python.side_effect = ImportError("Python not available")

                with pytest.raises(
                    ImportError, match="Tree-sitter language python not available"
                ):
                    get_parser("python")

    def test_extract_chunk_complete(self):
        """Test extract_chunk complete functionality."""
        from synapse.chunking.treesitter import extract_chunk

        content = "line1\nline2\ndef test():\n    pass\nline5"
        mock_node = Mock()
        mock_node.start_byte = len("line1\nline2\n")
        mock_node.end_byte = len("line1\nline2\ndef test():\n    ")
        mock_node.type = "function_definition"

        result = extract_chunk(mock_node, content, "python")

        assert result["id"].startswith("chunk:")
        assert result["text"] == "def test():\n    "
        assert result["language"] == "python"
        assert result["node_type"] == "function_definition"
        assert result["line_start"] == 3
        assert result["line_end"] == 4

    def test_extract_chunk_single_line(self):
        """Test extract_chunk with single line."""
        from synapse.chunking.treesitter import extract_chunk

        content = "def test(): pass"
        mock_node = Mock()
        mock_node.start_byte = 0
        mock_node.end_byte = len("def test(): pass")
        mock_node.type = "function_definition"

        result = extract_chunk(mock_node, content, "python")

        assert result["line_start"] == 1
        assert result["line_end"] == 1

    def test_extract_chunk_empty_text(self):
        """Test extract_chunk with empty text."""
        from synapse.chunking.treesitter import extract_chunk

        content = "line1\nline2\nline3"
        mock_node = Mock()
        mock_node.start_byte = 5
        mock_node.end_byte = 5  # Same as start, empty
        mock_node.type = "empty"

        result = extract_chunk(mock_node, content, "python")

        assert result["text"] == ""

    def test_fallback_chunk_by_lines_basic(self):
        """Test fallback line-based chunking basic functionality."""
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
        """Test fallback chunking with overlap."""
        from synapse.chunking.treesitter import fallback_chunk_by_lines

        content = "line1\nline2\nline3\nline4\nline5\nline6"

        result = fallback_chunk_by_lines(content, chunk_size=3, overlap=1)

        assert len(result) == 2
        assert result[0]["text"] == "line1\nline2\nline3"
        assert result[1]["text"] == "line3\nline4\nline5\nline6"

    def test_fallback_chunk_by_lines_single_line(self):
        """Test fallback chunking with single line."""
        from synapse.chunking.treesitter import fallback_chunk_by_lines

        content = "single line"

        result = fallback_chunk_by_lines(content, chunk_size=5, overlap=1)

        assert len(result) == 1
        assert result[0]["text"] == "single line"
        assert result[0]["line_start"] == 1
        assert result[0]["line_end"] == 1

    def test_fallback_chunk_by_lines_empty_content(self):
        """Test fallback chunking with empty content."""
        from synapse.chunking.treesitter import fallback_chunk_by_lines

        content = ""

        result = fallback_chunk_by_lines(content, chunk_size=2, overlap=0)

        assert len(result) == 1
        assert result[0]["text"] == ""

    def test_fallback_chunk_by_lines_large_overlap(self):
        """Test fallback chunking with large overlap."""
        from synapse.chunking.treesitter import fallback_chunk_by_lines

        content = "line1\nline2\nline3"

        result = fallback_chunk_by_lines(content, chunk_size=3, overlap=2)

        assert len(result) == 1
        assert result[0]["text"] == "line1\nline2\nline3"

    def test_extension_mapping_comprehensive(self):
        """Test extension mapping comprehensively."""
        from synapse.chunking.treesitter import EXTENSION_MAP

        # Test common extensions
        test_cases = [
            (".py", "python"),
            (".js", "javascript"),
            (".ts", "typescript"),
            (".jsx", "javascript"),
            (".tsx", "typescript"),
            (".java", "java"),
            (".cpp", "cpp"),
            (".c", "c"),
            (".h", "c"),
            (".hpp", "cpp"),
            (".cs", "c_sharp"),
            (".go", "go"),
            (".rs", "rust"),
            (".rb", "ruby"),
            (".php", "php"),
            (".swift", "swift"),
            (".kt", "kotlin"),
            (".scala", "scala"),
            (".r", "r"),
            (".m", "objc"),
            (".sh", "bash"),
            (".sql", "sql"),
            (".html", "html"),
            (".css", "css"),
            (".json", "json"),
            (".yaml", "yaml"),
            (".yml", "yaml"),
            (".xml", "xml"),
            (".md", "markdown"),
        ]

        for ext, expected_lang in test_cases:
            assert EXTENSION_MAP[ext] == expected_lang

    def test_chunk_node_types_comprehensive(self):
        """Test chunk node types set comprehensively."""
        from synapse.chunking.treesitter import CHUNK_NODE_TYPES

        expected_types = {
            "function_definition",
            "function_declaration",
            "class_definition",
            "class_declaration",
            "method_definition",
            "block",
            "statement_block",
            "for_statement",
            "while_statement",
            "if_statement",
            "switch_statement",
            "try_statement",
            "with_statement",
            "async_function_definition",
            "async_method_definition",
            "lambda_function",
            "decorated_definition",
        }

        assert CHUNK_NODE_TYPES == expected_types

    def test_chunk_by_treesitter_nested_structure(self):
        """Test chunking with nested structure."""
        from synapse.chunking.treesitter import chunk_by_treesitter

        content = """
class MyClass:
    def method1(self):
        pass

    def method2(self):
        def inner_function():
            pass
        pass
"""

        with (
            patch("synapse.chunking.treesitter.get_parser") as mock_get_parser,
            patch("synapse.chunking.treesitter.extract_chunk") as mock_extract,
        ):
            mock_parser = Mock()
            mock_tree = Mock()
            mock_root = Mock()
            mock_class = Mock()
            mock_method1 = Mock()
            mock_method2 = Mock()
            mock_inner = Mock()

            # Set up nested structure
            mock_parser.parse.return_value = mock_tree
            mock_tree.root_node = mock_root
            mock_root.children = [mock_class]
            mock_class.type = "class_definition"
            mock_class.children = [mock_method1, mock_method2]
            mock_method1.type = "method_definition"
            mock_method1.children = []
            mock_method2.type = "method_definition"
            mock_method2.children = [mock_inner]
            mock_inner.type = "function_definition"
            mock_inner.children = []

            mock_get_parser.return_value = mock_parser
            mock_extract.return_value = {"id": "test", "text": "test"}

            result = chunk_by_treesitter(content, ".py")

            # Should find all functions/methods
            assert len(result) == 3
            assert mock_extract.call_count == 3

    def test_chunk_by_treesitter_mixed_node_types(self):
        """Test chunking with mixed node types."""
        from synapse.chunking.treesitter import chunk_by_treesitter

        content = """
def function():
    pass

class MyClass:
    pass

if True:
    pass
"""

        with (
            patch("synapse.chunking.treesitter.get_parser") as mock_get_parser,
            patch("synapse.chunking.treesitter.extract_chunk") as mock_extract,
        ):
            mock_parser = Mock()
            mock_tree = Mock()
            mock_root = Mock()
            mock_function = Mock()
            mock_class = Mock()
            mock_if = Mock()
            mock_comment = Mock()  # Should not be chunked

            mock_parser.parse.return_value = mock_tree
            mock_tree.root_node = mock_root
            mock_root.children = [mock_function, mock_class, mock_if, mock_comment]

            mock_function.type = "function_definition"
            mock_function.children = []
            mock_class.type = "class_definition"
            mock_class.children = []
            mock_if.type = "if_statement"
            mock_if.children = []
            mock_comment.type = "comment"
            mock_comment.children = []

            mock_get_parser.return_value = mock_parser
            mock_extract.return_value = {"id": "test", "text": "test"}

            result = chunk_by_treesitter(content, ".py")

            # Should only chunk the specified types
            assert len(result) == 3
            assert mock_extract.call_count == 3

    def test_import_error_handling(self):
        """Test import error handling for tree-sitter."""
        # Test that the module handles missing tree-sitter gracefully
        from synapse.chunking import treesitter

        # Should have fallback imports
        assert hasattr(treesitter, "Language")
        assert hasattr(treesitter, "Parser")
        assert hasattr(treesitter, "Node")

        # Should have the constants
        assert hasattr(treesitter, "EXTENSION_MAP")
        assert hasattr(treesitter, "CHUNK_NODE_TYPES")
