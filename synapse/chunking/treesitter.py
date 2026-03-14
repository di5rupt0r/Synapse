"""Tree-sitter based code chunking."""

from typing import List, Dict, Any
import uuid

try:
    from tree_sitter import Language, Parser, Node
except ImportError:
    # Fallback for testing without tree-sitter
    Language = None
    Parser = None
    Node = None


# Extension to language mapping (copied from codebase-rag)
EXTENSION_MAP = {
    ".py": "python",
    ".js": "javascript", 
    ".ts": "typescript",
    ".jsx": "javascript",
    ".tsx": "typescript",
    ".java": "java",
    ".cpp": "cpp",
    ".c": "c",
    ".h": "c",
    ".hpp": "cpp",
    ".cs": "c_sharp",
    ".go": "go",
    ".rs": "rust",
    ".rb": "ruby",
    ".php": "php",
    ".swift": "swift",
    ".kt": "kotlin",
    ".scala": "scala",
    ".r": "r",
    ".m": "objc",
    ".sh": "bash",
    ".sql": "sql",
    ".html": "html",
    ".css": "css",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".xml": "xml",
    ".md": "markdown",
}

# Node types that should be chunked (copied from codebase-rag)
CHUNK_NODE_TYPES = {
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


def chunk_by_treesitter(content: str, extension: str) -> List[Dict[str, Any]]:
    """Chunk code using tree-sitter AST parsing."""
    if not content.strip():
        return []
    
    # Get language from extension
    language = EXTENSION_MAP.get(extension.lower())
    if not language:
        raise ValueError(f"Unsupported file extension: {extension}")
    
    # Parse the code
    try:
        parser = get_parser(language)
        tree = parser.parse(bytes(content, "utf8"))
    except Exception as e:
        # Fallback to simple line-based chunking if parsing fails
        return fallback_chunk_by_lines(content)
    
    # Find chunk nodes
    chunks = []
    root = tree.root_node
    
    def traverse(node: Node, depth: int = 0) -> None:
        if node.type in CHUNK_NODE_TYPES:
            chunk = extract_chunk(node, content, language)
            if chunk:
                chunks.append(chunk)
        
        # Recursively traverse children
        for child in node.children:
            traverse(child, depth + 1)
    
    traverse(root)
    
    return chunks


def get_parser(language: str) -> Parser:
    """Get tree-sitter parser for the given language."""
    if Language is None or Parser is None:
        raise ImportError("tree-sitter not available")
    
    try:
        # Try to import the language
        lang_module = __import__(f"tree_sitter_{language}", fromlist=[language])
        lang = Language(lang_module.language())
    except (ImportError, AttributeError):
        # Fallback to Python parser if language not available
        try:
            import tree_sitter_python
            lang = Language(tree_sitter_python.language())
        except ImportError:
            raise ImportError(f"Tree-sitter language {language} not available")
    
    return Parser(lang)


def extract_chunk(node: Node, content: str, language: str) -> Dict[str, Any]:
    """Extract chunk information from a tree-sitter node."""
    # Get the text content
    start_byte = node.start_byte
    end_byte = node.end_byte
    text = content.encode("utf8")[start_byte:end_byte].decode("utf8")
    
    # Calculate line numbers
    lines_before = content[:start_byte].count('\n')
    line_start = lines_before + 1
    lines_in_chunk = text.count('\n')
    line_end = line_start + lines_in_chunk
    
    return {
        "id": f"chunk:{uuid.uuid4()}",
        "text": text.strip(),
        "language": language,
        "node_type": node.type,
        "line_start": line_start,
        "line_end": line_end,
    }


def fallback_chunk_by_lines(content: str, chunk_size: int = 50, overlap: int = 5) -> List[Dict[str, Any]]:
    """Fallback line-based chunking when tree-sitter fails."""
    lines = content.split('\n')
    chunks = []
    
    for i in range(0, len(lines), chunk_size - overlap):
        chunk_lines = lines[i:i + chunk_size]
        if not chunk_lines:
            continue
            
        chunk_text = '\n'.join(chunk_lines)
        
        chunks.append({
            "id": f"chunk:{uuid.uuid4()}",
            "text": chunk_text,
            "language": "unknown",
            "node_type": "line_chunk",
            "line_start": i + 1,
            "line_end": min(i + chunk_size, len(lines)),
        })
    
    return chunks
