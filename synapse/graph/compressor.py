"""Graph YAML Compressor - Token-efficient output formatting."""

import re
from textwrap import shorten
from typing import Any, Dict

import yaml


class GraphCompressor:
    """Compresses graph data into token-efficient YAML format."""

    def __init__(self, max_content_length: int = 50) -> None:
        """Initialize compressor with content length limit."""
        self.max_content_length = max_content_length
        self.stopwords = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "from",
            "up",
            "about",
            "into",
            "through",
            "during",
            "before",
            "after",
            "above",
            "below",
            "between",
            "under",
            "along",
            "following",
            "across",
            "behind",
            "beyond",
            "plus",
            "except",
            "but",
            "yet",
            "so",
            "nor",
            "as",
        }

    def compress_yaml(self, graph_data: Dict[str, Any]) -> str:
        """Compress graph data to token-efficient YAML."""
        # Process nodes
        compressed_nodes = []
        for node in graph_data.get("matched_nodes", []):
            compressed_node = {
                "id": node["id"],
                "domain": node["domain"],
                "type": node["type"],
                "content": self._compress_content(node.get("content", "")),
            }
            compressed_nodes.append(compressed_node)

        # Process edges
        compressed_edges = []
        for edge in graph_data.get("resolved_edges", []):
            compressed_edge = {
                "source": edge["source"],
                "target": edge["target"],
                "relation_type": edge["relation_type"],
            }
            compressed_edges.append(compressed_edge)

        # Create compressed data structure
        compressed_data = {
            "matched_nodes": compressed_nodes,
            "resolved_edges": compressed_edges,
        }

        # Convert to YAML with inline formatting for token efficiency
        yaml_str = yaml.dump(
            compressed_data,
            default_flow_style=True,
            sort_keys=False,
            allow_unicode=True,
        )

        # Further compression: remove extra whitespace and newlines
        compressed_yaml = yaml_str.replace("\n", " ").replace("  ", " ")

        return self._remove_stopwords(compressed_yaml)

    def _compress_content(self, content: str) -> str:
        """Compress content to telegraphic format."""
        if not content:
            return ""

        # Remove stopwords first, then truncate
        words = content.split()
        key_words = [
            word
            for word in words
            if word.lower() not in self.stopwords and len(word) > 2
        ]

        if key_words:
            # Take only first few key words for maximum compression
            compressed = " ".join(key_words[:5])
            if len(key_words) > 5:
                compressed += "..."
        else:
            # Fallback to simple truncation if no key words
            compressed = shorten(
                content, width=self.max_content_length, placeholder="..."
            )

        return compressed

    def _remove_stopwords(self, text: str) -> str:
        """Remove stopwords from compressed text."""
        # This is a simple approach - remove common stopwords with spaces around them
        for stopword in self.stopwords:
            # Remove with word boundaries
            pattern = r"\b" + re.escape(stopword) + r"\b"
            text = re.sub(pattern, "", text, flags=re.IGNORECASE)

        # Clean up extra spaces
        text = re.sub(r"\s+", " ", text).strip()

        return text
