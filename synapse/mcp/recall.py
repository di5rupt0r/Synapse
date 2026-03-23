"""MCP Recall Handler - Hybrid Search + Graph Resolution."""

from textwrap import shorten
from typing import Any, Dict


class MCPRecall:
    """MCP handler for recall context operations."""

    def __init__(self, redis_client: Any, embedding_service: Any) -> None:
        """Initialize with Redis client and embedding service."""
        self.redis = redis_client
        self.embeddings = embedding_service

    def handle_recall(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle recall request: embed → search → resolve → compress."""
        try:
            # Validate required fields
            self._validate_params(params)

            # Extract parameters
            query = params["query"]
            domain_filter = params.get("domain_filter")
            type_filter = params.get("type_filter")
            depth = params.get("depth", 1)
            limit = params.get("limit", 10)

            # Hybrid search
            search_results = self.redis.search_hybrid(
                query=query,
                embedding=self.embeddings.embed(query),
                domain_filter=domain_filter,
                type_filter=type_filter,
                limit=limit,
            )

            # Convert to matched nodes
            matched_nodes = []
            for result in search_results:
                node_data = result.get("json", result)
                matched_node = {
                    "id": node_data.get("id", ""),
                    "domain": node_data.get("domain", ""),
                    "type": node_data.get("type", ""),
                    "content_summary": shorten(node_data.get("content", ""), width=100),
                }
                matched_nodes.append(matched_node)

            # Graph resolution (1-degree depth)
            resolved_edges = []
            if depth > 0:
                for node in matched_nodes:
                    linked_nodes = self.redis.get_linked_nodes(node["id"], depth=depth)
                    for linked in linked_nodes:
                        edge = {
                            "source": node["id"],
                            "target": linked.get("id", ""),
                            "relation_type": "linked",
                        }
                        resolved_edges.append(edge)

            return {
                "results": matched_nodes,
                "total": len(matched_nodes),
                "query_time_ms": 0,
            }

        except Exception as e:
            return {"format": "error", "content": str(e)}

    def _validate_params(self, params: Dict[str, Any]) -> None:
        """Validate recall request parameters."""
        if "query" not in params:
            raise ValueError("Missing required field: query")

        query = params["query"]
        if not query or not isinstance(query, str):
            raise ValueError("Query must be a non-empty string")

        # Validate optional parameters
        if "limit" in params:
            limit = params["limit"]
            if not isinstance(limit, int) or limit < 1 or limit > 50:
                raise ValueError("Limit must be an integer between 1 and 50")

        if "depth" in params:
            depth = params["depth"]
            if not isinstance(depth, int) or depth < 1 or depth > 3:
                raise ValueError("Depth must be an integer between 1 and 3")
