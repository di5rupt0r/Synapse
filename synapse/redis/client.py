"""SynapseRedis — Wrapper que implementa interface esperada pelos handlers MCP."""

import json
import struct
import time
from typing import Any, Dict, List, Optional

import redis
from redis.commands.search.query import Query


class SynapseRedis:
    """Wrapper sobre redis.Redis com métodos de alto nível."""

    INDEX_NAME = "synapse_idx"
    NODE_PREFIX = "node:"

    def __init__(self, redis_client: Any) -> None:
        """Initialize with raw Redis client (sync or async)."""
        self._client = redis_client
        self._is_async = hasattr(redis_client, 'async_execute') or 'redis.asyncio' in str(type(redis_client))

    # ---- Low-level pass-through ----
    def ping(self) -> bool:  # pragma: no cover
        return self._client.ping()

    def close(self) -> None:  # pragma: no cover
        self._client.close()

    # ---- Node CRUD ----
    def store_node(
        self,
        node_id: str,
        domain: str,
        node_type: str,
        content: str,
        embedding: List[float],
        metadata: Optional[Dict] = None,
        links: Optional[Dict] = None,
    ) -> str:
        """Store node as RedisJSON. Returns node_id."""
        node = {
            "id": node_id,
            "domain": domain,
            "type": node_type,
            "content": content,
            "embedding": embedding,
            "metadata": metadata or {},
            "links": links or {"inbound": [], "outbound": []},
            "created_at": time.time(),
        }
        
        # Store as JSON document
        self._client.json().set(node_id, "$", node)
        
        # Ensure the document is indexed by checking index info
        # This forces RediSearch to update the index for the new document
        try:
            self._client.ft(self.INDEX_NAME).info()
        except Exception:
            # If index doesn't exist, it will be created on server startup
            pass
        
        return node_id

    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get node by ID. Returns None if not found."""
        try:
            data = self._client.json().get(node_id)
            if isinstance(data, list) and data:
                return data[0]
            return data
        except Exception:  # pragma: no cover
            return None

    def update_node(self, node_id: str, operations: List[Dict]) -> bool:
        """Apply JSON patch operations to node."""
        node = self.get_node(node_id)
        if not node:
            return False

        for op in operations:
            path = op["path"]  # e.g. "$.metadata.foo"
            action = op["op"]  # set|delete|append
            value = op.get("value")

            # Convert path for json().set (remove leading $.)
            json_path = path if path.startswith("$") else f"$.{path}"

            if action == "set":
                self._client.json().set(node_id, json_path, value)
            elif action == "delete":
                self._client.json().delete(node_id, json_path)
            elif action == "append":
                current = self._client.json().get(node_id, json_path)
                if isinstance(current, list):
                    current.append(value)
                    self._client.json().set(node_id, json_path, current)

        return True

    # ---- Hybrid Search ----
    def search_hybrid(
        self,
        query: str,
        embedding: Optional[List[float]] = None,
        domain_filter: Optional[List[str]] = None,
        type_filter: Optional[List[str]] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Hybrid search: KNN (dense) + BM25 (sparse) → RRF fusion."""
        # Build RediSearch query
        q_parts = []

        # Domain filter
        if domain_filter:
            domains = "|".join(domain_filter)
            q_parts.append(f"@domain:{{{domains}}}")

        # Type filter
        if type_filter:
            types = "|".join(type_filter)
            q_parts.append(f"@type:{{{types}}}")

        # BM25 text search
        if query:
            q_parts.append(f"(@content:{query}) | (@chunk_text:{query})")

        filter_str = " ".join(q_parts) if q_parts else "*"

        # KNN search if embedding provided
        if embedding:
            query_obj = (
                Query(f"{filter_str}=>[KNN {limit} @embedding $vec AS score]")
                .sort_by("score")
                .return_fields("id", "domain", "type", "content", "score")
                .dialect(2)
                .paging(0, limit)
            )
            try:
                results = self._client.ft(self.INDEX_NAME).search(
                    query_obj, query_params={"vec": self._float_to_bytes(embedding)}
                )
                return [self._doc_to_dict(doc) for doc in results.docs]
            except Exception:  # nosec B110: Intentional silent fail for KNN search fallback
                pass

        # Fallback: pure BM25
        query_obj = (
            Query(filter_str)
            .return_fields("id", "domain", "type", "content")
            .paging(0, limit)
        )
        try:
            results = self._client.ft(self.INDEX_NAME).search(query_obj)
            return [self._doc_to_dict(doc) for doc in results.docs]
        except Exception:
            return []

    # ---- Graph Traversal ----
    def get_linked_nodes(
        self,
        node_id: str,
        direction: str = "both",
        depth: int = 1,
    ) -> List[Dict[str, Any]]:
        """Get nodes linked to given node via inbound/outbound edges."""
        node = self.get_node(node_id)
        if not node:
            return []

        linked_ids = set()
        links = node.get("links", {})

        if direction in ("both", "inbound"):
            linked_ids.update(links.get("inbound", []))
        if direction in ("both", "outbound"):
            linked_ids.update(links.get("outbound", []))

        # Fetch linked nodes
        linked_nodes = []
        for lid in linked_ids:
            ln = self.get_node(lid)
            if ln:
                linked_nodes.append(ln)

        return linked_nodes

    # ---- Helpers ----
    def _float_to_bytes(self, vec: List[float]) -> bytes:
        """Convert float list to bytes for RediSearch KNN."""

        return struct.pack(f"{len(vec)}f", *vec)

    def _doc_to_dict(self, doc) -> Dict[str, Any]:
        """Convert RediSearch Document to dict."""
        d = doc.__dict__
        # Parse JSON content if present
        if "json" in d and isinstance(d["json"], str):
            try:
                d["json"] = json.loads(d["json"])
            except Exception:  # nosec B110: JSON parsing failure is acceptable, use raw value
                pass
        return d
