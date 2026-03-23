"""MCP Server implementation using FastMCP."""

from mcp.server.fastmcp import FastMCP

from synapse.config import get_settings
from synapse.embeddings.cache import EmbeddingCache
from synapse.mcp.memorize import MCPMemorize
from synapse.mcp.patch import MCPPatch
from synapse.mcp.recall import MCPRecall
from synapse.redis.client import SynapseRedis

settings = get_settings()

# Create FastMCP server
mcp = FastMCP(
    name="synapse",
    instructions="Synapse AKG Server - Agentic Knowledge Graph with Hybrid Search",
)

# Global instances (initialized in app lifespan)
synapse_redis: SynapseRedis | None = None
embedding_cache: EmbeddingCache | None = None


@mcp.tool()
async def memorize(
    domain: str,
    type: str,
    content: str,
    metadata: dict | None = None,
    links: dict | None = None,
) -> dict:
    """
    Store a node in the knowledge graph.

    Args:
        domain: Domain/namespace for the node (e.g., "github", "docs")
        type: Node type (e.g., "entity", "chunk", "relation")
        content: Text content to store and embed
        metadata: Optional metadata dict
        links: Optional links dict with "inbound"/"outbound" node IDs

    Returns:
        dict: {"id": "node:domain:uuid", "status": "success"}
    """
    if not synapse_redis or not embedding_cache:
        raise RuntimeError("MCP server not initialized")

    handler = MCPMemorize(synapse_redis, embedding_cache)
    return handler.handle_memorize(
        {
            "domain": domain,
            "type": type,
            "content": content,
            "metadata": metadata or {},
            "links": links or {"inbound": [], "outbound": []},
        }
    )


@mcp.tool()
async def recall(
    query: str,
    domain: list[str] | None = None,
    type: list[str] | None = None,
    limit: int = 10,
    include_embedding: bool = False,
) -> dict:
    """
    Hybrid search across the knowledge graph.

    Args:
        query: Search query text
        domain: Filter by domain(s)
        type: Filter by node type(s)
        limit: Max number of results (default 10)
        include_embedding: Include embedding vectors in response

    Returns:
        dict: {"results": [...], "total": N, "query_time_ms": X}
    """
    if not synapse_redis:
        raise RuntimeError("MCP server not initialized")

    handler = MCPRecall(synapse_redis, embedding_cache)
    return handler.handle_recall(
        {
            "query": query,
            "domain_filter": domain,
            "type_filter": type,
            "limit": limit,
            "include_embedding": include_embedding,
        }
    )


@mcp.tool()
async def patch(node_id: str, operations: list[dict]) -> dict:
    """
    Apply JSON Patch operations to update a node.

    Args:
        node_id: ID of node to update
        operations: List of patch ops: {"op":"set|delete|append","path":"$.field","value":...}

    Returns:
        dict: {"status":"success|error","message":str,"node_id":str}
    """
    if not synapse_redis:
        raise RuntimeError("MCP server not initialized")

    handler = MCPPatch(synapse_redis)
    return handler.handle_patch({"node_id": node_id, "operations": operations})


def initialize(redis_client: SynapseRedis, cache: EmbeddingCache) -> None:
    """Initialize MCP server with dependencies."""
    global synapse_redis, embedding_cache
    synapse_redis = redis_client
    embedding_cache = cache
