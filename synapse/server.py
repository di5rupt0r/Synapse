"""FastAPI Server for Synapse AKG."""

import time
from contextlib import asynccontextmanager

import redis.asyncio as redis_async
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .config import get_settings
from .embeddings.cache import EmbeddingCache
from .embeddings.unixcoder import UniXCoderBackend
from .index.setup import IndexManager
from .mcp_server import initialize as init_mcp
from .mcp_server import mcp
from .redis.client import SynapseRedis

settings = get_settings()

# Global instances
synapse_redis: SynapseRedis | None = None
embedding_cache: EmbeddingCache | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):  # pragma: no cover
    """Manage application lifespan."""
    global synapse_redis, embedding_cache

    # Startup
    settings = get_settings()

    # Initialize Redis
    redis_url = f"redis://{settings.redis_host}:{settings.redis_port}"
    redis_client = redis_async.from_url(
        redis_url,
        decode_responses=True,
        socket_connect_timeout=5,
        socket_keepalive=True,
    )
    await redis_client.ping()

    # Initialize SynapseRedis wrapper
    synapse_redis = SynapseRedis(redis_client)

    # Initialize embedding backend with cache
    embedding_backend = UniXCoderBackend()
    embedding_cache = EmbeddingCache(embedding_backend, max_size=settings.cache_size)

    # Setup RediSearch index
    index_manager = IndexManager(redis_client)
    index_manager.ensure_index()

    # Initialize MCP server with dependencies
    init_mcp(synapse_redis, embedding_cache)

    try:
        yield
    finally:
        # Shutdown
        if synapse_redis:
            await synapse_redis.close()


# Initialize FastAPI app
app = FastAPI(
    title="Synapse AKG",
    description="Agentic Knowledge Graph with Hybrid Search",
    version="0.1.0",
    lifespan=lifespan,
)

# Mount FastMCP as ASGI sub-application at /mcp
# This provides full MCP protocol support:
#   POST /mcp  → initialize, tools/list, tools/call (JSON-RPC 2.0)
app.mount("/mcp", mcp.streamable_http_app())


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Test Redis connection
        synapse_redis.ping()

        # Test embedding backend
        test_embedding = embedding_cache.embed("test")
        settings = get_settings()

        return {
            "status": "healthy",
            "timestamp": time.time(),
            "embedding_model": settings.embedding_model,
            "embedding_dim": len(test_embedding) if test_embedding else 0,
            "services": {
                "redis": "connected",
                "embedding": "available",
                "cache_stats": embedding_cache.get_stats() if embedding_cache else None,
                "mcp": "running",
            },
        }
    except Exception as e:  # pragma: no cover
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": str(e), "timestamp": time.time()},
        )


@app.get("/metrics")
async def metrics_endpoint():
    """Metrics endpoint."""
    try:
        # Get Redis info
        redis_info = await synapse_redis._client.info()

        # Get index stats
        try:
            index_info = await synapse_redis._client.ft("synapse_idx").info()
            index_stats = {
                "num_docs": index_info.get("num_docs", 0),
                "max_doc_id": index_info.get("max_doc_id", 0),
                "num_terms": index_info.get("num_terms", 0),
                "num_records": index_info.get("num_records", 0),
            }
        except Exception:  # pragma: no cover
            index_stats = {"error": "Index not available"}

        # Get cache stats
        cache_stats = embedding_cache.get_stats() if embedding_cache else {}

        return {
            "timestamp": time.time(),
            "redis": {
                "connected_clients": redis_info.get("connected_clients", 0),
                "used_memory": redis_info.get("used_memory_human", "0B"),
                "total_commands_processed": redis_info.get(
                    "total_commands_processed", 0
                ),
            },
            "index": index_stats,
            "cache": cache_stats,
        }

    except Exception as e:  # pragma: no cover
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.exception_handler(Exception)
async def global_exception_handler(
    request: Request, exc: Exception
):  # pragma: no cover
    """Global exception handler."""
    return JSONResponse(
        status_code=500, content={"error": "Internal server error", "detail": str(exc)}
    )


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "synapse.server:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )
