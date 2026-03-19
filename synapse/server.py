"""FastAPI Server for Synapse AKG."""

import asyncio
import time
from contextlib import asynccontextmanager
from typing import Any, Dict

import redis.asyncio as redis_async
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from .config import get_settings
from .embeddings.cache import EmbeddingCache
from .embeddings.unixcoder import UniXCoderBackend
from .index.setup import IndexManager
from .mcp_discovery import MCPDiscovery
from .mcp_server import initialize as init_mcp, mcp
from .redis.client import SynapseRedis

settings = get_settings()

# Global instances
synapse_redis: SynapseRedis | None = None
embedding_cache: EmbeddingCache | None = None
mcp_discovery: MCPDiscovery | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan."""
    global synapse_redis, embedding_cache, mcp_discovery

    # Startup
    settings = get_settings()

    # Initialize Redis
    redis_url = f"redis://{settings.redis_host}:{settings.redis_port}"
    redis_client = redis_async.from_url(
        redis_url,
        decode_responses=True,
        socket_connect_timeout=5,
        socket_keepalive=True
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

    # Initialize MCP Discovery
    mcp_discovery = MCPDiscovery(synapse_redis)

    # Auto-register synapse server
    server_info = {
        "name": "synapse",
        "description": "Synapse AKG Server - Agentic Knowledge Graph",
        "version": "0.1.0",
        "capabilities": ["memorize", "recall", "patch", "hybrid-search"],
        "endpoints": ["/mcp", "/health", "/metrics"],
        "transport": "MCP-HTTP"
    }
    mcp_discovery.register_server("synapse", server_info)

    # Start FastMCP in background task
    mcp_task = asyncio.create_task(
        mcp.run_sse(host="0.0.0.0", port=8080)
    )

    try:
        yield
    finally:
        # Shutdown
        mcp_task.cancel()
        try:
            await mcp_task
        except asyncio.CancelledError:
            pass
        if synapse_redis:
            await synapse_redis.close()


# Initialize FastAPI app
app = FastAPI(
    title="Synapse AKG",
    description="Agentic Knowledge Graph with Hybrid Search",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Test Redis connection
        synapse_redis.ping()

        # Test embedding backend
        test_embedding = embedding_cache.embed("test")
        settings = get_settings()

        health_data = {
            "status": "healthy",
            "timestamp": time.time(),
            "embedding_model": settings.embedding_model,
            "embedding_dim": len(test_embedding) if test_embedding else 0,
            "services": {
                "redis": "connected",
                "embedding": "available",
                "cache_stats": embedding_cache.get_stats() if embedding_cache else None,
            },
        }
        
        # Update MCP discovery health data
        if mcp_discovery:
            mcp_discovery.update_health("synapse", health_data)

        return health_data
    except Exception as e:
        error_data = {"status": "unhealthy", "error": str(e), "timestamp": time.time()}
        
        # Update MCP discovery health data on error
        if mcp_discovery:
            mcp_discovery.update_health("synapse", error_data)
            
        return JSONResponse(status_code=503, content=error_data)


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
        except Exception:
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

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/mcp/servers")
async def list_mcp_servers():
    """List all registered MCP servers."""
    try:
        servers = mcp_discovery.list_servers()
        return {
            "servers": servers,
            "count": len(servers),
            "timestamp": time.time()
        }
    except Exception as e:
        return JSONResponse(
            status_code=500, 
            content={"error": "Failed to list servers", "detail": str(e)}
        )


@app.get("/mcp/server/{server_name}")
async def get_mcp_server_info(server_name: str):
    """Get detailed information about a specific MCP server."""
    try:
        server_info = mcp_discovery.get_server_info(server_name)
        
        if not server_info:
            return JSONResponse(
                status_code=404,
                content={"error": f"Server '{server_name}' not found"}
            )
        
        return server_info
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to get server info", "detail": str(e)}
        )


@app.get("/mcp/server/{server_name}/tools")
async def get_mcp_server_tools(server_name: str):
    """Get available tools for a specific MCP server."""
    try:
        tools = mcp_discovery.get_server_tools(server_name)
        return {
            "tools": tools,
            "count": len(tools),
            "server": server_name,
            "timestamp": time.time()
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to get server tools", "detail": str(e)}
        )


@app.post("/mcp/server/{server_name}/register")
async def register_mcp_server(server_name: str, request: Dict[str, Any]):
    """Register a new MCP server."""
    try:
        success = mcp_discovery.register_server(server_name, request)
        
        if success:
            return {
                "status": "registered",
                "server": server_name,
                "timestamp": time.time()
            }
        else:
            return JSONResponse(
                status_code=500,
                content={"error": "Failed to register server"}
            )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to register server", "detail": str(e)}
        )


@app.post("/mcp/server/{server_name}/health")
async def update_server_health(server_name: str, health_data: Dict[str, Any]):
    """Update health status for a specific server."""
    try:
        success = mcp_discovery.update_health(server_name, health_data)
        
        if success:
            return {
                "status": "updated",
                "server": server_name,
                "timestamp": time.time()
            }
        else:
            return JSONResponse(
                status_code=500,
                content={"error": "Failed to update health"}
            )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to update health", "detail": str(e)}
        )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    return JSONResponse(
        status_code=500, content={"error": "Internal server error", "detail": str(exc)}
    )


# For testing purposes
def create_test_client():
    """Create test client for the app."""
    return TestClient(app)


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "synapse.server:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )
