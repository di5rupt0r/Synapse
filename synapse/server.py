"""FastAPI Server for Synapse AKG."""

import json
import time
from contextlib import asynccontextmanager

import redis.asyncio as redis_async
import redis
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from mcp.types import JSONRPCRequest, JSONRPCResponse

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
    redis_client = redis.from_url(
        redis_url,
        decode_responses=True,
        socket_connect_timeout=5,
        socket_keepalive=True,
    )
    redis_client.ping()

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

# Note: FastMCP's streamable_http_app() cannot be mounted in FastAPI due to task group issues
# We need to create a custom MCP endpoint that handles the JSON-RPC protocol directly


@app.post("/mcp")
async def mcp_endpoint(request: Request) -> Response:
    """Custom MCP endpoint that handles JSON-RPC protocol."""
    try:
        # Parse JSON-RPC request
        body = await request.json()
        
        # Handle different JSON-RPC methods
        if "method" not in body:
            return Response(
                content=json.dumps({"jsonrpc": "2.0", "error": {"code": -32600, "message": "Invalid Request"}}),
                status_code=400,
                media_type="application/json"
            )
        
        method = body["method"]
        request_id = body.get("id")
        params = body.get("params", {})
        
        # Initialize response
        response = {"jsonrpc": "2.0", "id": request_id}
        
        if method == "initialize":
            response["result"] = {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {},
                    "logging": {}
                },
                "serverInfo": {
                    "name": "synapse",
                    "version": "0.1.0"
                }
            }
        elif method == "tools/list":
            # Get tools from FastMCP instance
            tools_list = await mcp.list_tools()
            tools = []
            for tool in tools_list:
                tools.append({
                    "name": tool.name,
                    "description": tool.description or f"Tool: {tool.name}",
                    "inputSchema": tool.inputSchema or {"type": "object", "properties": {}}
                })
            response["result"] = {"tools": tools}
        elif method == "tools/call":
            tool_name = params.get("name")
            arguments = params.get("arguments", {})
            
            # Call the tool using FastMCP's call_tool method
            try:
                result = await mcp.call_tool(tool_name, arguments)
                
                # Handle different result types from FastMCP
                if isinstance(result, list):
                    # Raw list returned - convert to MCP format
                    response["result"] = {"content": [{"type": "text", "text": str(item)} for item in result]}
                elif hasattr(result, 'data') and result.data is not None:
                    # CallToolResult with data
                    response["result"] = {"content": [{"type": "text", "text": str(result.data)}]}
                else:
                    # Fallback to content array or convert raw result
                    if hasattr(result, 'content'):
                        response["result"] = {"content": result.content or []}
                    else:
                        # Convert raw result to content
                        response["result"] = {"content": [{"type": "text", "text": str(result)}]}
            except Exception as e:
                response["error"] = {"code": -32603, "message": str(e)}
        else:
            response["error"] = {"code": -32601, "message": f"Method '{method}' not found"}
        
        return Response(
            content=json.dumps(response),
            media_type="application/json"
        )
        
    except Exception as e:
        return Response(
            content=json.dumps({
                "jsonrpc": "2.0", 
                "error": {"code": -32603, "message": f"Internal error: {str(e)}"}
            }),
            status_code=500,
            media_type="application/json"
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
