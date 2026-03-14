"""FastAPI Server for Synapse AKG."""

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
import time
import uuid
from typing import Dict, Any
import redis
from contextlib import asynccontextmanager

from .config import get_settings
from .embeddings.unixcoder import UniXCoderBackend
from .embeddings.cache import EmbeddingCache
from .mcp.memorize import MCPMemorize
from .mcp.recall import MCPRecall
from .mcp.patch import MCPPatch
from .index.setup import IndexManager

# Global instances
redis_client = None
embedding_backend = None
embedding_cache = None
mcp_memorize = None
mcp_recall = None
mcp_patch = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan."""
    global redis_client, embedding_backend, embedding_cache
    global mcp_memorize, mcp_recall, mcp_patch
    
    # Startup
    settings = get_settings()
    
    # Initialize Redis
    redis_client = redis.Redis(
        host=settings.redis_host,
        port=settings.redis_port,
        decode_responses=True
    )
    
    # Initialize embedding backend with cache
    embedding_backend = UniXCoderBackend()
    embedding_cache = EmbeddingCache(embedding_backend, max_size=settings.cache_size)
    
    # Setup RediSearch index
    index_manager = IndexManager(redis_client)
    index_manager.ensure_index()
    
    # Initialize MCP handlers
    mcp_memorize = MCPMemorize(redis_client, embedding_cache)
    mcp_recall = MCPRecall(redis_client, embedding_cache)
    mcp_patch = MCPPatch(redis_client)
    
    yield
    
    # Shutdown
    if redis_client:
        redis_client.close()

# Initialize FastAPI app
app = FastAPI(
    title="Synapse AKG",
    description="Agentic Knowledge Graph with Hybrid Search",
    version="0.1.0",
    lifespan=lifespan
)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Test Redis connection
        redis_client.ping()
        
        # Test embedding backend
        test_embedding = embedding_cache.embed("test")
        
        return {
            "status": "healthy",
            "timestamp": time.time(),
            "services": {
                "redis": "connected",
                "embedding": "available",
                "cache_stats": embedding_cache.get_stats() if embedding_cache else None
            }
        }
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e)
            }
        )


@app.post("/mcp/memorize")
async def memorize_endpoint(request: Dict[str, Any]):
    """MCP memorize endpoint - store knowledge."""
    start_time = time.time()
    
    try:
        # Add request ID for tracking
        request["request_id"] = str(uuid.uuid4())
        
        # Process through MCP handler
        result = mcp_memorize.handle_request(request)
        
        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000
        
        # Add metadata
        if "result" in result:
            result["result"]["metadata"] = {
                "request_id": request["request_id"],
                "latency_ms": round(latency_ms, 2),
                "timestamp": time.time()
            }
        
        return result
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "jsonrpc": "2.0",
                "id": request.get("id", ""),
                "error": {
                    "code": -32603,
                    "message": str(e)
                }
            }
        )


@app.post("/mcp/recall")
async def recall_endpoint(request: Dict[str, Any]):
    """MCP recall endpoint - hybrid search."""
    start_time = time.time()
    
    try:
        # Add request ID for tracking
        request["request_id"] = str(uuid.uuid4())
        
        # Process through MCP handler
        result = mcp_recall.handle_request(request)
        
        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000
        
        # Add metadata
        if "result" in result:
            result["result"]["metadata"] = {
                "request_id": request["request_id"],
                "latency_ms": round(latency_ms, 2),
                "timestamp": time.time(),
                "query": request.get("params", {}).get("query", "")
            }
        
        return result
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "jsonrpc": "2.0",
                "id": request.get("id", ""),
                "error": {
                    "code": -32603,
                    "message": str(e)
                }
            }
        )


@app.post("/mcp/patch")
async def patch_endpoint(request: Dict[str, Any]):
    """MCP patch endpoint - atomic mutations."""
    start_time = time.time()
    
    try:
        # Add request ID for tracking
        request["request_id"] = str(uuid.uuid4())
        
        # Process through MCP handler
        result = mcp_patch.handle_request(request)
        
        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000
        
        # Add metadata
        if "result" in result:
            result["result"]["metadata"] = {
                "request_id": request["request_id"],
                "latency_ms": round(latency_ms, 2),
                "timestamp": time.time()
            }
        
        return result
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "jsonrpc": "2.0",
                "id": request.get("id", ""),
                "error": {
                    "code": -32603,
                    "message": str(e)
                }
            }
        )


@app.get("/metrics")
async def metrics_endpoint():
    """Metrics endpoint."""
    try:
        # Get Redis info
        redis_info = redis_client.info()
        
        # Get index stats
        try:
            index_info = redis_client.ft("synapse_idx").info()
            index_stats = {
                "num_docs": index_info.get("num_docs", 0),
                "max_doc_id": index_info.get("max_doc_id", 0),
                "num_terms": index_info.get("num_terms", 0),
                "num_records": index_info.get("num_records", 0)
            }
        except:
            index_stats = {"error": "Index not available"}
        
        # Get cache stats
        cache_stats = embedding_cache.get_stats() if embedding_cache else {}
        
        return {
            "timestamp": time.time(),
            "redis": {
                "connected_clients": redis_info.get("connected_clients", 0),
                "used_memory": redis_info.get("used_memory_human", "0B"),
                "total_commands_processed": redis_info.get("total_commands_processed", 0)
            },
            "index": index_stats,
            "cache": cache_stats
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc)
        }
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
        reload=settings.debug
    )
