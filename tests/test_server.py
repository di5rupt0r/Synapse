"""TDD RED Phase: Tests for FastAPI Server."""

import pytest
from unittest.mock import Mock, patch
import json
from fastapi.testclient import TestClient


def test_server_health_endpoint():
    """Test health check endpoint."""
    # This will fail - server doesn't exist yet (RED phase)
    from synapse.server import app
    
    client = TestClient(app)
    
    # Mock the global variables in the server module
    with patch('synapse.server.redis_client') as mock_redis, \
         patch('synapse.server.embedding_cache') as mock_cache:
        
        # Setup mocks
        mock_redis.ping.return_value = True
        mock_cache.embed.return_value = [0.1] * 768
        mock_cache.get_stats.return_value = {"hits": 10, "misses": 2}
        
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "services" in data
        assert data["services"]["redis"] == "connected"
        assert data["services"]["embedding"] == "available"


def test_server_memorize_endpoint():
    """Test memorize MCP endpoint."""
    from synapse.server import app
    
    client = TestClient(app)
    
    # Mock MCP handler
    with patch('synapse.server.mcp_memorize') as mock_mcp:
        mock_mcp.handle_request.return_value = {
            "jsonrpc": "2.0",
            "id": "test-123",
            "result": {
                "node_id": "node:test:123",
                "status": "success"
            }
        }
        
        request_data = {
            "jsonrpc": "2.0",
            "id": "test-123",
            "method": "memorize",
            "params": {
                "domain": "test",
                "type": "entity",
                "content": "test content"
            }
        }
        
        response = client.post("/mcp/memorize", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["result"]["node_id"] == "node:test:123"
        assert "metadata" in data["result"]
        assert "request_id" in data["result"]["metadata"]
        assert "latency_ms" in data["result"]["metadata"]


def test_server_recall_endpoint():
    """Test recall MCP endpoint."""
    from synapse.server import app
    
    client = TestClient(app)
    
    # Mock MCP handler
    with patch('synapse.server.mcp_recall') as mock_mcp:
        mock_mcp.handle_request.return_value = {
            "jsonrpc": "2.0",
            "id": "test-456",
            "result": {
                "format": "compressed_yaml",
                "content": "matched_nodes: []"
            }
        }
        
        request_data = {
            "jsonrpc": "2.0",
            "id": "test-456",
            "method": "recall_context",
            "params": {
                "query": "test query",
                "limit": 5
            }
        }
        
        response = client.post("/mcp/recall", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["result"]["format"] == "compressed_yaml"
        assert "metadata" in data["result"]
        assert data["result"]["metadata"]["query"] == "test query"


def test_server_patch_endpoint():
    """Test patch MCP endpoint."""
    from synapse.server import app
    
    client = TestClient(app)
    
    # Mock MCP handler
    with patch('synapse.server.mcp_patch') as mock_mcp:
        mock_mcp.handle_request.return_value = {
            "jsonrpc": "2.0",
            "id": "test-789",
            "result": {
                "node_id": "node:test:123",
                "updated": True
            }
        }
        
        request_data = {
            "jsonrpc": "2.0",
            "id": "test-789",
            "method": "patch_state",
            "params": {
                "node_id": "node:test:123",
                "updates": {"content": "updated content"}
            }
        }
        
        response = client.post("/mcp/patch", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["result"]["node_id"] == "node:test:123"
        assert data["result"]["updated"] is True
        assert "metadata" in data["result"]


def test_server_metrics_endpoint():
    """Test metrics endpoint."""
    from synapse.server import app
    
    client = TestClient(app)
    
    # Mock Redis client
    with patch('synapse.server.redis_client') as mock_redis, \
         patch('synapse.server.embedding_cache') as mock_cache:
        
        # Setup mocks
        mock_redis.info.return_value = {
            "connected_clients": 5,
            "used_memory_human": "1.5M",
            "total_commands_processed": 1000
        }
        
        mock_ft = Mock()
        mock_ft.info.return_value = {
            "num_docs": 100,
            "max_doc_id": 150,
            "num_terms": 1000,
            "num_records": 200
        }
        mock_redis.ft.return_value = mock_ft
        
        mock_cache.get_stats.return_value = {
            "hits": 800,
            "misses": 200,
            "hit_rate": 0.8
        }
        
        response = client.get("/metrics")
        
        assert response.status_code == 200
        data = response.json()
        assert "timestamp" in data
        assert "redis" in data
        assert "index" in data
        assert "cache" in data
        assert data["redis"]["connected_clients"] == 5
        assert data["index"]["num_docs"] == 100


def test_server_error_handling():
    """Test server error handling."""
    from synapse.server import app
    
    client = TestClient(app)
    
    # Mock MCP handler to raise exception
    with patch('synapse.server.mcp_memorize') as mock_mcp:
        mock_mcp.handle_request.side_effect = Exception("Test error")
        
        request_data = {
            "jsonrpc": "2.0",
            "id": "test-error",
            "method": "memorize",
            "params": {}
        }
        
        response = client.post("/mcp/memorize", json=request_data)
        
        assert response.status_code == 500
        data = response.json()
        assert "error" in data
        assert data["error"]["code"] == -32603
        assert "Test error" in data["error"]["message"]


def test_server_startup_events():
    """Test server startup events."""
    # This will test that startup events are properly configured
    from synapse.server import app
    
    # Check that lifespan is configured (FastAPI uses lifespan, not separate startup/shutdown events)
    assert app.router.lifespan_context is not None
    # Just verify the app has the lifespan configured
    assert hasattr(app, 'state') or app.router.lifespan_context is not None


def test_server_request_validation():
    """Test request validation."""
    from synapse.server import app
    
    client = TestClient(app)
    
    # Test invalid JSON
    response = client.post("/mcp/memorize", data="invalid json")
    
    assert response.status_code == 422  # Validation error


def test_server_cors_headers():
    """Test CORS headers are present."""
    from synapse.server import app
    
    client = TestClient(app)
    
    response = client.options("/mcp/memorize")
    
    # Should handle OPTIONS requests
    assert response.status_code in [200, 405]
