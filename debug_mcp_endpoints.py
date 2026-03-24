"""Debug FastMCP redirect issue."""

import asyncio
import json
import logging
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
import sys

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_mcp_endpoints():
    """Test different MCP endpoints to identify redirect issues."""
    
    # Mock dependencies to avoid import issues
    torch_mock = MagicMock()
    torch_mock.__version__ = "2.0.0"
    torch_mock.tensor = lambda x: x
    
    transformers_mock = MagicMock()
    transformers_mock.AutoModel = MagicMock()
    transformers_mock.AutoTokenizer = MagicMock()
    
    numpy_mock = MagicMock()
    numpy_mock.array = lambda x: x
    
    # Mock pydantic root_model to fix MCP import issues
    pydantic_root_model_mock = MagicMock()
    
    with patch.dict('sys.modules', {
        'torch': torch_mock,
        'transformers': transformers_mock,
        'numpy': numpy_mock,
        'pydantic.root_model': pydantic_root_model_mock
    }):
        try:
            # Import after patching
            from synapse.server import app
            
            client = TestClient(app)
            
            # Test different endpoints
            endpoints = [
                "/mcp",
                "/mcp/",
                "/health"  # Control endpoint
            ]
            
            results = {}
            
            for endpoint in endpoints:
                print(f"\n=== Testing {endpoint} ===")
                
                # Test GET request
                try:
                    resp = client.get(endpoint)
                    results[f"{endpoint}_GET"] = {
                        "status_code": resp.status_code,
                        "headers": dict(resp.headers),
                        "content": resp.text[:200] if resp.text else None
                    }
                    print(f"GET {endpoint}: {resp.status_code}")
                    if resp.status_code >= 300:
                        print(f"  Location: {resp.headers.get('location', 'No location')}")
                except Exception as e:
                    results[f"{endpoint}_GET"] = {"error": str(e)}
                    print(f"GET {endpoint}: ERROR - {e}")
                
                # Test POST request (MCP protocol uses POST)
                try:
                    test_payload = {
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "initialize",
                        "params": {
                            "protocolVersion": "2024-11-05",
                            "capabilities": {},
                            "clientInfo": {"name": "test", "version": "1.0"}
                        }
                    }
                    resp = client.post(endpoint, json=test_payload)
                    results[f"{endpoint}_POST"] = {
                        "status_code": resp.status_code,
                        "headers": dict(resp.headers),
                        "content": resp.text[:200] if resp.text else None
                    }
                    print(f"POST {endpoint}: {resp.status_code}")
                    if resp.status_code >= 300:
                        print(f"  Location: {resp.headers.get('location', 'No location')}")
                except Exception as e:
                    results[f"{endpoint}_POST"] = {"error": str(e)}
                    print(f"POST {endpoint}: ERROR - {e}")
            
            # Print summary
            print(f"\n=== SUMMARY ===")
            for key, result in results.items():
                print(f"{key}: {result}")
            
            return results
            
        except Exception as e:
            print(f"Failed to import or test: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e)}


if __name__ == "__main__":
    test_mcp_endpoints()
