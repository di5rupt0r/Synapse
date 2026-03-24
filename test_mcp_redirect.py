"""Simple test to check MCP redirect without complex imports."""

import requests
import json


def test_mcp_redirect():
    """Test MCP endpoint redirect by making actual HTTP requests."""
    
    base_url = "http://localhost:8000"
    
    # Test different endpoints
    endpoints = [
        "/mcp",
        "/mcp/",
        "/health"  # Control endpoint
    ]
    
    results = {}
    
    for endpoint in endpoints:
        print(f"\n=== Testing {endpoint} ===")
        url = f"{base_url}{endpoint}"
        
        # Test GET request
        try:
            resp = requests.get(url, timeout=5)
            results[f"{endpoint}_GET"] = {
                "status_code": resp.status_code,
                "headers": dict(resp.headers),
                "content": resp.text[:200] if resp.text else None
            }
            print(f"GET {endpoint}: {resp.status_code}")
            if resp.status_code >= 300:
                print(f"  Location: {resp.headers.get('location', 'No location')}")
            if resp.status_code == 200:
                print(f"  Response: {resp.text[:100]}...")
        except requests.exceptions.ConnectionError:
            results[f"{endpoint}_GET"] = {"error": "Connection refused - server not running"}
            print(f"GET {endpoint}: Connection refused - server not running")
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
            resp = requests.post(url, json=test_payload, timeout=5)
            results[f"{endpoint}_POST"] = {
                "status_code": resp.status_code,
                "headers": dict(resp.headers),
                "content": resp.text[:200] if resp.text else None
            }
            print(f"POST {endpoint}: {resp.status_code}")
            if resp.status_code >= 300:
                print(f"  Location: {resp.headers.get('location', 'No location')}")
            if resp.status_code == 200:
                print(f"  Response: {resp.text[:100]}...")
        except requests.exceptions.ConnectionError:
            results[f"{endpoint}_POST"] = {"error": "Connection refused - server not running"}
            print(f"POST {endpoint}: Connection refused - server not running")
        except Exception as e:
            results[f"{endpoint}_POST"] = {"error": str(e)}
            print(f"POST {endpoint}: ERROR - {e}")
    
    # Print summary
    print(f"\n=== SUMMARY ===")
    for key, result in results.items():
        print(f"{key}: {result}")
    
    return results


if __name__ == "__main__":
    test_mcp_redirect()
