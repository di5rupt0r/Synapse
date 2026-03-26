#!/usr/bin/env python3
"""
Test script to verify the recall fix for codebase_rag_project domain.
"""

import json
import requests
import time

def test_recall_fix():
    """Test recall with codebase_rag_project domain."""
    
    base_url = "http://localhost:8000"
    
    print("🔧 Testing Recall Fix for codebase_rag_project domain")
    print("=" * 60)
    
    # Test 1: String domain parameter (should coerce to list)
    print("\n1. Testing string domain parameter...")
    payload1 = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": "recall",
            "arguments": {
                "query": "test",
                "domain": "codebase_rag_project"
            }
        }
    }
    
    try:
        response1 = requests.post(f"{base_url}/mcp", json=payload1, timeout=10)
        result1 = response1.json()
        print(f"✅ String domain test: {response1.status_code}")
        if "result" in result1 and "results" in result1["result"]:
            print(f"   Found {result1['result']['total']} results")
        else:
            print(f"   Error: {result1}")
    except Exception as e:
        print(f"❌ String domain test failed: {e}")
    
    # Test 2: List domain parameter (correct format)
    print("\n2. Testing list domain parameter...")
    payload2 = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/call",
        "params": {
            "name": "recall",
            "arguments": {
                "query": "test",
                "domain": ["codebase_rag_project"]
            }
        }
    }
    
    try:
        response2 = requests.post(f"{base_url}/mcp", json=payload2, timeout=10)
        result2 = response2.json()
        print(f"✅ List domain test: {response2.status_code}")
        if "result" in result2 and "results" in result2["result"]:
            print(f"   Found {result2['result']['total']} results")
        else:
            print(f"   Error: {result2}")
    except Exception as e:
        print(f"❌ List domain test failed: {e}")
    
    # Test 3: Multiple domains
    print("\n3. Testing multiple domains...")
    payload3 = {
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {
            "name": "recall",
            "arguments": {
                "query": "test",
                "domain": ["codebase_rag_project", "copilot"]
            }
        }
    }
    
    try:
        response3 = requests.post(f"{base_url}/mcp", json=payload3, timeout=10)
        result3 = response3.json()
        print(f"✅ Multiple domains test: {response3.status_code}")
        if "result" in result3 and "results" in result3["result"]:
            print(f"   Found {result3['result']['total']} results")
        else:
            print(f"   Error: {result3}")
    except Exception as e:
        print(f"❌ Multiple domains test failed: {e}")
    
    # Test 4: No domain filter (should work without regression)
    print("\n4. Testing no domain filter...")
    payload4 = {
        "jsonrpc": "2.0",
        "id": 4,
        "method": "tools/call",
        "params": {
            "name": "recall",
            "arguments": {
                "query": "test"
            }
        }
    }
    
    try:
        response4 = requests.post(f"{base_url}/mcp", json=payload4, timeout=10)
        result4 = response4.json()
        print(f"✅ No domain filter test: {response4.status_code}")
        if "result" in result4 and "results" in result4["result"]:
            print(f"   Found {result4['result']['total']} results")
        else:
            print(f"   Error: {result4}")
    except Exception as e:
        print(f"❌ No domain filter test failed: {e}")
    
    print("\n" + "=" * 60)
    print("🎯 Recall fix testing completed!")
    print("Check server logs for DEBUG output to see query generation.")

if __name__ == "__main__":
    test_recall_fix()
