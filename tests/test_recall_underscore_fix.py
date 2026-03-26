#!/usr/bin/env python3
"""
TDD Tests for Synapse Recall Fix - RED → GREEN → REFACTOR
Tests for underscore escaping bug fix in RediSearch queries.
"""

import pytest
import json
from unittest.mock import Mock, patch
from io import StringIO
import sys
from synapse.redis.client import SynapseRedis
from synapse.mcp_server import recall


class TestRecallUnderscoreFix:
    """TDD test suite for underscore escaping fix."""

    def test_domain_filter_no_escaping(self):
        """RED: Test that underscores are NOT escaped in domain filters."""
        # Setup
        mock_client = Mock()
        mock_ft = Mock()
        mock_client.ft.return_value = mock_ft
        
        redis = SynapseRedis(mock_client)
        
        # Capture debug output
        captured_output = StringIO()
        with patch('sys.stdout', captured_output):
            # Test domain filter with underscores
            domain_filter = ["codebase_rag_project"]
            query = "test"
            
            # Execute
            redis.search_hybrid(
                query=query,
                domain_filter=domain_filter,
                limit=10
            )
        
        # Check debug output
        output = captured_output.getvalue()
        assert "DEBUG: domain_filter=['codebase_rag_project']" in output
        assert "DEBUG: final_query=@domain:{codebase_rag_project}" in output
        # Should NOT contain escaped underscores
        assert "codebase\\_rag\\_project" not in output

    def test_type_filter_no_escaping(self):
        """RED: Test that underscores are NOT escaped in type filters."""
        # Setup
        mock_client = Mock()
        mock_ft = Mock()
        mock_client.ft.return_value = mock_ft
        
        redis = SynapseRedis(mock_client)
        
        # Capture debug output
        captured_output = StringIO()
        with patch('sys.stdout', captured_output):
            # Test type filter with underscores
            type_filter = ["test_type"]
            query = "test"
            
            # Execute
            redis.search_hybrid(
                query=query,
                type_filter=type_filter,
                limit=10
            )
        
        # Check debug output
        output = captured_output.getvalue()
        assert "DEBUG: type_filter=['test_type']" in output
        assert "DEBUG: final_query=@type:{test_type}" in output
        # Should NOT contain escaped underscores
        assert "test\\_type" not in output

    def test_multiple_domains_no_escaping(self):
        """RED: Test multiple domains with underscores are not escaped."""
        # Setup
        mock_client = Mock()
        mock_ft = Mock()
        mock_client.ft.return_value = mock_ft
        
        redis = SynapseRedis(mock_client)
        
        # Capture debug output
        captured_output = StringIO()
        with patch('sys.stdout', captured_output):
            # Test multiple domain filters
            domain_filter = ["codebase_rag_project", "test_domain"]
            query = "test"
            
            # Execute
            redis.search_hybrid(
                query=query,
                domain_filter=domain_filter,
                limit=10
            )
        
        # Check debug output
        output = captured_output.getvalue()
        assert "DEBUG: domain_filter=['codebase_rag_project', 'test_domain']" in output
        assert "DEBUG: final_query=@domain:{codebase_rag_project|test_domain}" in output
        # Should NOT contain escaped underscores
        assert "codebase\\_rag\\_project" not in output

    @patch('synapse.mcp_server.synapse_redis')
    @patch('synapse.mcp_server.embedding_cache')
    def test_recall_string_domain_coercion(self, mock_cache, mock_redis):
        """RED: Test string domain parameter is coerced to list."""
        # Setup
        mock_handler = Mock()
        mock_handler.handle_recall.return_value = {"results": [], "total": 0}
        
        # Capture debug output
        captured_output = StringIO()
        with patch('sys.stdout', captured_output):
            with patch('synapse.mcp_server.MCPRecall', return_value=mock_handler):
                # Test with string domain (should coerce to list)
                recall(
                    query="test",
                    domain="codebase_rag_project",  # String, not list
                    limit=10
                )
        
        # Check debug output shows coercion
        output = captured_output.getvalue()
        assert "DEBUG: Coerced string domain to list: ['codebase_rag_project']" in output
        
        # Verify handler was called with list
        mock_handler.handle_recall.assert_called_once()
        call_args = mock_handler.handle_recall.call_args[0][0]
        
        assert call_args["domain_filter"] == ["codebase_rag_project"]

    @patch('synapse.mcp_server.synapse_redis')
    @patch('synapse.mcp_server.embedding_cache')
    def test_recall_list_domain_unchanged(self, mock_cache, mock_redis):
        """GREEN: Test list domain parameter works correctly."""
        # Setup
        mock_handler = Mock()
        mock_handler.handle_recall.return_value = {"results": [], "total": 0}
        
        with patch('synapse.mcp_server.MCPRecall', return_value=mock_handler):
            # Test with list domain (should remain unchanged)
            domain_list = ["codebase_rag_project", "copilot"]
            recall(
                query="test",
                domain=domain_list,  # Already a list
                limit=10
            )
            
            # Verify handler was called with same list
            mock_handler.handle_recall.assert_called_once()
            call_args = mock_handler.handle_recall.call_args[0][0]
            
            assert call_args["domain_filter"] == domain_list

    def test_no_domain_filter_regression(self):
        """GREEN: Test that no domain filter still works."""
        # Setup
        mock_client = Mock()
        mock_ft = Mock()
        mock_client.ft.return_value = mock_ft
        
        redis = SynapseRedis(mock_client)
        
        # Capture debug output
        captured_output = StringIO()
        with patch('sys.stdout', captured_output):
            # Test without domain filter
            query = "test"
            
            # Execute
            redis.search_hybrid(
                query=query,
                domain_filter=None,
                limit=10
            )
        
        # Check debug output
        output = captured_output.getvalue()
        assert "DEBUG: final_query=(@content:test)" in output


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
