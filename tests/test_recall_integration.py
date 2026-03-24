"""Integration tests for recall function indexing issue."""

import pytest
import asyncio
import redis.asyncio as redis_async
from synapse.redis.client import SynapseRedis
from tests.test_mock_embeddings import MockEmbeddingBackend
from synapse.mcp.memorize import MCPMemorize
from synapse.mcp.recall import MCPRecall


class TestRecallIntegration:
    """Test memorize → recall end-to-end flow."""
    
    @pytest.fixture
    def redis_client(self):
        """Create sync Redis client for testing."""
        import redis
        client = redis.Redis.from_url('redis://ubuntuserver.tail70104d.ts.net:6379', decode_responses=True)
        yield client
        client.close()
    
    @pytest.fixture
    def synapse_redis(self, redis_client):
        """Create SynapseRedis wrapper."""
        return SynapseRedis(redis_client)
    
    @pytest.fixture
    def embedding_backend(self):
        """Create mock embedding backend for testing."""
        return MockEmbeddingBackend()
    
    @pytest.fixture
    def memorize_handler(self, synapse_redis, embedding_backend):
        """Create memorize handler."""
        return MCPMemorize(synapse_redis, embedding_backend)
    
    @pytest.fixture
    def recall_handler(self, synapse_redis, embedding_backend):
        """Create recall handler."""
        return MCPRecall(synapse_redis, embedding_backend)
    
    @pytest.mark.asyncio
    async def test_memorize_recall_e2e_flow(self, memorize_handler, recall_handler):
        """RED Test: End-to-end memorize → recall should work."""
        # Arrange
        test_content = "Test content for recall integration"
        test_domain = "integration_test"
        
        # Act: Memorize
        memorize_result = memorize_handler.handle_memorize({
            "domain": test_domain,
            "type": "entity",
            "content": test_content
        })
        
        # Assert: Memorize succeeded
        assert memorize_result["status"] == "success"
        node_id = memorize_result["id"]
        assert node_id.startswith("node:integration_test:")
        
        # Act: Recall
        recall_result = recall_handler.handle_recall({
            "query": "Test content recall",
            "limit": 5
        })
        
        # Assert: Recall should find the memorized content
        assert recall_result["total"] > 0, f"Expected results, got {recall_result}"
        assert len(recall_result["results"]) > 0
        
        # Verify the memorized node is in results
        result_ids = [r.get("id", "") for r in recall_result["results"]]
        assert node_id in result_ids, f"Expected {node_id} in results {result_ids}"
    
    @pytest.mark.asyncio
    async def test_index_document_count_increases(self, synapse_redis):
        """RED Test: Index document count should increase after memorize."""
        # Arrange: Get initial document count
        ft = synapse_redis._client.ft("synapse_idx")
        initial_info = ft.info()
        initial_docs = initial_info.get("num_docs", 0)
        
        # Act: Store a document directly
        test_node_id = "node:test_count:12345"
        synapse_redis.store_node(
            node_id=test_node_id,
            domain="test_count",
            node_type="entity",
            content="Test document for count verification",
            embedding=[0.1] * 768
        )
        
        # Assert: Document count should increase
        final_info = ft.info()
        final_docs = final_info.get("num_docs", 0)
        assert final_docs > initial_docs, f"Expected {initial_docs + 1} <= {final_docs}"
    
    @pytest.mark.asyncio
    async def test_search_returns_stored_documents(self, synapse_redis):
        """RED Test: Search should return documents that were stored."""
        # Arrange: Store a test document
        test_node_id = "node:test_search:67890"
        test_content = "Unique search test content"
        synapse_redis.store_node(
            node_id=test_node_id,
            domain="test_search",
            node_type="entity", 
            content=test_content,
            embedding=[0.2] * 768
        )
        
        # Act: Search for the content
        results = synapse_redis.search_hybrid(
            query="Unique search test",
            limit=5
        )
        
        # Assert: Should find the stored document
        assert len(results) > 0, "Expected search results"
        
        # Find our document in results
        found = False
        for result in results:
            if test_content in result.get("content", ""):
                found = True
                break
        
        assert found, f"Expected to find content '{test_content}' in results {results}"
    
    @pytest.mark.asyncio
    async def test_domain_filtering_works(self, synapse_redis):
        """RED Test: Domain filtering should work in search."""
        # Arrange: Store documents in different domains
        synapse_redis.store_node(
            node_id="node:domain_a:11111",
            domain="domain_a",
            node_type="entity",
            content="Content in domain A",
            embedding=[0.3] * 768
        )
        
        synapse_redis.store_node(
            node_id="node:domain_b:22222", 
            domain="domain_b",
            node_type="entity",
            content="Content in domain B",
            embedding=[0.4] * 768
        )
        
        # Act: Search with domain filter
        results_a = synapse_redis.search_hybrid(
            query="Content",
            domain_filter=["domain_a"],
            limit=10
        )
        
        # Assert: Should only return domain_a results
        for result in results_a:
            assert result.get("domain") == "domain_a", f"Expected domain_a, got {result.get('domain')}"
    
    @pytest.mark.asyncio
    async def test_no_indexing_failures(self, synapse_redis):
        """RED Test: Should have no indexing failures in FT.INFO."""
        # Arrange: Store various document types
        synapse_redis.store_node(
            node_id="node:no_failures:33333",
            domain="test_failures",
            node_type="entity",
            content="Test content for failure check",
            embedding=[0.5] * 768,
            metadata={"test": True}
        )
        
        # Act: Get index info
        ft = synapse_redis._client.ft("synapse_idx")
        info = ft.info()
        
        # Assert: No indexing failures
        indexing_failures = info.get("indexing_failures", 0)
        assert indexing_failures == 0, f"Expected 0 indexing failures, got {indexing_failures}"
        
        last_error = info.get("last_indexing_error", "")
        assert last_error == "", f"Expected no last error, got '{last_error}'"
