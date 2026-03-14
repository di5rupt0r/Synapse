"""MCP Memorize Handler - State → Embedding → Node."""

from typing import Dict, Any, Optional
import uuid
from datetime import datetime

from .base import MCPBase
from ..schema.node import SynapseNode


class MCPMemorize(MCPBase):
    """MCP handler for memorize operations."""
    
    def __init__(self, redis_client: Any, embedding_service: Any) -> None:
        """Initialize with Redis client and embedding service."""
        super().__init__()
        self.redis = redis_client
        self.embeddings = embedding_service
        
        # Register memorize method
        self.register("memorize")(self.handle_memorize)
    
    def handle_memorize(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle memorize request: validate → embed → store → respond."""
        try:
            # Validate required fields
            self._validate_params(params)
            
            # Extract parameters
            domain = params["domain"]
            node_type = params["type"]
            content = params["content"]
            metadata = params.get("metadata")
            links = params.get("links")
            
            # Generate embedding
            embedding = self.embeddings.embed(content)
            
            # Generate node ID
            node_id = f"node:{domain}:{uuid.uuid4()}"
            
            # Store node in Redis
            self.redis.store_node(
                domain=domain,
                node_type=node_type,
                content=content,
                embedding=embedding,
                metadata=metadata,
                links=links
            )
            
            return {
                "status": "success",
                "id": node_id
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def _validate_params(self, params: Dict[str, Any]) -> None:
        """Validate memorize request parameters."""
        required_fields = ["domain", "type", "content"]
        
        for field in required_fields:
            if field not in params:
                raise ValueError(f"Missing required field: {field}")
        
        # Validate node type
        valid_types = ["entity", "observation", "relation", "chunk"]
        if params["type"] not in valid_types:
            raise ValueError(f"Invalid type: {params['type']}. Must be one of: {valid_types}")
        
        # Validate domain format
        domain = params["domain"]
        if not domain or not isinstance(domain, str) or not domain.replace("_", "").isalnum():
            raise ValueError("Domain must be a non-empty alphanumeric string")
