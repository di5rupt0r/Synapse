"""Node schema for Synapse AKG."""

import re
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


class Chunk(BaseModel):
    """Code chunk schema for AST-based segmentation."""

    id: str = Field(..., pattern=r"^chunk:[a-f0-9-]{36}$")
    text: str = Field(..., description="Chunk text content")
    language: str = Field(..., description="Programming language")
    node_type: str = Field(..., description="AST node type")
    line_start: int = Field(..., ge=1, description="Starting line number")
    line_end: int = Field(..., ge=1, description="Ending line number")
    embedding: List[float] = Field(..., description="768-dim vector")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

    @field_validator("embedding")
    @classmethod
    def validate_embedding_dim(cls, v):
        """Validate embedding dimension."""
        if len(v) != 768:
            raise ValueError("Embedding must be 768 dimensions")
        return v

    @field_validator("line_end")
    @classmethod
    def validate_line_range(cls, v, info):
        """Validate line range."""
        if "line_start" in info.data and v < info.data["line_start"]:
            raise ValueError("line_end must be >= line_start")
        return v

    @field_validator("id")
    @classmethod
    def validate_id_format(cls, v):
        """Validate chunk ID format."""
        pattern = r"^chunk:[a-f0-9-]{36}$"
        if not re.match(pattern, v):
            raise ValueError("Invalid chunk ID format, expected: chunk:uuid")
        return v


class SynapseNode(BaseModel):
    """Polymorphic node schema for AKG."""

    id: str = Field(..., pattern=r"^node:[a-z_]+:[a-f0-9-]{36}$")
    domain: str = Field(..., description="TAG indexable domain")
    type: Literal["entity", "observation", "relation", "chunk"]
    content: str = Field(..., description="Core payload")
    embedding: List[float] = Field(..., description="768-dim vector")
    chunks: Optional[List[Chunk]] = Field(
        default_factory=list, description="Code chunks"
    )
    links: Optional[Dict[str, List[str]]] = Field(
        default_factory=lambda: {"inbound": [], "outbound": []}
    )
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime

    @field_validator("embedding")
    @classmethod
    def validate_embedding_dim(cls, v):
        """Validate embedding dimension."""
        if len(v) != 768:
            raise ValueError("Embedding must be 768 dimensions")
        return v

    @field_validator("id")
    @classmethod
    def validate_id_format(cls, v):
        """Validate node ID format."""
        pattern = r"^node:[a-z_]+:[a-f0-9-]{36}$"
        if not re.match(pattern, v):
            raise ValueError("Invalid node ID format, expected: node:domain:uuid")
        return v
