"""Configuration settings for Synapse AKG."""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class Settings:
    """Application settings."""
    
    # Redis configuration
    redis_host: str = os.getenv("REDIS_HOST", "localhost")
    redis_port: int = int(os.getenv("REDIS_PORT", "6379"))
    redis_password: Optional[str] = os.getenv("REDIS_PASSWORD", None)
    
    # Server configuration
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8000"))
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"
    
    # Embedding configuration
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "microsoft/unixcoder-base")
    embedding_device: str = os.getenv("EMBEDDING_DEVICE", "cpu")
    
    # Search configuration
    default_top_k: int = int(os.getenv("DEFAULT_TOP_K", "10"))
    rrf_k: int = int(os.getenv("RRF_K", "60"))
    
    # Cache configuration
    cache_size: int = int(os.getenv("CACHE_SIZE", "1000"))
    
    # Performance targets
    max_query_latency_ms: float = float(os.getenv("MAX_QUERY_LATENCY_MS", "80.0"))


def get_settings() -> Settings:
    """Get application settings."""
    return Settings()
