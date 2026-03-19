"""RediSearch index setup for Synapse AKG."""

from typing import Any


class IndexManager:
    """Manages RediSearch index creation and maintenance."""

    INDEX_NAME = "synapse_idx"
    NODE_PREFIX = "node:"

    def __init__(self, redis_client: Any) -> None:
        """Initialize with Redis client."""
        self.redis = redis_client

    def get_create_command(self) -> str:
        """Generate FT.CREATE command as string for testing."""
        return (
            'FT.CREATE synapse_idx ON JSON PREFIX 1 "node:" '
            "SCHEMA "
            '$.domain AS domain TAG SEPARATOR "|" '
            "$.type AS type TAG "
            "$.content AS content TEXT WEIGHT 1.0 "
            "$.chunks[*].text AS chunk_text TEXT WEIGHT 1.5 "
            "$.chunks[*].language AS chunk_lang TAG "
            "$.embedding AS embedding VECTOR FLAT 6 TYPE FLOAT32 DIM 768 DISTANCE_METRIC COSINE "
            "$.metadata.* AS metadata TEXT "
            "$.created_at AS created_at NUMERIC SORTABLE"
        )

    def ensure_index(self) -> None:
        """Create index if not exists, drop and recreate if exists."""
        try:
            # Check if index exists
            ft = self.redis.ft(self.INDEX_NAME)
            ft.info()

            # Index exists, drop it
            ft.dropindex()

        except Exception:  # nosec B110: Intentional silent fail for index existence check
            # Index doesn't exist, continue to creation
            pass

        # Create new index
        self._create_index()

    def _create_index(self) -> None:
        """Create the RediSearch index with ADR-001 schema."""
        # Import here to avoid module issues
        from redis.commands.search.field import (
            NumericField,
            TagField,
            TextField,
            VectorField,
        )
        from redis.commands.search.index_definition import IndexDefinition, IndexType

        schema = (
            TagField("$.domain", as_name="domain", separator="|"),
            TagField("$.type", as_name="type"),
            TextField("$.content", as_name="content", weight=1.0),
            TextField("$.chunks[*].text", as_name="chunk_text", weight=1.5),
            TagField("$.chunks[*].language", as_name="chunk_lang"),
            VectorField(
                "$.embedding",
                "FLAT",
                {"TYPE": "FLOAT32", "DIM": 768, "DISTANCE_METRIC": "COSINE"},
                as_name="embedding",
            ),
            TextField("$.metadata.*", as_name="metadata"),
            NumericField("$.created_at", as_name="created_at", sortable=True),
        )

        definition = IndexDefinition(
            prefix=[self.NODE_PREFIX], index_type=IndexType.JSON
        )

        ft = self.redis.ft(self.INDEX_NAME)
        ft.create_index(fields=schema, definition=definition)
