"""MCP Patch Handler - Atomic Node Mutation."""

from typing import Any, Dict, List


class MCPPatch:
    """MCP handler for patch state operations."""

    def __init__(self, redis_client: Any) -> None:
        """Initialize with Redis client."""
        self.redis = redis_client

    def handle_patch(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle patch request: validate → check existence → atomic update."""
        try:
            self._validate_params(params)

            node_id = params["node_id"]
            operations = params["operations"]

            self._validate_operations(operations)

            node = self.redis.get_node(node_id)
            if not node:
                return {"status": "error", "error": f"Node {node_id} not found"}

            redis_operations = []
            for op in operations:
                redis_op = {
                    "path": op["path"],
                    "op": op["op"],
                    "value": op.get("value"),
                }
                redis_operations.append(redis_op)

            success = self.redis.update_node(node_id, redis_operations)

            if success:
                return {"status": "success", "node_id": node_id, "updated": True}
            else:
                return {
                    "status": "error",
                    "node_id": node_id,
                    "error": "Failed to apply operations",
                }

        except Exception as e:
            return {"status": "error", "error": str(e)}

    def _validate_params(self, params: Dict[str, Any]) -> None:
        """Validate patch request parameters."""
        if "node_id" not in params:
            raise ValueError("Missing required field: node_id")

        if "operations" not in params:
            raise ValueError("Missing required field: operations")

        node_id = params["node_id"]
        if not node_id or not isinstance(node_id, str):
            raise ValueError("Node ID must be a non-empty string")

        operations = params["operations"]
        if not isinstance(operations, list) or len(operations) == 0:
            raise ValueError("Operations must be a non-empty list")

    def _validate_operations(self, operations: List[Dict[str, Any]]) -> None:
        """Validate individual operations."""
        valid_ops = ["set", "delete", "append"]

        for i, op in enumerate(operations):
            if not isinstance(op, dict):
                raise ValueError(f"Operation {i} must be a dictionary")

            if "path" not in op:
                raise ValueError(f"Operation {i} missing required field: path")

            if "op" not in op:
                raise ValueError(f"Operation {i} missing required field: op")

            if op["op"] not in valid_ops:
                raise ValueError(
                    f"Operation {i} has invalid op: {op['op']}. Must be one of: {valid_ops}"
                )

            # Validate value requirement
            if op["op"] in ["set", "append"] and "value" not in op:
                raise ValueError(
                    f"Operation {i} with op='{op['op']}' requires value field"
                )
