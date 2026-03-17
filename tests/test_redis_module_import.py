"""TDD RED Phase: Tests for Redis module import."""


def test_redis_module_import():
    """Test that synapse.redis.client can be imported."""
    # This will fail - module doesn't exist yet (RED phase)
    from synapse.redis.client import SynapseRedis

    assert SynapseRedis is not None
    assert hasattr(SynapseRedis, 'store_node')
    assert hasattr(SynapseRedis, 'get_node')
    assert hasattr(SynapseRedis, 'update_node')
    assert hasattr(SynapseRedis, 'search_hybrid')
    assert hasattr(SynapseRedis, 'get_linked_nodes')
    assert hasattr(SynapseRedis, 'ping')
    assert hasattr(SynapseRedis, 'close')


def test_redis_package_init():
    """Test that synapse.redis package can be imported."""
    # This will fail - package doesn't exist yet (RED phase)
    import synapse.redis

    assert synapse.redis is not None
