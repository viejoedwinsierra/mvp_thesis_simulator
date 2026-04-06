from simulator.allocator import largest_remainder_allocation


def test_largest_remainder_preserves_total():
    allocation = largest_remainder_allocation(10, {"a": 0.33, "b": 0.33, "c": 0.34})
    assert sum(allocation.values()) == 10
