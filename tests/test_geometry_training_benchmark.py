from src.common.tensors.youngman.benchmark_geometry_training import _csv_tuple


def test_benchmark_csv_tuple_is_deterministic():
    assert _csv_tuple("1729, 2718,31415", int) == (1729, 2718, 31415)
    assert _csv_tuple("ripple, saddle", str) == ("ripple", "saddle")
