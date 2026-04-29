import pytest


def test_benchmark_name_is_reserved_for_trading_comparison_references():
    from alpha_os.benchmark import Benchmark

    with pytest.raises(NotImplementedError, match="Use EvaluationSpec"):
        Benchmark()
