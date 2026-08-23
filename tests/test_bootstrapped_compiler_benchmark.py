from tools.benchmark_bootstrapped_compiler import compare_results


def test_comparison_reports_direction_and_requires_signature_equality():
    source = {
        "signature": {"functions": ["sample"]},
        "warm_compile_median_seconds": 2.0,
        "first_compile_seconds": 3.0,
        "activation_seconds": 0.5,
        "fresh_process_seconds": 4.0,
    }
    bootstrapped = {
        "signature": {"functions": ["sample"]},
        "warm_compile_median_seconds": 1.0,
        "first_compile_seconds": 2.5,
        "activation_seconds": 1.0,
        "fresh_process_seconds": 3.5,
    }

    comparison = compare_results(source, bootstrapped)

    assert comparison == {
        "equivalent": True,
        "warm_seconds_difference": -1.0,
        "warm_speedup": 2.0,
        "first_compile_seconds_difference": -0.5,
        "activation_seconds_difference": 0.5,
        "activation_plus_first_seconds_difference": 0.0,
        "fresh_process_seconds_difference": -0.5,
    }
