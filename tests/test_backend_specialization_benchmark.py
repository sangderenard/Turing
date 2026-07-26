from src.common.tensors.benchmark_backend_specializations import run_comparison


def test_specialization_benchmark_reports_parity_for_numpy_and_c():
    table = run_comparison(
        backends=("numpy", "c"),
        tasks=("neural", "metric", "laplace"),
        size=9,
        warmup=0,
        repeats=1,
    )

    assert len(table) == 6
    assert table["parity_ok"].all()
    assert set(table["task_key"]) == {"neural", "metric", "laplace"}
    assert (table["warm_median_sec"] >= 0.0).all()
