from types import SimpleNamespace


def test_stage_failure_rollup_is_current_concise_and_points_to_receipts(tmp_path):
    from tools.bootstrap_compiler_concordance import _print_outcome

    current = tmp_path / "growth/product/units/unit_007/failure.json"
    current.parent.mkdir(parents=True)
    current.write_text("{}")
    receipt = {"stage": "native-growth-stage-persisted", "native_growth": {
        "stage_admitted_units": 2,
        "stage_admitted_unit_indices": [7, 8],
        "concurrent_workers_peak": 2,
        "pending_unit_indices": [9],
        "counts": {"verified": 0, "compiled-unverified": 0,
                   "partial": 0, "failed": 2},
        "units": [
            {"unit_index": 3, "status": "failed", "error": "old failure"},
            {"unit_index": 7, "status": "failed", "error_type": "ValueError",
             "error": "first line\nsecond line",
             "qualified_names": ["package.current"],
             "stage": {"phase": "ssa-lowering"}},
            {"unit_index": 8, "status": "compiled-unverified"},
        ],
    }}
    lines = []

    _print_outcome(receipt, tmp_path, emit=lines.append)

    rendered = "\n".join(lines)
    assert "cumulative=3/4" in rendered
    assert "failures from this stage (1)" in rendered
    assert "unit 007 | package.current | ssa-lowering | ValueError" in rendered
    assert "first line second line" in rendered
    assert str(current.resolve()) in rendered
    assert "old failure" not in rendered


def test_stopped_stage_reports_all_blocking_failures_and_exits_nonzero(
    tmp_path, monkeypatch, capsys,
):
    import tools.bootstrap_compiler_concordance as cli

    failures = []
    for index in (3, 7):
        path = tmp_path / f"growth/product/units/unit_{index:03d}/failure.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
        failures.append({
            "unit_index": index, "status": "failed", "error_type": "ValueError",
            "error": f"failure {index}", "qualified_names": [f"package.unit_{index}"],
        })
    receipt = {"stage": "native-growth-failed", "native_growth": {
        "stage_admitted_units": 1,
        "stage_admitted_unit_indices": [7],
        "blocking_failure_unit_indices": [3, 7],
        "stopped_after_failure": True,
        "concurrent_workers_peak": 1,
        "pending_unit_indices": [8],
        "counts": {"failed": 2},
        "units": failures,
    }}
    monkeypatch.setattr(cli, "build_compartment", lambda *args, **kwargs: receipt)

    assert cli.main(["--output", str(tmp_path)]) == 1
    rendered = capsys.readouterr().out
    assert "blocking failures (2)" in rendered
    assert "unit 003" in rendered and "unit 007" in rendered


def test_completed_repository_ssa_resumes_and_changed_artifact_rebuilds(
    tmp_path, monkeypatch,
):
    from src.compiler import fortran_c_shell
    from src.compiler import project_compilation_product as product
    from tools.bootstrap_compiler_concordance import build_compartment

    monkeypatch.setattr(
        product, "compiler_toolchain_fingerprint",
        lambda: {"schema": "test-toolchain", "files": []},
    )
    calls = []

    def lower(*args, **kwargs):
        calls.append("lower")
        return SimpleNamespace(functions=["compiled"]), {"result": 1}, ("entry",)

    monkeypatch.setattr(fortran_c_shell, "lower_ast_source_to_ssa", lower)

    first = build_compartment(
        tmp_path, lower_only=True, grow_stage=False, progress=None,
    )
    assert first["stage"] == "repository-ssa"
    assert calls == ["lower"]

    second = build_compartment(
        tmp_path, lower_only=True, grow_stage=False, progress=None,
    )
    assert second["stage"] == "repository-ssa"
    assert second["resumed_from"] == "provenance-repository-ssa"
    assert second["checkpoint_status"] == "hit"
    assert calls == ["lower"]

    (tmp_path / "repository-ssa.pkl").write_bytes(b"changed")
    third = build_compartment(
        tmp_path, lower_only=True, grow_stage=False, progress=None,
    )
    assert third["stage"] == "repository-ssa"
    assert third["checkpoint_resume_status"] == "miss: external artifact digest mismatch"
    assert third["checkpoint_status"] == "stored"
    assert calls == ["lower", "lower"]


def test_pickle_publication_preserves_previous_artifact_when_replace_fails(
    tmp_path, monkeypatch,
):
    from src.compiler import project_compilation_product as product

    artifact = tmp_path / "repository-ssa.pkl"
    product._dump_pickle_artifact(artifact, {"generation": 1})

    def reject_replace(source, destination):
        raise PermissionError("occupied")

    monkeypatch.setattr(product.os, "replace", reject_replace)
    try:
        product._dump_pickle_artifact(artifact, {"generation": 2})
    except PermissionError:
        pass
    else:
        raise AssertionError("failed atomic replacement was accepted")

    assert product._load_pickle_artifact(artifact) == {"generation": 1}
    assert list(tmp_path.iterdir()) == [artifact]
