from types import SimpleNamespace


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
