"""Whole-module selection and the exclusive first-stage barrier."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

from src.compiler import compiler_bootstrap_runtime as runtime


def test_missing_link_leaves_normal_python_module_unchanged(tmp_path):
    import src.compiler.identity_concordance as core
    assert not runtime.activate_native_provenance_core(tmp_path / "absent.json")
    assert sys.modules[runtime.PROVENANCE_CORE_MODULE] is core
    assert core.IdentityBook.__module__ == runtime.PROVENANCE_CORE_MODULE


def test_unverified_or_changed_core_cannot_be_published(tmp_path):
    library = tmp_path / "identity_concordance.so"
    library.write_bytes(b"not a native module")
    receipt = tmp_path / "verification.json"
    registry = tmp_path / "registry.json"
    source = Path(runtime.__file__).with_name("identity_concordance.py")
    value = {"schema": "turing.native-provenance-core-verification.v1",
             "status": "verified", "scope": "whole-module",
             "module": runtime.PROVENANCE_CORE_MODULE,
             "python_cache_tag": sys.implementation.cache_tag,
             "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
             "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
             "native_probe_count": 1, "fallback_probe_count": 0}
    repository = source.parents[2]
    value["compiler_toolchain"] = {"files": [
        {"path": f"src/compiler/{name}", "sha256": hashlib.sha256(
            (repository / "src/compiler" / name).read_bytes()).hexdigest()}
        for name in ("identity_concordance.py", "id_space.py", "monotonic_ids.py",
                     "concordance_declarations.py")]}
    for changed in ({"status": "built"}, {"scope": "accessor"},
                    {"source_sha256": "stale"}, {"fallback_probe_count": 1}):
        receipt.write_text(json.dumps({**value, **changed}))
        with pytest.raises(ValueError, match="whole-module verification"):
            runtime.publish_native_provenance_core(library, receipt, registry_path=registry)
        assert not registry.exists()
    # Even matching hashes and a claimed receipt cannot turn a bind(C) library
    # into an installed whole module: real fresh-process import must succeed.
    receipt.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="fresh-process verification"):
        runtime.publish_native_provenance_core(library, receipt, registry_path=registry)
    assert not registry.exists()


def test_unverified_build_blocks_bootstrap_before_outer_source_submission(tmp_path, monkeypatch):
    from src.compiler import fortran_c_shell
    from src.compiler.whole_program_compiler_bootstrap import build_whole_program_compiler
    events = []
    monkeypatch.setattr(runtime, "provenance_core_state", lambda: {"backend": "python"})
    monkeypatch.setenv(runtime.COMPILER_BOOTSTRAP_REGISTRY_ENV, str(tmp_path / "registry.json"))

    def isolated_compile(command, **kwargs):
        events.append("core-build")
        assert kwargs["check"] is True
        assert kwargs["env"][runtime.PROVENANCE_SOURCE_ONLY_ENV] == "1"
        isolated = Path(kwargs["env"][runtime.COMPILER_BOOTSTRAP_REGISTRY_ENV])
        assert json.loads(isolated.read_text())["products"] == []
        assert command[2] == "tools.bootstrap_compiler_concordance"
        destination = Path(command[-1])
        (destination / "compartment.json").write_text(json.dumps({"stage": "native-built-unverified"}))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("subprocess.run", isolated_compile)
    monkeypatch.setattr(fortran_c_shell, "lower_ast_source_to_ssa",
                        lambda *a, **k: events.append("outer-build"))
    with pytest.raises(RuntimeError, match="not verified"):
        build_whole_program_compiler(tmp_path / "product")
    assert events == ["core-build"]


def test_selected_native_core_needs_no_build_or_operator_dispatch(tmp_path, monkeypatch):
    selected = {"backend": "native", "library": "whole-module"}
    monkeypatch.setattr(runtime, "provenance_core_state", lambda: selected)
    monkeypatch.setattr("subprocess.run", lambda *a, **k: pytest.fail("unexpected rebuild"))
    assert runtime.ensure_native_provenance_core(tmp_path) is selected


def test_installation_requires_restart_before_any_outer_worker(tmp_path, monkeypatch):
    from src.compiler import fortran_c_shell
    from src.compiler.whole_program_compiler_bootstrap import build_whole_program_compiler
    events = []

    def install_then_restart(*args, **kwargs):
        events.extend(("verify", "install"))
        raise runtime.ProvenanceCoreRestartRequired("installed")

    monkeypatch.setattr(runtime, "ensure_native_provenance_core", install_then_restart)
    monkeypatch.setattr(fortran_c_shell, "lower_ast_source_to_ssa",
                        lambda *a, **k: events.append("outer-build"))
    with pytest.raises(runtime.ProvenanceCoreRestartRequired):
        build_whole_program_compiler(tmp_path)
    assert events == ["verify", "install"]
