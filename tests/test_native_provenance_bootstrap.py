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
        assert kwargs["jobs"] == 3
        assert kwargs["schedule_options"]["worker_resident_reservation_bytes"] == 256 << 20
        events.extend(("verify", "install"))
        raise runtime.ProvenanceCoreRestartRequired("installed")

    monkeypatch.setattr(runtime, "ensure_native_provenance_core", install_then_restart)
    monkeypatch.setattr(fortran_c_shell, "lower_ast_source_to_ssa",
                        lambda *a, **k: events.append("outer-build"))
    with pytest.raises(runtime.ProvenanceCoreRestartRequired):
        build_whole_program_compiler(tmp_path, jobs=3, worker_resident_reservation_bytes=256 << 20)
    assert events == ["verify", "install"]


def test_interior_growth_restarts_without_publishing_an_unfinished_core(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "provenance_core_state", lambda: {"backend": "python"})
    monkeypatch.setenv(runtime.COMPILER_BOOTSTRAP_REGISTRY_ENV, str(tmp_path / "registry.json"))

    def stage(command, **kwargs):
        assert "--grow-stage" in command
        assert "--continue-after-failure" in command
        assert command[command.index("--jobs") + 1] == "3"
        options = json.loads(command[command.index("--schedule-json") + 1])
        assert options["max_total_resident_bytes"] == 4 << 30
        destination = Path(command[-1])
        (destination / "compartment.json").write_text(json.dumps({
            "stage": "native-growth-stage-persisted",
        }))

    monkeypatch.setattr("subprocess.run", stage)
    monkeypatch.setattr(runtime, "publish_native_provenance_core",
                        lambda *a, **k: pytest.fail("unfinished core published"))
    with pytest.raises(
        runtime.ProvenanceCoreRestartRequired, match="interior stage",
    ) as restart:
        runtime.ensure_native_provenance_core(tmp_path / "core", jobs=3,
            schedule_options={"max_total_resident_bytes": 4 << 30},
            stop_after_failure=False)
    assert restart.value.requires_process_restart is False


@pytest.mark.parametrize(
    "jobs,stage_units,stop_after_failure,expected_stage_units",
    [(None, None, True, None), (2, 3, False, 3), (2, None, False, None)],
)
def test_provenance_stage_plans_once_and_resumes_ssa_enclosures(
    tmp_path, monkeypatch, jobs, stage_units, stop_after_failure,
    expected_stage_units,
):
    from tools.bootstrap_compiler_concordance import build_compartment
    from src.compiler import project_compilation_product as products
    planned, stages = [], []

    def plan(source, directory, **kwargs):
        planned.append(kwargs)
        assert kwargs["ssa_composition"] and kwargs["plan_only"]
        destination = directory / "inputs/plan.json"
        destination.parent.mkdir(parents=True)
        value = {"compiler_toolchain": products.compiler_toolchain_fingerprint(), "units": []}
        destination.write_text(json.dumps(value))
        return {"plan": value}

    def stage(graph, plan, directory, **kwargs):
        stages.append((graph, plan, directory, kwargs))
        assert kwargs["max_stage_units"] == expected_stage_units
        assert kwargs["stop_after_failure"] is stop_after_failure
        assert kwargs["jobs"] == (12 if jobs is None else jobs)
        assert kwargs["worker_resident_reservation_bytes"] == 256 << 20
        return {"pending_unit_indices": [1], "units": [], "status": "growing"}

    monkeypatch.setattr(products, "compile_partitioned_source", plan)
    monkeypatch.setattr(products, "compile_resolved_process_graph_plan", stage)
    monkeypatch.setattr(products, "load_composed_source_units",
                        lambda *a: pytest.fail("assembled unfinished module"))
    for _ in range(2):
        receipt = build_compartment(tmp_path / "core", jobs=jobs, max_stage_units=stage_units,
            schedule_options={"worker_resident_reservation_bytes": 256 << 20},
            stop_after_failure=stop_after_failure, progress=None)
        assert receipt["stage"] == "native-growth-stage-persisted"
        assert receipt["installed"] is False
    assert len(planned) == 1 and len(stages) == 2
    assert stages[0][:3] == stages[1][:3]
    triage = json.loads((tmp_path / "core/growth/source-complexity.json").read_text())
    assert triage["cut_authority"] == "resolved-function-table"


def test_provenance_failure_is_terminal_before_repository_assembly(tmp_path, monkeypatch):
    from tools.bootstrap_compiler_concordance import build_compartment
    from src.compiler import project_compilation_product as products

    def plan(source, directory, **kwargs):
        destination = directory / "inputs/plan.json"
        destination.parent.mkdir(parents=True)
        value = {"compiler_toolchain": products.compiler_toolchain_fingerprint(), "units": []}
        destination.write_text(json.dumps(value))
        return {"plan": value}

    monkeypatch.setattr(products, "compile_partitioned_source", plan)
    monkeypatch.setattr(products, "compile_resolved_process_graph_plan", lambda *args, **kwargs: {
        "pending_unit_indices": [], "units": [{"unit_index": 4, "status": "failed"}],
        "counts": {"failed": 1}, "stopped_after_failure": True,
        "blocking_failure_unit_indices": [4],
    })
    monkeypatch.setattr(
        products, "load_composed_source_units",
        lambda *args: pytest.fail("assembled a failed provenance stage"),
    )

    receipt = build_compartment(tmp_path / "core", progress=None)

    assert receipt["stage"] == "native-growth-failed"


def test_surface_contract_includes_constants_and_class_declarations():
    import src.compiler.identity_concordance as core
    source = Path(core.__file__).read_text()
    assert runtime._provenance_core_surface_findings(core, source) == []
    surface = dict(vars(core))
    surface.pop("REGISTRY")
    findings = runtime._provenance_core_surface_findings(SimpleNamespace(**surface), source)
    assert findings == [{"kind": "missing-module-symbol", "symbol": "REGISTRY"}]
    surface["REGISTRY"] = core.REGISTRY
    surface["IdentityPage"] = type("IdentityPage", (), {})
    findings = runtime._provenance_core_surface_findings(SimpleNamespace(**surface), source)
    assert {"kind": "missing-class-member", "symbol": "IdentityPage._stamp"} in findings


@pytest.mark.parametrize("initializer", ["identity_concordance", "wrong_core"])
def test_real_extension_import_is_insufficient_without_complete_core_surface(tmp_path, initializer):
    """A real Python ABI library must still satisfy the whole-unit contract."""
    import os
    import shutil
    import subprocess
    import sysconfig
    compiler = shutil.which(os.environ.get("CC", "cc"))
    if compiler is None:
        pytest.skip("a C compiler is required for the Python extension ABI check")
    include = Path(sysconfig.get_paths()["include"])
    if not (include / "Python.h").is_file():
        pytest.skip("Python development headers are required for the extension ABI check")
    source = tmp_path / "incomplete.c"
    source.write_text('''#include <Python.h>
static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT, "identity_concordance", NULL, -1, NULL
};
PyMODINIT_FUNC PyInit_''' + initializer + '''(void) { return PyModule_Create(&module); }
''')
    library = tmp_path / ("identity_concordance" + sysconfig.get_config_var("EXT_SUFFIX"))
    subprocess.run([compiler, "-shared", "-fPIC", "-I" + str(include),
                    str(source), "-o", str(library)], check=True, capture_output=True, text=True)
    authored = Path(runtime.__file__).with_name("identity_concordance.py")
    repository = authored.parents[2]
    receipt = tmp_path / "verification.json"
    receipt.write_text(json.dumps({
        "schema": "turing.native-provenance-core-verification.v1",
        "status": "verified", "scope": "whole-module", "module": runtime.PROVENANCE_CORE_MODULE,
        "python_cache_tag": sys.implementation.cache_tag,
        "source_sha256": hashlib.sha256(authored.read_bytes()).hexdigest(),
        "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
        "native_probe_count": 1, "fallback_probe_count": 0,
        "compiler_toolchain": {"files": [
            {"path": f"src/compiler/{name}", "sha256": hashlib.sha256(
                (repository / "src/compiler" / name).read_bytes()).hexdigest()}
            for name in ("identity_concordance.py", "id_space.py", "monotonic_ids.py",
                         "concordance_declarations.py")]},
    }))
    registry = tmp_path / "registry.json"
    original = {"schema": runtime.COMPILER_BOOTSTRAP_REGISTRY_SCHEMA, "products": []}
    registry.write_text(json.dumps(original))
    with pytest.raises(ValueError, match="fresh-process verification") as failure:
        runtime.publish_native_provenance_core(library, receipt, registry_path=registry)
    if initializer == "identity_concordance":
        assert "missing-module-symbol" in str(failure.value)
        assert "REGISTRY" in str(failure.value)
    else:
        assert "PyInit_identity_concordance" in str(failure.value)
    assert json.loads(registry.read_text()) == original
    assert not registry.with_suffix(".json.tmp").exists()
