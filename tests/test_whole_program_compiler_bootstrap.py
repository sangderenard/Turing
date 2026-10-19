from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace


def test_boundary_plan_composes_existing_foundations_around_one_outer_target(tmp_path):
    from src.compiler.whole_program_compiler_bootstrap import (
        BOOTSTRAP_SCHEMA, STRATEGY, plan_whole_program_compiler,
    )

    receipt = plan_whole_program_compiler(tmp_path)
    by_name = {boundary["name"]: boundary for boundary in receipt["boundaries"]}

    assert receipt["schema"] == BOOTSTRAP_SCHEMA
    assert receipt["strategy"] == STRATEGY
    assert receipt["semantic_authority"] == "python-coordinator-and-identity-book"
    assert receipt["next_partition"] == by_name["provenance-core"]
    assert receipt["outer_partition"] == by_name["process-graph-lowering"]
    assert receipt["bootstrap_schedule"][0]["exclusive"] is True
    assert by_name["process-graph-lowering"]["entrypoint"] == (
        "_lower_process_graph_to_compilation"
    )
    assert by_name["process-graph-lowering"]["kind"] == "outer-partition"
    assert by_name["whole-module-storage-effects"]["semantic"] == (
        "compiler.ssa.read_only_module"
    )
    assert by_name["metadata-copy"]["disposition"] == "rejected-hot-boundary"
    assert by_name["sympy-ingestion"]["disposition"] == (
        "candidate-after-process-graph-lowering"
    )
    assert "keep the active fingerprint" in receipt["product_retention_policy"]
    assert all(len(boundary["version"]) == 64 for boundary in by_name.values())
    assert json.loads((tmp_path / "plan.json").read_text()) == receipt


def test_boundary_version_changes_with_source_or_toolchain(tmp_path):
    from src.compiler.whole_program_compiler_bootstrap import CompilerBoundary

    source = tmp_path / "stage.py"
    source.write_text("def stage(value):\n    return value\n")
    boundary = CompilerBoundary(
        "stage", "outer-partition", "stage.py", "stage", "compiler.stage",
        "one input", "one output",
    )
    first = boundary.receipt(tmp_path, {"sha256": "a" * 64})
    source.write_text("def stage(value):\n    return value + 1\n")
    changed_source = boundary.receipt(tmp_path, {"sha256": "a" * 64})
    changed_toolchain = boundary.receipt(tmp_path, {"sha256": "b" * 64})

    assert first["version"] != changed_source["version"]
    assert changed_source["version"] != changed_toolchain["version"]


def test_bootstrap_retention_keeps_active_failure_receipt_and_unrelated_paths(
    tmp_path, monkeypatch,
):
    from src.compiler import compiler_bootstrap_runtime as runtime
    from src.compiler.whole_program_compiler_bootstrap import (
        bootstrap_provenance_core, plan_whole_program_compiler,
    )

    plan = plan_whole_program_compiler(tmp_path)
    boundary = plan["next_partition"]
    root = tmp_path / "products" / boundary["name"]
    active = root / boundary["version"]
    active_failure = active / "product" / "units" / "unit_029" / "failure.json"
    active_failure.parent.mkdir(parents=True)
    active_failure.write_text('{"status": "failed"}')
    stale = root / ("0" * 64)
    (stale / "temporary-tiles").mkdir(parents=True)
    unrelated = root / "manually-named-diagnostic"
    unrelated.mkdir()
    monkeypatch.setattr(
        runtime, "ensure_native_provenance_core",
        lambda *args, **kwargs: {"backend": "native"},
    )

    bootstrap_provenance_core(tmp_path)

    assert active_failure.is_file()
    assert not stale.exists()
    assert unrelated.is_dir()


def test_build_submits_authored_pieces_then_whole_to_public_source_scheduler(
    tmp_path, monkeypatch,
):
    from src.compiler import fortran_c_shell
    from src.compiler.whole_program_compiler_bootstrap import (
        build_whole_program_compiler,
    )

    from src.compiler import compiler_bootstrap_runtime as runtime
    monkeypatch.setattr(runtime, "ensure_native_provenance_core", lambda *a, **k: {"backend": "native"})
    observed = []

    def compile_partitioned(requests, **kwargs):
        observed.append((requests, kwargs))
        manifest = Path(requests[0]["product_directory"]) / "manifest.json"
        manifest.parent.mkdir(parents=True)
        manifest.write_text(json.dumps({"status": "sealed"}))
        return object()

    monkeypatch.setattr(
        fortran_c_shell, "lower_ast_source_to_ssa", compile_partitioned,
    )
    product = build_whole_program_compiler(
        tmp_path, jobs=2,
    )

    assert len(observed) == 1
    requests, options = observed[0]
    names = [request["entrypoint"] for request in requests]
    assert names[-1] == "_lower_process_graph_to_compilation"
    assert any(".<locals>." in name for name in names[:-1])
    assert set(requests[-1]["after"]).issubset(names[:-1])
    assert "def _lower_process_graph_to_compilation(" in Path(requests[0]["source_path"]).read_text()
    assert options["partitioned_options"]["jobs"] == 2
    assert options["source_compilation"] == "parallel"
    assert Path(requests[0]["product_directory"]).parent.name == product.boundary["version"]
    assert product.product["status"] == "sealed"
    assert product.manifest.is_file()


def test_build_uses_requested_bootstrap_memory_contract_by_default(
    tmp_path, monkeypatch,
):
    from src.compiler import fortran_c_shell
    from src.compiler.whole_program_compiler_bootstrap import (
        build_whole_program_compiler,
    )

    from src.compiler import compiler_bootstrap_runtime as runtime
    monkeypatch.setattr(runtime, "ensure_native_provenance_core", lambda *a, **k: {"backend": "native"})
    observed = {}

    def compile_partitioned(requests, **kwargs):
        observed.update(kwargs["partitioned_options"])
        manifest = Path(requests[0]["product_directory"]) / "manifest.json"
        manifest.parent.mkdir(parents=True)
        manifest.write_text(json.dumps({"status": "sealed"}))
        return object()

    monkeypatch.setattr(
        fortran_c_shell, "lower_ast_source_to_ssa", compile_partitioned,
    )
    build_whole_program_compiler(tmp_path)

    assert observed["max_total_resident_bytes"] == 10 << 30
    assert observed["max_worker_memory_bytes"] == 10 << 30
    assert observed["worker_resident_reservation_bytes"] == 2 << 30
    assert observed["minimum_total_resident_bytes"] == 2 << 30
    assert observed["system_reserve_bytes"] == 2 << 30
    assert observed["worker_elastic_available_percent"] == 75
    assert observed["minimum_worker_jobs"] == 1
    assert observed["adaptive_memory_ceiling"] is True
    assert "jobs" not in observed


def test_provenance_only_cli_stops_before_outer_partition(tmp_path, monkeypatch, capsys):
    import tools.bootstrap_compiler_whole_program as cli

    events = []
    plan = {"next_partition": {"version": "a" * 64}}

    def provenance_only(directory, **kwargs):
        events.append((Path(directory), kwargs["expansion_policy"]))
        return plan, {"backend": "native", "library": "whole-module"}

    monkeypatch.setattr(cli, "bootstrap_provenance_core", provenance_only)
    monkeypatch.setattr(
        cli, "build_whole_program_compiler",
        lambda *args, **kwargs: events.append("outer-build"),
    )
    arguments = SimpleNamespace(provenance_only=True, output=tmp_path)
    display = SimpleNamespace(log=lambda message: None)
    policy = {"unroll_depth": 0, "inline_depth": 0, "custom_version_depth": 0}

    assert cli._build_and_report(arguments, policy, display) == 0
    assert events == [(tmp_path, policy)]
    assert "outer partition not started" in capsys.readouterr().out


def test_provenance_cli_does_not_open_outer_display_while_child_owns_terminal(
    tmp_path, monkeypatch, capsys,
):
    import tools.bootstrap_compiler_whole_program as cli
    from src.compiler import shell_telemetry

    plan = {"next_partition": {"version": "b" * 64}}
    monkeypatch.setattr(
        cli, "bootstrap_provenance_core",
        lambda *args, **kwargs: (
            plan, {"backend": "native", "library": "whole-module"},
        ),
    )
    monkeypatch.setattr(
        shell_telemetry, "compile_progress",
        lambda *args, **kwargs: pytest.fail("outer display opened"),
    )

    assert cli.main(["--provenance-only", "--output", str(tmp_path)]) == 0
    assert "outer partition not started" in capsys.readouterr().out


def test_provenance_child_failure_returns_its_status_without_parent_traceback(
    tmp_path, monkeypatch,
):
    import subprocess
    import tools.bootstrap_compiler_whole_program as cli

    def failed_child(*args, **kwargs):
        raise subprocess.CalledProcessError(1, ["provenance-child"])

    monkeypatch.setattr(cli, "bootstrap_provenance_core", failed_child)

    assert cli.main(["--provenance-only", "--output", str(tmp_path)]) == 1


def test_provenance_cli_continues_clean_interior_stage_without_process_overlay(
    tmp_path, monkeypatch, capsys,
):
    import tools.bootstrap_compiler_whole_program as cli
    from src.compiler.compiler_bootstrap_runtime import (
        ProvenanceCoreRestartRequired,
    )

    calls = []
    plan = {"next_partition": {"version": "c" * 64}}

    def staged(*args, **kwargs):
        calls.append(kwargs["stop_after_failure"])
        if len(calls) == 1:
            raise ProvenanceCoreRestartRequired(
                "interior stage",
                requires_process_restart=False,
            )
        return plan, {"backend": "native", "library": "whole-module"}

    monkeypatch.setattr(cli, "bootstrap_provenance_core", staged)
    monkeypatch.setattr(
        cli.os, "execve",
        lambda *args, **kwargs: pytest.fail("clean stage used process overlay"),
    )

    assert cli.main(["--provenance-only", "--output", str(tmp_path)]) == 0
    assert calls == [True, True]
    assert "outer partition not started" in capsys.readouterr().out


def test_outer_partition_is_an_authored_project_root():
    from src.compiler.project_compilation_product import discover_authored_calls

    repository = Path(__file__).resolve().parents[1]
    source = repository / "src/common/tensors/accelerator_backends/aot_compile.py"
    calls = {call.qualified_name for call in discover_authored_calls(source.read_text())}
    assert "_lower_process_graph_to_compilation" in calls


def test_pursued_nested_class_method_keeps_outer_mapping_identity():
    """The real compiler's loop pass uses this exact closure geometry."""

    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
    from src.compiler.loop_interchange import interchange_reduction_loops

    plans = []
    lower_ast_source_to_ssa(
        "def root(source):\n    return interchange(source)\n",
        "root",
        python_bindings={"interchange": interchange_reduction_loops},
        extraction_contract="extraction_contracts/program_extraction.yaml",
        name="nested_class_closure_mapping",
        compilation_unit_plan_sink=plans.append,
        stop_after_compilation_unit_plan=True,
    )

    assert len(plans) == 1
    qualified_names = {
        name
        for unit in plans[0]["units"]
        for name in unit["qualified_names"]
    }
    assert any(
        "interchange_reduction_loops.<locals>" in name
        and name.endswith(".visit_For")
        for name in qualified_names
    )


def test_pursued_nested_function_keeps_annotated_outer_mapping_identity():
    from src.compiler.fortran_c_shell import (
        _bind_sequence_storage_members, lower_ast_source_to_ssa,
    )

    plans = []
    lower_ast_source_to_ssa(
        "def root(bindings, callee, caller):\n"
        "    return bind_members(bindings, callee, caller)\n",
        "root",
        python_bindings={"bind_members": _bind_sequence_storage_members},
        extraction_contract="extraction_contracts/program_extraction.yaml",
        name="annotated_closure_mapping",
        compilation_unit_plan_sink=plans.append,
        stop_after_compilation_unit_plan=True,
    )

    assert len(plans) == 1
    assert any(
        name.endswith("_bind_sequence_storage_members.<locals>.bind")
        for unit in plans[0]["units"]
        for name in unit["qualified_names"]
    )
