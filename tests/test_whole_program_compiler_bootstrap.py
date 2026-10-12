from __future__ import annotations

import json
from pathlib import Path


def test_boundary_plan_composes_existing_foundations_around_one_outer_target(tmp_path):
    from src.compiler.whole_program_compiler_bootstrap import (
        BOOTSTRAP_SCHEMA, STRATEGY, plan_whole_program_compiler,
    )

    receipt = plan_whole_program_compiler(tmp_path)
    by_name = {boundary["name"]: boundary for boundary in receipt["boundaries"]}

    assert receipt["schema"] == BOOTSTRAP_SCHEMA
    assert receipt["strategy"] == STRATEGY
    assert receipt["semantic_authority"] == "python-coordinator-and-identity-book"
    assert receipt["next_partition"] == by_name["process-graph-lowering"]
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


def test_build_delegates_the_complete_outer_partition_to_bootstrap_creep(
    tmp_path, monkeypatch,
):
    from src.compiler import project_compilation_product
    from src.compiler.whole_program_compiler_bootstrap import (
        build_whole_program_compiler,
    )

    observed = []

    def compile_creep(source, directory, **kwargs):
        observed.append((Path(source), Path(directory), kwargs))
        return {"status": "frontier", "rounds": []}

    monkeypatch.setattr(
        project_compilation_product,
        "compile_project_bootstrap_creep",
        compile_creep,
    )
    product = build_whole_program_compiler(
        tmp_path, jobs=2, unit_timeout_seconds=17, max_rounds=3,
    )

    assert len(observed) == 1
    source, directory, options = observed[0]
    assert source.name == "aot_compile.py"
    assert options["entries"] == ("_lower_process_graph_to_compilation",)
    assert options["expand_entry_dependencies"] is True
    assert options["jobs"] == 2
    assert options["unit_timeout_seconds"] == 17
    assert options["max_rounds"] == 3
    assert directory.name == product.boundary["version"]
    assert product.product["status"] == "frontier"
    assert product.manifest.is_file()


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
