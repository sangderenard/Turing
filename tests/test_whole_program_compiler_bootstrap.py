from __future__ import annotations

import json
import os

import pytest

from src.compiler.whole_program_compiler_bootstrap import (
    BOOTSTRAP_SCHEMA,
    _root_export,
    _without_incremental_bootstrap,
)


def test_single_image_bootstrap_disables_and_restores_incremental_products(
    tmp_path, monkeypatch,
):
    from src.compiler.compiler_bootstrap_runtime import (
        COMPILER_BOOTSTRAP_PRODUCTS_ENV,
        COMPILER_BOOTSTRAP_REGISTRY_ENV,
    )

    monkeypatch.setenv(COMPILER_BOOTSTRAP_PRODUCTS_ENV, "old-product")
    monkeypatch.setenv(COMPILER_BOOTSTRAP_REGISTRY_ENV, "old-registry")
    with _without_incremental_bootstrap(tmp_path):
        assert os.environ[COMPILER_BOOTSTRAP_PRODUCTS_ENV] == ""
        assert os.environ[COMPILER_BOOTSTRAP_REGISTRY_ENV] == str(
            tmp_path / "no-incremental-bootstrap-registry.json"
        )
    assert os.environ[COMPILER_BOOTSTRAP_PRODUCTS_ENV] == "old-product"
    assert os.environ[COMPILER_BOOTSTRAP_REGISTRY_ENV] == "old-registry"


def test_single_image_bootstrap_requires_one_program_root():
    assert _root_export((
        "whole_program_compiler__helper",
        "whole_program_compiler__compile_program",
    )) == "whole_program_compiler__compile_program"
    with pytest.raises(RuntimeError, match="exactly one compile_program"):
        _root_export(("whole_program_compiler__helper",))


def test_whole_program_plan_uses_one_canonical_lowering(monkeypatch, tmp_path):
    from src.compiler import fortran_c_shell
    from src.compiler.whole_program_compiler_bootstrap import (
        plan_whole_program_compiler,
    )

    observed = []

    def lower(source, entrypoint, **kwargs):
        observed.append((source, entrypoint, kwargs))
        kwargs["compilation_unit_plan_sink"]({
            "schema": "test-plan",
            "units": [{"qualified_names": ["compile_program", "helper"]}],
        })
        return None, {}, ()

    monkeypatch.setattr(fortran_c_shell, "lower_ast_source_to_ssa", lower)
    receipt = plan_whole_program_compiler(
        tmp_path,
        source_compiler_impl=lambda source, entrypoint, **kwargs: None,
    )

    assert len(observed) == 1
    _source, entrypoint, options = observed[0]
    assert entrypoint == "compile_program"
    assert options["stop_after_compilation_unit_plan"] is True
    assert set(options["python_bindings"]) == {"source_compiler_impl"}
    assert receipt["schema"] == BOOTSTRAP_SCHEMA
    assert receipt["strategy"] == "single-root-single-module-single-library"
    assert receipt["incremental_bootstrap_products"] == []
    assert json.loads((tmp_path / "plan.json").read_text()) == receipt


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
        _bind_sequence_storage_members,
        lower_ast_source_to_ssa,
    )

    plans = []
    lower_ast_source_to_ssa(
        "def root(bindings, callee, caller):\n"
        "    return bind_members(bindings, callee, caller)\n",
        "root",
        python_bindings={
            "bind_members": _bind_sequence_storage_members,
        },
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
