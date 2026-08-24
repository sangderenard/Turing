import json
import os
import hashlib

import tools.compile_project_catalogue as compile_project_catalogue
from tools.diagnose_translation import (
    load_compilation_product,
    load_compilation_unit,
    planned_dependency_repair_order,
    stage_0_compilation_product,
    stage_0_compilation_unit,
)
from tools.compile_project_catalogue import _planned_unit_progress_writer


def test_partial_meta_unit_reports_name_based_dependency_repair_order(
    tmp_path, capsys,
):
    plan_path = tmp_path / "process-graph-units.json"
    plan_path.write_text(json.dumps({
        "schema": "turing.compilation-unit-plan.v1",
        "units": [
            {"qualified_names": ["Compiler.leaf"], "dependency_units": []},
            {"qualified_names": ["Compiler.middle"], "dependency_units": [0]},
            {"qualified_names": ["Compiler.other"], "dependency_units": []},
            {"qualified_names": ["Compiler.root"], "dependency_units": [1, 2]},
        ],
    }), encoding="utf-8")
    receipt = {
        "status": "partial",
        "unit_index": 3,
        "unit": {"dependency_units": [1, 2]},
        "process_graph_unit_plan": plan_path.as_posix(),
        "process_graph_unit_plan_sha256": hashlib.sha256(
            plan_path.read_bytes()
        ).hexdigest(),
        "linked_verified_units": [],
        "repository_ssa_complete": False,
        "repository_ssa_accounting": [{
            "qualified_name": "Compiler.root",
            "shortfalls": [{"kind": "unresolved-authored-calls"}],
        }],
    }
    _write_json(tmp_path / "unit.json", receipt)
    (tmp_path / "repository-ssa.pkl").write_bytes(b"published")

    order = planned_dependency_repair_order(receipt)
    healthy = stage_0_compilation_unit(load_compilation_unit(tmp_path))
    output = capsys.readouterr().out

    assert [row["qualified_names"] for row in order] == [
        ("Compiler.leaf",), ("Compiler.middle",), ("Compiler.other",),
    ]
    assert healthy is True
    assert "Meta-compilation repair order (dependencies first)" in output
    assert "unit 0: Compiler.leaf [leaf]" in output
    assert "unit 1: Compiler.middle [after 0]" in output


def _write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def test_stage_zero_names_stale_compiler_toolchain_plan(tmp_path, capsys):
    from src.compiler.project_compilation_product import (
        compiler_toolchain_fingerprint,
    )

    stale_toolchain = compiler_toolchain_fingerprint()
    stale_toolchain["sha256"] = "0" * 64
    for item in stale_toolchain["files"]:
        if item["path"] == "src/compiler/precompile_to_ssa.py":
            item["sha256"] = "0" * 64
    _write_json(tmp_path / "unit.json", {
        "status": "partial",
        "unit_index": 0,
        "unit": {"dependency_units": []},
        "repository_ssa_complete": False,
        "repository_ssa_accounting": [],
        "compiler_toolchain": stale_toolchain,
    })
    (tmp_path / "repository-ssa.pkl").write_bytes(b"published")

    stage_0_compilation_unit(load_compilation_unit(tmp_path))
    output = capsys.readouterr().out

    assert "stale compiler-toolchain plan" in output
    assert "src/compiler/precompile_to_ssa.py" in output


def test_compilation_unit_routes_resource_failure_before_ssa(tmp_path):
    _write_json(tmp_path / "compile-progress.json", {
        "qualified_name": "lower_stage",
        "current": {
            "elapsed_seconds": 12.0,
            "message": "planning complete graph",
        },
    })
    _write_json(tmp_path / "failure.json", {
        "qualified_name": "lower_stage",
        "error_type": "ResourceLimitExceeded",
        "error": "elapsed time exceeded limit",
    })

    snapshot = load_compilation_unit(tmp_path)

    assert snapshot["state"] == "resource-failure"
    assert snapshot["qualified_name"] == "lower_stage"
    assert snapshot["repository"] is None


def test_resource_failure_routes_from_durable_meta_phase(tmp_path, capsys):
    _write_json(tmp_path / "compile-progress.json", {
        "qualified_names": ["Compiler.recursive_scc"],
        "current": {
            "phase": "deployment-instantiation",
            "elapsed_seconds": 226.0,
            "message": "ssa-source: instantiating complete deployment",
        },
    })
    _write_json(tmp_path / "failure.json", {
        "error_type": "ResourceLimitExceeded",
        "error": "private memory exceeded limit",
        "stage": {
            "phase": "deployment-instantiation",
            "elapsed_seconds": 226.0,
            "message": "ssa-source: instantiating complete deployment",
        },
    })

    healthy = stage_0_compilation_unit(load_compilation_unit(tmp_path))
    output = capsys.readouterr().out

    assert healthy is False
    assert "last durable phase deployment-instantiation" in output
    assert "Divide the selected authored/SCC activation closure" in output


def test_resolved_worker_persists_meta_compilation_phase_and_memory(tmp_path):
    report = _planned_unit_progress_writer(
        tmp_path, 7, {"qualified_names": ["Compiler.lower"]},
    )

    report("ssa-source: instantiating complete control/operator deployment")

    progress = json.loads(
        (tmp_path / "compile-progress.json").read_text(encoding="utf-8")
    )
    assert progress["unit_index"] == 7
    assert progress["qualified_names"] == ["Compiler.lower"]
    assert progress["current"]["phase"] == "deployment-instantiation"
    assert progress["current"]["private_bytes"] > 0


def test_resolved_worker_retries_transient_windows_progress_replace(
    tmp_path, monkeypatch,
):
    real_replace = os.replace
    attempts = []

    def briefly_denied(source, destination):
        attempts.append((source, destination))
        if len(attempts) < 4:
            raise PermissionError("observer retains sharing handle")
        real_replace(source, destination)

    monkeypatch.setattr(compile_project_catalogue.os, "replace", briefly_denied)
    report = _planned_unit_progress_writer(
        tmp_path, 2, {"qualified_names": ["Compiler.lower"]},
    )

    report("ssa-source: lowering full planned source to repository SSA")

    assert len(attempts) == 4
    assert json.loads((tmp_path / "compile-progress.json").read_text())[
        "current"
    ]["phase"] == "repository-ssa-lowering"


def test_compilation_unit_routes_published_repository_into_ssa_tree(tmp_path):
    _write_json(tmp_path / "unit.json", {
        "qualified_name": "lower_stage",
        "repository_ssa_complete": True,
    })
    (tmp_path / "repository-ssa.pkl").write_bytes(b"published")

    snapshot = load_compilation_unit(tmp_path)

    assert snapshot["state"] == "published"
    assert snapshot["qualified_name"] == "lower_stage"
    assert snapshot["repository"] == tmp_path / "repository-ssa.pkl"


def test_compilation_unit_reads_single_resolved_graph_name_and_partial_state(
    tmp_path,
):
    _write_json(tmp_path / "unit.json", {
        "schema": "turing.resolved-process-graph-unit.v1",
        "status": "partial",
        "qualified_names": ["Compiler.lower"],
        "repository_ssa_complete": False,
        "repository_ssa_accounting": [{
            "qualified_name": "Compiler.lower",
            "complete": False,
            "shortfalls": [{"kind": "conditional-accounting-mismatch"}],
        }],
    })
    (tmp_path / "repository-ssa.pkl").write_bytes(b"published")

    snapshot = load_compilation_unit(tmp_path)

    assert snapshot["state"] == "published"
    assert snapshot["qualified_name"] == "Compiler.lower"
    assert snapshot["receipt"]["status"] == "partial"


def test_source_only_subdivision_is_terminal_without_inventing_ssa(
    tmp_path, capsys,
):
    _write_json(tmp_path / "unit.json", {
        "schema": "turing.process-graph-subdivision-product.v1",
        "status": "source-only",
        "qualified_names": ["Compiler.orchestrate"],
        "repository_ssa_complete": False,
        "repository_ssa_accounting": [{
            "qualified_name": "Compiler.orchestrate",
            "shortfalls": [{
                "kind": "no-numeric-regions",
                "action": "retain-authored-source",
            }],
        }],
    })

    snapshot = load_compilation_unit(tmp_path)
    healthy = stage_0_compilation_unit(snapshot)
    output = capsys.readouterr().out

    assert snapshot["state"] == "source-only"
    assert snapshot["repository"] is None
    assert healthy is True
    assert "authored source remains authoritative" in output
    assert "no-numeric-regions -> retain-authored-source" in output


def test_compilation_unit_distinguishes_live_worker_from_interruption(tmp_path):
    _write_json(tmp_path / "compile-progress.json", {
        "qualified_name": "lower_stage",
        "process_id": os.getpid(),
        "current": {
            "elapsed_seconds": 3.0,
            "message": "planning",
        },
    })

    snapshot = load_compilation_unit(tmp_path)

    assert snapshot["state"] == "running"


def test_compilation_unit_routes_structured_subdivision_frontier(
    tmp_path, capsys,
):
    _write_json(tmp_path / "failure.json", {
        "qualified_names": ["Compiler.lower"],
        "error_type": "CompilationSubdivisionRequired",
        "error": "loop control owner is not lowerable",
        "frontier_kind": "compilation-subdivision-required",
        "subdivision_boundaries": [{
            "kind": "loop-control-owner",
            "loop_node_id": 33,
            "region_indices": [0],
            "blockers": ["opaque-state-effect"],
        }],
    })

    healthy = stage_0_compilation_unit(load_compilation_unit(tmp_path))
    output = capsys.readouterr().out

    assert healthy is False
    assert "subdivide at loop-control-owner node=33 regions=(0,)" in output
    assert "enqueue the named boundary" in output


def test_compilation_product_reads_live_terminal_and_pending_frontier(tmp_path):
    _write_json(tmp_path / "progress.json", {
        "completed": [
            {"qualified_name": "good", "status": "complete"},
            {
                "qualified_name": "large",
                "status": "failed",
                "error_type": "ResourceLimitExceeded",
                "error": "committed/private memory 10 exceeded 9 bytes",
            },
        ],
        "running": [{"qualified_name": "current", "process_id": os.getpid()}],
        "pending": ["later"],
    })

    snapshot = load_compilation_product(tmp_path)

    assert snapshot["sealed"] is False
    assert len(snapshot["records"]) == 2
    assert snapshot["pending"] == ("later",)


def test_resolved_product_reports_authored_names_and_structural_causes(
    tmp_path, capsys,
):
    _write_json(tmp_path / "manifest.json", {
        "units": [
            {
                "status": "failed",
                "error_type": "ValueError",
                "error": "loop body could not compile: opaque effect",
                "unit": {"qualified_names": ["Compiler.loop"]},
            },
            {
                "status": "partial",
                "qualified_names": ["Compiler.branch"],
                "repository_ssa_accounting": [{
                    "shortfalls": [{
                        "kind": "conditional-accounting-mismatch",
                    }],
                }],
            },
        ],
    })

    healthy = stage_0_compilation_product(load_compilation_product(tmp_path))
    output = capsys.readouterr().out

    assert healthy is False
    assert "Compiler.loop" in output
    assert "conditional-accounting-mismatch" in output


def test_resolved_product_counts_recursive_scc_as_one_failed_unit(
    tmp_path, capsys,
):
    _write_json(tmp_path / "manifest.json", {
        "units": [{
            "status": "failed",
            "error_type": "ResourceLimitExceeded",
            "error": "committed/private memory exceeded limit",
            "unit": {"qualified_names": [
                "Compiler.lower", "Compiler.lower_branch",
            ]},
        }],
    })

    healthy = stage_0_compilation_product(load_compilation_product(tmp_path))
    output = capsys.readouterr().out

    assert healthy is False
    assert "1 failed unit(s)" in output
    assert "1 x ResourceLimitExceeded:memory" in output
    assert "Compiler.lower, Compiler.lower_branch" in output


def test_subdivision_product_routes_integrals_and_unverified_work(
    tmp_path, capsys,
):
    _write_json(tmp_path / "manifest.json", {
        "schema": "turing.process-graph-subdivision-product-plan.v1",
        "integrals": [
            {
                "status": "source-only",
                "qualified_names": ["Compiler.control_only"],
            },
            {
                "status": "compiled-unverified",
                "qualified_names": ["Compiler.numeric_region"],
            },
        ],
    })

    snapshot = load_compilation_product(tmp_path)
    healthy = stage_0_compilation_product(snapshot)
    output = capsys.readouterr().out

    assert snapshot["kind"] == "subdivision"
    assert len(snapshot["records"]) == 2
    assert healthy is False
    assert "meta-compilation integral product" in output
    assert "1 source-only integral(s)" in output
    assert "1 compiled integral(s) still require exact semantic" in output
    assert "Compiler.numeric_region" in output


def test_standalone_verified_subdivision_unit_is_a_clean_product(
    tmp_path, capsys,
):
    _write_json(tmp_path / "unit.json", {
        "schema": "turing.process-graph-subdivision-product.v1",
        "status": "verified",
        "qualified_names": ["Compiler.lower"],
        "regions": [{
            "verification_status": "verified",
            "probe_count": 3,
        }],
    })

    snapshot = load_compilation_product(tmp_path)
    healthy = stage_0_compilation_product(snapshot)
    output = capsys.readouterr().out

    assert snapshot["kind"] == "subdivision"
    assert snapshot["sealed"] is True
    assert healthy is True
    assert "1 verified integral" in output
    assert "probe count=3" in output


def test_subdivision_product_reports_region_shortfall_kinds(tmp_path, capsys):
    _write_json(tmp_path / "manifest.json", {
        "schema": "turing.process-graph-subdivision-product-plan.v1",
        "integrals": [{
            "status": "partial",
            "qualified_names": ["Compiler.loop"],
            "regions": [{
                "region_index": 3,
                "shortfalls": [{
                    "kind": "unresolved-boundary-types",
                    "value_identities": [{
                        "value_id": 41,
                        "identity_token_chain": [
                            "line:000000000123",
                            "column:000000000007",
                            "field:node.op",
                            "value:add",
                            "version:2",
                        ],
                    }],
                }],
            }],
        }],
    })

    healthy = stage_0_compilation_product(load_compilation_product(tmp_path))
    output = capsys.readouterr().out

    assert healthy is False
    assert "1 x unresolved-boundary-types" in output
    assert "Compiler.loop: add@123:7/version:2" in output
    assert "NEXT: no deeper deterministic subdivision was published" in output


def test_subdivision_product_renders_boundary_input_by_authored_label(
    tmp_path, capsys,
):
    _write_json(tmp_path / "manifest.json", {
        "schema": "turing.process-graph-subdivision-product-plan.v1",
        "integrals": [{
            "status": "partial",
            "qualified_names": ["Compiler.loop"],
            "regions": [{
                "shortfalls": [{
                    "kind": "unresolved-boundary-types",
                    "value_identities": [{
                        "identity_token_chain": [
                            "line:1000000000000",
                            "column:1000000000000",
                            "field:node.label",
                            "value:target_id",
                            "field:node.op",
                            "value:input",
                            "version:4",
                        ],
                    }],
                }],
            }],
        }],
    })

    stage_0_compilation_product(load_compilation_product(tmp_path))
    output = capsys.readouterr().out

    assert "Compiler.loop: input[target_id]/version:4" in output
    assert "1000000000000" not in output


def test_subdivision_product_renders_boundary_expression_not_source_slot(
    tmp_path, capsys,
):
    _write_json(tmp_path / "manifest.json", {
        "schema": "turing.process-graph-subdivision-product-plan.v1",
        "integrals": [{
            "status": "partial",
            "qualified_names": ["Compiler.loop"],
            "regions": [{
                "shortfalls": [{
                    "kind": "unresolved-boundary-types",
                    "value_identities": [{
                        "identity_token_chain": [
                            "line:000000005251",
                            "column:000000000016",
                            "field:node.ast",
                            "value:Subscript",
                            "value:(",
                            "value:'",
                            "value:self",
                            "value:'",
                            "value:'",
                            "value:external_values",
                            "value:'",
                            "value:'",
                            "value:initial_id",
                            "value:'",
                            "field:node.op",
                            "value:IndexedStore",
                            "version:0",
                        ],
                    }],
                }],
            }],
        }],
    })

    stage_0_compilation_product(load_compilation_product(tmp_path))
    output = capsys.readouterr().out

    assert "Compiler.loop: self.external_values[initial_id]/version:0" in output
    assert "IndexedStore@5251:16" not in output


def test_subdivision_product_routes_published_child_plan(tmp_path, capsys):
    _write_json(tmp_path / "manifest.json", {
        "schema": "turing.process-graph-subdivision-product-plan.v1",
        "integrals": [{"status": "failed"}],
        "subdivision_integral_count": 2,
    })

    stage_0_compilation_product(load_compilation_product(tmp_path))
    output = capsys.readouterr().out

    assert "2 deterministic child integral(s) queued" in output
    assert "NEXT: compile the queued subdivision plan" in output


def test_live_product_is_not_clean_while_worker_is_running(tmp_path):
    _write_json(tmp_path / "progress.json", {
        "completed": [],
        "running": [{"unit_index": 0, "process_id": os.getpid()}],
        "pending": [],
    })

    snapshot = load_compilation_product(tmp_path)

    assert stage_0_compilation_product(snapshot) is False
