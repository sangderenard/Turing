from __future__ import annotations

import ast
import hashlib
import json
import pickle
from types import SimpleNamespace

from src.compiler.project_compilation_product import (
    compilation_creep_frontier,
    compile_project_bootstrap_creep,
    compile_process_graph_creep,
    compile_process_graph_subdivision_plan,
    compile_resolved_process_graph_unit,
    compiler_toolchain_fingerprint,
    authored_control_contract,
    authored_definition_sha256,
    authored_parameter_contract,
    authored_return_contract,
    detach_repository_ssa_frontend,
    discover_authored_calls,
    dependency_ordered_records,
    dependency_ordered_authored_calls,
    encoded_call_name,
    native_unit_name,
    partition_authored_source,
    open_project_compilation_product,
    publish_process_graph_subdivision_plan,
    ready_process_graph_unit_indices,
    resident_bytes,
    source_region_integral_accounting,
    verify_structural_resident_table_integral,
    verify_project_scalar_units_automatically,
)


def test_compiler_toolchain_fingerprint_is_deterministic_and_auditable():
    first = compiler_toolchain_fingerprint()
    second = compiler_toolchain_fingerprint()

    assert first == second
    assert len(first["sha256"]) == 64
    paths = [item["path"] for item in first["files"]]
    assert paths == sorted(paths)
    assert "src/compiler/precompile_to_ssa.py" in paths
    assert "src/common/tensors/topological_reducer.py" in paths


def test_resolved_unit_refuses_a_stale_compiler_toolchain_plan(tmp_path):
    graph_path = tmp_path / "graph.pkl"
    graph_path.write_bytes(b"not reached")
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps({
        "schema": "turing.compilation-unit-plan.v1",
        "compiler_toolchain": {
            "schema": "turing.compiler-toolchain-fingerprint.v1",
            "sha256": "0" * 64,
            "files": [],
        },
        "units": [{
            "function_references": [1],
            "qualified_names": ["Compiler.stage"],
        }],
    }), encoding="utf-8")

    try:
        compile_resolved_process_graph_unit(
            graph_path, plan_path, 0, tmp_path / "output",
        )
    except ValueError as error:
        assert "compiler toolchain changed" in str(error)
    else:
        raise AssertionError("stale compiler plan was accepted")


def test_empty_subdivision_plan_seals_a_bounded_product(tmp_path):
    plan_path = tmp_path / "subdivision-integrals.json"
    plan_path.write_text(json.dumps({
        "schema": "turing.process-graph-subdivision-plan.v1",
        "integrals": [],
    }), encoding="utf-8")

    manifest = compile_process_graph_subdivision_plan(
        plan_path, tmp_path / "product", jobs=2,
    )

    assert manifest["integrals"] == []
    assert manifest["counts"] == {
        "compiled-unverified": 0,
        "partial": 0,
        "source-only": 0,
        "failed": 0,
    }
    assert manifest["subdivision_integral_count"] == 0
    assert json.loads(
        (tmp_path / "product" / "manifest.json").read_text(encoding="utf-8")
    ) == manifest


def test_compiler_creep_crawls_nested_plans_and_stops_repeated_cut(
    tmp_path, monkeypatch,
):
    graph = tmp_path / "resolved.pkl"
    graph.write_bytes(b"resolved")
    plan = tmp_path / "units.json"
    plan.write_text(json.dumps({
        "schema": "turing.compilation-unit-plan.v1",
        "units": [{"qualified_names": ["Compiler.root"]}],
    }), encoding="utf-8")
    child = {
        "schema": "turing.process-graph-subdivision-plan.v1",
        "resolved_process_graph": graph.as_posix(),
        "process_graph_unit_plan": plan.as_posix(),
        "integrals": [{
            "identity_token_chain": [
                "process-graph-subdivision", "Compiler.root",
                "function-shell",
            ],
            "qualified_names": ["Compiler.root"],
        }],
    }
    calls = []

    def write_json(path, value):
        path.write_text(
            json.dumps(value), encoding="utf-8", newline="\n",
        )

    def resolved(_graph, _plan, output, **_kwargs):
        calls.append("resolved")
        output.mkdir(parents=True, exist_ok=True)
        write_json(output / "subdivision-integrals.json", child)
        return {
            "counts": {"failed": 1},
            "units": [{"status": "failed"}],
            "subdivision_integral_count": 1,
            "subdivision_integrals": "subdivision-integrals.json",
        }

    def subdivision(_plan, output, **_kwargs):
        calls.append("subdivision")
        output.mkdir(parents=True, exist_ok=True)
        write_json(output / "subdivision-integrals.json", child)
        return {
            "counts": {"verified": 1},
            "integrals": [{
                "status": "verified",
                "qualified_names": ["Compiler.root"],
                "integral": child["integrals"][0],
            }],
            "subdivision_integral_count": 1,
            "subdivision_integrals": "subdivision-integrals.json",
        }

    monkeypatch.setattr(
        "src.compiler.project_compilation_product."
        "compile_resolved_process_graph_plan",
        resolved,
    )
    monkeypatch.setattr(
        "src.compiler.project_compilation_product."
        "compile_process_graph_subdivision_plan",
        subdivision,
    )

    manifest = compile_process_graph_creep(
        graph, plan, tmp_path / "creep", max_subdivision_depth=8,
    )

    assert calls == ["resolved", "subdivision"]
    assert len(manifest["rounds"]) == 2
    assert len(manifest["verified_products"]) == 1
    assert manifest["fixed_points"][0]["kind"] == (
        "repeated-subdivision-plan"
    )
    assert manifest["status"] == "frontier"
    assert json.loads(
        (tmp_path / "creep" / "manifest.json").read_text(encoding="utf-8")
    ) == manifest


def test_project_bootstrap_creep_feeds_only_newly_verified_products(
    tmp_path, monkeypatch,
):
    source = tmp_path / "compiler_part.py"
    source.write_text("def leaf(value):\n    return value + 1\n", encoding="utf-8")
    calls = []

    def compile_round(_source, output, **kwargs):
        round_index = len(calls)
        calls.append({
            "output": output,
            "bootstrap_products": tuple(kwargs["bootstrap_products"]),
            "seed_product": kwargs["seed_product"],
            "emit_native": kwargs["emit_native"],
        })
        verified = round_index == 0
        return {
            "units": [{
                "qualified_name": "leaf",
                "status": "complete",
                "source_region_integrals": [],
            }],
            "automatic_native_verification": [{
                "qualified_name": "leaf",
                "status": "verified" if verified else "unsupported",
            }],
            "creep_frontier": [],
        }

    monkeypatch.setattr(
        "src.compiler.project_compilation_product.compile_project_product",
        compile_round,
    )
    monkeypatch.setattr(
        "src.compiler.compiler_bootstrap_runtime."
        "publish_compiler_bootstrap_products",
        lambda _paths: tmp_path / "registry.json",
    )

    prior_seed = tmp_path / "previous-generation"
    manifest = compile_project_bootstrap_creep(
        source, tmp_path / "creep", seed_product=prior_seed,
        max_rounds=8,
    )

    assert len(calls) == 2
    assert calls[0]["bootstrap_products"] == ()
    assert calls[0]["emit_native"] is True
    assert calls[0]["seed_product"] == prior_seed.resolve()
    assert calls[1]["bootstrap_products"] == (
        (tmp_path / "creep" / "round_000").resolve(),
    )
    assert calls[1]["seed_product"] == (
        tmp_path / "creep" / "round_000"
    ).resolve()
    assert manifest["installed_qualified_names"] == ["leaf"]
    assert manifest["fixed_point"]["kind"] == "no-new-proven-deployments"


def test_project_bootstrap_creep_automatically_crawls_failed_unit_plan(
    tmp_path, monkeypatch,
):
    source = tmp_path / "compiler_part.py"
    source.write_text("def root(value):\n    return value\n", encoding="utf-8")
    observed = []

    def compile_round(_source, output, **_kwargs):
        unit_root = output / "units" / "root"
        unit_root.mkdir(parents=True)
        (unit_root / "resolved-process-graph.pkl").write_bytes(b"graph")
        plan = unit_root / "process-graph-units.json"
        plan.write_text("{}", encoding="utf-8")
        return {
            "units": [{
                "qualified_name": "root",
                "status": "failed",
                "path": "units/root",
                "process_graph_unit_plan": (
                    "units/root/process-graph-units.json"
                ),
            }],
            "automatic_native_verification": [],
            "creep_frontier": [{"qualified_name": "root"}],
        }

    def crawl(graph, plan, output, **kwargs):
        observed.append((graph, plan, output, kwargs["bootstrap_products"]))
        return {
            "status": "frontier",
            "verified_products": [{"status": "verified"}],
            "fixed_points": [{"kind": "minimum-integral"}],
        }

    monkeypatch.setattr(
        "src.compiler.project_compilation_product.compile_project_product",
        compile_round,
    )
    monkeypatch.setattr(
        "src.compiler.project_compilation_product.compile_process_graph_creep",
        crawl,
    )

    manifest = compile_project_bootstrap_creep(
        source, tmp_path / "creep", max_rounds=4,
    )

    assert len(observed) == 1
    assert observed[0][0].name == "resolved-process-graph.pkl"
    assert observed[0][1].name == "process-graph-units.json"
    assert manifest["rounds"][0]["process_graph_creeps"] == [{
        "qualified_name": "root",
        "product": (
            tmp_path / "creep" / "round_000" / "process-graph-creeps"
            / encoded_call_name("root")
        ).as_posix(),
        "status": "frontier",
        "verified_product_count": 1,
        "fixed_point_count": 1,
    }]


def test_process_graph_workers_wait_for_terminal_dependencies_not_success():
    units = (
        {"qualified_names": ["leaf"], "dependency_units": []},
        {"qualified_names": ["middle"], "dependency_units": [0]},
        {"qualified_names": ["other"], "dependency_units": []},
        {"qualified_names": ["root"], "dependency_units": [1, 2]},
    )
    records = [None, None, None, None]

    assert ready_process_graph_unit_indices(
        [0, 1, 2, 3], units, records,
    ) == (0, 2)

    # A partial terminal dependency still releases its caller. The caller
    # will retain authored source for it rather than linking an unsafe unit.
    records[0] = {"status": "partial"}
    assert ready_process_graph_unit_indices(
        [1, 2, 3], units, records,
    ) == (1, 2)

    records[1] = {"status": "verified"}
    records[2] = {"status": "failed"}
    assert ready_process_graph_unit_indices([3], units, records) == (3,)


def test_structured_refusals_publish_name_based_deterministic_integrals(tmp_path):
    graph = tmp_path / "resolved.pkl"
    unit_plan = tmp_path / "units.json"
    graph.write_bytes(b"resolved graph")
    unit_plan.write_text("{}", encoding="utf-8")
    records = ({
        "frontier_kind": "compilation-subdivision-required",
        "unit_index": 4,
        "unit": {
            "qualified_names": ["Compiler.lower"],
            "function_references": [92],
        },
        "subdivision_boundaries": [{
            "kind": "loop-control-owner",
            "loop_node_id": 33,
            "region_indices": [7, 2],
            "blockers": ["opaque-state-effect"],
        }],
    },)

    first = publish_process_graph_subdivision_plan(
        tmp_path / "first", records, graph, unit_plan,
    )
    second = publish_process_graph_subdivision_plan(
        tmp_path / "second", records, graph, unit_plan,
    )

    assert first == second
    integral, = first["integrals"]
    assert integral["identity_token_chain"] == [
        "process-graph-subdivision",
        "Compiler.lower",
        "loop-control-owner",
        "loop:33",
        "region:2",
        "region:7",
    ]
    assert integral["function_references"] == [92]
    assert integral["region_indices"] == [2, 7]
    assert json.loads(
        (tmp_path / "first" / "subdivision-integrals.json").read_text()
    ) == first


def test_subdivision_plan_uses_actual_fallback_owner_not_selected_parent(
    tmp_path,
):
    graph = tmp_path / "resolved.pkl"
    unit_plan = tmp_path / "units.json"
    graph.write_bytes(b"resolved graph")
    unit_plan.write_text(json.dumps({
        "schema": "turing.compilation-unit-plan.v1",
        "units": [
            {
                "qualified_names": ["Compiler.loop_leaf"],
                "function_references": [19],
                "dependency_units": [],
            },
            {
                "qualified_names": ["Compiler.parent"],
                "function_references": [29],
                "dependency_units": [0],
            },
        ],
    }), encoding="utf-8")
    record = {
        "frontier_kind": "compilation-subdivision-required",
        "unit_index": 1,
        "unit": {
            "qualified_names": ["Compiler.parent"],
            "function_references": [29],
        },
        "subdivision_boundaries": [{
            "kind": "loop-control-owner",
            "loop_node_id": 33,
            "region_indices": [0],
            "blockers": ["opaque-state-effect"],
            "function_reference": 19,
            "qualified_name": "Compiler.loop_leaf",
        }],
    }

    plan = publish_process_graph_subdivision_plan(
        tmp_path / "product", (record, record), graph, unit_plan,
    )

    integral, = plan["integrals"]
    assert integral["parent_unit_index"] == 0
    assert integral["function_references"] == [19]
    assert integral["qualified_names"] == ["Compiler.loop_leaf"]
    assert integral["identity_token_chain"] == [
        "process-graph-subdivision", "Compiler.loop_leaf",
        "loop-control-owner", "loop:33", "region:0",
    ]


def test_resource_bound_scc_divides_into_name_correlated_function_shells(
    tmp_path,
):
    graph = tmp_path / "resolved.pkl"
    unit_plan = tmp_path / "units.json"
    graph.write_bytes(b"resolved graph")
    unit_plan.write_text(json.dumps({
        "schema": "turing.compilation-unit-plan.v1",
        "units": [{
            "qualified_names": ["Compiler.left", "Compiler.right"],
            "function_references": [31, 47],
            "dependency_units": [],
            "recursive": True,
        }],
    }), encoding="utf-8")
    record = {
        "status": "failed",
        "error_type": "ResourceLimitExceeded",
        "resource": "private-memory",
        "stage": {"phase": "deployment-instantiation"},
        "unit_index": 0,
        "unit": {
            "qualified_names": ["Compiler.left", "Compiler.right"],
            "function_references": [31, 47],
            "recursive": True,
        },
    }

    plan = publish_process_graph_subdivision_plan(
        tmp_path / "product", (record,), graph, unit_plan,
    )

    assert [item["identity_token_chain"] for item in plan["integrals"]] == [
        ["process-graph-subdivision", "Compiler.left", "function-shell"],
        ["process-graph-subdivision", "Compiler.right", "function-shell"],
    ]
    assert [item["function_references"] for item in plan["integrals"]] == [
        [31], [47],
    ]
    assert all(
        item["blockers"] == [
            "resource:private-memory", "phase:deployment-instantiation",
        ]
        for item in plan["integrals"]
    )


def test_authored_parameter_contract_distinguishes_used_and_unused_formals():
    contract = authored_parameter_contract("""
def outer(used, unused, *, keyword=1):
    def nested():
        return unused
    return used + keyword
""", "outer")

    assert contract == {
        "parameters": ["used", "unused", "keyword"],
        "used_parameters": ["used", "keyword"],
    }


def test_authored_definition_hash_ignores_sibling_edits_but_not_body_edits():
    first = "def selected(value):\n    return value + 1\n\ndef sibling():\n    return 2\n"
    sibling_changed = first.replace("return 2", "return 200")
    selected_changed = first.replace("value + 1", "value + 2")

    fingerprint = authored_definition_sha256(first, "selected")

    assert authored_definition_sha256(sibling_changed, "selected") == fingerprint
    assert authored_definition_sha256(selected_changed, "selected") != fingerprint
from src.transmogrifier.function_table import FunctionTable
from src.transmogrifier.ssa import IRModule


def test_native_emission_guard_rejects_unlowered_root_source_values():
    import src.compiler.project_compilation_product as product_module
    from src.transmogrifier.ssa import BasicBlock, Function

    module = IRModule(functions={
        "root": Function(
            "root",
            [],
            {"entry": BasicBlock("entry")},
            metadata={
                "unresolved_required_source_values": ((7, "List", ()),),
            },
        ),
    })

    try:
        product_module._require_native_root_semantics(module, "root")
    except RuntimeError as error:
        assert "unlowered source values (7,)" in str(error)
    else:
        raise AssertionError("incomplete root was accepted for native emission")


def test_root_semantic_accounting_accepts_only_declared_workspace_and_projection():
    import src.compiler.project_compilation_product as product_module
    from src.transmogrifier.ssa import Function, SSAValue

    function = Function(
        "root",
        [
            SSAValue(1, dtype="int64", accounting={
                "compiler_frame_storage": "root",
            }),
            SSAValue(2, dtype="int64", accounting={
                "linked_call_frame_storage": "callee",
            }),
            SSAValue(3, dtype="int64"),
            SSAValue(4, dtype="int64"),
        ],
        {},
        metadata={
            "scalar_source_transforms": (
                (3, "items", "materialized_length"),
            ),
        },
    )

    assert product_module._unexplained_root_argument_ids(function) == (4,)


def test_unexplained_root_argument_details_persist_structural_use_sites():
    import src.compiler.project_compilation_product as product_module
    from src.transmogrifier.ssa import BasicBlock, Function, Instr, SSAValue

    escaped = SSAValue(4, dtype="int64")
    result = SSAValue(5, dtype="bool")
    function = Function(
        "root",
        [escaped],
        {"entry": BasicBlock("entry", [Instr(
            "Call", [escaped], result,
            attributes={
                "callee": "root__planned_region_2",
                "region_index": 2,
                "feed_ids": (4,),
            },
        )])},
    )

    assert product_module._unexplained_root_argument_details(function) == ({
        "value_id": 4,
        "dtype": "int64",
        "shape": [],
        "accounting": {},
        "uses": [{
            "block": "entry",
            "instruction_index": 0,
            "operation": "Call",
            "role": "arg:0",
            "callee": "root__planned_region_2",
            "region_index": 2,
        }],
    },)


def test_source_region_accounting_accepts_only_a_complete_physical_abi():
    from src.transmogrifier.ssa import BasicBlock, Function, Instr, SSAValue

    index = SSAValue(2, dtype="float64")
    zero = SSAValue(10, dtype="int")
    converted = SSAValue(32, dtype="int")
    predicate = SSAValue(33, dtype="bool")
    function = Function(
        "builder__planned_region_1",
        [index],
        {"entry": BasicBlock("entry", [
            Instr("Const", [], zero, {"value": 0}),
            Instr("Cast", [index], converted, {"target_dtype": "int"}),
            Instr("Ge", [converted, zero], predicate),
        ])},
        metadata={
            "source_region_integral": {
                "schema": "turing.source-region-integral.v1",
                "owner": "builder",
                "plan_name": "region_1",
                "region_index": 1,
                "closure_id": 4,
                "identity_token_chain": (
                    "source-region", "builder", "closure:4", "region_1",
                ),
            },
        },
    )
    module = IRModule({function.name: function})

    records = source_region_integral_accounting(
        module, {function.name: (predicate, zero, converted)},
    )

    assert len(records) == 1
    assert records[0]["complete"] is True
    assert records[0]["shortfalls"] == []
    assert records[0]["identity_token_chain"] == [
        "source-region", "builder", "closure:4", "region_1",
    ]


def test_source_region_accounting_rejects_an_unranked_address_base():
    from src.transmogrifier.ssa import BasicBlock, Function, Instr, SSAValue

    base = SSAValue(1, dtype="float64")
    index = SSAValue(2, dtype="int")
    address = SSAValue(3, dtype="int64")
    value = SSAValue(4, dtype="float64")
    function = Function(
        "builder__planned_region_2",
        [base, index],
        {"entry": BasicBlock("entry", [
            Instr("GetElementPtr", [base, index], address),
            Instr("Load", [address], value),
        ])},
        metadata={
            "source_region_integral": {
                "schema": "turing.source-region-integral.v1",
                "owner": "builder",
                "plan_name": "region_2",
                "region_index": 2,
                "closure_id": 5,
                "identity_token_chain": (
                    "source-region", "builder", "closure:5", "region_2",
                ),
            },
        },
    )

    record = source_region_integral_accounting(
        IRModule({function.name: function}), {function.name: (value,)},
    )[0]

    assert record["complete"] is False
    assert record["shortfalls"] == [{
        "kind": "unresolved-address-base-contracts",
        "value_ids": [1],
    }]


def test_structural_resident_table_verifier_proves_mutation_and_abi():
    from src.common.tensors.fused_ir import FusedProgram, OpStep
    from src.compiler.precompile_to_ssa import (
        lower_fused_integral_to_repository_ssa,
    )

    program = FusedProgram(
        version=1,
        feeds={13, 19, 289},
        steps=[OpStep(0, "IndexedStore", [289, 13, 19], {}, 290)],
        outputs={"mutated": 290},
        meta={},
        extras={
            "structural_resident_table_contract": {
                "schema": "turing.structural-resident-table-integral.v1",
                "sequences": [{
                    "sequence_id": 289,
                    "policy": "unique",
                    "column_count": 2,
                    "writable": True,
                    "column_dtypes": ["int64", "int64"],
                    "storage_identity": "Builder.external_values",
                    "value_record": "SSAValue",
                    "value_optional": True,
                }],
                "stores": [{
                    "effect_value_id": 290,
                    "key_value_id": 13,
                    "stored_value_id": 19,
                    "sequence_value_id": 289,
                }],
            },
        },
    )
    module, outputs, _exports, shortfalls = (
        lower_fused_integral_to_repository_ssa(
            program, function_name="restore_external_value",
        )
    )
    function = module.functions["restore_external_value"]
    function.metadata["subdivision_identity_token_chain"] = (
        "process-graph-subdivision", "Builder.lower", "region:6",
    )

    verification = verify_structural_resident_table_integral(
        module,
        outputs,
        "restore_external_value",
        repository_ssa_sha256="a" * 64,
    )

    assert shortfalls == ()
    assert verification["status"] == "verified"
    assert verification["probe_count"] == 3
    assert [probe["outcome"] for probe in verification["probes"]] == [
        "inserted", "updated", "capacity-exhausted",
    ]
    assert verification["abi"]["stored_record_identity"] == "SSAValue"


def test_project_discovery_includes_lexically_addressable_nested_functions():
    calls = discover_authored_calls("""
def first(value):
    def nested(item):
        return item
    return nested(value)

class Surface:
    def method(self, value):
        return value

    async def asynchronous(self):
        return 1
""")

    assert [(call.qualified_name, call.kind) for call in calls] == [
        ("Surface.asynchronous", "method"),
        ("Surface.method", "method"),
        ("first", "function"),
        ("first.<locals>.nested", "nested-function"),
    ]


def test_nested_function_partition_keeps_exact_lexical_shell_and_body():
    source = """import math

def outer(value):
    discarded = value * 1000
    def integral(item):
        return math.sin(item)
    return integral(value) + discarded

def unrelated(value):
    return value
"""

    partitioned, symbols = partition_authored_source(
        source, "outer.<locals>.integral",
    )

    assert "def outer(value):" in partitioned
    assert "discarded =" not in partitioned
    assert "def integral(item):" in partitioned
    assert "return math.sin(item)" in partitioned
    assert "return integral(value)" not in partitioned
    assert "def unrelated" not in partitioned
    assert symbols == ("integral", "math", "outer")
    ast.parse(partitioned)


def test_parent_call_depends_on_its_nested_authored_integral():
    source = """
def outer(value):
    def integral(item):
        return item + 1
    return integral(value)
"""

    assert dependency_ordered_authored_calls(source, ("outer",)) == (
        "outer.<locals>.integral", "outer",
    )


def test_authored_return_contract_does_not_borrow_nested_returns():
    source = """
def outer(flag):
    def child():
        return 3
    if flag:
        return child()
    return
"""

    assert authored_return_contract(source, "outer") == {
        "explicit_return_count": 2,
        "valued_return_count": 1,
        "requires_value_publication": True,
        "annotation": None,
    }


def test_authored_control_contract_excludes_children_and_counts_loop_exits():
    source = """
def encode(value):
    if value < 0:
        raise ValueError(value)
    while True:
        if value:
            value -= 1
        else:
            break
    def child():
        while True:
            continue
    return value
"""

    assert authored_control_contract(source, "encode") == {
        "if_count": 2,
        "raise_guard_count": 1,
        "loop_count": 1,
        "break_count": 1,
        "continue_count": 0,
        "loop_early_return_count": 0,
    }


def test_authored_control_contract_counts_return_that_terminates_a_loop():
    assert authored_control_contract(
        "def encode(value):\n"
        "    while True:\n"
        "        if value:\n"
        "            value -= 1\n"
        "        else:\n"
        "            return value\n",
        "encode",
    )["loop_early_return_count"] == 1


def test_bounded_failure_names_the_next_compilation_creep_action():
    records = [{
        "qualified_name": "outer",
        "status": "failed",
        "error_type": "ResourceLimitExceeded",
        "failure_stage": "planning complete control/operator graph",
    }]

    assert compilation_creep_frontier(
        records,
        {"outer": ("leaf", "outer.<locals>.integral")},
        {"outer": ("outer.<locals>.integral",)},
    ) == [{
        "qualified_name": "outer",
        "status": "failed",
        "action": "compile-deeper-authored-integrals",
        "authored_subunits": ["outer.<locals>.integral"],
        "resolved_process_graph_subunits": [],
        "dependencies": ["leaf", "outer.<locals>.integral"],
        "minimum_authored_integral": False,
        "failure_stage": "planning complete control/operator graph",
        "error_type": "ResourceLimitExceeded",
    }]


def test_bounded_failure_uses_resolved_process_graph_units_as_deeper_cut():
    frontier = compilation_creep_frontier([{
        "qualified_name": "outer",
        "status": "failed",
        "error_type": "ResourceLimitExceeded",
        "process_graph_subunits": ["stdlib.worker", "outer.helper"],
    }], {"outer": ()})

    assert frontier[0]["action"] == "compile-resolved-process-graph-units"
    assert frontier[0]["resolved_process_graph_subunits"] == [
        "outer.helper", "stdlib.worker",
    ]
    assert frontier[0]["minimum_authored_integral"] is False


def test_partial_return_failure_names_return_lowering_frontier():
    assert compilation_creep_frontier([{
        "qualified_name": "value",
        "status": "partial",
        "root_return_publication_complete": False,
    }], {"value": ()})[0]["action"] == "repair-root-return-lowering"


def test_partial_required_source_value_names_semantic_materialization_frontier():
    assert compilation_creep_frontier([{
        "qualified_name": "value",
        "status": "partial",
        "root_return_publication_complete": True,
        "unresolved_required_source_values": [[4, "input", []]],
    }], {"value": ()})[0]["action"] == (
        "materialize-required-source-values"
    )


def test_partial_link_failure_names_link_frame_frontier_before_extraction():
    frontier = compilation_creep_frontier([{
        "qualified_name": "value",
        "status": "partial",
        "root_return_publication_complete": True,
        "unmaterialized_extraction_boundaries": 0,
        "unresolved_call_count": 1,
    }], {"value": ("leaf",)})

    assert frontier[0]["action"] == "repair-linked-call-frame"


def test_compiler_graph_reference_leaf_names_table_abi_frontier_first():
    frontier = compilation_creep_frontier([{
        "qualified_name": "planner_phase",
        "status": "partial",
        "control_frontier_action": "repair-conditionals-control-lowering",
        "unresolved_program_abi_references": [{
            "parameter": "graph", "field": "G.nodes",
        }],
    }], {"planner_phase": ()})

    assert frontier[0]["action"] == "lower-compiler-graph-table-abi"


def test_compilation_creep_names_known_compiler_semantic_boundaries():
    cases = {
        "a static Python reference cannot be assigned through a runtime "
        "tensor index": "model-static-value-table-boundary",
        "blockers=('iterable-access=closure_aggregate',)":
            "model-resident-closure-aggregate",
        "generator consumer has no safe resident query lowering":
            "lower-generator-sequence-query",
        "resolved-schema-conflict": "repair-shared-sequence-schema",
        "cyclic loop-control containment":
            "repair-control-region-containment",
    }

    for error, expected in cases.items():
        frontier = compilation_creep_frontier([{
            "qualified_name": "phase",
            "status": "failed",
            "error_type": "ValueError",
            "error": error,
        }], {"phase": ()})
        assert frontier[0]["action"] == expected


def test_compilation_creep_names_stale_planned_node_keyerror():
    frontier = compilation_creep_frontier([{
        "qualified_name": "phase",
        "status": "failed",
        "error_type": "KeyError",
        "error": "56",
    }], {"phase": ()})

    assert frontier[0]["action"] == "repair-stale-planned-node-reference"


def test_call_filename_encoding_is_deterministic_reversible_in_spelling():
    assert encoded_call_name("Surface.method") == "Surface.method"
    assert encoded_call_name("operator<λ>") == "operator_u00003c__u0003bb__u00003e_"


def test_native_parameter_surface_uses_correlations_not_physical_order():
    from src.compiler.project_compilation_product import (
        _same_authored_parameter_surface,
    )

    assert _same_authored_parameter_surface(
        ("function_name", "body", "data"),
        ("function_name", "data", "body"),
    )
    assert not _same_authored_parameter_surface(
        ("function_name", "body", "data"),
        ("function_name", "data", "data"),
    )


def test_native_unit_name_is_target_safe_bounded_and_deterministic():
    identity = "Builder.method.<locals>.a_very_long_nested_compiler_integral"

    first = native_unit_name(identity)

    assert first == native_unit_name(identity)
    assert len(first) <= 63
    assert first.startswith("project_unit__")
    assert "." not in first


def test_worker_resident_memory_can_be_measured():
    import os

    measured = resident_bytes(os.getpid())
    assert measured is not None
    assert measured > 0


def test_source_partition_retains_exact_dependency_closure_and_line_numbers():
    source = """from __future__ import annotations
import json
import pathlib

UNUSED = 41
USED = 7

def leaf(value):
    return json.dumps(value + USED)

def root(value):
    return leaf(value)

def unrelated(value):
    return pathlib.Path(value)
"""

    partitioned, symbols = partition_authored_source(source, "root")

    assert symbols == ("USED", "annotations", "json", "leaf", "root")
    assert "import pathlib" not in partitioned
    assert "UNUSED = 41" not in partitioned
    assert "def unrelated" not in partitioned
    assert "def root" in partitioned
    assert ast.parse(partitioned).body[-1].lineno == 11


def test_worker_receipts_link_in_dependency_order(tmp_path):
    records = [
        {"qualified_name": "root", "status": "complete", "path": "root"},
        {"qualified_name": "leaf", "status": "complete", "path": "leaf"},
    ]
    plans = {
        "root": {
            "units": [
                {"qualified_names": ["leaf"], "dependency_units": []},
                {"qualified_names": ["root"], "dependency_units": [0]},
            ],
        },
        "leaf": {
            "units": [
                {"qualified_names": ["leaf"], "dependency_units": []},
            ],
        },
    }
    for name, plan in plans.items():
        directory = tmp_path / name
        directory.mkdir()
        (directory / "unit.json").write_text(json.dumps({
            "compilation_unit_plan": plan,
        }), encoding="utf-8")

    ordered = dependency_ordered_records(tmp_path, records)

    assert [record["qualified_name"] for record in ordered] == ["leaf", "root"]


def test_source_calls_launch_dependencies_before_callers():
    source = """
def root(value):
    return middle(value)

def leaf(value):
    return value

def middle(value):
    return leaf(value)
"""

    assert dependency_ordered_authored_calls(
        source, ("root", "middle", "leaf"),
    ) == ("leaf", "middle", "root")


def test_requested_root_expands_and_orders_its_authored_dependency_closure():
    source = """
def root(value):
    return middle(value)

def leaf(value):
    return value

def middle(value):
    return leaf(value)
"""

    assert dependency_ordered_authored_calls(source, ("root",)) == (
        "leaf", "middle", "root",
    )


def test_dependency_closure_follows_constructed_object_methods():
    source = """
class Builder:
    def __init__(self, value):
        self.value = value

    def lower(self):
        return self.finish()

    def finish(self):
        return self.value

def compile_phase(value):
    builder = Builder(value)
    builder.lower()
    return builder.finish()
"""

    ordered = dependency_ordered_authored_calls(source, ("compile_phase",))

    assert set(ordered) == {
        "Builder.__init__", "Builder.lower", "Builder.finish", "compile_phase",
    }
    assert ordered.index("Builder.finish") < ordered.index("Builder.lower")
    assert ordered[-1] == "compile_phase"


def test_dependency_closure_follows_annotated_parameter_methods():
    source = """
class Builder:
    def finish(self):
        return 1

def compile_phase(builder: Builder):
    return builder.finish()
"""

    assert dependency_ordered_authored_calls(source, ("compile_phase",)) == (
        "Builder.finish", "compile_phase",
    )


def test_linked_dependency_retains_authored_definition_as_call_frame_contract():
    source = """
def leaf(value):
    return value + 1

def root(value):
    return leaf(value)
"""

    partitioned, symbols = partition_authored_source(
        source, "root", linked_dependencies=("leaf",),
    )

    assert "def leaf" in partitioned
    assert "def root" in partitioned
    assert "return leaf(value)" in partitioned
    assert symbols == ("leaf", "root")
    assert "def leaf" in source


def test_method_partition_keeps_class_shell_and_selected_method_only():
    source = """
class Builder:
    category = "compiler"

    def first(self, value):
        return value + 1

    def second(self, value):
        return value * 2
"""

    partitioned, symbols = partition_authored_source(
        source, "Builder.second",
    )

    assert "class Builder:" in partitioned
    assert 'category = "compiler"' in partitioned
    assert "def first" not in partitioned
    assert "def second" in partitioned
    assert "Builder.second" in {
        call.qualified_name for call in discover_authored_calls(partitioned)
    }
    assert symbols == ("Builder", "category", "second")


def test_link_product_loads_the_selected_repository_ssa_unit(tmp_path):
    artifact = tmp_path / "units" / "leaf" / "repository-ssa.pkl"
    artifact.parent.mkdir(parents=True)
    with artifact.open("wb") as stream:
        pickle.dump(("module", "outputs", "exports"), stream)
    (tmp_path / "manifest.json").write_text(json.dumps({
        "schema": "turing.project-compilation-product.v1",
    }), encoding="utf-8")
    (tmp_path / "links.json").write_text(json.dumps({
        "schema": "turing.project-compilation-links.v1",
        "links": [{
            "qualified_name": "leaf",
            "artifact": "units/leaf/repository-ssa.pkl",
            "exports": ["leaf"],
        }],
    }), encoding="utf-8")

    product = open_project_compilation_product(tmp_path)

    assert product.load_repository_ssa("leaf") == (
        "module", "outputs", "exports",
    )


def test_automatic_scalar_verification_is_abi_selected(
    tmp_path, monkeypatch,
):
    owner = SimpleNamespace()

    def leaf(value):
        return value + 1

    owner.leaf = leaf
    observed = {}

    class Product:
        root = tmp_path
        links = {
            "leaf": {
                "source_module": "compiler_leaf_module",
                "native_api": "leaf.api.yaml",
                "native_entrypoint": "leaf_native",
            },
        }

        def verify_native_scalar_callable(
            self, qualified_name, authored, probes, *, activation_adapter=None,
        ):
            observed.update({
                "qualified_name": qualified_name,
                "authored": authored,
                "probes": tuple(probes),
                "activation_adapter": activation_adapter,
            })

            def deployed(value):
                return value + 1

            deployed.__turing_native_verification__ = {
                "probe_count": 3,
                "native_probe_count": 3,
                "fallback_probe_count": 0,
            }
            return deployed

    (tmp_path / "leaf.api.yaml").write_text("api", encoding="utf-8")
    monkeypatch.setattr(
        "src.compiler.project_compilation_product."
        "open_project_compilation_product",
        lambda _path: Product(),
    )
    monkeypatch.setattr(
        "src.compiler.project_compilation_product._resolve_product_callable",
        lambda _module, _name: (owner, leaf),
    )
    monkeypatch.setattr(
        "src.compiler.compiled_program_api.load_api",
        lambda _path: {"entry_points": [{
            "name": "leaf_native",
            "parameters": [
                {
                    "name": "value", "source_name": "value",
                    "role": "input", "ctypes": "c_int64",
                },
                {"name": "result", "role": "output", "ctypes": "c_int64"},
            ],
        }]},
    )

    result, = verify_project_scalar_units_automatically(tmp_path)

    assert result["status"] == "verified"
    assert observed["authored"] is leaf
    assert observed["probes"] == (
        {"value": -3}, {"value": 1}, {"value": 5},
    )
    assert observed["activation_adapter"] == "qualified-scalar-call-v1"


def _verified_install_fixture(tmp_path):
    source_path = tmp_path / "authored_module.py"
    source = "def leaf(value):\n    return ('source', value)\n"
    source_path.write_text(source, encoding="utf-8")
    namespace = {}
    exec(compile(source, str(source_path), "exec"), namespace)
    owner = SimpleNamespace(leaf=namespace["leaf"])
    api_path = tmp_path / "units" / "leaf" / "native" / "leaf.api.yaml"
    library_path = api_path.with_name("leaf.dll")
    api_path.parent.mkdir(parents=True)
    api_path.write_bytes(b"api")
    library_path.write_bytes(b"library")
    source_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()
    verification = {
        "schema": "turing.native-callable-verification.v1",
        "qualified_name": "leaf",
        "abi_kind": "sequence",
        "source_sha256": source_hash,
        "api_sha256": hashlib.sha256(api_path.read_bytes()).hexdigest(),
        "library_sha256": hashlib.sha256(library_path.read_bytes()).hexdigest(),
        "probe_count": 2,
        "native_probe_count": 2,
        "fallback_probe_count": 0,
        "status": "verified",
    }
    (library_path.parent / "native-verification.json").write_text(
        json.dumps(verification), encoding="utf-8",
    )
    deployed = lambda value: ("native", value)
    deployed.__turing_native_verification__ = verification
    (tmp_path / "manifest.json").write_text(json.dumps({
        "schema": "turing.project-compilation-product.v1",
        "source": str(source_path),
        "source_sha256": source_hash,
    }), encoding="utf-8")
    (tmp_path / "links.json").write_text(json.dumps({
        "schema": "turing.project-compilation-links.v1",
        "links": [{
            "qualified_name": "leaf",
            "artifact": "units/leaf/repository-ssa.pkl",
            "native_api": "units/leaf/native/leaf.api.yaml",
            "native_library": "units/leaf/native/leaf.dll",
            "native_entrypoint": "leaf",
        }],
    }), encoding="utf-8")
    return open_project_compilation_product(tmp_path), owner, deployed, source_path


def test_verified_install_uses_native_and_preserves_authored_source(tmp_path):
    from src.common.tensors.source_realization import authored_source_realization

    product, owner, deployed, _ = _verified_install_fixture(tmp_path)
    product.install_callable("leaf", owner, deployed)

    assert owner.leaf(3) == ("native", 3)
    with authored_source_realization():
        assert owner.leaf(3) == ("source", 3)


def test_verified_install_preserves_property_descriptor_and_source(tmp_path):
    from src.common.tensors.source_realization import authored_source_realization

    product, _owner, deployed, _ = _verified_install_fixture(tmp_path)

    class Owner:
        @property
        def leaf(self):
            return ("source", 3)

    # The fixture's source revision proof is intentionally about its authored
    # module, so give this local getter that exact source callable while testing
    # descriptor preservation independently.
    descriptor = Owner.__dict__["leaf"]
    descriptor.fget.__turing_authored_source_callable__ = _owner.leaf
    native = lambda self: ("native", 3)
    native.__turing_native_verification__ = (
        deployed.__turing_native_verification__
    )
    product.install_callable("leaf", Owner, native)

    assert isinstance(Owner.__dict__["leaf"], property)
    instance = Owner()
    assert instance.leaf == ("native", 3)
    with authored_source_realization():
        assert instance.leaf == ("source", instance)


def test_verified_install_rejects_changed_authored_source(tmp_path):
    product, owner, deployed, source_path = _verified_install_fixture(tmp_path)
    source_path.write_text(
        "def leaf(value):\n    return ('changed', value)\n", encoding="utf-8",
    )

    try:
        product.install_callable("leaf", owner, deployed)
    except ValueError as error:
        assert "changed after compilation" in str(error)
    else:
        raise AssertionError("stale native deployment was installed")


def test_verified_install_rejects_changed_native_library(tmp_path):
    product, owner, deployed, _ = _verified_install_fixture(tmp_path)
    (tmp_path / "units" / "leaf" / "native" / "leaf.dll").write_bytes(
        b"tampered",
    )

    try:
        product.install_callable("leaf", owner, deployed)
    except ValueError as error:
        assert "changed after verification" in str(error)
    else:
        raise AssertionError("tampered native deployment was installed")


def test_verified_install_rejects_scalar_record_probe_fallback(tmp_path):
    product, owner, deployed, _ = _verified_install_fixture(tmp_path)
    deployed.__turing_native_verification__ = {
        **deployed.__turing_native_verification__,
        "abi_kind": "scalar-record",
        "native_probe_count": 1,
        "fallback_probe_count": 1,
    }

    try:
        product.install_callable("leaf", owner, deployed)
    except ValueError as error:
        assert "did not prove every probe" in str(error)
    else:
        raise AssertionError("fallback-dependent record deployment was installed")


def test_verified_install_accepts_accounted_record_return_fallback(tmp_path):
    product, owner, deployed, _ = _verified_install_fixture(tmp_path)
    verification = {
        **deployed.__turing_native_verification__,
        "abi_kind": "record-return",
        "native_probe_count": 1,
        "fallback_probe_count": 1,
    }
    deployed.__turing_native_verification__ = verification
    receipt = tmp_path / "units" / "leaf" / "native" / "native-verification.json"
    receipt.write_text(json.dumps(verification), encoding="utf-8")

    product.install_callable("leaf", owner, deployed)

    assert owner.leaf(3) == ("native", 3)


def test_verified_install_rejects_record_return_without_native_probe(tmp_path):
    product, owner, deployed, _ = _verified_install_fixture(tmp_path)
    verification = {
        **deployed.__turing_native_verification__,
        "abi_kind": "record-return",
        "native_probe_count": 0,
        "fallback_probe_count": 2,
    }
    deployed.__turing_native_verification__ = verification

    try:
        product.install_callable("leaf", owner, deployed)
    except ValueError as error:
        assert "record-return probe" in str(error)
    else:
        raise AssertionError("fallback-only record deployment was installed")


def test_finished_ssa_artifact_releases_unpickleable_frontend_graphs():
    import ast as live_module

    table = FunctionTable()
    reference = table.declare("leaf", qualified_name="package.leaf")
    entry = table.resolve_graph(reference, object())
    entry.python_callable = lambda value: value
    entry.implementations["python"] = entry.python_callable
    entry.metadata.update({"source_node": 17, "live_module": live_module})
    module = IRModule({}, function_table=table)

    detach_repository_ssa_frontend(module)

    assert entry.reference.address == reference.address
    assert entry.qualified_name == "package.leaf"
    assert entry.graph is None
    assert entry.python_callable is None
    assert entry.implementations == {}
    assert entry.metadata == {"source_node": 17}
    pickle.dumps(module)
