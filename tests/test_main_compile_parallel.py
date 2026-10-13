"""The normal native-program lowering must execute bounded compiler units."""
from pathlib import Path
import dataclasses
import json
import pickle
import subprocess
import sys
import numpy as np
import pytest

from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa


RECORD_SOURCE = '''
def left(state):
    return state.x * 2.0

def right(state):
    return state.y * 3.0

def root(state):
    return left(state) + right(state)
'''


def record_contract():
    return ExtractionContract("extraction_contracts/program_extraction.yaml").with_program_abi({
        "records": {"State": {"fields": {
            "x": {"storage": "scalar", "dtype": "float64", "rank": 0},
            "y": {"storage": "scalar", "dtype": "float64", "rank": 0},
        }}},
        "bindings": [{"function": name, "parameter": "state", "record": "State"}
                     for name in ("left", "right", "root")],
        "values": [],
    })


def assert_native_outputs(tmp_path, module, outputs, exports, feeds, expected):
    from src.compiler.identity_concordance import concordance_report
    from src.compiler.ssa_llvm_backend import emit_ssa_function_to_llvm, compile_artifact
    assert concordance_report(module).splitlines()[0].endswith(", 0 finding(s)")
    root = module.functions[exports[0]]
    artifact = emit_ssa_function_to_llvm(module, root.name, entry_name="parallel_main_check")
    assert artifact.complete, artifact.shortfalls
    artifact = compile_artifact(artifact, directory=tmp_path / "native")
    artifact = dataclasses.replace(artifact, emission=None, _entry=None, _library=None,
        _validation_error_reset=None, _validation_error_take=None)
    payload = tmp_path / "native-check.pkl"
    payload.write_bytes(pickle.dumps((artifact, {}, ("program_outputs",), {
        "inputs": feeds, "outputs": tuple(int(value.id) for value in outputs[root.name]),
    })))
    child = subprocess.run([sys.executable, "-u", str(Path(__file__).resolve().parents[1] /
        "tools/compiler_probes/graph_kernel_native_worker.py"), str(payload)],
        text=True, capture_output=True, timeout=10)
    assert child.returncode == 0, child.stdout + child.stderr
    result = json.loads(next(line.split("=", 1)[1] for line in child.stdout.splitlines()
                             if line.startswith("GRAPH_PROGRAM_OUTPUTS=")))
    assert result == expected


def test_main_native_units_use_existing_record_boundaries(tmp_path):
    module, outputs, exports = lower_ast_source_to_ssa(
        RECORD_SOURCE, "root", extraction_contract=record_contract(),
        partitioned_directory=tmp_path / "units",
        partitioned_options={"jobs": 2, "max_unit_source_nodes": 128,
            "worker_resident_reservation_bytes": 256 * 1024 ** 2,
            "max_worker_memory_bytes": 2 * 1024 ** 3},
        progress=lambda message: None,
    )
    manifest = module.metadata["partitioned_source_product"]
    assert manifest["concurrent_workers_peak"] == 2
    assert manifest["counts"]["failed"] == manifest["counts"]["partial"] == 0
    assert all(not function.blocks for function in module.functions.values())
    assert exports and outputs[exports[0]]
    root = module.functions[exports[0]]
    feeds = {int(value_id): np.array({"x": 2.0, "y": 3.0}[field.name])
             for record in module.record_tables[root.name].records.values()
             for field in record.fields for value_id in field.value_ids}
    assert_native_outputs(tmp_path, module, outputs, exports, feeds, [13.0])


@pytest.mark.parametrize("nested", [False, True])
def test_multi_result_interfaces_use_the_existing_projection_graph(tmp_path, nested):
    source = ("def leaf(x):\n    return x * 2.0, (x * 3.0, x * 4.0)\ndef root(x):\n    a, b = leaf(x)\n    return a + b[0] + b[1]\n" if nested else
              "def leaf(x):\n    return x * 2.0, x * 3.0\ndef root(x):\n    a, b = leaf(x)\n    return a + b\n")
    contract = ExtractionContract("extraction_contracts/program_extraction.yaml").with_program_abi({
        "records": {}, "bindings": [], "values": [
            {"function": name, "parameter": "x", "storage": "scalar", "dtype": "float64",
             "rank": 0, "shape": [], "python_type": "builtins.float"} for name in ("leaf", "root")],
    })
    module, outputs, exports = lower_ast_source_to_ssa(
        source, "root", extraction_contract=contract, partitioned_directory=tmp_path / "units",
        partitioned_options={"jobs": 2, "max_unit_source_nodes": 128,
            "worker_resident_reservation_bytes": 256 * 1024 ** 2,
            "max_worker_memory_bytes": 2 * 1024 ** 3},
        progress=lambda message: None,
    )
    assert module.metadata["partitioned_source_product"]["counts"]["failed"] == 0
    root = module.functions[exports[0]]
    feeds = {int(value.id): np.array(21.0) for value in root.args}
    assert_native_outputs(tmp_path, module, outputs, exports, feeds, [189.0] if nested else [105.0])


def test_linked_piece_boundary_is_not_recompiled_as_an_empty_body(tmp_path):
    import sympy as sp
    from src.compiler.native_package import piece_from_law
    from src.compiler.symbolic_equation_compiler import compile_sympy_equations
    from src.compiler.native_law_kernels import batch_contract
    x = sp.Symbol("x")
    law = compile_sympy_equations([sp.Eq(sp.Symbol("result"), x * 2.0, evaluate=False)], name="parallel_link")
    piece = piece_from_law(law, "parallel_link", 1, directory=tmp_path / "law")
    module, outputs, exports = lower_ast_source_to_ssa(
        "def root(x):\n    return law(x)\n", "root", python_bindings={"law": piece},
        extraction_contract=batch_contract("root", ("x",), 1),
        retain_linked_piece_bodies=False, partitioned_directory=tmp_path / "units",
        partitioned_options={"jobs": 2, "max_unit_source_nodes": 128,
            "worker_resident_reservation_bytes": 256 * 1024 ** 2,
            "max_worker_memory_bytes": 2 * 1024 ** 3},
        progress=lambda message: None,
    )
    manifest = module.metadata["partitioned_source_product"]
    assert manifest["counts"]["failed"] == manifest["counts"]["partial"] == 0
    assert exports and outputs[exports[0]]
    feeds = {int(value.id): np.array([21.0]) for value in module.functions[exports[0]].args}
    assert_native_outputs(tmp_path, module, outputs, exports, feeds, [[42.0]])


@pytest.mark.parametrize("source_compilation", [None, "units", "serial"])
def test_normal_linked_dt_entry_requests_existing_unit_workers(tmp_path, monkeypatch, source_compilation):
    """Plumbing check only; actual DT helper execution is a separate probe."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
    import llvm_dt_system as dt
    import src.compiler.fortran_c_shell as compiler
    from types import SimpleNamespace
    piece = SimpleNamespace(batch=1, argument_names=("x",), output_names=("x_next",), entry="step")
    monkeypatch.setattr(dt.LLVMPiece, "load", lambda path: piece)
    monkeypatch.setattr(dt, "bind_pieces", lambda pieces, **kwargs: {})
    monkeypatch.setattr(dt, "generated_source", lambda pieces, **kwargs: "")
    from src.common.tensors.accelerator_backends import c_backend_llvm_ssa
    monkeypatch.setattr(c_backend_llvm_ssa, "c_backend_repository_ssa_reference", lambda: None)
    class ObservedRoute(Exception):
        pass
    def observe(source, entrypoint, **options):
        assert entrypoint == "dt_system_over"
        assert "def dt_system_over(" in source
        if source_compilation == "serial":
            assert "partitioned_options" not in options
            assert "partitioned_directory" not in options
        else:
            assert options["partitioned_options"]["jobs"] >= 2
            assert Path(options["partitioned_directory"]).is_relative_to(tmp_path)
        assert options["retain_linked_piece_bodies"] is False
        raise ObservedRoute()
    monkeypatch.setattr(compiler, "lower_ast_source_to_ssa", observe)
    with pytest.raises(ObservedRoute):
        dt.lowered_system(["probe.piece"], directory=tmp_path, progress=lambda message: None,
                          **({} if source_compilation is None else {"source_compilation": source_compilation}))


def test_partitioned_public_entry_preserves_module_import_identity(
    tmp_path, monkeypatch,
):
    import src.compiler.project_compilation_product as products

    observed = {}

    def compile_partitioned(source, directory, **options):
        observed.update(options)
        return {"status": "sealed"}

    monkeypatch.setattr(products, "compile_partitioned_source", compile_partitioned)
    monkeypatch.setattr(
        products, "load_partitioned_source_interfaces", lambda manifest: manifest,
    )

    result = lower_ast_source_to_ssa(
        "from .helper import leaf\ndef root(value):\n    return leaf(value)\n",
        "root",
        source_module="package.program",
        python_package="package",
        partitioned_directory=tmp_path / "units",
    )

    assert result == {"status": "sealed"}
    assert observed["source_options"]["source_module"] == "package.program"
    assert observed["source_options"]["python_package"] == "package"


def test_native_tensor_index_is_not_a_tuple_projection(tmp_path):
    source = "def leaf(x):\n    return x * 2.0\ndef root(x):\n    return leaf(x)[0] + 1.0\n"
    contract = ExtractionContract("extraction_contracts/program_extraction.yaml").with_program_abi({
        "records": {}, "bindings": [], "values": [
            {"function": name, "parameter": "x", "storage": "span", "dtype": "float64",
             "rank": 1, "shape": [4], "python_type": "src.common.tensors.abstraction.AbstractTensor"}
            for name in ("leaf", "root")],
    })
    module, outputs, exports = lower_ast_source_to_ssa(
        source, "root", extraction_contract=contract, partitioned_directory=tmp_path / "units",
        partitioned_options={"jobs": 2, "max_unit_source_nodes": 128,
            "worker_resident_reservation_bytes": 256 * 1024 ** 2,
            "max_worker_memory_bytes": 2 * 1024 ** 3},
        progress=lambda message: None,
    )
    feeds = {int(value.id): np.array([21.0, 22.0, 23.0, 24.0])
             for value in module.functions[exports[0]].args}
    assert_native_outputs(tmp_path, module, outputs, exports, feeds, [43.0])


def test_native_call_keeps_authored_order_and_omitted_defaults(tmp_path):
    source = ("def leaf(value, default=3.0):\n    return value * 2.0 + default\n"
              "def root(x):\n    return leaf(x) + leaf(default=5.0, value=x)\n")
    contract = ExtractionContract("extraction_contracts/program_extraction.yaml").with_program_abi({
        "records": {}, "bindings": [], "values": [
            {"function": name, "parameter": parameter, "storage": "scalar", "dtype": "float64",
             "rank": 0, "shape": [], "python_type": "builtins.float"}
            for name, parameter in (("leaf", "value"), ("leaf", "default"), ("root", "x"))],
    })
    module, outputs, exports = lower_ast_source_to_ssa(
        source, "root", extraction_contract=contract, partitioned_directory=tmp_path / "units",
        partitioned_options={"jobs": 2, "max_unit_source_nodes": 128,
            "worker_resident_reservation_bytes": 256 * 1024 ** 2,
            "max_worker_memory_bytes": 2 * 1024 ** 3},
        progress=lambda message: None,
    )
    feeds = {int(value.id): np.array(21.0) for value in module.functions[exports[0]].args}
    assert_native_outputs(tmp_path, module, outputs, exports, feeds, [92.0])


def test_native_boundary_retains_literal_argument_of_compiled_dependency(tmp_path):
    source = ("def leaf(value):\n    return value * 2.0\n"
              "def root(x):\n    return leaf(3.0) + leaf(x)\n")
    contract = ExtractionContract("extraction_contracts/program_extraction.yaml").with_program_abi({
        "records": {}, "bindings": [], "values": [
            {"function": name, "parameter": parameter, "storage": "scalar", "dtype": "float64",
             "rank": 0, "shape": [], "python_type": "builtins.float"}
            for name, parameter in (("leaf", "value"), ("root", "x"))],
    })
    module, outputs, exports = lower_ast_source_to_ssa(
        source, "root", extraction_contract=contract, partitioned_directory=tmp_path / "units",
        partitioned_options={"jobs": 2, "max_unit_source_nodes": 128,
            "worker_resident_reservation_bytes": 256 * 1024 ** 2,
            "max_worker_memory_bytes": 2 * 1024 ** 3},
        progress=lambda message: None,
    )
    feeds = {int(value.id): np.array(21.0) for value in module.functions[exports[0]].args}
    assert_native_outputs(tmp_path, module, outputs, exports, feeds, [48.0])


def test_native_boundary_uses_only_surviving_specialized_formals(tmp_path):
    source = ("def leaf(value, multiplier=2.0):\n    return value * multiplier\n"
              "def root(x):\n    return leaf(x, 3.0)\n")
    contract = ExtractionContract("extraction_contracts/program_extraction.yaml").with_program_abi({
        "records": {}, "bindings": [], "values": [
            {"function": name, "parameter": parameter, "storage": "scalar", "dtype": "float64",
             "rank": 0, "shape": [], "python_type": "builtins.float"}
            for name, parameter in (("leaf", "value"), ("leaf", "multiplier"), ("root", "x"))],
    })
    module, outputs, exports = lower_ast_source_to_ssa(
        source, "root", extraction_contract=contract, partitioned_directory=tmp_path / "units",
        partitioned_options={"jobs": 2, "max_unit_source_nodes": 128,
            "worker_resident_reservation_bytes": 256 * 1024 ** 2,
            "max_worker_memory_bytes": 2 * 1024 ** 3},
        progress=lambda message: None,
    )
    feeds = {int(value.id): np.array(21.0) for value in module.functions[exports[0]].args}
    assert_native_outputs(tmp_path, module, outputs, exports, feeds, [63.0])


def test_main_workers_use_graph_size_cuts_and_bounded_instruction_snapshots(tmp_path, monkeypatch):
    # Same authored source, not an alternate numerical lowering. The existing
    # dependency/control quotient splits two long causal chains into regions.
    # The main unit workers must actually lower/reload those bounded snapshots.
    monkeypatch.setenv("TURING_GRAPH_REGION_NODE_LIMIT", "8")
    monkeypatch.setenv("TURING_SSA_RESIDENT_REGIONS", "1")
    lines = []
    for name, multiplier in (("left", 2), ("right", 3)):
        lines.extend((f"def {name}(x):", "    v = x"))
        lines.extend(f"    v = v * {multiplier}.0 + 1.0" for _ in range(16))
        lines.append("    return v")
    lines.extend(("def root(x):", "    if x > 0.0:", "        return left(x) + right(x)",
                  "    return -4.0"))
    source = "\n".join(lines) + "\n"
    contract = ExtractionContract("extraction_contracts/program_extraction.yaml").with_program_abi({
        "records": {}, "bindings": [], "values": [
            {"function": name, "parameter": "x", "storage": "scalar", "dtype": "float64",
             "rank": 0, "shape": [], "python_type": "builtins.float"}
            for name in ("left", "right", "root")],
    })
    module, outputs, exports = lower_ast_source_to_ssa(
        source, "root", extraction_contract=contract, partitioned_directory=tmp_path / "units",
        partitioned_options={"jobs": 2, "max_unit_source_nodes": 128,
            "worker_resident_reservation_bytes": 256 * 1024 ** 2,
            "max_worker_memory_bytes": 2 * 1024 ** 3},
        progress=lambda message: None,
    )
    manifest = module.metadata["partitioned_source_product"]
    assert manifest["concurrent_workers_peak"] == 2
    from src.compiler.project_compilation_product import _load_pickle_artifact
    for unit in manifest["units"]:
        if tuple(unit["qualified_names"]) == ("root",):
            continue
        unit_root = tmp_path / "units/product/units" / f"unit_{unit['unit_index']:03d}"
        lowered, _outputs, _exports = _load_pickle_artifact(unit_root / unit["artifact"])
        receipts = [f.metadata["region_instruction_residency"] for f in lowered.functions.values()
                    if "region_instruction_residency" in f.metadata]
        assert receipts and all(r["limit"] == r["resident_peak"] == 1 for r in receipts)
        assert all(r["snapshots"] > 2 and r["spills"] > 0 and r["reloads"] > 0 for r in receipts)
        regions = [f for f in lowered.functions.values() if f.metadata.get("source_region_integral")]
        assert len(regions) > 1
        # Eight arithmetic graph members plus their literal instructions and
        # Ret. The graph limit is not an arbitrary SSA expansion byte bound.
        assert all(sum(len(b.instrs) for b in f.blocks.values()) <= 17 for f in regions)
    feeds = {int(value.id): np.array(1.0) for value in module.functions[exports[0]].args}
    assert_native_outputs(tmp_path, module, outputs, exports, feeds, [64701152.0])
    feeds = {int(value.id): np.array(-1.0) for value in module.functions[exports[0]].args}
    assert_native_outputs(tmp_path, module, outputs, exports, feeds, [-4.0])


@pytest.mark.parametrize("partitioned,nested", [(False, False), (True, False), (True, True)])
def test_record_specialization_dependencies_are_indivisible_before_native_cut(tmp_path, partitioned, nested):
    source = '''
def leaf(state):
    return state.x * 2.0
def relay(state):
    if state.x > 0.0:
        return leaf(state) + 1.0
    return 0.0
def left(x):
    return x * 3.0
def right(x):
    return x * 5.0
def root(state, x):
    return relay(state) + left(x) + right(x)
def top(state, x):
    return root(state, x)
'''
    contract = ExtractionContract("extraction_contracts/program_extraction.yaml").with_program_abi({
        "records": {name: {"fields": {"x": {"storage": "scalar", "dtype": "float64", "rank": 0}}}
                    for name in ("Actual", "Declared")},
        "bindings": [{"function": name, "parameter": "state", "record": record}
                     for name, record in (("root", "Actual"), ("top", "Actual"), ("relay", "Declared"), ("leaf", "Declared"))],
        "values": [{"function": name, "parameter": "x", "storage": "scalar", "dtype": "float64",
                    "rank": 0, "shape": [], "python_type": "builtins.float"}
                   for name in ("root", "top", "left", "right")],
    })
    options = {"partitioned_directory": tmp_path / "units",
        "partitioned_options": {"jobs": 2, "max_unit_source_nodes": 128,
            "worker_resident_reservation_bytes": 256 * 1024 ** 2,
            "max_worker_memory_bytes": 2 * 1024 ** 3}} if partitioned else {}
    module, outputs, exports = lower_ast_source_to_ssa(
        source, "top" if nested else "root", extraction_contract=contract, **options, progress=lambda message: None,
    )
    root = module.functions[exports[0]]
    fields = {int(value_id) for record in module.record_tables[root.name].records.values()
              for field in record.fields for value_id in field.value_ids}
    feeds = {int(value.id): np.array(21.0 if int(value.id) in fields else 2.0) for value in root.args}
    assert_native_outputs(tmp_path, module, outputs, exports, feeds, [59.0])
    if partitioned:
        manifest = module.metadata["partitioned_source_product"]
        assert manifest["concurrent_workers_peak"] == 2
        groups = {tuple(unit["qualified_names"]) for unit in manifest["units"]}
        assert groups == {("leaf", "relay", "root"), ("left",), ("right",), ("top",)}
        assert all(not unit["unit"]["recursive"] for unit in manifest["units"])
        assert all(unit["scoped_input"]["source_nodes"] <= 128 and
                   set(unit["scoped_input"]["function_references"]) == set(unit["unit"]["function_references"])
                   for unit in manifest["units"])
        plan = json.loads((tmp_path / "units/inputs/plan.json").read_text())
        assert len(plan["interface_coupling_edges"]) == 2
        if nested:
            # The accepted header's source identity, not export order/name,
            # binds the later caller to this contracted component.
            header_receipt = next(unit for unit in manifest["units"]
                                  if unit["qualified_names"] == ["leaf", "relay", "root"])
            assert len(header_receipt["exports"]) == 3
