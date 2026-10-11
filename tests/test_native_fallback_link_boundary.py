"""Installed native pieces remain link leaves, not Python source dependencies."""

import dataclasses
import inspect
import json
import pickle
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pytest

from src.common.tensors.source_realization import (
    authored_source_realization, deployed_with_authored_fallback,
)
from src.compiler.extraction_contract import (
    ExtractionAction, ExtractionContract, llvm_piece_of,
)
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import concordance_report
from src.compiler.native_law_kernels import LLVMPiece, batch_contract
from src.compiler.ssa_llvm_backend import compile_artifact, emit_ssa_function_to_llvm


def _authored_leaf(x):
    return x * x + 1.0, x + 2.0


@pytest.fixture(scope="module")
def native_leaf(tmp_path_factory):
    # The SAME authored Python, compiled by the public entry once.
    module, outputs, exports = lower_ast_source_to_ssa(
        inspect.getsource(_authored_leaf), "_authored_leaf",
        extraction_contract=batch_contract("_authored_leaf", ("x",), 3),
        progress=lambda _message: None, identity_log_level="summary",
    )
    root = module.functions[exports[0]]
    artifact = compile_artifact(emit_ssa_function_to_llvm(module, root.name),
                               directory=tmp_path_factory.mktemp("link-leaf"))
    parameters = dict(root.metadata["parameter_names"])
    return LLVMPiece(
        artifact, ("x",), (int(parameters["x"]),), ("result", "other"),
        {"result": int(outputs[root.name][0].id),
         "other": int(outputs[root.name][1].id)}, batch=3,
        module=module, entry=root.name, outputs=outputs,
        # Deliberately unparsable: a link must use the declared ABI, not source.
        source="THIS RETAINED SOURCE MUST NOT BE PARSED",
    )


def _binding(piece):
    return deployed_with_authored_fallback(
        _authored_leaf, piece, identity="_authored_leaf", targeted=True,
    )


def test_installed_piece_is_the_same_declared_native_link(native_leaf):
    installed = _binding(native_leaf)
    assert installed.__turing_deployed_callable__ is native_leaf
    assert llvm_piece_of(installed) is native_leaf
    contract = batch_contract("caller", ("x",), 3)
    decision = contract.decide(installed)
    assert decision.action is ExtractionAction.USE_NATIVE
    assert not decision.ingest_parent
    assert decision.parameters["piece_abi"] == contract.decide(native_leaf).parameters["piece_abi"]


def test_runtime_native_and_explicit_source_routes_still_agree(native_leaf):
    installed = _binding(native_leaf)
    values = np.array([-2.0, 0.0, 3.5])
    native = installed(values)
    with authored_source_realization(targets=("_authored_leaf",)):
        eager = installed(values)
    for actual, expected in zip(native, eager):
        np.testing.assert_array_equal(actual, expected)


def test_explicit_authored_root_is_still_compiled_not_replaced_by_its_installation(native_leaf):
    module, _outputs, exports = lower_ast_source_to_ssa(
        inspect.getsource(_authored_leaf), "_authored_leaf",
        python_bindings={"_authored_leaf": _binding(native_leaf)},
        extraction_contract=batch_contract("_authored_leaf", ("x",), 3),
        retain_linked_piece_bodies=False,
        progress=lambda _message: None, identity_log_level="summary",
    )
    root = module.functions[exports[0]]
    assert root.blocks
    assert not root.metadata.get("llvm_piece")


@pytest.mark.parametrize("wrapped", [False, True])
def test_public_lowerer_links_without_reading_fallback_source(native_leaf, monkeypatch, tmp_path, wrapped):
    binding = _binding(native_leaf) if wrapped else native_leaf
    original_getsource = inspect.getsource
    fallback_reads = []

    def guarded_source(value):
        if value is binding or value is _authored_leaf:
            fallback_reads.append(value)
            raise AssertionError("native link tried to ingest Python fallback source")
        return original_getsource(value)

    monkeypatch.setattr(inspect, "getsource", guarded_source)
    started = time.perf_counter()
    module, outputs, exports = lower_ast_source_to_ssa(
        "def caller(x):\n    result, other = law(x)\n    return result\n", "caller",
        python_bindings={"law": binding},
        extraction_contract=batch_contract("caller", ("x",), 3),
        retain_linked_piece_bodies=False,
        progress=lambda _message: None, identity_log_level="summary",
    )
    lowering_seconds = time.perf_counter() - started
    assert not fallback_reads
    linked = module.functions[native_leaf.entry]
    assert not linked.blocks
    assert linked.metadata["linked_piece_interface"]["native_symbol"] == native_leaf.artifact.name
    assert set(module.functions) == {exports[0], native_leaf.entry}
    report = concordance_report(module)
    assert report.splitlines()[0].endswith(", 0 finding(s)"), report
    root = module.functions[exports[0]]
    artifact = emit_ssa_function_to_llvm(module, root.name)
    assert artifact.complete, artifact.shortfalls
    artifact = compile_artifact(artifact, directory=tmp_path / "native")
    portable = dataclasses.replace(artifact, emission=None, _entry=None, _library=None,
        _validation_error_reset=None, _validation_error_take=None)
    values = np.array([-2.0, 0.0, 3.5])
    payload = tmp_path / "native-check.pkl"
    payload.write_bytes(pickle.dumps((portable, {}, ("program_outputs",), {
        "inputs": {int(root.args[0].id): values},
        "outputs": tuple(int(value.id) for value in outputs[root.name]),
    })))
    # No diagnostic timeout; the child isolates only native execution crashes.
    child = subprocess.run([sys.executable, "-u", str(Path(__file__).resolve().parents[1] /
        "tools/compiler_probes/graph_kernel_native_worker.py"), str(payload)],
        capture_output=True, text=True)
    assert child.returncode == 0, child.stdout + child.stderr
    actual = json.loads(next(line.split("=", 1)[1] for line in child.stdout.splitlines()
                            if line.startswith("GRAPH_PROGRAM_OUTPUTS=")))
    assert actual == [[5.0, 1.0, 13.25]]
    print(f"NATIVE_FALLBACK_LINK wrapped={wrapped} lowering_seconds={lowering_seconds:.6f} "
          f"functions={len(module.functions)} fallback_source_reads={len(fallback_reads)} "
          "native_outputs=exact concordance_findings=0", flush=True)


def test_runtime_only_wrapped_piece_does_not_fall_back_to_source(native_leaf):
    with pytest.raises(ValueError, match="runtime-only.*source linking requires"):
        lower_ast_source_to_ssa(
            "def caller(x):\n    return law(x)\n", "caller",
            python_bindings={"law": _binding(native_leaf.for_runtime())},
            extraction_contract=batch_contract("caller", ("x",), 3),
            progress=lambda _message: None, identity_log_level="summary",
        )


def test_native_runtime_receipt_without_declared_abi_is_not_a_link():
    def python_deployment(x):
        return _authored_leaf(x)

    python_deployment.__turing_native_verification__ = {"status": "verified"}
    installed = _binding(python_deployment)
    assert llvm_piece_of(installed) is None
    assert ExtractionContract("extraction_contracts/program_extraction.yaml").decide(installed).ingest_parent


def test_cyclic_deployment_reference_is_not_a_link():
    installed = _binding(lambda x: x)
    installed.__turing_deployed_callable__ = installed
    assert llvm_piece_of(installed) is None


def test_nested_installations_keep_the_same_piece_identity(native_leaf):
    installed = _binding(_binding(native_leaf))
    assert llvm_piece_of(installed) is native_leaf


def test_method_binding_keeps_the_same_declared_piece(native_leaf):
    from types import MethodType

    installed = _binding(native_leaf)
    assert llvm_piece_of(MethodType(installed, object())) is native_leaf


def test_unrelated_decorator_does_not_acquire_native_link_status(native_leaf):
    def decorator(x):
        return native_leaf(x)

    decorator.__wrapped__ = native_leaf
    assert llvm_piece_of(decorator) is None


def test_external_slot_reuses_installed_piece_abi_without_compilation(native_leaf, monkeypatch):
    import ctypes
    from src.compiler import fortran_c_shell
    from src.compiler.external_functions import ExternalSignature, declare_external, _slot_address

    def forbidden_compile(*_args, **_kwargs):
        raise AssertionError("accepted native dependency tried to compile a new signature")

    monkeypatch.setattr(fortran_c_shell, "lower_ast_source_to_ssa", forbidden_compile)
    installed = _binding(native_leaf)
    external = declare_external("installed_leaf", ExternalSignature(
        ("float64",), ((3,),), result_shape=(3,)), implementation=installed)
    assert external.implementation is installed
    assert external.argument_ids == native_leaf.argument_ids
    assert external.output_ids == native_leaf.output_ids
    assert external.artifact.buffer_order == native_leaf.artifact.buffer_order
    assert external.module is not native_leaf.module
    row = {"external": external.external, "buffer_order": external.artifact.buffer_order}
    keep, errors = [], []
    address = _slot_address(row, installed, keep, errors)
    assert address == ctypes.cast(native_leaf.artifact.entry(), ctypes.c_void_p).value
    assert keep == [native_leaf.artifact] and errors == []


def _piece_mapping(_extent):
    return {}


def _piece_sequence(_extent):
    return ()


class _MappingOwner:
    def __init__(self):
        self.pieces = _piece_mapping(3)

    def choose(self, x):
        return self.pieces["leaf"](x)


class _SequenceOwner:
    def __init__(self):
        batch = len(self.items)
        self.pieces = _piece_sequence(batch)


@pytest.mark.parametrize("owner,factory", [
    (_MappingOwner, "_piece_mapping"), (_SequenceOwner, "_piece_sequence"),
])
def test_retained_piece_containers_publish_the_resolved_native_identity(native_leaf, monkeypatch, owner, factory):
    from src.transmogrifier.graph.graph_express2 import ProcessGraph

    installed = _binding(native_leaf)
    result = {"leaf": installed} if owner is _MappingOwner else (installed,)
    contract = batch_contract("caller", ("x",), 3).with_program_abi({
        "records": {owner.__name__: {
            "identity": f"{owner.__module__}.{owner.__qualname__}",
            "fields": {"items": {"storage": "span", "dtype": "float64",
                                 "rank": 1, "shape": [3]}},
        }}, "bindings": [], "values": [],
    })

    class BoundaryObserved(Exception):
        pass

    def observe(graph, tree, **_options):
        native_bindings = [value for value in graph.python_bindings.values()
                           if llvm_piece_of(value) is not None]
        assert native_bindings == [native_leaf]
        assert native_bindings[0] is native_leaf
        raise BoundaryObserved()

    monkeypatch.setattr(ProcessGraph, "build_from_ast", observe)
    with pytest.raises(BoundaryObserved):
        lower_ast_source_to_ssa(
            "def caller(x):\n    return x\n", "caller", retain=(owner,),
            python_bindings={factory: lambda _extent: result},
            extraction_contract=contract, retain_linked_piece_bodies=False,
            progress=lambda _message: None, identity_log_level="summary",
        )
