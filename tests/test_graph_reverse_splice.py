"""The canonical graph reverse spliced into its host graph.

``splice_graph_reverse`` differentiates one host value with the existing
``BACKWARD_RULES`` reverse and copies that reverse back into the host
(``splice_process_graph``), moving the rule functions into the host's
function table.  The SymPy importer uses it for a Derivative SymPy cannot
take, once the graph carries its descriptors.
"""

from pathlib import Path
from types import SimpleNamespace

import sympy as sp

from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import concordance_report
from src.compiler.symbolic_equation_compiler import compile_sympy_equations
from src.compiler.symbolic_process_graph import GRAPH_REVERSE_SOCKET


CONTRACT = ExtractionContract(
    Path(__file__).resolve().parents[1] / "extraction_contracts/program_extraction.yaml"
)


def _floor_product_derivative():
    x = sp.Symbol("x")
    return compile_sympy_equations((
        sp.Eq(
            sp.Symbol("result"), sp.Derivative(x*sp.floor(x), x),
            evaluate=False,
        ),
    ), name="floor_product_derivative")


def _called_functions(graph):
    return {
        graph.function_table.entry(int(data["attributes"]["callee_ref"])).qualified_name
        for _node, data in graph.G.nodes(data=True)
        if data.get("op") == "Call"
    }


def test_derivative_sympy_cannot_take_becomes_a_spliced_graph_reverse():
    envelope = _floor_product_derivative()
    graph = envelope.process_graph

    assert not any(
        data.get("op") == GRAPH_REVERSE_SOCKET
        for _node, data in graph.G.nodes(data=True)
    )
    assert _called_functions(graph) == {"bw_mul", "bw_floor"}
    for _node, data in graph.G.nodes(data=True):
        assert data["tensor"]["dtype"] in {"float64", "int64"}
    for entry in graph.function_table:
        if entry.graph is not None:
            assert entry.graph.function_table is graph.function_table


def test_spliced_graph_reverse_executes_through_the_source_compiler(tmp_path):
    envelope = _floor_product_derivative()
    module, outputs, exports = lower_ast_source_to_ssa(
        "def root(x):\n    return floor_product_derivative(x)\n",
        "root",
        name="floor_product_derivative_host",
        linked_process_graphs={
            "floor_product_derivative": envelope.process_graph,
        },
        extraction_contract=CONTRACT,
        progress=lambda _message: None,
    )
    root = module.functions[exports[0]]
    report = concordance_report(module)
    assert report.splitlines()[0].endswith("0 finding(s)"), report

    from src.compiler.ssa_llvm_backend import (
        compile_artifact, emit_ssa_function_to_llvm, prepare_artifact_execution,
    )
    from src.compiler.vehicle_python_compilation import _managed_native_feeds_by_id

    artifact = emit_ssa_function_to_llvm(module, root.name)
    assert artifact.complete, artifact.shortfalls
    artifact = compile_artifact(artifact, directory=tmp_path, optimization="O3")
    for x_value in (2.5, -1.25, 7.0):
        feeds = _managed_native_feeds_by_id(
            SimpleNamespace(module=module, root_name=root.name), {"x": x_value},
        )
        execution = prepare_artifact_execution(artifact, feeds)
        execution.run()
        result, = outputs[root.name]
        # d/dx (x*floor(x)) = floor(x) + x * d floor(x)/dx, and floor is flat
        # between its integer steps.
        assert float(execution.buffers[result.id]) == float(sp.floor(x_value))
