"""Opaque mathematical function sockets resolved to ProcessGraph bodies."""

from pathlib import Path
from types import SimpleNamespace
import ast

import numpy as np
import sympy as sp

from src.compiler.dual_ir_math import MathProgram
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import concordance_report
from src.compiler.process_graph_function_linking import (
    bind_process_graph_function_sockets,
)
from src.compiler.symbolic_equation_compiler import compile_sympy_equations


CONTRACT = ExtractionContract(
    Path(__file__).resolve().parents[1] / "extraction_contracts/program_extraction.yaml"
)


def _execute_linked_scalar(envelope, feeds, tmp_path, *, name):
    parameters = ", ".join(feeds)
    module, outputs, exports = lower_ast_source_to_ssa(
        f"def root({parameters}):\n    return {name}({parameters})\n",
        "root",
        name=f"{name}_host",
        linked_process_graphs={name: envelope.process_graph},
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
    native_feeds = _managed_native_feeds_by_id(
        SimpleNamespace(module=module, root_name=root.name), feeds,
    )
    execution = prepare_artifact_execution(artifact, native_feeds)
    execution.run()
    result, = outputs[root.name]
    return float(execution.buffers[result.id])


def _execute_linked_tensor_python(envelope, feeds, *, name):
    parameters = ", ".join(feeds)
    module, _outputs, exports = lower_ast_source_to_ssa(
        f"def root({parameters}):\n    return {name}({parameters})\n",
        "root",
        name=f"{name}_host",
        linked_process_graphs={name: envelope.process_graph},
        extraction_contract=CONTRACT,
        progress=lambda _message: None,
    )
    report = concordance_report(module)
    assert report.splitlines()[0].endswith("0 finding(s)"), report

    from src.common.tensors import AbstractTensor
    from src.compiler.ssa_python_materializer import materialize_ir_module

    emitted, skipped = materialize_ir_module(module, tensor_vocabulary=True)
    assert not skipped, skipped
    namespace = {"AbstractTensor": AbstractTensor}
    exec(compile(ast.fix_missing_locations(emitted), "<graph-socket>", "exec"), namespace)
    result = namespace[exports[0]](*(
        AbstractTensor.get_tensor(np.asarray(value)) for value in feeds.values()
    ))
    return np.asarray(getattr(result, "data", result))


def _envelope_and_body():
    x = sp.Symbol("x")
    unresolved = sp.Function("opaque_body")
    envelope = compile_sympy_equations((
        sp.Eq(sp.Symbol("result"), unresolved(x) + 2*x, evaluate=False),
    ), name="opaque_envelope")
    body = MathProgram.symbolic((x*x,), inputs=(x,))
    envelope_node_count = len(envelope.process_graph.G)
    bindings = bind_process_graph_function_sockets(
        envelope.process_graph, {unresolved: body.numeric},
    )
    return envelope, body, unresolved, bindings, envelope_node_count


def test_unresolved_function_is_spliced_by_class_identity_as_graph_nodes():
    envelope, body, unresolved, bindings, envelope_node_count = _envelope_and_body()

    binding, = bindings
    call_id, = binding.call_node_ids
    assert call_id not in envelope.process_graph.G
    assert binding.function is unresolved
    assert binding.graph is body.numeric
    assert len(envelope.process_graph.G) == (
        envelope_node_count - 1 + len(binding.inserted_node_ids)
    )
    assert binding.inserted_node_ids
    inserted_root = binding.inserted_node_ids[-1]
    assert envelope.process_graph.G.nodes[inserted_root]["op"] == "Pow"
    assert envelope.process_graph.G.nodes[inserted_root]["attributes"][
        "graph_socket_body_value"
    ] == body.numeric.roots[0]
    assert body.numeric is not envelope.process_graph


def test_identity_body_replaces_a_root_socket_without_tensorizing_the_port():
    x = sp.Symbol("x")
    unresolved = sp.Function("identity_body")
    envelope = compile_sympy_equations((
        sp.Eq(sp.Symbol("result"), unresolved(x), evaluate=False),
    ), name="identity_envelope")
    body = MathProgram.symbolic((x,), inputs=(x,))
    binding, = bind_process_graph_function_sockets(
        envelope.process_graph, {unresolved: body.numeric},
    )

    graph = envelope.process_graph
    x_id, = graph.G.graph["identity_table"]["x"]
    result_id, = graph.G.graph["identity_table"]["result"]
    assert not binding.inserted_node_ids
    assert binding.call_node_ids[0] not in graph.G
    assert graph.roots == [x_id]
    assert result_id == x_id
    assert graph.G.nodes[x_id]["tensor"]["shape"] == ()


def test_spliced_outer_equation_executes_through_separate_linked_graph(tmp_path):
    envelope, body, _unresolved, bindings, _node_count = _envelope_and_body()
    module, outputs, exports = lower_ast_source_to_ssa(
        "def root(x):\n    return opaque_envelope(x)\n",
        "root",
        name="spliced_outer",
        linked_process_graphs={
            "opaque_envelope": envelope.process_graph,
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
    feeds = _managed_native_feeds_by_id(
        SimpleNamespace(module=module, root_name=root.name), {"x": 3.0},
    )
    execution = prepare_artifact_execution(artifact, feeds)
    execution.run()
    result, = outputs[root.name]
    assert float(execution.buffers[result.id]) == 15.0


def test_graph_function_socket_inside_finite_sum_executes(tmp_path):
    k, n, x = sp.symbols("k n x")
    term = sp.Function("finite_sum_term")
    envelope = compile_sympy_equations((
        sp.Eq(sp.Symbol("result"), sp.Sum(term(k, x), (k, 1, n)), evaluate=False),
    ), name="finite_sum_envelope")
    body = MathProgram.symbolic((k*x,), inputs=(k, x))
    binding, = bind_process_graph_function_sockets(
        envelope.process_graph, {term: body.numeric},
    )

    graph = envelope.process_graph
    x_id, = graph.G.graph["identity_table"]["x"]
    assert graph.G.nodes[x_id]["tensor"]["shape"] == ()
    assert all(graph.G.nodes[node_id]["op"] != "input"
               for node_id in binding.inserted_node_ids)
    multiplied = [node_id for node_id in binding.inserted_node_ids
                  if graph.G.nodes[node_id]["op"] == "Mul"]
    assert len(multiplied) == 1
    assert x_id in {parent for parent, _role in graph.G.nodes[multiplied[0]]["parents"]}

    assert any(data.get("op") == "sum" for _node, data in graph.G.nodes(data=True))
    got = _execute_linked_tensor_python(
        envelope, {"n": 4.0, "x": 2.5}, name="finite_sum_envelope",
    )
    assert got == 25.0


def test_graph_function_socket_inside_definite_integral_executes(tmp_path):
    t, x = sp.symbols("t x")
    integrand = sp.Function("integrand")
    envelope = compile_sympy_equations((
        sp.Eq(
            sp.Symbol("result"),
            sp.Integral(integrand(t, x), (t, 0, 1)),
            evaluate=False,
        ),
    ), name="integral_envelope")
    body = MathProgram.symbolic((x*t,), inputs=(t, x))
    bind_process_graph_function_sockets(
        envelope.process_graph, {integrand: body.numeric},
    )

    got = _execute_linked_tensor_python(
        envelope, {"x": 6.0}, name="integral_envelope",
    )
    assert np.allclose(got, 3.0, rtol=1e-13, atol=0.0)


def test_graph_function_socket_builds_a_truncated_zeta_series(tmp_path):
    k, n, s = sp.symbols("k n s")
    zeta_term = sp.Function("zeta_term")
    envelope = compile_sympy_equations((
        sp.Eq(
            sp.Symbol("result"),
            sp.Sum(zeta_term(k, s), (k, 1, n)),
            evaluate=False,
        ),
    ), name="zeta_series_envelope")
    body = MathProgram.symbolic((k**(-s),), inputs=(k, s))
    bind_process_graph_function_sockets(
        envelope.process_graph, {zeta_term: body.numeric},
    )

    assert any(data.get("op") == "sum"
               for _node, data in envelope.process_graph.G.nodes(data=True))
    got = _execute_linked_tensor_python(
        envelope, {"n": 6.0, "s": 2.0}, name="zeta_series_envelope",
    )
    want = sum(1.0 / (index*index) for index in range(1, 7))
    assert np.isclose(got, want, rtol=1e-13, atol=0.0)
