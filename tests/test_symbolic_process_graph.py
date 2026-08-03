from __future__ import annotations

import ast
import contextlib
import inspect
import io
import textwrap

import numpy as np
import pytest
import sympy

from src.common.tensors.topological_reducer import (
    reduce_abstract_tensor_topology,
)
from src.compiler.symbolic_process_graph import (
    SYMPY_PROCESS_GRAPH_TRANSLATIONS,
    aggressively_simplify_expression,
    aggressively_simplify_process_relations,
    boolean_domain_constraint,
    boolean_polynomial,
    polynomial_select,
    ingest_sympy_process_model,
    process_graph_to_sympy_expressions,
    process_graph_to_sympy_relations,
    symbolically_reduce_process_graph,
    unroll_symbolic_transition,
    unsigned_bit_expression,
)
from src.compiler.bitops_translator import BitOpsTranslator
from src.compiler.process_graph_helper import provenance_to_process_graph
from src.compiler.ssa_builder import process_graph_to_ssa_instrs
from src.transmogrifier.graph.graph_express2 import ProcessGraph


def _direct_precompile_length(graph):
    # Function-table extraction can retain the owning module's scheduler.
    # Rebind it so this precompile measures the selected function graph only.
    graph.scheduler = type(graph.scheduler)(graph)
    return len(process_graph_to_ssa_instrs(graph, schedule="asap"))


def test_sympy_ingestion_round_trips_semantically():
    x, y = sympy.symbols("x y")
    expression = (x + 2 * y) ** 2 + sympy.sin(x)
    graph = ProcessGraph(materialize_memory=False)

    graph.build_from_expression(expression)
    rebuilt, = process_graph_to_sympy_expressions(graph)

    assert graph.G.number_of_nodes() > 1
    assert sympy.simplify(rebuilt - expression) == 0


def test_sympy_reverse_translation_table_recovers_control_index_and_calls():
    x = sympy.Symbol("x")
    samples = sympy.IndexedBase("samples")
    opaque = sympy.Function("opaque")
    expression = sympy.Piecewise(
        (sympy.sin(x) + opaque(samples[1]), x > 0),
        (x**2, True),
    )
    graph = ProcessGraph(materialize_memory=False)

    graph.build_from_expression(expression)
    rebuilt, = process_graph_to_sympy_expressions(graph)

    operations = {
        data.get("op") for _node_id, data in graph.G.nodes(data=True)
    }
    select = next(
        data
        for _node_id, data in graph.G.nodes(data=True)
        if data.get("op") == "Select"
    )
    indexed = next(
        data
        for _node_id, data in graph.G.nodes(data=True)
        if data.get("op") == "Indexed"
    )
    call = next(
        data
        for _node_id, data in graph.G.nodes(data=True)
        if data.get("op") == "Call"
    )

    assert (
        SYMPY_PROCESS_GRAPH_TRANSLATIONS[sympy.Piecewise].operation
        == "Select"
    )
    assert {role for _parent, role in select["parents"]} == {
        "condition", "if_true", "if_false",
    }
    assert [role for _parent, role in indexed["parents"]] == ["base", "index"]
    assert call["attributes"]["callee"] == "opaque"
    assert {"Select", "Indexed", "Call", "Sin"} <= operations
    assert graph.G.graph["sympy_translation_fallbacks"] == ()
    assert rebuilt == expression


def test_reduced_ast_function_projects_through_canonical_schema():
    source = (
        "def f(x, y):\n"
        "    z = x + y * 2\n"
        "    return z\n"
    )
    module = ProcessGraph(materialize_memory=False)
    with contextlib.redirect_stdout(io.StringIO()):
        module.build_from_ast(ast.parse(source))
    reduce_abstract_tensor_topology(module)
    graph = module.function_table.entry("f").graph

    rebuilt, = process_graph_to_sympy_expressions(graph)
    x, y = sympy.symbols("x y")

    assert sympy.simplify(rebuilt - (x + y * 2)) == 0


def test_to_sympy_compatibility_package_uses_current_expression_tensor_api():
    x = sympy.Symbol("x")
    graph = ProcessGraph(materialize_memory=False)
    graph.build_from_expression(x + 1)

    registry, tensor = graph.to_sympy()

    assert registry == [x + 1]
    assert tuple(tensor.domain_shape) == (1,)


def test_first_filtered_mandelbrot_avi_region_round_trips_into_compiler():
    from src.common.tensors.accelerator_backends.demo_mandelbrot_fusion import (
        build_parametric_mandelbrot_glsl_deployment,
    )
    from src.transmogrifier.graph.graph_deep_compiler import (
        GraphDeepCompiler,
    )
    from src.transmogrifier.operator_defs import (
        abstract_tensor_funcs,
        abstract_tensor_sigs,
    )
    from src.common.tensors.abstraction import AbstractTensor

    deployment, _module = build_parametric_mandelbrot_glsl_deployment(8)
    filtered = deployment.dispatch_subgraphs[0]
    rebuilt, report = symbolically_reduce_process_graph(filtered)
    rebuilt.compute_levels("asap")
    compiler = GraphDeepCompiler(
        rebuilt,
        dict(abstract_tensor_funcs),
        abstract_tensor_sigs,
    )

    compiled = compiler.build_function()
    with AbstractTensor.use_backend("numpy"):
        frames = AbstractTensor.tensor(
            np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        )
        result = compiled(
            floatframes=frames,
            floatframe_index=1,
        )[0]

    assert callable(compiled)
    assert filtered.G.graph["deployment_nodes"] == (27,)
    assert report.original == report.reduced
    assert any(
        data.get("type") == "Indexed"
        for _node_id, data in rebuilt.G.nodes(data=True)
    )
    assert "op_" in compiler._code
    np.testing.assert_array_equal(
        result.numpy(),
        np.arange(12, 24, dtype=np.float32).reshape(3, 4),
    )


def test_deep_compiler_preserves_string_literal_for_tensor_dtype():
    from src.transmogrifier.graph.graph_deep_compiler import GraphDeepCompiler
    from src.transmogrifier.operator_defs import (
        abstract_tensor_funcs,
        abstract_tensor_sigs,
    )

    graph = ProcessGraph(materialize_memory=False)
    graph.G.clear()
    graph.G.add_node(
        1, type="Constant", label="int64", parents=[], constant="int64"
    )
    graph.G.add_node(
        2, type="Store", label="result", parents=[(1, "value")]
    )
    graph.G.add_edge(1, 2)
    graph.levels = {1: 0, 2: 1}
    compiler = GraphDeepCompiler(
        graph,
        dict(abstract_tensor_funcs),
        abstract_tensor_sigs,
    )
    compiled = compiler.build_function()

    result, = compiled()

    assert "b'int64'" not in compiler._code
    assert "'int64'" in compiler._code
    assert result == "int64"


def test_branch_phi_projects_as_an_exact_piecewise_expression():
    module = ProcessGraph(materialize_memory=False)
    with contextlib.redirect_stdout(io.StringIO()):
        module.build_from_ast(ast.parse(
            "def choose(condition, left, right):\n"
            "    if condition:\n"
            "        selected = left\n"
            "    else:\n"
            "        selected = right\n"
            "    return selected\n"
        ))
    reduce_abstract_tensor_topology(module)
    graph = module.function_table.entry("choose").graph

    expression, = process_graph_to_sympy_expressions(graph)
    condition, left, right = sympy.symbols("condition left right")

    assert expression.subs({condition: 1, left: 11, right: 7}) == 11
    assert expression.subs({condition: 0, left: 11, right: 7}) == 7


def test_bitops_nand_algebra_can_solve_a_process_graph_backwards():
    translator = BitOpsTranslator(1)
    left_bits = translator.bits_from_int(0)
    right_bits = translator.bits_from_int(0)
    translator.graph.bind_input(
        left_bits, name="left", metadata={"result_length": 1}
    )
    translator.graph.bind_input(
        right_bits, name="right", metadata={"result_length": 1}
    )
    result = translator.apply_bits("bitand", left_bits, right_bits)
    result_id = translator.graph.producer_index(result)
    graph = provenance_to_process_graph(translator.graph)
    graph.roots = [result_id]
    model = process_graph_to_sympy_relations(graph)
    left = model.inputs["left"]
    right = model.inputs["right"]
    output, = model.outputs

    solutions = sympy.solve(
        (*model.relations, sympy.Eq(output, 1)),
        tuple(dict.fromkeys(model.expressions.values())),
        dict=True,
    )

    assert model.uninterpreted == ()
    assert solutions
    assert all(solution[left] == 1 and solution[right] == 1 for solution in solutions)


def test_boolean_polynomials_and_bit_recombination_are_exact_on_bits():
    left, right = sympy.symbols("left right")

    assert boolean_polynomial("nand", left, right) == 1 - left * right
    assert unsigned_bit_expression((left, right)) == left + 2 * right
    assert boolean_domain_constraint(left) == sympy.Eq(left * (left - 1), 0)
    for left_value in (0, 1):
        for right_value in (0, 1):
            actual = boolean_polynomial("xor", left, right).subs({
                left: left_value,
                right: right_value,
            })
            assert actual == (left_value ^ right_value)


def test_backend_minimum_maximum_and_tanh_have_registered_sympy_semantics():
    x, y, z = sympy.symbols("x y z")
    expected = sympy.tanh(sympy.Max(x, sympy.Min(y, z)))
    graph = ProcessGraph(materialize_memory=False)
    graph.build_from_expression(expected)
    operation_aliases = {
        "Min": "minimum",
        "Max": "maximum",
        "Tanh": "tanh",
    }
    for _node_id, data in graph.G.nodes(data=True):
        operation = str(data.get("op"))
        if operation in operation_aliases:
            data["op"] = operation_aliases[operation]

    expression, = process_graph_to_sympy_expressions(graph)
    model = process_graph_to_sympy_relations(graph)

    assert expression == expected
    assert model.uninterpreted == ()


def test_symbolic_transition_unroll_uses_simultaneous_state_updates():
    state, choose_forward = sympy.symbols("state choose_forward")
    unrolled = unroll_symbolic_transition(
        {
            "state": polynomial_select(
                choose_forward,
                state + 2,
                state - 1,
            ),
            "choose_forward": 1 - choose_forward,
        },
        2,
        initial={"state": 0, "choose_forward": 1},
    )

    solution = sympy.solve(unrolled.equations, dict=True)[0]

    assert solution[unrolled.states[1]["state"]] == 2
    assert solution[unrolled.states[1]["choose_forward"]] == 0
    assert solution[unrolled.states[2]["state"]] == 1
    assert solution[unrolled.states[2]["choose_forward"]] == 1


def test_ast_precompile_sympy_aggressive_round_trip_reduces_program_length():
    module = ProcessGraph(materialize_memory=False)
    with contextlib.redirect_stdout(io.StringIO()):
        module.build_from_ast(ast.parse(
            "def redundant_polynomial(left, right):\n"
            "    summed = left + right\n"
            "    square = summed * summed\n"
            "    expanded = left * left + 2 * left * right + right * right\n"
            "    cancelled = square - expanded\n"
            "    return cancelled + 3 * right\n"
        ))
    reduce_abstract_tensor_topology(module)
    original_graph = module.function_table.entry("redundant_polynomial").graph
    original_precompile_length = _direct_precompile_length(original_graph)

    rebuilt_graph, report = symbolically_reduce_process_graph(
        original_graph,
        aggressive=True,
    )
    rebuilt_precompile_length = _direct_precompile_length(rebuilt_graph)
    rebuilt_expression, = process_graph_to_sympy_expressions(rebuilt_graph)

    assert sympy.simplify(report.original[0] - report.reduced[0]) == 0
    assert rebuilt_expression == report.reduced[0]
    assert rebuilt_graph.G.graph["sympy_translation_fallbacks"] == ()
    assert sympy.count_ops(report.reduced[0]) < sympy.count_ops(report.original[0])
    assert rebuilt_precompile_length < original_precompile_length


@pytest.mark.stress
def test_ast_ingested_pixel_to_jpeg_aggressive_round_trip_completes(
    record_property,
):
    """Exercise the real hierarchical JPEG source without a fused shortcut.

    This intentionally waits for both aggressive SymPy passes and reports each
    compiler stage. The complete equation model, rather than only the compact
    output expression, is used for reconstruction so effects remain present.
    """

    from src.common.tensors.compression.jpeg.frame import encode_jfif_resident
    from src.common.tensors.operator_catalog import (
        include_ast_parent_outside_abstract_tensor,
    )

    module = ProcessGraph(materialize_memory=False)
    module.python_bindings = dict(encode_jfif_resident.__globals__)
    module.python_package = encode_jfif_resident.__module__.rpartition(".")[0]
    source = textwrap.dedent(inspect.getsource(encode_jfif_resident))
    with contextlib.redirect_stdout(io.StringIO()):
        module.build_from_ast(
            ast.parse(source),
            resolve_unresolved_parents=True,
            parent_include=include_ast_parent_outside_abstract_tensor,
        )
    reduce_abstract_tensor_topology(module)
    encoder_graph = module.function_table.entry("encode_jfif_resident").graph

    first_reduction_nodes = encoder_graph.G.number_of_nodes()
    original_precompile_length = _direct_precompile_length(encoder_graph)
    post_precompile_nodes = encoder_graph.G.number_of_nodes()
    original_expression, = process_graph_to_sympy_expressions(encoder_graph)
    relational_model = process_graph_to_sympy_relations(
        encoder_graph,
        live_only=False,
    )
    reduced_expression = aggressively_simplify_expression(
        original_expression,
        rounds=1,
    )
    reduced_equations = aggressively_simplify_process_relations(
        relational_model,
        rounds=1,
    )
    changed_equations = sum(
        original.rhs != reduced.rhs
        for original, reduced in zip(
            relational_model.equations,
            reduced_equations,
        )
    )

    rebuilt_graph = ProcessGraph(materialize_memory=False)
    rebuilt_mapping = ingest_sympy_process_model(
        rebuilt_graph,
        relational_model,
        equations=reduced_equations,
    )
    reverse_translated_nodes = rebuilt_graph.G.number_of_nodes()
    reduce_abstract_tensor_topology(rebuilt_graph)
    second_reduction_nodes = rebuilt_graph.G.number_of_nodes()
    rebuilt_precompile_length = _direct_precompile_length(rebuilt_graph)
    post_rebuilt_precompile_nodes = rebuilt_graph.G.number_of_nodes()

    record_property("ast_module_nodes", module.G.number_of_nodes())
    record_property("first_reduction_nodes", first_reduction_nodes)
    record_property("original_precompile_length", original_precompile_length)
    record_property("post_precompile_nodes", post_precompile_nodes)
    record_property("sympy_before_ops", int(sympy.count_ops(original_expression)))
    record_property("sympy_before_srepr", len(sympy.srepr(original_expression)))
    record_property("sympy_after_ops", int(sympy.count_ops(reduced_expression)))
    record_property("sympy_after_srepr", len(sympy.srepr(reduced_expression)))
    record_property("sympy_expression_changed", reduced_expression != original_expression)
    record_property("sympy_relation_equations", len(relational_model.equations))
    record_property("sympy_relation_changed_equations", changed_equations)
    record_property("reverse_translated_nodes", reverse_translated_nodes)
    record_property("second_reduction_nodes", second_reduction_nodes)
    record_property("rebuilt_precompile_length", rebuilt_precompile_length)
    record_property(
        "post_rebuilt_precompile_nodes",
        post_rebuilt_precompile_nodes,
    )
    record_property("uninterpreted_nodes", len(relational_model.uninterpreted))

    assert module.G.number_of_nodes() > 5_000
    assert first_reduction_nodes > 100
    assert relational_model.uninterpreted
    assert rebuilt_graph.G.graph["sympy_translation_fallbacks"] == ()
    assert len(relational_model.node_specs) == post_precompile_nodes
    assert len(rebuilt_mapping) == post_precompile_nodes
    assert all(
        rebuilt_graph.G.has_edge(
            rebuilt_mapping[source], rebuilt_mapping[target]
        )
        for source, target in relational_model.ordering_edges
    )
    assert sympy.count_ops(reduced_expression) <= sympy.count_ops(
        original_expression
    )
    assert rebuilt_precompile_length > 0
