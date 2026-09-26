import ast
import contextlib
import io
from pathlib import Path
from types import SimpleNamespace

import networkx as nx
import pytest

from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.ir_identities import PRECISION_PIPELINE_METADATA
from src.common.tensors.fused_ir import (
    flatten_tensor_constant,
    uniform_tensor_constant,
)
from src.compiler.ir_identities import (
    NUMERIC_FEATURE_PIPELINE_METADATA,
    apply_numeric_feature_pipeline,
)
from src.compiler.identity_concordance import (
    begin_identity_book,
    current_identity_book,
    end_identity_book,
)
from src.common.tensors.topological_reducer import (
    _record_numeric_annotation_descriptors,
    join_numeric_feature_descriptors,
    numeric_feature_descriptor,
    reduce_abstract_tensor_topology,
    specialize_python_precision_widths,
)
from src.common.tensors.extended_precision import (
    ComplexRationalPrecision,
    Precision,
    RationalPrecision,
    _ComplexRationalBase,
    _RationalBase,
)
from src.transmogrifier.graph.graph_express2 import ProcessGraph
from src.transmogrifier.ssa import BasicBlock, Function, IRModule, Instr, SSAValue


@pytest.mark.parametrize(("annotation", "features", "components"), (
    ("Precision[2]", {"precision"}, 2),
    ("ComplexPrecision[2]", {"complex", "precision"}, 4),
    ("Rational", {"rational"}, 2),
    ("RationalPrecision[2]", {"rational", "precision"}, 4),
    ("ComplexRational", {"complex", "rational"}, 4),
    (
        "ComplexRationalPrecision[2]",
        {"complex", "rational", "precision"},
        8,
    ),
))
def test_numeric_annotation_descriptor_covers_feature_shape_and_surface(
    annotation, features, components,
):
    descriptor = numeric_feature_descriptor(annotation)

    assert descriptor is not None
    assert descriptor.features == frozenset(features)
    assert descriptor.scalar_components == components
    assert {"add", "sub", "mul", "truediv", "neg"} <= set(
        descriptor.operators
    )
    assert "collapse" in descriptor.methods
    if "rational" in features:
        assert {"components", "reciprocal"} <= set(descriptor.methods)


def test_numeric_annotation_descriptors_join_to_the_canonical_feature_union():
    precision = numeric_feature_descriptor("Precision[3, float64]")
    complex_rational = numeric_feature_descriptor("ComplexRational")

    joined = join_numeric_feature_descriptors(precision, complex_rational)

    assert joined is not None
    assert joined.type_name == "ComplexRationalPrecision"
    assert joined.limbs == 3
    assert joined.element_type == "float64"
    assert joined.scalar_components == 12


def test_topology_entry_publishes_parameter_method_and_component_contracts():
    graph = nx.DiGraph()
    graph.graph["function_parameter_annotations"] = {
        "f": {
            "value": "ComplexRationalPrecision[3, float64]",
            "count": "int",
        },
    }

    _record_numeric_annotation_descriptors(SimpleNamespace(G=graph))

    described = graph.graph["function_parameter_numeric_descriptors"]
    value = described["f"]["value"]
    assert "count" not in described["f"]
    assert value["features"] == ("complex", "precision", "rational")
    assert value["scalar_components"] == 12
    assert value["coefficient_paths"] == (
        ("real", "numerator"),
        ("real", "denominator"),
        ("imag", "numerator"),
        ("imag", "denominator"),
    )
    assert {"components", "reciprocal", "collapse"} <= set(value["methods"])


def test_repository_ssa_receives_composite_descriptor_before_refusing_lowering():
    descriptor = numeric_feature_descriptor("RationalPrecision[2]")
    assert descriptor is not None
    left = SSAValue(1, dtype="float64")
    right = SSAValue(2, dtype="float64")
    result = SSAValue(3, dtype="float64")
    function = Function("f", [left, right], {
        "entry": BasicBlock("entry", [
            Instr("Div", [left, right], result, attributes={
                "numeric_composite_pending": True,
                "numeric_feature_descriptor": descriptor.receipt(),
            }),
            Instr("Ret", [result], None),
        ]),
    })
    module = IRModule({"f": function})

    with pytest.raises(
        NotImplementedError,
        match="repository SSA reached the outer-algebra lowering seam",
    ):
        apply_numeric_feature_pipeline(module)

    receipt = module.metadata[NUMERIC_FEATURE_PIPELINE_METADATA]
    assert receipt["status"] == "outer-lowering-required"
    assert receipt["sections"] == [{
        "function": "f",
        "block": "entry",
        "value_id": 3,
        "operand_ids": (1, 2),
        "operation": "truediv",
        "descriptor": descriptor.receipt(),
    }]


def test_generic_numeric_annotation_binds_the_existing_operator_method():
    graph = ProcessGraph(materialize_memory=False)
    source = ast.parse("""
def divide(
    left: ComplexRationalPrecision[2],
    right: ComplexRationalPrecision[2],
):
    return left / right
""")

    with contextlib.redirect_stdout(io.StringIO()):
        graph.build_from_ast(
            source,
            retain=(ComplexRationalPrecision, _ComplexRationalBase),
        )
    reduce_abstract_tensor_topology(graph)

    function_graph = graph.function_table.entry("divide").graph.G
    division = next(
        data
        for _node_id, data in function_graph.nodes(data=True)
        if (data.get("attributes") or {}).get("operator_method")
        == "__truediv__"
    )
    inputs = {
        (data.get("attributes") or {}).get("binding_name"): data
        for _node_id, data in function_graph.nodes(data=True)
        if data.get("type") == "Input"
    }

    assert division["type"] == "Call"
    assert division["attributes"]["operator_receiver_class"] == (
        "ComplexRationalPrecision"
    )
    assert division["attributes"]["result_class_ref"] == (
        "ComplexRationalPrecision"
    )
    assert graph.G.graph["class_table"]["ComplexRationalPrecision"][
        "methods"
    ]["__truediv__"] == division["attributes"]["callee_ref"]
    for parameter in ("left", "right"):
        attributes = inputs[parameter]["attributes"]
        assert "class_ref" not in attributes
        assert attributes["result_class_ref"] == "ComplexRationalPrecision"
        assert attributes["precision_limbs"] == 2
        assert attributes["numeric_feature_descriptor"]["features"] == (
            "complex", "precision", "rational",
        )

    dunder = next(
        entry
        for entry in graph.function_table
        if entry.name == "__truediv__"
        and entry.graph is not None
        and entry.graph.G.graph.get("method_owner")
        == "_ComplexRationalBase"
    )
    dunder_inputs = {
        (data.get("attributes") or {}).get("binding_name"): (
            int(node_id), data.get("attributes") or {}
        )
        for node_id, data in dunder.graph.G.nodes(data=True)
        if data.get("type") == "Input"
    }
    class_page = current_identity_book().page(
        "source_value_class_concordance"
    )
    for parameter in ("self", "other"):
        value_id, attributes = dunder_inputs[parameter]
        assert "class_ref" not in attributes
        assert attributes["result_class_ref"] == "ComplexRationalPrecision"
        assert attributes["precision_limbs"] == 2
        assert tuple(class_page.latest((
            "_ComplexRationalBase.__truediv__", value_id,
        )))[:2] == (
            "ComplexRationalPrecision", 2,
        )


def test_composite_field_projection_preserves_arbitrary_precision_width():
    _book, token = begin_identity_book()
    try:
        graph = ProcessGraph(materialize_memory=False)
        source = ast.parse("""
def coefficient(value: ComplexRationalPrecision[7]):
    return value.real.numerator
""")

        with contextlib.redirect_stdout(io.StringIO()):
            graph.build_from_ast(
                source,
                retain=(
                    ComplexRationalPrecision, _ComplexRationalBase,
                    RationalPrecision, _RationalBase, Precision,
                ),
            )
        reduce_abstract_tensor_topology(graph)

        function_graph = graph.function_table.entry("coefficient").graph.G
        projections = {
            (data.get("attributes") or {}).get("attribute"): (
                int(node_id), data.get("attributes") or {}
            )
            for node_id, data in function_graph.nodes(data=True)
            if data.get("type") == "GetAttr"
        }
        real_id, real = projections["real"]
        numerator_id, numerator = projections["numerator"]
        assert (
            real["result_class_ref"], real["precision_limbs"],
            real["numeric_component_path"],
        ) == ("RationalPrecision", 7, ("real",))
        assert (
            numerator["result_class_ref"], numerator["precision_limbs"],
            numerator["numeric_component_path"],
        ) == ("Precision", 7, ("real", "numerator"))

        page = current_identity_book().page(
            "source_numeric_component_concordance"
        )
        facts = {
            tuple(page.latest(row)[2]): page.latest(row)
            for row in page.rows()
            if isinstance(row, tuple) and row[0] == "coefficient"
        }
        real_fact = facts[("real",)]
        numerator_fact = facts[("real", "numerator")]
        assert real_fact[2] == ("real",)
        assert real_fact[3]["type_name"] == "RationalPrecision"
        assert real_fact[3]["limbs"] == 7
        assert numerator_fact[2] == ("real", "numerator")
        assert numerator_fact[3]["type_name"] == "Precision"
        assert numerator_fact[3]["limbs"] == 7
    finally:
        end_identity_book(token)


def test_precision_width_is_concorded_per_exact_specialization():
    _book, token = begin_identity_book()
    try:
        graph = ProcessGraph(materialize_memory=False)
        source = ast.parse("""
def evaluate(parts, limbs):
    wide = Precision(parts, limbs)
    return (wide + 1.0).collapse()
""")
        with contextlib.redirect_stdout(io.StringIO()):
            graph.build_from_ast(source, retain=(Precision,))
        reduce_abstract_tensor_topology(graph)

        function = graph.function_table.entry("evaluate").graph
        scopes = []
        for width in (2, 7):
            function.G.graph["planner_specializations"] = {"limbs": width}
            specialize_python_precision_widths(function)
            scope = function.G.graph["source_numeric_scope"]
            scopes.append(scope)

            constructor_id, constructor = next(
                (int(node_id), data)
                for node_id, data in function.G.nodes(data=True)
                if isinstance(data.get("expr_obj"), ast.Call)
                and isinstance(data["expr_obj"].func, ast.Name)
                and data["expr_obj"].func.id == "Precision"
            )
            operation_id, operation = next(
                (int(node_id), data)
                for node_id, data in function.G.nodes(data=True)
                if isinstance(data.get("expr_obj"), ast.BinOp)
            )
            collapse_id, collapse = next(
                (int(node_id), data)
                for node_id, data in function.G.nodes(data=True)
                if isinstance(data.get("expr_obj"), ast.Call)
                and isinstance(data["expr_obj"].func, ast.Attribute)
                and data["expr_obj"].func.attr == "collapse"
            )

            assert constructor["attributes"]["precision_limbs"] == width
            assert operation["attributes"]["precision_limbs"] == width
            assert collapse["attributes"]["precision_limbs"] == width
            class_page = current_identity_book().page(
                "source_value_class_concordance"
            )
            assert tuple(class_page.latest((scope, constructor_id)))[:2] == (
                "Precision", width,
            )
            assert tuple(class_page.latest((scope, operation_id)))[:2] == (
                "Precision", width,
            )
            boundary_page = current_identity_book().page(
                "source_precision_boundary_concordance"
            )
            assert boundary_page.latest((scope, collapse_id))[2] == width

        assert scopes[0] != scopes[1]
        specialization_page = current_identity_book().page(
            "source_numeric_specialization_concordance"
        )
        assert len(tuple(specialization_page.rows())) == 2
    finally:
        end_identity_book(token)


def test_projected_coefficients_enter_precision_lowering_at_authored_width():
    _book, token = begin_identity_book()
    try:
        graph = ProcessGraph(materialize_memory=False)
        source = ast.parse("""
def coefficient_product(
    left: RationalPrecision[7],
    right: RationalPrecision[7],
):
    return left.numerator * right.denominator
""")

        with contextlib.redirect_stdout(io.StringIO()):
            graph.build_from_ast(
                source,
                retain=(RationalPrecision, _RationalBase, Precision),
            )
        reduce_abstract_tensor_topology(graph)

        function_graph = graph.function_table.entry(
            "coefficient_product"
        ).graph.G
        product_id, product = next(
            (int(node_id), data)
            for node_id, data in function_graph.nodes(data=True)
            if isinstance(data.get("expr_obj"), ast.BinOp)
        )
        attributes = product.get("attributes") or {}
        assert product["type"] == "precision_mul"
        assert attributes["result_class_ref"] == "Precision"
        assert attributes["precision_limbs"] == 7

        page = current_identity_book().page(
            "source_precision_operator_concordance"
        )
        assert page.latest(("coefficient_product", product_id)) == (
            "precision_mul", product["parents"][0][0], "Precision", 7,
        )
    finally:
        end_identity_book(token)


def test_exact_large_integer_is_not_misclassified_as_tensor_payload():
    exact_bound = 2 ** 4096
    with pytest.raises(ValueError, match="finite binary64 payload range"):
        flatten_tensor_constant(exact_bound)
    assert uniform_tensor_constant(exact_bound) is None


def test_concorded_same_type_dunder_reaches_existing_wrapper_body():
    _book, token = begin_identity_book()
    try:
        graph = ProcessGraph(materialize_memory=False)
        source = ast.parse("""
def multiply(left: RationalPrecision[7], right: RationalPrecision[7]):
    return left * right
""")
        with contextlib.redirect_stdout(io.StringIO()):
            graph.build_from_ast(
                source,
                retain=(RationalPrecision, _RationalBase, Precision),
            )
        reduce_abstract_tensor_topology(graph)

        dunder = next(
            entry.graph.G
            for entry in graph.function_table
            if entry.name == "__mul__"
            and entry.graph is not None
            and entry.graph.G.graph.get("method_owner") == "_RationalBase"
        )
        call_id, call = next(
            (int(node_id), data)
            for node_id, data in dunder.nodes(data=True)
            if (data.get("attributes") or {}).get(
                "numeric_same_type_specialization"
            )
        )
        attributes = call["attributes"]
        assert attributes["result_class_ref"] == "RationalPrecision"
        assert attributes["precision_limbs"] == 7
        target = graph.function_table.entry(attributes["callee_ref"])
        assert target.name == "_binary_same"
        assert target.graph.G.graph["method_owner"] == "_RationalBase"

        page = current_identity_book().page(
            "source_numeric_operator_specialization_concordance"
        )
        fact = page.latest(("_RationalBase.__mul__", call_id))
        assert fact is not None
        assert fact[3]["type_name"] == "RationalPrecision"
        assert fact[3]["limbs"] == 7
        assert fact[4] == attributes["callee_ref"]
    finally:
        end_identity_book(token)


@pytest.fixture(scope="module")
def compiled_complex_rational_precision_division():
    source = """
def repeated_division(
    x: ComplexRationalPrecision[2],
    y: ComplexRationalPrecision[2],
):
    first = x / y
    return first / y
"""
    return lower_ast_source_to_ssa(
        source,
        "repeated_division",
        name="complex_rational_precision_probe",
        extraction_contract=ExtractionContract(
            Path("extraction_contracts/program_extraction.yaml")
        ),
    )


def test_same_type_numeric_division_specializes_authoritative_dependency(
    compiled_complex_rational_precision_division,
):
    module, _outputs, _exports = (
        compiled_complex_rational_precision_division
    )
    specializations = module.metadata["numeric_source_specializations"]

    assert {
        (item["class"], item["method"], item["target"])
        for item in specializations
    } >= {
        ("_ComplexRationalBase", "__truediv__", "_binary_same"),
        ("_ComplexRationalBase", "__rtruediv__", "_binary_same"),
    }
    assert any(
        "_binary_same" in function_name
        for function_name in module.functions
    )
    assert not any(
        "_composed_binary" in function_name
        for function_name in module.functions
    )


def test_complex_rational_precision_repeated_division_lowers_all_components(
    compiled_complex_rational_precision_division,
):
    """A width-two complex-rational-precision value crosses as its leaves.

    It owns real and imaginary rational coefficients, each with numerator
    and denominator Precision[2] expansions: 2 * 2 leaves of 2 limbs each.
    Every authored leaf of ``x`` and ``y`` is a root input carrying both
    limbs, and every leaf of the returned value leaves the root with both
    limbs.  Caller-provided frame storage is ABI by design and not counted.
    """

    module, outputs, _exports = (
        compiled_complex_rational_precision_division
    )
    root = module.functions[
        "complex_rational_precision_probe__repeated_division"
    ]
    receipt = module.metadata.get(PRECISION_PIPELINE_METADATA)

    assert receipt is not None
    assert receipt["status"] == "lowered"
    lowered = dict(root.metadata["precision_lowered_values"])
    leaf_paths = (
        "real.numerator", "real.denominator",
        "imag.numerator", "imag.denominator",
    )
    for parameter in ("x", "y"):
        leaves = {
            str((argument.accounting or {}).get("program_abi_field")):
                int(argument.id)
            for argument in root.args
            if (argument.accounting or {}).get("program_abi_parameter")
            == parameter
            and not (argument.accounting or {}).get(
                "linked_call_frame_storage"
            )
        }
        assert set(leaf_paths) <= set(leaves), (parameter, sorted(leaves))
        assert all(
            len(lowered[leaves[path]]) == 2 for path in leaf_paths
        ), (parameter, {path: lowered.get(leaves[path]) for path in leaf_paths})
    results = outputs[root.name]
    assert len(results) == len(leaf_paths)
    assert all(len(lowered[int(value.id)]) == 2 for value in results)
