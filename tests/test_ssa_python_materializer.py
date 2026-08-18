"""Prove the SSA definitions materialize back into Python that runs and agrees.

The materializer's claim is that a definition which crossed a boundary can be
made executable Python again, body and all. That claim is only worth having if
the Python it produces computes what the SSA said, so the central test here is
the same calibration case ``test_ssa_reference_evaluator`` uses: the symbolic
fluid step, whose authored mathematics is independently available as SymPy.

If the materialized Python matches that oracle, the materializer preserved the
meaning across a 291-instruction body. If it does not, this file is what says
so, before anything is round-tripped through it.
"""
from __future__ import annotations

import ast

import pytest

from src.compiler.oop_schema import (
    ClassSchema,
    FieldSchema,
    MethodSchema,
    ParameterSchema,
)
from src.compiler.ssa_python_materializer import (
    INVENTED,
    UNIMPLEMENTED,
    MaterializationError,
    materialize_class,
    materialize_function_body,
    materialize_module,
    to_source,
)
from src.transmogrifier.ssa import (
    BasicBlock,
    Function,
    Instr,
    SSAClassDefinition,
    SSAClassField,
    SSAClassMethod,
    SSAClassTable,
    SSAValue,
)


# -- vocabulary -----------------------------------------------------------


def test_vocabulary_is_derived_not_invented():
    """No opcode may be spelled here that the compiler's own table lacks."""

    assert INVENTED == frozenset()


def test_the_unimplemented_edge_is_declared_rather_than_guessed():
    """Opcodes needing a stated bit width stay out, and stay named."""

    assert {"ULt", "ULe", "SExt", "ZExt", "FpToUi"} <= UNIMPLEMENTED


def _one_block(*instructions, name="f", args=()):
    return Function(name, list(args), {"entry": BasicBlock("entry", list(instructions))})


def test_an_unknown_operation_raises_rather_than_emitting_something():
    result = SSAValue(1, dtype="float64")
    function = _one_block(
        Instr("NoSuchOperation", [SSAValue(0)], result),
        args=[SSAValue(0, dtype="float64")],
    )

    with pytest.raises(MaterializationError, match="no Python form"):
        materialize_function_body(function)


def test_an_unimplemented_opcode_explains_the_missing_width():
    result = SSAValue(2, dtype="int64")
    function = _one_block(
        Instr("ULt", [SSAValue(0), SSAValue(1)], result),
        args=[SSAValue(0), SSAValue(1)],
    )

    with pytest.raises(MaterializationError, match="stated bit width"):
        materialize_function_body(function)


def test_control_flow_is_refused_by_name_rather_than_flattened():
    """A CFG is not silently emitted as straight-line Python."""

    function = Function(
        "branchy",
        [],
        {
            "entry": BasicBlock("entry", []),
            "then": BasicBlock("then", []),
        },
    )

    with pytest.raises(MaterializationError) as raised:
        materialize_function_body(function)
    assert "'entry'" in str(raised.value) and "'then'" in str(raised.value)


# -- the constant that motivates the whole exercise -----------------------


def test_a_constant_authored_inside_a_body_survives_materialization():
    """The custom trig epsilon case, which a bare method name cannot carry."""

    epsilon = SSAValue(1, dtype="float64")
    scaled = SSAValue(2, dtype="float64")
    argument = SSAValue(0, dtype="float64")
    function = _one_block(
        Instr("Const", [], epsilon, attributes={"value": 1.0000000001e-07}),
        Instr("Add", [argument, epsilon], scaled),
        Instr("Ret", [scaled], SSAValue(3)),
        name="with_epsilon",
        args=[argument],
    )

    body, _ = materialize_function_body(function, parameter_names=("x",))
    source = to_source(ast.Module(body=body, type_ignores=[]))

    assert "1.0000000001e-07" in source
    assert "x + t1" in source


# -- classes ---------------------------------------------------------------


def test_the_ssa_field_layout_becomes_source_order():
    definition = SSAClassDefinition(
        identity="fluid.Cell",
        fields=(SSAClassField("v", 1), SSAClassField("u", 0)),
        methods=(),
    )

    node, _ = materialize_class(definition)
    source = to_source(node)

    assert node.name == "Cell"
    # Emitted in slot order, not in the order the SSA record happened to list.
    assert source.index("u: object") < source.index("v: object")
    # The dotted identity is not silently dropped when Python cannot spell it.
    assert "fluid.Cell" in source


def test_a_gap_in_the_slot_layout_is_reported_not_closed_up():
    """Closing a gap would move every field after it, silently."""

    definition = SSAClassDefinition(
        identity="Gapped",
        fields=(SSAClassField("a", 0), SSAClassField("c", 2)),
    )

    with pytest.raises(MaterializationError, match="gapless"):
        materialize_class(definition)


def test_a_method_with_no_supplied_body_raises_instead_of_passing():
    """A class that silently no-ops is worse than one that will not run."""

    definition = SSAClassDefinition(
        identity="Cell",
        methods=(SSAClassMethod("advance", 3, "cell_advance"),),
    )

    node, _ = materialize_class(definition)
    source = to_source(node)

    assert "NotImplementedError" in source
    assert "cell_advance" in source
    assert "pass" not in source


def test_a_class_materializes_into_something_that_actually_runs():
    """Fields, a constructor, and a real body, executed."""

    receiver = SSAValue(0)
    gain = SSAValue(1, dtype="float64")
    read = SSAValue(2, dtype="float64")
    scaled = SSAValue(3, dtype="float64")
    body = _one_block(
        Instr("Const", [], gain, attributes={"value": 2.5}),
        Instr("GetAttr", [receiver], read, attributes={"attribute": "u"}),
        Instr("Mul", [read, gain], scaled),
        Instr("SetAttr", [receiver, scaled], SSAValue(4), attributes={"attribute": "u"}),
        Instr("Ret", [scaled], SSAValue(5)),
        name="cell_amplify",
        args=[receiver],
    )

    schema = ClassSchema(
        identity="fluid.Cell",
        fields=(
            FieldSchema(name="u", type_name="float", slot=0, initial=1.5),
            FieldSchema(name="v", type_name="float", slot=1, initial=0.0),
        ),
        methods=(
            MethodSchema(
                name="amplify",
                parameters=(),
                returns="float",
                function_name="cell_amplify",
            ),
        ),
    )

    module = materialize_module([schema], functions={"cell_amplify": body})
    namespace: dict = {}
    exec(compile(module, "<materialized>", "exec"), namespace)

    cell = namespace["Cell"]()
    assert cell.u == 1.5
    assert cell.amplify() == 3.75
    # The SetAttr was a real write, not a discarded value.
    assert cell.u == 3.75


def test_the_schema_recovered_from_ssa_still_projects_back_onto_it():
    """Materializing must not be a place where the layout quietly changes."""

    definition = SSAClassDefinition(
        identity="Cell",
        fields=(SSAClassField("u", 0), SSAClassField("v", 1)),
        methods=(SSAClassMethod("advance", 7, "cell_advance"),),
    )
    schema = ClassSchema.from_ssa_class_definition(definition)

    assert schema.ssa_projection_agrees(definition) == ()
    node, _ = materialize_class(definition)
    assert [
        statement.target.id
        for statement in node.body
        if isinstance(statement, ast.AnnAssign)
    ] == ["u", "v"]


def test_a_class_table_materializes_as_one_module():
    table = SSAClassTable(
        classes=(
            SSAClassDefinition(identity="A", fields=(SSAClassField("x", 0),)),
            SSAClassDefinition(identity="B", fields=(SSAClassField("y", 0),)),
        )
    )

    module = materialize_module(table)
    assert [node.name for node in module.body if isinstance(node, ast.ClassDef)] == [
        "A",
        "B",
    ]


def test_a_static_method_keeps_its_decorator_and_drops_the_receiver():
    schema = ClassSchema(
        identity="Util",
        methods=(
            MethodSchema(
                name="scale",
                parameters=(ParameterSchema(name="x", type_name="float"),),
                returns="float",
                is_static=True,
            ),
        ),
    )

    source = to_source(materialize_class(schema)[0])
    assert "@staticmethod" in source
    assert "def scale(x: float) -> float:" in source


# -- the loop closed at AbstractTensor.ssa --------------------------------


def test_a_class_stated_through_the_mirrors_comes_back_as_python():
    """State it with the shared vocabulary, get Python back."""

    from src.common.tensors import AbstractTensor

    ssa = AbstractTensor.ssa
    definition = ssa.define_class(
        "fluid.Cell",
        fields=(ssa.define_field("u", 0), ssa.define_field("v", 1)),
        methods=(ssa.define_method("advance", 3, "cell_advance"),),
    )

    source = to_source(ssa.to_python(definition))
    assert source.startswith("class Cell:")
    assert "u: object" in source and "v: object" in source


def test_the_materializer_import_stays_deferred():
    """``abstraction`` must remain importable without the compiler tables.

    The mirrors module already defers its ``transmogrifier`` import for this
    reason; ``to_python`` reaches further still, into the likeness tables, so
    the same rule has to hold or every consumer of AbstractTensor pays for it.
    """

    import subprocess
    import sys

    probe = (
        "import sys;"
        "import src.common.tensors.abstraction_methods.ssa_structure;"
        "print('src.compiler.ssa_python_materializer' in sys.modules)"
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        cwd=str(__import__("pathlib").Path(__file__).resolve().parents[1]),
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip().splitlines()[-1] == "False"


# -- calibration against independently known truth ------------------------


@pytest.fixture(scope="module")
def fluid_step():
    from src.compiler.symbolic_fluid_model import compile_symbolic_fluid_step

    return compile_symbolic_fluid_step()


def test_materialized_python_reproduces_the_authored_mathematics(fluid_step):
    """A real 291-instruction body, executed, against its SymPy oracle.

    This is the same case the reference evaluator is calibrated on, for the
    same reason: three independent paths to one set of numbers. If the
    materialized Python joins them, the materializer preserved the meaning.
    """

    import sympy

    from src.compiler.symbolic_fluid_model import (
        symbolic_viscous_shallow_water_equations,
    )

    function = fluid_step.function
    argument_names = tuple(function.metadata["argument_names"])
    output_names = tuple(function.metadata["output_names"])

    body, uses_math = materialize_function_body(
        function, parameter_names=argument_names
    )
    definition = ast.FunctionDef(
        name="materialized",
        args=ast.arguments(
            posonlyargs=[],
            args=[ast.arg(arg=name) for name in argument_names],
            vararg=None,
            kwonlyargs=[],
            kw_defaults=[],
            kwarg=None,
            defaults=[],
        ),
        body=body,
        decorator_list=[],
        returns=None,
        type_comment=None,
        type_params=[],
    )
    prologue = [ast.Import(names=[ast.alias(name="math")])] if uses_math else []
    module = ast.fix_missing_locations(
        ast.Module(body=prologue + [definition], type_ignores=[])
    )
    namespace: dict = {}
    exec(compile(module, "<materialized>", "exec"), namespace)

    values = dict.fromkeys(argument_names, 0.0)
    values.update(
        coriolis=0.0, dt=0.2, dx=0.25, gravity=1.0, linear_drag=0.0,
        minimum_height=0.001, tracer_diffusivity=0.0, viscosity=0.0,
        height_center=1.008425, height_east=1.0, height_north=1.00037,
        height_south=1.00037, height_west=1.0,
        momentum_x_center=0.01, momentum_x_east=0.002, momentum_x_north=0.003,
        momentum_x_south=0.004, momentum_x_west=0.005,
        momentum_y_center=0.01, momentum_y_east=0.006, momentum_y_north=0.007,
        momentum_y_south=0.008, momentum_y_west=0.009,
        tracer_center=0.5, tracer_east=0.1, tracer_north=0.2,
        tracer_south=0.3, tracer_west=0.4,
    )
    positional = [values[name] for name in argument_names]
    produced = namespace["materialized"](*positional)

    model = symbolic_viscous_shallow_water_equations()
    by_name = {str(equation.lhs): equation.rhs for equation in model.equations}
    ordered = [model.symbols[name] for name in argument_names]
    for name, value in zip(output_names, produced):
        oracle = float(sympy.lambdify(ordered, by_name[name], "numpy")(*positional))
        assert float(value) == pytest.approx(oracle, abs=1e-12)


def test_materialized_python_agrees_with_the_ssa_reference_evaluator(fluid_step):
    """The two independent readings of the same SSA must not diverge."""

    from src.compiler.ssa_reference_evaluator import SSAReferenceEvaluator

    function = fluid_step.function
    argument_names = tuple(function.metadata["argument_names"])

    values = dict.fromkeys(argument_names, 0.25)
    values.update(dt=0.2, dx=0.25, gravity=1.0, minimum_height=0.001)
    positional = [values[name] for name in argument_names]

    body, uses_math = materialize_function_body(
        function, parameter_names=argument_names
    )
    definition = ast.FunctionDef(
        name="materialized",
        args=ast.arguments(
            posonlyargs=[],
            args=[ast.arg(arg=name) for name in argument_names],
            vararg=None,
            kwonlyargs=[],
            kw_defaults=[],
            kwarg=None,
            defaults=[],
        ),
        body=body,
        decorator_list=[],
        returns=None,
        type_comment=None,
        type_params=[],
    )
    prologue = [ast.Import(names=[ast.alias(name="math")])] if uses_math else []
    module = ast.fix_missing_locations(
        ast.Module(body=prologue + [definition], type_ignores=[])
    )
    namespace: dict = {}
    exec(compile(module, "<materialized>", "exec"), namespace)
    produced = namespace["materialized"](*positional)

    evaluated = SSAReferenceEvaluator(fluid_step.module).run(
        function.name,
        {int(formal.id): value for formal, value in zip(function.args, positional)},
    )
    for materialized_value, reference_value in zip(produced, evaluated.returned):
        assert float(materialized_value) == pytest.approx(
            float(reference_value), abs=1e-12
        )
