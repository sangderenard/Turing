from __future__ import annotations

from src.compiler.class_emission_plan import plan_class_emission
from src.compiler.oop_schema import (
    ClassSchema,
    FieldSchema,
    MethodSchema,
    ParameterSchema,
)
from src.transmogrifier.function_table import FunctionTable, ParameterContract
from src.transmogrifier.ssa import (
    BasicBlock,
    Function,
    Instr,
    IRModule,
    SSAClassDefinition,
    SSAClassField,
    SSAClassMethod,
    SSAClassTable,
    SSAValue,
)


def test_canonical_python_class_plan_joins_logical_and_physical_method_abis():
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa

    module, _outputs, _exports = lower_ast_source_to_ssa("""
class Counter:
    def __init__(self, start):
        self.value = start

    def bump(self, amount):
        self.value = self.value + amount
        return self.value

    def twice(self, amount):
        return self.bump(amount) + self.bump(amount)
""")
    plan = plan_class_emission(module)

    assert plan.complete, tuple(issue.format() for issue in plan.issues)
    counter = plan.classes[0]
    assert counter.identity == "Counter"
    assert [(field.name, field.slot) for field in counter.fields] == [
        ("value", 0),
    ]
    methods = {method.name: method for method in counter.methods}
    assert methods["__init__"].kind == "initializer"
    assert methods["__init__"].receiver_fields[0].formal_position == 0
    assert methods["bump"].receiver_evidence == "ssa-receiver-field-locations"
    assert [(item.name, item.position) for item in methods["bump"].parameters] == [
        ("amount", 1),
    ]
    # The nested-call linker appends the logical receiver after the scalar
    # formal. The shared plan follows physical SSA evidence, not source order.
    assert methods["twice"].receiver_position == 1
    assert methods["twice"].receiver_evidence == "linked-method-receiver-storage"
    assert [(item.name, item.position) for item in methods["twice"].parameters] == [
        ("amount", 0),
    ]
    assert all(
        record.resolution == "native_call"
        for record in module.call_table[methods["twice"].function_name]
    )

    # The authored return must point at an actual producer object after object
    # field reconstruction, not a disconnected same-id placeholder.
    bump = module.functions[methods["bump"].function_name]
    produced = {
        id(instruction.res)
        for block in bump.blocks.values()
        for instruction in block.instrs
        if instruction.res is not None
    }
    returned = next(
        instruction.args
        for block in bump.blocks.values()
        for instruction in block.instrs
        if instruction.op == "Ret"
    )
    assert all(id(value) in produced for value in returned)


def test_rich_schema_enriches_but_must_agree_with_the_ssa_projection():
    receiver = SSAValue(0, "object")
    start = SSAValue(1, "float64")
    function = Function("counter_init", [receiver, start], {
        "entry": BasicBlock("entry", [Instr("Ret", [], None)]),
    })
    function_table = FunctionTable()
    reference = function_table.declare(
        "__init__",
        qualified_name="Counter.__init__",
        parameter_contracts=(
            ParameterContract(
                "self", transfer="alias", access="inout",
                storage="record", scope="caller",
            ),
            ParameterContract("start"),
        ),
    )
    definition = SSAClassDefinition(
        "Counter",
        fields=(SSAClassField("value", 0),),
        methods=(SSAClassMethod("__init__", reference.address, "counter_init"),),
    )
    module = IRModule(
        {"counter_init": function},
        function_table=function_table,
        class_table=SSAClassTable((definition,)),
    )
    schema = ClassSchema(
        identity="Counter",
        fields=(FieldSchema("value", "float64", 0, initial=2.5),),
        methods=(MethodSchema(
            "__init__",
            parameters=(ParameterSchema("start", "float64"),),
            function_reference=reference.address,
            function_name="counter_init",
            is_constructor=True,
        ),),
        origin_language="python",
    )

    plan = plan_class_emission(module, schemas=(schema,))

    assert plan.complete
    assert plan.classes[0].fields[0].type_name == "float64"
    assert plan.classes[0].fields[0].initial == 2.5
    assert plan.classes[0].methods[0].parameters[0].type_name == "float64"
    assert plan.classes[0].origin_language == "python"

    mismatched = ClassSchema(
        identity="Counter",
        fields=(FieldSchema("value", "float64", 3),),
        methods=schema.methods,
    )
    rejected = plan_class_emission(module, schemas=(mismatched,))
    assert not rejected.complete
    assert any(
        issue.code == "schema-ssa-disagreement" for issue in rejected.issues
    )


def test_authored_parameter_order_is_independent_of_physical_ssa_positions():
    second = SSAValue(20, "float64")
    receiver = SSAValue(
        10,
        "object",
        accounting={"linked_method_receiver_storage": "self"},
    )
    first = SSAValue(11, "float64")
    function = Function(
        "counter_reordered",
        [second, receiver, first],
        {"entry": BasicBlock("entry", [Instr("Ret", [first], None)])},
        metadata={
            "parameter_names": (
                ("second", second.id),
                ("self", receiver.id),
                ("first", first.id),
            ),
        },
    )
    function_table = FunctionTable()
    reference = function_table.declare(
        "reordered",
        qualified_name="Counter.reordered",
        parameter_contracts=(
            ParameterContract(
                "self", transfer="alias", access="inout",
                storage="record", scope="caller",
            ),
            ParameterContract("first"),
            ParameterContract("second"),
        ),
    )
    module = IRModule(
        {"counter_reordered": function},
        function_table=function_table,
        class_table=SSAClassTable((SSAClassDefinition(
            "Counter",
            methods=(SSAClassMethod(
                "reordered", reference.address, "counter_reordered",
            ),),
        ),)),
    )

    method = plan_class_emission(module).classes[0].methods[0]

    assert method.receiver_position == 1
    assert [(item.name, item.position) for item in method.parameters] == [
        ("first", 2),
        ("second", 0),
    ]
