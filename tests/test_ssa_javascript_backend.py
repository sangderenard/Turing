from __future__ import annotations

import json
from pathlib import Path
import subprocess

from src.compiler.ssa_javascript_backend import emit_ssa_module_to_javascript
from src.compiler.fused_program_python_backend import (
    supported_elementwise_operations as python_oracle_operations,
)
from src.compiler.javascript_numeric_operators import (
    JAVASCRIPT_NUMERIC_OPERATORS,
    supported_javascript_numeric_operations,
)
from src.compiler.ssa_numeric_operators import TENSOR_SSA_OPERATOR_BY_NAME
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


def _run_module(tmp_path: Path, source: str, body: str):
    module_path = tmp_path / "program.mjs"
    runner_path = tmp_path / "runner.mjs"
    module_path.write_text(source, encoding="utf-8")
    runner_path.write_text(
        "import * as program from './program.mjs';\n" + body,
        encoding="utf-8",
    )
    completed = subprocess.run(
        ["node", str(runner_path)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout)


def test_none_value_emits_javascript_null_without_a_numeric_substitution(tmp_path):
    absent = SSAValue(1, "none")
    function = Function("absent", [], {"entry": BasicBlock("entry", [
        Instr("NoneValue", [], absent),
        Instr("Ret", [absent], None),
    ])})

    artifact = emit_ssa_module_to_javascript(
        IRModule({"absent": function}), function_name="absent",
    )

    assert artifact.complete, artifact.shortfalls
    assert "t1 = null;" in artifact.source
    assert _run_module(
        tmp_path, artifact.source,
        "console.log(JSON.stringify(program.default.run([])));",
    ) == [None]


def test_emitter_supplies_named_runtime_utilities_by_deterministic_identity(tmp_path):
    output = SSAValue(1, "int32")
    function = Function("one", [], {"entry": BasicBlock("entry", [
        Instr("Const", [], output, attributes={"value": 1}),
        Instr("Ret", [output], None),
    ])})
    requested = (
        "turing.wasm.registry",
        "turing.world.registry",
        "turing.revision.channel",
    )
    artifact = emit_ssa_module_to_javascript(
        IRModule({"one": function}), "one", runtime_utilities=requested,
    )

    assert artifact.complete, artifact.shortfalls
    utilities = artifact.api.metadata["runtime_utilities"]
    assert tuple(item["identity"] for item in utilities) == requested
    assert all(item["content_key"].startswith("javascript-utility:sha256:") for item in utilities)
    assert all(f"turing-runtime-utility: {identity}" in artifact.source for identity in requested)
    observed = _run_module(
        tmp_path,
        artifact.source,
        """
const world = program.RUNTIME_UTILITIES["turing.world.registry"].exports.create({
  identity: "world", objects: [{identity: "room", parent: "world", semantic_parts: []}]
});
const revisions = program.RUNTIME_UTILITIES["turing.revision.channel"].exports.create("edits");
const event = revisions.publish({revision: 1, identity: "room"});
console.log(JSON.stringify({room: world.object("room").identity, revision: event.revision}));
""",
    )
    assert observed == {"room": "room", "revision": 1}


def test_emitted_function_performance_label_preserves_inline_intent(tmp_path):
    value = SSAValue(0, "float64")
    result = SSAValue(1, "float64")
    function = Function(
        "hot_leaf", [value], {"entry": BasicBlock("entry", [
            Instr("Add", [value, value], result),
            Instr("Ret", [result], None),
        ])},
        metadata={"performance": {
            "inline": "prefer", "hot_path": True,
            "frequency": "per-frame", "allocation_risk": "none",
        }},
    )
    artifact = emit_ssa_module_to_javascript(
        IRModule({"hot_leaf": function}), "hot_leaf",
    )

    (label,) = artifact.api.metadata["performance_labels"]
    assert label == {
        "identity": "function:hot_leaf", "function": "hot_leaf",
        "role": "function", "inline": "prefer", "basis": "authored-metadata",
        "hot_path": True, "frequency": "per-frame", "instruction_count": 2,
        "block_count": 1, "call_count": 0, "branch_count": 0,
        "async_boundary": False, "allocation_risk": "none",
    }
    observed = _run_module(
        tmp_path, artifact.source,
        'console.log(JSON.stringify(program.PERFORMANCE_LABELS[0]));',
    )
    assert observed["inline"] == "prefer"
    assert observed["hot_path"] is True


def test_authored_python_none_reaches_javascript_as_ssa_none(tmp_path):
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa

    module, _outputs, _exports = lower_ast_source_to_ssa(
        "def absent():\n    return None\n"
    )
    function_name = next(iter(module.functions))
    instructions = [
        instruction
        for block in module.functions[function_name].blocks.values()
        for instruction in block.instrs
    ]

    assert instructions[0].op == "NoneValue"
    assert instructions[0].res.dtype == "none"
    assert instructions[0].attributes == {}

    artifact = emit_ssa_module_to_javascript(
        module, function_name=function_name,
    )
    assert artifact.complete, artifact.shortfalls
    assert _run_module(
        tmp_path, artifact.source,
        "console.log(JSON.stringify(program.default.run([])));",
    ) == [None]


def test_ssa_class_table_emits_executable_javascript_classes_first(tmp_path):
    receiver = SSAValue(0, "object")
    start = SSAValue(1, "float64")
    constructor = Function("counter_init", [receiver, start], {
        "entry": BasicBlock("entry", [
            Instr("SetAttr", [receiver, start], None, attributes={"attribute": "value"}),
            Instr("Ret", [], None),
        ]),
    }, metadata={"argument_names": ("self", "start")})

    amount = SSAValue(2, "float64")
    current = SSAValue(3, "float64")
    updated = SSAValue(4, "float64")
    bump = Function("counter_bump", [receiver, amount], {
        "entry": BasicBlock("entry", [
            Instr("GetAttr", [receiver], current, attributes={"attribute": "value"}),
            Instr("Add", [current, amount], updated),
            Instr("SetAttr", [receiver, updated], None, attributes={"attribute": "value"}),
            Instr("Ret", [updated], None),
        ]),
    }, metadata={"argument_names": ("self", "amount")})

    table = SSAClassTable((SSAClassDefinition(
        identity="model.Counter",
        # Deliberately record fields out of slot order. The generated class
        # preserves appearance order while retaining independent slot ids.
        fields=(SSAClassField("spare", 1), SSAClassField("value", 0)),
        methods=(
            SSAClassMethod("__init__", 7, "counter_init"),
            SSAClassMethod("bump", 8, "counter_bump"),
        ),
    ),))
    artifact = emit_ssa_module_to_javascript(
        IRModule(
            {"counter_init": constructor, "counter_bump": bump},
            class_table=table,
        )
    )

    assert artifact.complete, artifact.shortfalls
    assert artifact.entry is None
    assert "export class model_Counter" in artifact.source
    assert artifact.api.metadata["class_table"][0]["identity"] == "model.Counter"
    assert tuple(
        {"name": field["name"], "slot": field["slot"]}
        for field in artifact.api.metadata["class_table"][0]["fields"]
    ) == (
        {"name": "spare", "slot": 1},
        {"name": "value", "slot": 0},
    )

    observed = _run_module(
        tmp_path,
        artifact.source,
        """
const Counter = program.SSA_CLASSES["model.Counter"];
const counter = new Counter(4);
const returned = counter.bump(3);
console.log(JSON.stringify({
  identity: Counter.ssaIdentity,
  layout: Counter.fieldLayout,
  returned,
  resident: counter.value,
  spareExists: Object.hasOwn(counter, "spare")
}));
""",
    )
    assert observed == {
        "identity": "model.Counter",
        "layout": [
            {"name": "spare", "slot": 1},
            {"name": "value", "slot": 0},
        ],
        "returned": 7,
        "resident": 7,
        "spareExists": True,
    }


def test_javascript_numeric_table_covers_python_oracle_and_portable_catalogue():
    from src.common.tensors.fused_ir import ELEMENTWISE_BINARY, ELEMENTWISE_UNARY

    assert python_oracle_operations() <= supported_javascript_numeric_operations()
    assert supported_javascript_numeric_operations() == (
        ELEMENTWISE_UNARY | ELEMENTWISE_BINARY
    )


def test_all_portable_numeric_operators_emit_and_execute_in_node(tmp_path):
    left = SSAValue(0, "int32")
    right = SSAValue(1, "int32")
    instructions = []
    results = []
    names = sorted(supported_javascript_numeric_operations())
    for index, name in enumerate(names, start=2):
        specification = JAVASCRIPT_NUMERIC_OPERATORS[name]
        result = SSAValue(index, "float64")
        row = TENSOR_SSA_OPERATOR_BY_NAME[name]
        arguments = [left] if specification.arity == 1 else [left, right]
        attributes = {} if row.is_direct else {"tensor_operation": name}
        instructions.append(Instr(
            row.handler.value,
            arguments,
            result,
            attributes=attributes,
        ))
        results.append(result)
    instructions.append(Instr("Ret", results, None))
    function = Function(
        "numeric_catalogue",
        [left, right],
        {"entry": BasicBlock("entry", instructions)},
    )

    artifact = emit_ssa_module_to_javascript(
        IRModule({"numeric_catalogue": function}),
        function_name="numeric_catalogue",
    )

    assert artifact.complete, tuple(item.format() for item in artifact.shortfalls)
    observed = _run_module(
        tmp_path,
        artifact.source,
        "console.log(JSON.stringify(program.default.run([6, 2])));",
    )
    by_name = dict(zip(names, observed))
    assert len(observed) == len(names)
    assert by_name["bitand"] == 2
    assert by_name["bitor"] == 6
    assert by_name["bitxor"] == 4
    assert by_name["invert"] == -7
    assert by_name["logical_and"] is True
    assert by_name["logical_not"] is False
    assert by_name["maximum"] == 6
    assert by_name["minimum"] == 2


def test_missing_ssa_method_body_is_a_shortfall_and_a_throwing_method(tmp_path):
    table = SSAClassTable((SSAClassDefinition(
        "Cell",
        methods=(SSAClassMethod("advance", 3, "cell_advance"),),
    ),))

    artifact = emit_ssa_module_to_javascript(IRModule({}, class_table=table))

    assert not artifact.complete
    assert artifact.shortfalls[0].operation == "class-plan"
    assert "cell_advance" in artifact.shortfalls[0].reason
    observed = _run_module(
        tmp_path,
        artifact.source,
        """
const cell = new program.SSA_CLASSES.Cell();
let message = "";
try { cell.advance(); } catch (error) { message = error.message; }
console.log(JSON.stringify({message}));
""",
    )
    assert "Cell.advance" in observed["message"]
    assert "cell_advance" in observed["message"]


def test_actual_frontend_class_constructor_uses_preserved_field_slots(tmp_path):
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa

    module, _outputs, _exports = lower_ast_source_to_ssa(
        """
class Counter:
    def __init__(self, start):
        self.value = start

    def bump(self, amount):
        self.value = self.value + amount
        return self.value

    def twice(self, amount):
        return self.bump(amount) + self.bump(amount)
""",
    )
    artifact = emit_ssa_module_to_javascript(module)

    assert artifact.complete, tuple(item.format() for item in artifact.shortfalls)
    observed = _run_module(
        tmp_path,
        artifact.source,
        """
const Counter = program.SSA_CLASSES.Counter;
const left = new Counter(1);
const right = new Counter(10);
const twice = left.twice(2);
const bumped = right.bump(5);
console.log(JSON.stringify({
  twice,
  left: left.value,
  bumped,
  right: right.value,
  layout: Counter.fieldLayout
}));
""",
    )
    assert observed == {
        "twice": 8,
        "left": 5,
        "bumped": 15,
        "right": 15,
        "layout": [{"name": "value", "slot": 0}],
    }


def test_generic_cfg_loop_executes_through_the_block_dispatch_template(tmp_path):
    values = {index: SSAValue(index, "int32") for index in range(9)}
    array = SSAValue(20, "float64")
    count = values[0]
    zero_i, zero_sum, one = values[1], SSAValue(2, "float64"), values[3]
    index, total = values[4], SSAValue(5, "float64")
    condition, address = values[6], SSAValue(7, "ptr")
    loaded = SSAValue(8, "float64")
    next_total, next_index = SSAValue(9, "float64"), SSAValue(10, "int32")
    function = Function("sum_values", [array, count], {
        "entry": BasicBlock("entry", [
            Instr("Const", [], zero_i, attributes={"value": 0}),
            Instr("Const", [], zero_sum, attributes={"value": 0.0}),
            Instr("Const", [], one, attributes={"value": 1}),
            Instr("Br", [], None, attributes={"target": "header"}),
        ]),
        "header": BasicBlock("header", [
            Instr("Phi", [zero_i, next_index], index, attributes={"incoming_blocks": ("entry", "body")}),
            Instr("Phi", [zero_sum, next_total], total, attributes={"incoming_blocks": ("entry", "body")}),
            Instr("Lt", [index, count], condition),
            Instr("CondBr", [condition], None, attributes={"true_target": "body", "false_target": "exit"}),
        ]),
        "body": BasicBlock("body", [
            Instr("GetElementPtr", [array, index], address),
            Instr("Load", [address], loaded),
            Instr("Add", [total, loaded], next_total),
            Instr("Add", [index, one], next_index),
            Instr("Br", [], None, attributes={"target": "header"}),
        ]),
        "exit": BasicBlock("exit", [Instr("Ret", [total], None)]),
    })
    artifact = emit_ssa_module_to_javascript(
        IRModule({"sum_values": function}), "sum_values",
    )

    assert artifact.complete, artifact.shortfalls
    assert artifact.buffer_order == (20, 0)
    assert artifact.pointer_formals == (20,)
    assert "for (;;)" in artifact.source
    assert "switch (block)" in artifact.source
    observed = _run_module(
        tmp_path,
        artifact.source,
        "console.log(JSON.stringify(program.sum_values([new Float64Array([1.5, 2.5, 4]), 3])));\n",
    )
    assert observed == [8]


def test_multidimensional_gep_walks_nested_javascript_storage(tmp_path):
    matrix = SSAValue(0, "float64", accounting={"ssa_call_rank": 2})
    row = SSAValue(1, "int32")
    column = SSAValue(2, "int32")
    address = SSAValue(3, "ptr")
    value = SSAValue(4, "float64")
    function = Function("read_cell", [matrix, row, column], {
        "entry": BasicBlock("entry", [
            Instr("GetElementPtr", [matrix, row, column], address),
            Instr("Load", [address], value),
            Instr("Ret", [value], None),
        ]),
    })

    artifact = emit_ssa_module_to_javascript(
        IRModule({"read_cell": function}), "read_cell",
    )

    assert artifact.complete, artifact.shortfalls
    assert "turingPointer(t0, t1, t2)" in artifact.source
    observed = _run_module(
        tmp_path,
        artifact.source,
        "console.log(JSON.stringify(program.read_cell([[[1, 2], [3, 4]], 1, 0])));\n",
    )
    assert observed == [3]


def test_string_token_uses_exact_bigint_identity(tmp_path):
    expected = SSAValue(0, "int64")
    token = SSAValue(1, "int64")
    matches = SSAValue(2, "bool")
    identity = 3_430_348_804_014_172_829
    function = Function("token_matches", [expected], {
        "entry": BasicBlock("entry", [
            Instr(
                "string_token", [], token,
                attributes={"token": identity, "text": "hard_failure"},
            ),
            Instr("Eq", [expected, token], matches),
            Instr("Ret", [matches], None),
        ]),
    })

    artifact = emit_ssa_module_to_javascript(
        IRModule({"token_matches": function}), "token_matches",
    )

    assert artifact.complete, artifact.shortfalls
    assert f"{identity}n" in artifact.source
    observed = _run_module(
        tmp_path,
        artifact.source,
        f"console.log(JSON.stringify(program.token_matches([{identity}n])));\n",
    )
    assert observed == [True]


def test_unsupported_instruction_is_reported_instead_of_silently_omitted():
    output = SSAValue(1, "float64")
    function = Function("probe", [], {
        "entry": BasicBlock("entry", [
            Instr("Teleport", [], output),
            Instr("Ret", [output], None),
        ]),
    })

    artifact = emit_ssa_module_to_javascript(IRModule({"probe": function}), "probe")

    assert not artifact.complete
    assert artifact.shortfalls[0].format() == (
        "Teleport in probe:entry: no JavaScript SSA spelling"
    )


def test_operand_defined_only_after_its_use_is_refused():
    condition = SSAValue(0, "bool")
    late = SSAValue(1, "float64")
    result = SSAValue(2, "float64")
    function = Function("late_definition", [condition], {
        "entry": BasicBlock("entry", [
            Instr(
                "CondBr", [condition], None,
                attributes={"true_target": "use", "false_target": "define"},
            ),
        ]),
        "use": BasicBlock("use", [
            Instr("Add", [late, late], result),
            Instr("Ret", [result], None),
        ]),
        "define": BasicBlock("define", [
            Instr("Const", [], late, attributes={"value": 1.0}),
            Instr("Ret", [late], None),
        ]),
    })

    artifact = emit_ssa_module_to_javascript(
        IRModule({"late_definition": function}), "late_definition",
    )

    assert not artifact.complete
    assert any(
        shortfall.function == "late_definition"
        and shortfall.block == "use"
        and "operand value 1 has no definition dominating" in shortfall.reason
        for shortfall in artifact.shortfalls
    )


def test_unlowered_authored_value_refuses_a_printable_javascript_body():
    output = SSAValue(1, "float64")
    function = Function(
        "false_complete",
        [],
        {"entry": BasicBlock("entry", [
            Instr("Const", [], output, attributes={"value": 0.0}),
            Instr("Ret", [output], None),
        ])},
        metadata={
            "structural_output_shortfalls": (
                (7, "loopresult", "carried-value"),
            ),
        },
    )

    artifact = emit_ssa_module_to_javascript(
        IRModule({"false_complete": function}), "false_complete",
    )

    assert not artifact.complete
    assert artifact.shortfalls[0].format() == (
        "source-value in false_complete: unlowered authored value 7 "
        "(loopresult, carried-value)"
    )


def test_external_compiled_function_is_joined_through_generated_runtime(tmp_path):
    value = SSAValue(0, "float64")
    deployed = SSAValue(1, "ssa.aggregate")
    index = SSAValue(2, "int32")
    address = SSAValue(3, "ptr")
    result = SSAValue(4, "float64")
    device = Function("device_step", [value], {
        "entry": BasicBlock("entry", [Instr("Ret", [value], None)]),
    })
    host = Function("managed_frame", [value], {
        "entry": BasicBlock("entry", [
            Instr("Deploy", [], None, attributes={"deployment_frame": True, "region_id": 7}),
            Instr("Call", [value], deployed, attributes={
                "callee": "device_step", "region_index": 9,
                "feed_ids": (0,), "output_ids": (4,),
                "result_convention": "ssa.aggregate",
            }),
            Instr("Join", [], None, attributes={"deployment_frame": True, "region_id": 7}),
            Instr("Const", [], index, attributes={"value": 0}),
            Instr("GetElementPtr", [deployed, index], address),
            Instr("Load", [address], result),
            Instr("Ret", [result], None),
        ]),
    })
    artifact = emit_ssa_module_to_javascript(
        IRModule({"managed_frame": host, "device_step": device}),
        "managed_frame",
        external_functions=("device_step",),
    )

    assert artifact.complete, artifact.shortfalls
    assert "function impl_device_step" not in artifact.source
    assert "await turingRuntime.call" in artifact.source
    observed = _run_module(
        tmp_path,
        artifact.source,
        """
const calls = [];
const runtime = {call: async (descriptor, args) => {
  calls.push({descriptor, args});
  return [args[0] * 3];
}};
const result = await program.managed_frame([4], [], runtime);
console.log(JSON.stringify({result, calls}));
""",
    )
    assert observed["result"] == [12]
    assert observed["calls"][0]["descriptor"]["callee"] == "device_step"
    assert observed["calls"][0]["descriptor"]["regionIndex"] == 9
