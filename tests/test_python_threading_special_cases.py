import ast

import pytest

from src.transmogrifier.graph.python_special_cases import (
    interpret_python_special_case,
    lower_python_threading,
)


def test_condition_scope_releases_captured_receiver_after_rebinding_and_exception():
    tree = ast.parse(
        "def run():\n"
        "    condition = Condition()\n"
        "    with condition:\n"
        "        condition = Condition()\n"
        "        raise ValueError('body failed')\n"
    )
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "Condition":
            node._extraction_contract = {"action": "use_native", "identity": "threading.Condition"}
    tree = lower_python_threading(tree)
    events = []
    def create():
        handle = object()
        events.append(("create", handle))
        return handle
    namespace = {
        "turing_dispatch_condition_create": create,
        "turing_dispatch_condition_acquire": lambda handle: events.append(("acquire", handle)),
        "turing_dispatch_condition_release": lambda handle: events.append(("release", handle)),
    }
    exec(compile(tree, "threading_scope", "exec"), namespace)
    with pytest.raises(ValueError, match="body failed"):
        namespace["run"]()
    assert [event[0] for event in events] == ["create", "acquire", "create", "release"]
    assert events[0][1] is events[1][1] is events[3][1]
    assert events[2][1] is not events[3][1]
    assert not any(isinstance(node, ast.With) for node in ast.walk(tree))
    calls = [node for node in ast.walk(tree) if getattr(node, "_turing_dispatch_operation", None)]
    assert len(calls) == 4
    for call in calls:
        assert interpret_python_special_case(call) is not None


def test_condition_dispatch_operations_survive_real_function_extraction():
    from pathlib import Path
    from src.compiler.extraction_contract import ExtractionContract
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa

    captured = []
    lower_ast_source_to_ssa(
        "import threading\n"
        "def root():\n"
        "    condition = threading.Condition()\n"
        "    with condition:\n"
        "        condition.notify_all()\n"
        "    return 1\n",
        "root",
        extraction_contract=ExtractionContract(
            Path(__file__).resolve().parents[1] / "extraction_contracts/program_extraction.yaml"
        ),
        stop_after_compilation_unit_plan=True,
        resolved_process_graph_sink=captured.append,
    )
    operations = []
    for entry in captured[0].function_table:
        graph = getattr(entry, "graph", None)
        if graph is None or graph.G.graph.get("function_name") != "root":
            continue
        for node_id, data in graph.G.nodes(data=True):
            attrs = data.get("attributes", {})
            operation = attrs.get("dispatch_operation", {}).get("operation")
            if operation:
                operations.append(operation)
                assert attrs["extraction_action"] == "intrinsic"
                assert attrs["extraction_identity"] == "turing.dispatch." + operation
                assert attrs["ordered_effect"]
                if operation != "condition_create":
                    assert any(
                        graph.G.nodes[parent].get("attributes", {}).get(
                            "dispatch_operation", {}
                        ).get("operation") == "condition_create"
                        for parent in graph.G.predecessors(node_id)
                    ), "synchronization must retain its constructed receiver"
    assert sorted(operations) == sorted([
        "condition_create", "condition_acquire", "condition_notify_all", "condition_release",
    ])


def test_explicit_condition_calls_materialize_ordered_dispatch_ssa():
    from pathlib import Path
    from src.compiler.extraction_contract import ExtractionContract
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
    from src.compiler.ssa_fusion_regions import discover_contiguous_ssa_regions

    module, _, _ = lower_ast_source_to_ssa(
        "import threading\n"
        "def root():\n"
        "    condition = threading.Condition()\n"
        "    condition.acquire()\n"
        "    condition.notify_all()\n"
        "    condition.release()\n"
        "    return 1\n",
        "root",
        extraction_contract=ExtractionContract(
            Path(__file__).resolve().parents[1] / "extraction_contracts/program_extraction.yaml"
        ),
    )
    function = module.functions["root__root"]
    operations = [i for block in function.blocks.values() for i in block.instrs if i.op == "Dispatch"]
    assert [i.attributes["dispatch_operation"] for i in operations] == [
        "condition_create", "condition_acquire", "condition_notify_all", "condition_release",
    ]
    assert function.args == []
    assert all(i.args == [operations[0].res] for i in operations[1:])
    assert all(i.attributes["required_capability"] == "communicating_tasks" for i in operations)
    assert all(i.attributes["ordered_effect"] for i in operations)
    assert not any("Dispatch" in region.operations for region in discover_contiguous_ssa_regions(
        function, {"Const", "Dispatch"},
    ))


def test_condition_scope_keeps_numeric_regions_inside_and_after_cleanup():
    from pathlib import Path
    from src.compiler.extraction_contract import ExtractionContract
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
    module, _, _ = lower_ast_source_to_ssa(
            "import threading\n"
            "def root(x):\n"
            "    condition = threading.Condition()\n"
            "    with condition:\n"
            "        y = x + 1\n"
            "        condition.notify_all()\n"
            "    return y * 2\n",
            "root",
            extraction_contract=ExtractionContract(
                Path(__file__).resolve().parents[1] / "extraction_contracts/program_extraction.yaml"
            ),
        )
    observed = []
    for block in module.functions["root__root"].blocks.values():
        for instruction in block.instrs:
            if instruction.op == "Dispatch":
                observed.append(instruction.attributes["dispatch_operation"])
            elif instruction.op == "Call":
                callee = module.functions[instruction.attributes["callee"]]
                observed.extend(i.op for b in callee.blocks.values() for i in b.instrs
                                if i.op in {"Add", "Mul"})
    assert observed == ["condition_create", "condition_acquire", "Add",
                        "condition_notify_all", "condition_release", "Mul"]


def test_dispatch_ssa_requires_backend_implementation():
    from src.compiler.control_source import ControlProgram, DispatchBlock
    from src.compiler.precompile_to_ssa import lower_control_program_to_ssa
    from src.compiler.ssa_c_backend import emit_ssa_module_to_c
    from src.transmogrifier.ssa import IRModule

    function, shortfalls = lower_control_program_to_ssa(
        ControlProgram(DispatchBlock(1, "condition_create", result_value_id=20)),
        output_value_ids=(20,),
    )
    assert not shortfalls
    artifact = emit_ssa_module_to_c(IRModule({function.name: function}), function.name)
    assert not artifact.complete
    assert any(item.operation == "Dispatch" for item in artifact.shortfalls)


def test_dispatch_wait_keeps_authored_keyword_literal():
    from pathlib import Path
    from src.compiler.extraction_contract import ExtractionContract
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa

    module, _, _ = lower_ast_source_to_ssa(
        "import threading\ndef root():\n"
        "    c = threading.Condition()\n"
        "    c.acquire()\n"
        "    c.wait(timeout=0.1)\n"
        "    c.release()\n"
        "    return 1\n", "root",
        extraction_contract=ExtractionContract(
            Path(__file__).resolve().parents[1] / "extraction_contracts/program_extraction.yaml"
        ),
    )
    function = module.functions["root__root"]
    assert not function.args
    instructions = [i for b in function.blocks.values() for i in b.instrs]
    wait = next(i for i in instructions if i.attributes.get("dispatch_operation") == "condition_wait")
    assert wait.attributes["keyword_names"] == ("timeout",)
    assert len(wait.args) == 2
    literal = next(i for i in instructions if i.res is not None and i.res.id == wait.args[1].id)
    assert literal.op == "Const" and literal.attributes["value"] == 0.1


def test_resource_wait_retains_loop_scope_and_unique_ssa_definitions():
    from pathlib import Path
    from src.compiler.extraction_contract import ExtractionContract
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa

    module, _, _ = lower_ast_source_to_ssa(
        "import threading\ndef root(n):\n"
        "    c = threading.Condition()\n"
        "    with c:\n"
        "        i = 0\n"
        "        while i < n:\n"
        "            c.wait(timeout=0.1)\n"
        "            i = i + 1\n"
        "    return i\n", "root",
        extraction_contract=ExtractionContract(
            Path(__file__).resolve().parents[1] / "extraction_contracts/program_extraction.yaml"
        ),
    )
    function = module.functions["root__root"]
    operations = {i.attributes["dispatch_operation"]: block.name
                  for block in function.blocks.values() for i in block.instrs if i.op == "Dispatch"}
    assert operations["condition_acquire"] == "entry"
    assert operations["condition_wait"] == "while_body"
    assert operations["condition_release"] == "while_exit"
    definitions = [i.res.id for b in function.blocks.values() for i in b.instrs if i.res is not None]
    assert len(definitions) == len(set(definitions))
    carried = next(i for i in function.blocks["while_header"].instrs
                   if i.attributes.get("binding") == "loop_carried")
    assert carried.args[1].id != carried.res.id
    call = next(i for i in function.blocks["while_body"].instrs if i.op == "Call")
    assert [value.id for value in call.args] == [carried.res.id]
    callee = module.functions[call.attributes["callee"]]
    add = next(i for b in callee.blocks.values() for i in b.instrs if i.op == "Add")
    assert add.args[0] == callee.args[0]


def test_literal_seeded_counter_executes_native_iterations(tmp_path):
    import json
    import pickle
    import subprocess
    import sys
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
    from src.compiler.ssa_c_backend import emit_ssa_module_to_c

    source = "def counter(n):\n    i = 0\n    while i < n:\n        i = i + 1\n    return i\n"
    module, outputs, exports = lower_ast_source_to_ssa(source, "counter")
    entry = exports[0]
    artifact = emit_ssa_module_to_c(module, entry)
    assert artifact.complete, artifact.shortfalls
    artifact.compile(tmp_path / "counter")
    argument = module.functions[entry].args[0].id
    result = outputs[entry][0].id
    saved = tmp_path / "artifact.pkl"
    saved.write_bytes(pickle.dumps((artifact, argument, result, source)))
    probe = subprocess.run([sys.executable, "-c", """
import json, pickle, sys
import numpy as np
with open(sys.argv[1], 'rb') as stream:
    artifact, argument, result, source = pickle.load(stream)
namespace = {}
exec(source, namespace)
observed = []
for n in (-3, 0, 1, 7, 19):
    native = artifact.prepare_execution({argument: np.array([n], dtype=np.float64)}).run()
    observed.append([native.buffers[result].item(), namespace['counter'](n)])
print(json.dumps(observed))
""", str(saved)], capture_output=True, text=True, timeout=20)
    assert probe.returncode == 0, probe.stderr
    assert all(native == eager for native, eager in json.loads(probe.stdout.splitlines()[-1]))


def test_resource_scope_unwinds_nested_handles_on_conditional_return():
    from src.compiler.control_source import (
        ControlProgram, DispatchBlock, LoopControlBlock, ResourceScopeBlock, SequenceBlock,
    )
    from src.compiler.precompile_to_ssa import lower_control_program_to_ssa

    inner = ResourceScopeBlock(
        SequenceBlock((
            LoopControlBlock("return", predicate_value_id=1, return_value_ids=(99,)),
            DispatchBlock(4, "condition_notify_all", (21,)),
        )),
        (DispatchBlock(5, "condition_release", (21,)),), 2,
    )
    outer = ResourceScopeBlock(inner, (DispatchBlock(6, "condition_release", (20,)),), 1)
    function, shortfalls = lower_control_program_to_ssa(
        ControlProgram(outer), output_value_ids=(99,),
    )
    assert not shortfalls
    # Follow the actual emitted CFG on both predicate outcomes. Ignore
    # unreachable compiler blocks; count the synchronization each path runs.
    for returning in (False, True):
        block_name = "entry"
        observed = []
        for _ in range(len(function.blocks) + 1):
            block = function.blocks[block_name]
            for instruction in block.instrs:
                if instruction.op == "Dispatch":
                    observed.append((instruction.attributes["dispatch_operation"], instruction.args[0].id))
            terminator = block.instrs[-1]
            if terminator.op == "Ret":
                break
            if terminator.op == "Br":
                block_name = terminator.attributes["target"]
            else:
                assert terminator.op == "CondBr"
                block_name = terminator.attributes["true_target" if returning else "false_target"]
        else:
            pytest.fail("cleanup CFG did not reach a return")
        assert observed == ([] if returning else [("condition_notify_all", 21)]) + [
            ("condition_release", 21), ("condition_release", 20),
        ]


def test_resource_scope_break_releases_only_scopes_exited_by_loop():
    from src.compiler.control_source import (
        ControlProgram, DispatchBlock, LoopBlock, LoopControlBlock, ResourceScopeBlock, SequenceBlock,
    )
    from src.compiler.precompile_to_ssa import lower_control_program_to_ssa

    loop = LoopBlock("i", "0", "1", "1", ResourceScopeBlock(
        LoopControlBlock("break"), (DispatchBlock(3, "condition_release", (21,)),), 2,
    ))
    program = ControlProgram(ResourceScopeBlock(
        SequenceBlock((loop, DispatchBlock(4, "condition_notify_all", (20,)))),
        (DispatchBlock(5, "condition_release", (20,)),), 1,
    ))
    function, shortfalls = lower_control_program_to_ssa(program)
    assert not shortfalls
    exits = [block for block in function.blocks.values() if block.instrs
             and block.instrs[-1].attributes.get("source_control") == "break"]
    assert len(exits) == 1
    operations = [i for i in exits[0].instrs if i.op == "Dispatch"]
    assert [(i.attributes["dispatch_operation"], i.args[0].id) for i in operations] == [
        ("condition_release", 21),
    ]
    # The outer scope's post-loop use still precedes its cleanup.
    post_loop = function.blocks[exits[0].instrs[-1].attributes["target"]]
    assert [(i.attributes["dispatch_operation"], i.args[0].id)
            for i in post_loop.instrs if i.op == "Dispatch"] == [
        ("condition_notify_all", 20), ("condition_release", 20),
    ]
