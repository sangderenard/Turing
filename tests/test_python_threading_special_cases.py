import ast

import pytest

from src.transmogrifier.graph.python_special_cases import (
    interpret_python_special_case,
    lower_python_threading,
)
from pathlib import Path
from src.compiler.thread_operations import PROCESS_OPERATORS


#: The repository's program extraction contract; the compiler refuses
#: to lower without one.
CONTRACT = (
    Path(__file__).resolve().parents[1]
    / "extraction_contracts"
    / "program_extraction.yaml"
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
    operations = [i for block in function.blocks.values() for i in block.instrs
                  if i.attributes.get("synchronization_receipt") or i.op in PROCESS_OPERATORS.values()]
    # Dictionary block order is not execution order once notify has a CFG
    # loop. Inspect each exact source procedure and its real continuation.
    operations = [i for kind in ("condition_create", "condition_acquire", "condition_notify_all", "condition_release")
                  for i in operations if i.attributes.get("synchronization_kind") == kind]
    from src.compiler.concordance_declarations import SSA_SYNCHRONIZATION_PROCEDURE
    book = module.metadata["identity_book"]
    (procedure,) = [book.page(SSA_SYNCHRONIZATION_PROCEDURE).latest(row)
                    for row in book.page(SSA_SYNCHRONIZATION_PROCEDURE).rows()
                    if book.page(SSA_SYNCHRONIZATION_PROCEDURE).latest(row)[0] == "condition_notify_all"]
    release_block = next(b for b in function.blocks.values() if any(i.op == "MutexRelease" for i in b.instrs))
    assert release_block.name == procedure[2].row[2]
    assert function.args == []
    handle = operations[0].res
    assert operations[2].args[0] is handle
    assert operations[3].args[0] is operations[2].res
    assert operations[-1].args[0] is operations[-3].res
    assert all(i.attributes["required_capability"] == "communicating_tasks" for i in operations)
    assert all(i.attributes["ordered_effect"] for i in operations)
    assert [i.op for i in operations] == [
        "MutexCreate", "WorkerToken", "WaitQueueMutex", "MutexAcquire", "MutexAcquired",
        "WorkerToken", "WaitQueueMutex", "MutexToken", "WaiterSelect", "LatchSignal",
        "WorkerToken", "WaitQueueMutex", "MutexToken", "MutexRelease",
    ]
    assert not any(set(PROCESS_OPERATORS.values()) & set(region.operations)
                   for region in discover_contiguous_ssa_regions(
                       function, {"Const", *PROCESS_OPERATORS.values()},
                   ))


def test_condition_scope_keeps_numeric_regions_inside_and_after_cleanup():
    from pathlib import Path
    from src.compiler.extraction_contract import ExtractionContract
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
    module, _, _ = lower_ast_source_to_ssa(
            "import threading\n"
            "def root(x: int):\n"
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
            kind = instruction.attributes.get("synchronization_kind")
            phase = instruction.attributes.get("synchronization_phase")
            if (kind, phase) in {("condition_create", "control"), ("condition_acquire", "acquire"),
                                 ("condition_notify_all", "lookup"), ("condition_release", "release")}:
                observed.append(kind)
            elif instruction.op == "Call":
                callee = module.functions[instruction.attributes["callee"]]
                observed.extend(i.op for b in callee.blocks.values() for i in b.instrs
                                if i.op in {"Add", "Mul"})
    assert observed == ["condition_create", "condition_acquire", "Add",
                        "condition_notify_all", "condition_release", "Mul"]


def test_condition_constructor_runs_in_standalone_c_shell(tmp_path):
    import subprocess
    import numpy as np
    from src.transmogrifier.dtype_layout import c_lane_numpy_dtype
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
    from src.compiler.identity_concordance import concordance_report
    from src.compiler.ssa_c_backend import emit_ssa_module_to_c

    module, _, exports = lower_ast_source_to_ssa(
        "import threading\ndef root() -> int:\n    condition = threading.Condition()\n    return 1\n",
        "root", extraction_contract=CONTRACT, progress=lambda _: None, progress_bars=False,
    )
    assert "0 finding(s)" in concordance_report(module)
    artifact = emit_ssa_module_to_c(module, exports[0])
    assert artifact.complete, artifact.shortfalls
    executable = artifact.compile_standalone(tmp_path / "condition", {})
    process = subprocess.run([str(executable.executable_path), "1", "0"], cwd=executable.directory,
                             capture_output=True, text=True, timeout=10)
    assert process.returncode == 0, process.stderr
    assert artifact.buffer_roles == ("output",)
    assert np.frombuffer(executable.final_outputs_path.read_bytes(),
                         dtype=c_lane_numpy_dtype(artifact.buffer_dtypes[0])).tolist() == [1]


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
    wait = next(i for i in instructions if i.op == "LatchWaitTimeout")
    timeout = wait.args[wait.arg_roles.index("timeout")]
    literal = next(i for i in instructions if i.res is not None and i.res.id == timeout.id)
    assert literal.op == "Const" and literal.attributes["value"] == 0.1


def test_resource_wait_retains_loop_scope_and_unique_ssa_definitions():
    from pathlib import Path
    from src.compiler.extraction_contract import ExtractionContract
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa

    module, _, _ = lower_ast_source_to_ssa(
        "import threading\ndef root(n: int):\n"
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
    operations = {i.attributes["synchronization_kind"]: block.name
                  for block in function.blocks.values() for i in block.instrs
                  if (i.attributes.get("synchronization_kind"), i.attributes.get("synchronization_phase")) in {
                      ("condition_acquire", "acquire"), ("condition_wait", "context"), ("condition_release", "release")}}
    assert operations["condition_acquire"] == "entry"
    assert operations["condition_wait"] == "while_body"
    assert operations["condition_release"] == "while_exit"
    definitions = [i.res.id for b in function.blocks.values() for i in b.instrs if i.res is not None]
    assert len(definitions) == len(set(definitions))
    carried = next(i for i in function.blocks["while_header"].instrs
                   if i.attributes.get("binding") == "loop_carried")
    assert carried.args[1].id != carried.res.id
    from src.compiler.concordance_declarations import SSA_SYNCHRONIZATION_PROCEDURE
    book = module.metadata["identity_book"]
    (procedure,) = [book.page(SSA_SYNCHRONIZATION_PROCEDURE).latest(row)
                    for row in book.page(SSA_SYNCHRONIZATION_PROCEDURE).rows()
                    if book.page(SSA_SYNCHRONIZATION_PROCEDURE).latest(row)[0] == "condition_wait"]
    call = next(i for i in function.blocks[procedure[2].row[2]].instrs if i.op == "Call")
    assert [value.id for value in call.args] == [carried.res.id]
    callee = module.functions[call.attributes["callee"]]
    add = next(i for b in callee.blocks.values() for i in b.instrs if i.op == "Add")
    assert add.args[0] == callee.args[0]


def test_literal_seeded_counter_executes_native_iterations(tmp_path):
    import subprocess
    import numpy as np
    from src.transmogrifier.dtype_layout import c_lane_numpy_dtype
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
    from src.compiler.ssa_c_backend import emit_ssa_module_to_c

    # The actual host feed is float64; state that entry contract explicitly.
    source = "def counter(n: float):\n    i = 0\n    while i < n:\n        i = i + 1\n    return i\n"
    module, outputs, exports = lower_ast_source_to_ssa(source, "counter", extraction_contract=CONTRACT)
    entry = exports[0]
    artifact = emit_ssa_module_to_c(module, entry)
    assert artifact.complete, artifact.shortfalls
    argument = module.functions[entry].args[0].id
    namespace = {}
    exec(source, namespace)
    for n in (-3, 0, 1, 7, 19):
        executable = artifact.compile_standalone(tmp_path / f"counter_{n}",
            {argument: np.array([n], dtype=np.float64)})
        process = subprocess.run([str(executable.executable_path), "1", "0"], cwd=executable.directory,
                                 capture_output=True, text=True, timeout=10)
        assert process.returncode == 0, process.stderr
        raw, offset, observed = executable.final_outputs_path.read_bytes(), 0, []
        for dtype, shape, role in zip(artifact.buffer_dtypes, artifact.buffer_shapes, artifact.buffer_roles, strict=True):
            count = int(np.prod(shape)) if shape else 1
            values = np.frombuffer(raw, dtype=c_lane_numpy_dtype(dtype), count=count, offset=offset)
            offset += values.nbytes
            if role == "output":
                observed.extend(values.tolist())
        assert observed == [namespace['counter'](n)]


def _source_cleanup_module(source, sealed=False):
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
    from src.compiler.identity_concordance import concordance_report
    from src.compiler.concordance_declarations import SOURCE_RESOURCE_CONVERSION, SOURCE_SPAN
    from collections import Counter
    import hashlib
    authored_scopes = Counter(hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()
                              for node in ast.walk(ast.parse(source)) if isinstance(node, ast.With))
    if sealed:
        from src.compiler.fortran_c_shell import lower_resolved_process_graph_unit_to_ssa
        captured = []
        lower_ast_source_to_ssa(
            source, "root", extraction_contract=CONTRACT, progress=lambda _: None, progress_bars=False,
            resolved_process_graph_sink=captured.append, stop_after_compilation_unit_plan=True,
        )
        graph = captured[0]
        references = tuple(int(entry.reference.address) for entry in graph.function_table
                           if getattr(entry, 'graph', None) is not None
                           and entry.graph.G.graph.get('function_name') == 'root')
        assert len(references) == 1
        module, _, _ = lower_resolved_process_graph_unit_to_ssa(
            graph, references, progress=lambda _: None, progress_bars=False,
        )
        entry_name = 'root__root'
    else:
        module, _, exports = lower_ast_source_to_ssa(
            source, "root", extraction_contract=CONTRACT, progress=lambda _: None, progress_bars=False,
        )
        entry_name = exports[0]
    assert "0 finding(s)" in concordance_report(module)
    book = module.metadata['identity_book']
    converted_scopes = Counter(book.page(SOURCE_SPAN).cell(row[0].row, row[0].column).dump_sha256
                               for row in book.page(SOURCE_RESOURCE_CONVERSION).rows())
    assert converted_scopes == authored_scopes
    return module.functions[entry_name], book


def _condition_trace(block, book):
    from src.compiler.concordance_declarations import SOURCE_SYNCHRONIZATION
    result = []
    for instruction in block.instrs:
        contract = instruction.attributes.get("synchronization_contract")
        if contract is None:
            continue
        kind, roles, _projections, _constants, operands, _source = book.page(
            SOURCE_SYNCHRONIZATION).cell(contract.row, contract.column)
        phase = instruction.attributes.get("synchronization_phase")
        if ((kind, phase) in {("condition_acquire", "acquire"), ("condition_release", "release"),
                              ("condition_notify", "lookup"), ("condition_notify_all", "lookup")}):
            result.append((kind, operands[roles.index("mutex")].row[1]))
    return result


def _acyclic_cleanup_paths(function, book, entry="entry"):
    """Project actual CFG paths onto resource effects, condensing pure notify loops."""
    import networkx as nx
    graph = nx.DiGraph()
    graph.add_nodes_from(function.blocks)
    graph.add_edges_from((block.name, target) for block in function.blocks.values() for target in block.successors)
    topology = nx.condensation(graph)
    traces, returns = {}, set()
    for node, data in topology.nodes(data=True):
        members = data["members"]
        if len(members) > 1:
            assert all(not _condition_trace(function.blocks[name], book) for name in members)
            traces[node] = ()
        else:
            traces[node] = tuple(_condition_trace(function.blocks[next(iter(members))], book))
        if any(function.blocks[name].instrs and function.blocks[name].instrs[-1].op == "Ret" for name in members):
            returns.add(node)
    pending = [(topology.graph["mapping"][entry], ())]
    result = set()
    while pending:
        node, observed = pending.pop()
        observed = (*observed, *traces[node])
        if node in returns:
            result.add(observed)
        else:
            assert topology.out_degree(node)
            pending.extend((target, observed) for target in topology.successors(node))
    return list(result)


@pytest.mark.parametrize("sealed", [False, True])
def test_resource_scope_unwinds_nested_handles_on_conditional_return(sealed):
    function, book = _source_cleanup_module("""import threading
def root(returning: bool) -> int:
    outer = threading.Condition()
    inner = threading.Condition()
    with outer:
        with inner:
            if returning:
                return 99
            inner.notify_all()
    return 99
""", sealed)
    handles = [i.res.id for b in function.blocks.values() for i in b.instrs
               if i.op == "MutexCreate" and i.attributes.get("synchronization_kind") == "condition_create"]
    assert len(handles) == 2
    outer, inner = handles
    entry = (("condition_acquire", outer), ("condition_acquire", inner))
    cleanup = (("condition_release", inner), ("condition_release", outer))
    assert sorted(_acyclic_cleanup_paths(function, book)) == sorted([
        (*entry, *cleanup),
        (*entry, ("condition_notify_all", inner), *cleanup),
    ])


@pytest.mark.parametrize("sealed", [False, True])
def test_resource_scope_break_releases_only_scopes_exited_by_loop(sealed):
    function, book = _source_cleanup_module("""import threading
def root() -> int:
    outer = threading.Condition()
    inner = threading.Condition()
    with outer:
        for i in range(1):
            with inner:
                break
        outer.notify_all()
    return 1
""", sealed)
    handles = [i.res.id for b in function.blocks.values() for i in b.instrs
               if i.op == "MutexCreate" and i.attributes.get("synchronization_kind") == "condition_create"]
    assert len(handles) == 2
    outer, inner = handles
    exits = [block for block in function.blocks.values() if block.instrs
             and block.instrs[-1].attributes.get("source_control") == "break"]
    assert len(exits) == 1
    before_break = _condition_trace(exits[0], book)
    assert before_break[-1] == ("condition_release", inner)
    assert ("condition_release", outer) not in before_break
    continuation = exits[0].instrs[-1].attributes["target"]
    assert _acyclic_cleanup_paths(function, book, continuation) == [(
        ("condition_notify_all", outer), ("condition_release", outer),
    )]
