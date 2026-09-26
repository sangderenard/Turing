"""Trace how ``break`` nested inside ``if`` inside ``for`` travels the pipeline.

Modelled on tools/audit_result_binding_nested_return.py and
tools/audit_structural_boolop_shapes.py: the REAL program ABI receipt from
``balloon_tire_managed_extraction_contract``, lowered with
``lower_ast_source_to_ssa``.  Never the vehicle build, never
``lower_balloon_tire_managed_python_ssa``, never a whole test file.

The authored function reproduces exactly the shape of
src/common/dt_system/dt_controller.py run_superstep lines 620-626::

    for boundary in boundary_values:
        if boundary > total_value + eps:
            dt_try = ...
            break

(for over a tuple parameter, compound predicate, assignment to a carried
scalar, then break).  A second variant nests the break two ifs deep so the
``guard`` composition in ``collect_loop_controls`` is observable.

Every stage is observed through ``sys.setprofile`` (closure frames read by
code name, exactly like audit_structural_boolop_shapes._ArrivalTrace) or a
``resolved_process_graph_sink``; nothing under src/ is edited.

Usage::

    python tools/audit_break_in_if_trace.py            # single if
    python tools/audit_break_in_if_trace.py nested     # break two ifs deep
    python tools/audit_break_in_if_trace.py all
"""

from __future__ import annotations

import ast
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.compiler.control_source import LoopBlock, LoopControlBlock  # noqa: E402
from src.compiler.extraction_contract import ExtractionContract  # noqa: E402
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa  # noqa: E402
from src.compiler.vehicle_python_compilation import (  # noqa: E402
    balloon_tire_managed_extraction_contract, BalloonTireManagedState,
)

CONTRACTS = Path(__file__).resolve().parents[1] / "extraction_contracts"

# Exact shape of dt_controller.run_superstep lines 618-626: ``dt_try`` is
# assigned BEFORE the loop (line 618), so it is in the reducer's
# ``before_loop`` environment and becomes a loop-carried binding.
SINGLE = """
def root(boundary_values, total_value, eps, dt_cap):
    dt_try = dt_cap - eps
    for boundary in boundary_values:
        if boundary > total_value + eps:
            dt_try = boundary - total_value
            break
    return dt_try
"""

# Same, with the break two ifs deep (guard composition question).
NESTED = """
def root(boundary_values, total_value, eps, dt_cap):
    dt_try = dt_cap - eps
    for boundary in boundary_values:
        if boundary > total_value + eps:
            if boundary < dt_try:
                dt_try = boundary - total_value
                break
    return dt_try
"""

# Control: ``dt_try`` as a PARAMETER never bound before the loop.  The
# reducer then records no carried binding at all (before_loop lacks it).
PARAM = """
def root(boundary_values, total_value, eps, dt_try):
    for boundary in boundary_values:
        if boundary > total_value + eps:
            dt_try = boundary - total_value
            break
    return dt_try
"""

# Control: the carried update is OUTSIDE the terminal arm, so the reducer
# records a carried binding and the break edge / exit Phi machinery fires.
CARRIED = """
def root(boundary_values, total_value, eps, dt_cap):
    dt_try = dt_cap - eps
    for boundary in boundary_values:
        dt_try = dt_try - boundary
        if boundary > total_value + eps:
            break
    return dt_try
"""

# run_superstep's while loop (dt_controller.py 612-660): a constant-arm
# break, an inner for-loop arm break, and a while-level break after the
# inner loop whose site sees ``dt_try`` computed BEFORE the if (not in the
# arm) -- that site must stay lexical, not arm-owned.
WHILE_BREAK = """
def root(boundary_values, total_value, eps, dt_cap, max_iters):
    total = 0.0
    iters = 0
    cap_hit = False
    dt_try = dt_cap
    while total < total_value:
        if iters >= max_iters:
            cap_hit = True
            break
        iters += 1
        dt_try = min(dt_cap, total_value - total)
        for boundary in boundary_values:
            if boundary > total + eps:
                dt_try = min(dt_try, boundary - total)
                break
        total = total + dt_try
        if dt_try <= eps:
            break
    return dt_try, cap_hit
"""

CASES = {
    "single": SINGLE, "nested": NESTED, "param": PARAM, "carried": CARRIED,
    "while_break": WHILE_BREAK,
}


def _base_records():
    import numpy as np

    stub = BalloonTireManagedState.__new__(BalloonTireManagedState)
    for name in (
        "inputs", "state", "output", "wheel_input_indices", "rest",
        "face_vertices", "face_rest", "face_scatter", "bending_incidence",
        "bending_scatter", "bending_weight", "vertex_area", "bead_mask",
        "face_material", "telemetry",
    ):
        setattr(stub, name, np.zeros((1,), dtype=np.float64))
    return balloon_tire_managed_extraction_contract(stub).program_abi.receipt()


def _contract():
    base = _base_records()
    scalar = {
        "storage": "scalar", "dtype": "float64", "rank": 0,
        "python_type": "builtins.float",
    }
    values = [
        {
            "function": "root", "parameter": "boundary_values",
            "storage": "span", "dtype": "float64", "rank": 1, "shape": [4],
            "python_type": "builtins.tuple",
        },
        {"function": "root", "parameter": "total_value", **scalar},
        {"function": "root", "parameter": "eps", **scalar},
        {"function": "root", "parameter": "dt_cap", **scalar},
        {"function": "root", "parameter": "dt_try", **scalar},
    ]
    return ExtractionContract(
        CONTRACTS / "program_extraction.yaml"
    ).with_program_abi({
        "records": dict(base["records"]),
        "bindings": [],
        "values": values,
    })


def _vid(value):
    return None if value is None else int(getattr(value, "id", value))


class _Trace:
    """Observe the closures/methods by code name; never touch src/."""

    def __init__(self):
        self.rows: list[str] = []
        self.loop_block: LoopBlock | None = None

    def say(self, text: str) -> None:
        self.rows.append(text)

    def __call__(self, frame, event, arg):
        name = frame.f_code.co_name
        f = frame.f_locals
        if name == "reduce_statement" and event == "return":
            statement = f.get("body_statement")
            if isinstance(statement, (ast.For, ast.If)):
                keys = (
                    "before_loop", "environment", "overwritten_before_read",
                    "live_after_loop", "loop_target_bindings",
                    "loop_carried_bindings", "body_terminal", "else_terminal",
                    "body_environment", "else_environment", "before",
                )
                self.say(
                    "[topological_reducer.reduce_statement return] "
                    f"{type(statement).__name__} node={f.get('node_id') or f.get('loop_id')} "
                    f"-> {arg!r}"
                )
                for key in keys:
                    if key in f:
                        value = f[key]
                        if isinstance(value, dict):
                            value = {
                                k: v for k, v in value.items()
                                if k in {"dt_try", "boundary", "dt_cap"}
                            }
                        self.say(f"    {key} = {value!r}")
            return
        if name == "collect_loop_controls":
            if event == "call":
                statements = tuple(f.get("statements") or ())
                self.say(
                    "[loop_composer.collect_loop_controls call] guard="
                    f"{f.get('guard')!r} statements="
                    f"{[type(s).__name__ for s in statements]}"
                )
            elif event == "return":
                self.say(
                    "[loop_composer.collect_loop_controls return] guard="
                    f"{f.get('guard')!r} loop_controls(so far)="
                    f"{f.get('loop_controls')!r}"
                )
            return
        if name == "analyze_shader_loop_reductions" and event == "return":
            loop = f.get("loop")
            self.say(
                "[loop_composer.analyze_shader_loop_reductions return] "
                f"returns {type(arg).__name__}; loop.node_id="
                f"{getattr(loop, 'node_id', None)} "
                f"break_nodes={getattr(loop, 'break_nodes', None)} "
                f"carried_bindings={getattr(loop, 'carried_bindings', None)}"
            )
            lexical = f.get("lexical_position")
            if lexical is not None:
                self.say(f"    lexical_position (node_id -> position) = "
                         f"{dict(sorted(lexical.items(), key=lambda kv: kv[1]))}")
            regions = f.get("regions")
            if regions is not None:
                self.say(f"    regions = {[sorted(map(int, r)) for r in regions]}")
            self.say(f"    body_region_indices = {f.get('body_region_indices')!r}"
                     f" body_region_positions = {f.get('body_region_positions')!r}")
            body_items = f.get("body_items")
            if body_items is not None:
                for position, block in sorted(body_items, key=lambda it: it[0]):
                    self.say(f"    body_item position={position}: "
                             f"{type(block).__name__} {block!r}"[:400])
            self.say(f"    carried_aliases = {f.get('carried_aliases')!r}")
            found = [
                item for item in (arg if isinstance(arg, tuple) else (arg,))
                if isinstance(item, LoopBlock)
            ]
            if found:
                self.loop_block = found[0]
            else:
                # Search one level of nesting in the returned tuple.
                for item in (arg if isinstance(arg, tuple) else ()):
                    for member in (
                        item if isinstance(item, (tuple, list)) else ()
                    ):
                        if isinstance(member, LoopBlock):
                            self.loop_block = member
            return
        if name == "lower" and event == "call":
            block = f.get("block")
            if isinstance(block, LoopControlBlock):
                lowerer = f.get("self")
                context = (
                    lowerer.loop_exit_contexts[-1]
                    if lowerer.loop_exit_contexts else None
                )
                self.say(
                    "[precompile_to_ssa.lower(LoopControlBlock) entry] "
                    f"block={block!r} current={lowerer.current.name} "
                    f"loop_targets(latch,exit)="
                    f"{[(l.name, e.name) for l, e in lowerer.loop_targets]}"
                )
                if context is not None:
                    for updated_id, initial_id, current in context["carried"]:
                        self.say(
                            "    carried: updated_id="
                            f"{updated_id} -> external_values="
                            f"%t{_vid(lowerer.external_values.get(int(updated_id)))}"
                            f" | initial_id={initial_id} -> external_values="
                            f"%t{_vid(lowerer.external_values.get(int(initial_id)))}"
                            f" | header phi current=%t{_vid(current)}"
                        )
            return
        if name == "_value_dominates_current_edge":
            caller = frame.f_back
            if caller is None or caller.f_code.co_name != "lower":
                return
            cl = caller.f_locals
            if not isinstance(cl.get("block"), LoopControlBlock):
                return
            if event == "call":
                lowerer = f.get("self")
                self.say(
                    "[precompile_to_ssa._value_dominates_current_edge call] "
                    f"candidate=%t{_vid(f.get('value'))} "
                    f"(updated_id={cl.get('updated_id')} "
                    f"initial_id={cl.get('initial_id')} "
                    f"header current=%t{_vid(cl.get('current'))}) "
                    f"self.current={lowerer.current.name}"
                )
            elif event == "return":
                self.say(
                    "[precompile_to_ssa._value_dominates_current_edge return] "
                    f"-> {arg!r}"
                )
            return
        if name == "_publish_loop_result_ports" and event == "call":
            loop = f.get("loop")
            self.say(
                "[precompile_to_ssa._publish_loop_result_ports call] "
                f"header={f['header'].name} exit={f['exit_block'].name} "
                f"result_ports={tuple(getattr(loop, 'result_ports', ()))}"
            )
            for updated_id, initial_id, initial, updated, current in f["carried"]:
                self.say(
                    f"    carried: updated_id={updated_id} initial_id={initial_id}"
                    f" initial=%t{_vid(initial)} updated=%t{_vid(updated)}"
                    f" current(header phi)=%t{_vid(current)}"
                )
            for predecessor, values, *bound in f["break_edges"]:
                self.say(
                    f"    break_edge: from block {predecessor.name!r} "
                    f"carried_values={tuple('%t' + str(_vid(v)) for v in values)} "
                    f"break_bound_values={tuple('%t' + str(_vid(v)) for group in bound for v in group)}"
                )
            return


def _dump_graph(graph) -> None:
    graphs = [("<root>", getattr(graph, "G", graph))]
    for entry in getattr(graph, "function_table", ()) or ():
        function_graph = getattr(getattr(entry, "graph", None), "G", None)
        if function_graph is not None:
            graphs.append((
                str(function_graph.graph.get("function_name") or entry.name),
                function_graph,
            ))
    print("--- resolved process graph: loop / if / break / compare nodes ---")
    for owner, G in graphs:
        identities = G.graph.get("identity_table") or {}
        print(
            f"  {owner:<10} function_outputs="
            f"{G.graph.get('function_outputs')!r} "
            f"identity_table[dt_try]={identities.get('dt_try')!r} "
            f"roots={list(getattr(G, 'roots', ()) or G.graph.get('roots', ()))!r}"
        )
        for node_id, data in G.nodes(data=True):
            expression = data.get("expr_obj")
            attributes = data.get("attributes") or {}
            interesting = isinstance(
                expression,
                (ast.For, ast.While, ast.If, ast.Break, ast.Compare, ast.BoolOp,
                 ast.Assign, ast.BinOp),
            ) or str(data.get("type")) in {"LoopResult", "LoopStatePort"}
            if owner == "<root>" and not isinstance(
                expression, (ast.For, ast.If, ast.Break)
            ):
                continue
            if not interesting:
                continue
            try:
                text = ast.unparse(expression) if expression is not None else (
                    f"label={data.get('label')!r}"
                )
            except Exception:  # pragma: no cover - defensive
                text = type(expression).__name__
            text = text.replace("\n", " | ")
            if len(text) > 100:
                text = text[:97] + "..."
            print(
                f"  {owner:<10} node={int(node_id):<16} "
                f"{type(expression).__name__ if expression is not None else '-':<9} "
                f"type={data.get('type')!s:<12} parents={data.get('parents')!r}"
            )
            print(f"      src: {text}")
            for key in (
                "loop_carried_bindings", "loop_result_ports",
                "loop_carried_updated_ids", "loop_target_bindings",
                "loop_target_initials", "binding_name", "loop_id",
                "initial_value_id", "updated_value_id", "result_kind",
                "iterator_kind", "source_conditional_id",
            ):
                if key in attributes:
                    print(f"      attr {key} = {attributes[key]!r}")


def _describe_block(block, indent: int = 0) -> None:
    pad = "  " * indent
    name = type(block).__name__
    if isinstance(block, LoopControlBlock):
        print(f"{pad}{name}({block!r})")
        return
    items = getattr(block, "items", None)
    if items is None:
        items = getattr(block, "blocks", None)
    if name == "StatementBlock":
        print(f"{pad}{name}{tuple(getattr(block, 'lines', ()) or getattr(block, 'statements', ()))}")
        return
    if isinstance(items, (tuple, list)):
        print(f"{pad}{name}[{len(items)}]")
        for item in items:
            _describe_block(item, indent + 1)
        return
    print(f"{pad}{name}: {block!r}"[:300])


def _dump_loop_block(loop: LoopBlock) -> None:
    print("--- LoopBlock returned by analyze_shader_loop_reductions ---")
    print(f"  induction={loop.induction!r} start={loop.start!r} "
          f"stop={loop.stop!r} step={loop.step!r}")
    print(f"  carried_aliases (updated_id, initial_id) = {loop.carried_aliases}")
    print(f"  result_ports (port_id, initial_id, updated_id) = {loop.result_ports}")
    print(f"  terminal_controls = {loop.terminal_controls}")
    print(f"  source_loop_node_id = {loop.source_loop_node_id}")
    print("  body:")
    _describe_block(loop.body, 2)


def _dump_ssa(module) -> None:
    functions = getattr(module, "functions", None) or {}
    items = (
        functions.items() if isinstance(functions, dict)
        else ((getattr(f, "name", "?"), f) for f in functions)
    )
    for name, function in items:
        print(f"--- SSA function {name} ---")
        print("  args: " + ", ".join(
            f"%t{_vid(a)}:{a.dtype}" for a in function.args
        ))
        metadata = function.metadata or {}
        for key in ("parameter_names", "named_outputs"):
            if key in metadata:
                print(f"  metadata[{key}] = {metadata[key]!r}")
        for block_name, block in function.blocks.items():
            print(f"  {block_name}:   (successors={block.successors})")
            for instruction in block.instrs:
                args = ", ".join(f"%t{_vid(a)}" for a in instruction.args)
                res = (
                    f"%t{_vid(instruction.res)} = "
                    if instruction.res is not None else ""
                )
                attributes = {
                    key: value
                    for key, value in (instruction.attributes or {}).items()
                    if key in {
                        "incoming_blocks", "binding", "initial_value_id",
                        "updated_value_id", "source_control", "value",
                        "comparison", "source_name", "target",
                        "true_target", "false_target", "cond", "label",
                    }
                }
                print(f"    {res}{instruction.op}({args})"
                      f"{'  ' + repr(attributes) if attributes else ''}")


def run(case: str) -> int:
    source = CASES[case]
    print(f"=== case {case} ===")
    print(source.strip())
    captured = {}

    def sink(graph):
        captured["graph"] = graph

    trace = _Trace()
    t0 = time.time()
    sys.setprofile(trace)
    try:
        module, outputs, exports = lower_ast_source_to_ssa(
            source, "root", name=f"audit_break_{case}",
            extraction_contract=_contract(),
            resolved_process_graph_sink=sink,
        )
    except Exception as error:  # noqa: BLE001
        sys.setprofile(None)
        print(f"FAILED after {time.time()-t0:.2f}s: {type(error).__name__}: "
              f"{str(error)[:1200]}", flush=True)
        import traceback
        traceback.print_exc()
        print("--- trace rows up to failure ---")
        for row in trace.rows:
            print("  " + row)
        if "graph" in captured:
            _dump_graph(captured["graph"])
        if trace.loop_block is not None:
            _dump_loop_block(trace.loop_block)
        return 1
    finally:
        sys.setprofile(None)
    print(f"LOWERED in {time.time()-t0:.2f}s", flush=True)
    if "graph" in captured:
        _dump_graph(captured["graph"])
    print("--- trace rows ---")
    for row in trace.rows:
        print("  " + row)
    if trace.loop_block is not None:
        _dump_loop_block(trace.loop_block)
    _dump_ssa(module)
    return 0


def main(argv) -> int:
    selected = argv[1:] or ["single"]
    if selected == ["all"]:
        selected = list(CASES)
    status = 0
    for case in selected:
        status |= run(case)
    return status


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
