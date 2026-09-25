"""Carry literal arguments across call edges during ABI settlement."""

from pathlib import Path

ANCHOR = """    polymorphic_formals: set[tuple[int, int]] = set()
"""

LEDGERS = """    polymorphic_formals: set[tuple[int, int]] = set()
    # A formal given two different literals by two callsites is parametric,
    # exactly as a formal given two shapes is.
    literal_conflicts: set[tuple[int, str]] = set()
"""

LOOP_ANCHOR = """        # Publish the current facts before deriving shapes of expressions."""

HELPERS = '''        # Literal arguments settle across a call edge the same way shapes do.
        # A callee cannot fold ``arange(n)`` while ``n`` is its own formal,
        # and the callsite specialization that would have supplied the
        # literal runs BEFORE the folds that make it one -- ``n`` is
        # ``get_shape()[-1]`` until this loop's own fold resolves it.
        # Carrying it here lets the next round's re-fold consume it, which is
        # what a fixed point is for: one fact unblocks the next.
        round_constants: dict[int, dict[int, Any]] = {}
        round_parameters: dict[int, dict[int, str]] = {}

        def graph_constants(candidate: Any) -> dict[int, Any]:
            table = round_constants.get(id(candidate))
            if table is None:
                table = {}
                for node_id, node_data in candidate.nodes(data=True):
                    if str(node_data.get("type")) not in {
                        "Constant", "Const", "const",
                    }:
                        continue
                    literal = node_data.get("constant")
                    if literal is None:
                        literal = (
                            node_data.get("attributes") or {}
                        ).get("value")
                    table[int(node_data.get("value_id", node_id))] = literal
                round_constants[id(candidate)] = table
            return table

        def parameter_of(candidate: Any, value_id: int) -> str | None:
            table = round_parameters.get(id(candidate))
            if table is None:
                table = {}
                declared = set(map(
                    str, candidate.graph.get("function_parameters") or ()
                ))
                for name, history in (
                    candidate.graph.get("identity_table") or {}
                ).items():
                    if str(name) not in declared:
                        continue
                    for member in history or ():
                        table[int(member)] = str(name)
                round_parameters[id(candidate)] = table
            return table.get(int(value_id))

        # Publish the current facts before deriving shapes of expressions.'''

EDGE_ANCHOR = """            existing = destination.get(int(callee_id))
            marker = (id(callee_graph), int(callee_id))"""

EDGE_LITERAL = '''            literal = graph_constants(caller_graph).get(
                int(caller_id), _LITERAL_ABSENT
            )
            if literal is not _LITERAL_ABSENT and isinstance(
                literal, (int, float, bool, str, tuple)
            ):
                parameter = parameter_of(callee_graph, int(callee_id))
                if parameter is not None:
                    conflict_key = (id(callee_graph), parameter)
                    specializations = callee_graph.graph.setdefault(
                        "planner_specializations", {}
                    )
                    if conflict_key in literal_conflicts:
                        pass
                    elif parameter not in specializations:
                        specializations[parameter] = literal
                        changed = True
                        settlement_witnesses.append((
                            str(callee_graph.graph.get("function_name")),
                            int(callee_id), f"literal:{parameter}",
                        ))
                    else:
                        try:
                            agrees = bool(
                                specializations[parameter] == literal
                            )
                        except Exception:
                            agrees = False
                        if not agrees:
                            specializations.pop(parameter, None)
                            literal_conflicts.add(conflict_key)
                            changed = True
            existing = destination.get(int(callee_id))
            marker = (id(callee_graph), int(callee_id))'''

path = Path(__file__).resolve().parents[2] / "src/compiler/fortran_c_shell.py"
text = path.read_text(encoding="utf-8")

assert text.count(ANCHOR) == 1, text.count(ANCHOR)
text = text.replace(ANCHOR, LEDGERS)

assert text.count(LOOP_ANCHOR) == 1, text.count(LOOP_ANCHOR)
text = text.replace(LOOP_ANCHOR, HELPERS)

assert text.count(EDGE_ANCHOR) == 1, text.count(EDGE_ANCHOR)
text = text.replace(EDGE_ANCHOR, EDGE_LITERAL)

# A sentinel distinct from None, which is itself a legitimate literal.
sentinel_anchor = "def _class_surface_ssa_program("
if "_LITERAL_ABSENT = object()" not in text:
    assert text.count(sentinel_anchor) >= 1
    text = text.replace(
        sentinel_anchor,
        "_LITERAL_ABSENT = object()\n\n\n" + sentinel_anchor,
        1,
    )

path.write_text(text, encoding="utf-8")
print("settlement now carries literal arguments across call edges")
