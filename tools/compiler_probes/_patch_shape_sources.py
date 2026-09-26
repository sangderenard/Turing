"""Record what each store believes a value's shape is, one page per store.

A shape is kept in several places -- the graph node's own `tensor` dict, the
settled `linked_value_abi`, the descriptor query's answer, and the SSAValue
field the lowering actually reads -- and each is filled at a different moment
by a different pass.  None of them defers to another, so "the shape of V" has
no single answer and continuity is lost wherever two of them were written at
different times.

`IdentityBook.disagreements(row)` exists for exactly this: one identity's
claim read across every page.  Giving each store its own page turns "the flow
is broken somewhere" into a list of the values where the stores differ and
which store is stale.
"""

from pathlib import Path

SETTLEMENT_ANCHOR = """    for planned_graph in planned_graphs_by_shell.values():
        contracts = linked_value_abi_by_graph.get(id(planned_graph), {})"""

SETTLEMENT = '''    # One page per store, so a value whose stores disagree is a row rather
    # than a hunt.
    _node_page = _shape_book().page("shape.node")
    _linked_page = _shape_book().page("shape.linked")
    for _graph in planned_graphs_by_shell.values():
        _name = str(_graph.graph.get("function_name"))
        _contracts = linked_value_abi_by_graph.get(id(_graph), {})
        for _node_id, _data in _graph.nodes(data=True):
            _value_id = int(_data.get("value_id", _node_id))
            _row = (_name, _value_id)
            _tensor = _data.get("tensor") or {}
            _extents = tuple(_tensor.get("shape") or ())
            if _extents:
                _node_page.set(_row, 0, _extents)
            _contract = _contracts.get(_value_id) or {}
            _declared = tuple(_contract.get("shape") or ())
            if _declared:
                _linked_page.set(_row, 0, tuple(map(int, _declared)))

    for planned_graph in planned_graphs_by_shell.values():
        contracts = linked_value_abi_by_graph.get(id(planned_graph), {})'''

SSA_ANCHOR = """def _used_value_ids(module: IRModule) -> set[int]:"""

SSA_RECORD = '''def _record_ssa_shape(function_name: str, value: Any) -> None:
    """What the lowering actually reads, under the authored function's name.

    The SSA name carries the module prefix and the callsite specialization
    hash; the other stores are keyed by the authored name, so strip both or
    the rows never line up and every value looks like it has one source.
    """

    try:
        from .identity_concordance import current_identity_book

        extents = tuple(int(e) for e in (value.shape or ()))
        if not extents:
            return
        name = str(function_name)
        if "__specialized_" in name:
            name = name.split("__specialized_")[0]
        if "__planned_region" in name:
            name = name.split("__planned_region")[0]
        name = name.rsplit("__", 1)[-1] if "__" in name else name
        page = current_identity_book().page("shape.ssa")
        page.set((name, int(value.id)), 0, extents)
    except Exception:
        pass


''' + SSA_ANCHOR

path = Path(__file__).resolve().parents[2]
shell = path / "src/compiler/fortran_c_shell.py"
text = shell.read_text(encoding="utf-8")
if text.count(SETTLEMENT_ANCHOR) == 1:
    shell.write_text(text.replace(SETTLEMENT_ANCHOR, SETTLEMENT), encoding="utf-8")

lowering = path / "src/compiler/tensor_ssa_lowering.py"
text = lowering.read_text(encoding="utf-8")
assert text.count(SSA_ANCHOR) >= 1, text.count(SSA_ANCHOR)
text = text.replace(SSA_ANCHOR, SSA_RECORD, 1)
marker = """                elif operation == "matmul" and len(data_args) == 2:
                    left, right = data_args"""
assert text.count(marker) == 1
text = text.replace(marker, marker + """
                    _record_ssa_shape(function_name, left)
                    _record_ssa_shape(function_name, right)""")
lowering.write_text(text, encoding="utf-8")
print("each store records its own page")
