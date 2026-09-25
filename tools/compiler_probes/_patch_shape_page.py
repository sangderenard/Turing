"""Record every settled shape on a concordance page.

The settlement's conclusions previously lived in local sets and graph
attributes, so "what does the system believe this value's shape is, and when
did it learn it" could only be answered by adding another print.  On a page
the history is readable by any later stage, and `oscillating_rows` and
`disagreements` work on it without further code.
"""

from pathlib import Path

LEDGER_ANCHOR = """    literal_conflicts: set[tuple[int, str]] = set()
"""

LEDGER_WITH_PAGE = '''    literal_conflicts: set[tuple[int, str]] = set()

    # The concordance is where a shape becomes readable across stages, so
    # every fact this phase settles is written there as it is settled: the
    # declared contract, each propagation over a call edge, and each formal
    # proved to have no single shape.  A row is one value identity; its
    # history is the order in which this phase learned about it.
    from .identity_concordance import current_identity_book as _shape_book

    shape_page = _shape_book().page("value_shape")

    def note_shape(owner: Any, value_id: int, fact: Any) -> None:
        row = (str(owner.graph.get("function_name")), int(value_id))
        if shape_page.latest(row) != fact:
            shape_page.set(row, len(shape_page.history(row)), fact)

'''

CONTRACT_ANCHOR = """            for value_id in identities.get(str(parameter_name), ()):
                value_contracts[int(value_id)] = dict(contract)"""

CONTRACT_WITH_PAGE = """            for value_id in identities.get(str(parameter_name), ()):
                value_contracts[int(value_id)] = dict(contract)
                note_shape(planned_graph, int(value_id), (
                    "contract",
                    tuple(contract.get("shape") or ()),
                    str(contract.get("dtype") or ""),
                    str(contract.get("storage") or ""),
                ))"""

INSTALL_ANCHOR = """            if existing is None:
                destination[int(callee_id)] = dict(source)
                changed = True
                continue"""

INSTALL_WITH_PAGE = """            if existing is None:
                destination[int(callee_id)] = dict(source)
                note_shape(callee_graph, int(callee_id), (
                    f"from {caller_graph.graph.get('function_name')}"
                    f" value {int(caller_id)}",
                    tuple(source.get("shape") or ()),
                    str(source.get("dtype") or ""),
                    str(source.get("storage") or ""),
                ))
                changed = True
                continue"""

POLYMORPHIC_ANCHOR = """                    if marker not in polymorphic_formals:
                        polymorphic_formals.add(marker)
                        changed = True"""

POLYMORPHIC_WITH_PAGE = """                    note_shape(callee_graph, int(callee_id), (
                        "polymorphic",
                        tuple(value),
                        str(source.get("dtype") or ""),
                        str(source.get("storage") or ""),
                    ))
                    if marker not in polymorphic_formals:
                        polymorphic_formals.add(marker)
                        changed = True"""

path = Path(__file__).resolve().parents[2] / "src/compiler/fortran_c_shell.py"
text = path.read_text(encoding="utf-8")

for anchor, replacement in (
    (LEDGER_ANCHOR, LEDGER_WITH_PAGE),
    (CONTRACT_ANCHOR, CONTRACT_WITH_PAGE),
    (INSTALL_ANCHOR, INSTALL_WITH_PAGE),
    (POLYMORPHIC_ANCHOR, POLYMORPHIC_WITH_PAGE),
):
    assert text.count(anchor) == 1, (text.count(anchor), anchor[:60])
    text = text.replace(anchor, replacement)

path.write_text(text, encoding="utf-8")
print("settled shapes are now written to the value_shape page")
