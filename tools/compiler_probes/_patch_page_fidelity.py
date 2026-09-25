"""Make the shape/edge pages record facts, not restatements.

Two flaws in the first cut showed up immediately in the output:

* a second callsite agreeing on the same shape appended a new column, because
  the recorded fact included the caller's name -- so ``oscillating_rows``
  reported round trips that were only two callers saying the same thing;
* only the FIRST install of a contract was recorded, so the interesting
  event -- a shape arriving on a formal that already had one -- was invisible.

One edge per row fixes the first for `call_edge`; comparing on the shape
rather than the whole fact fixes it for `value_shape`.
"""

from pathlib import Path

OLD_NOTE = """    def note_shape(owner: Any, value_id: int, fact: Any) -> None:
        row = (str(owner.graph.get("function_name")), int(value_id))
        if shape_page.latest(row) != fact:
            shape_page.set(row, len(shape_page.history(row)), fact)"""

NEW_NOTE = """    def note_shape(owner: Any, value_id: int, fact: Any) -> None:
        row = (str(owner.graph.get("function_name")), int(value_id))
        previous = shape_page.latest(row)
        # A fact is the shape, not who said it.  Two callsites agreeing is one
        # fact recorded once; otherwise every additional caller reads as a
        # change and a round trip between equal shapes reads as oscillation.
        if previous is not None and tuple(previous[1:]) == tuple(fact[1:]):
            return
        shape_page.set(row, len(shape_page.history(row)), fact)"""

OLD_EDGE = """                edge_row = (
                    str(callee_graph.graph.get("function_name")),
                    int(callee_id),
                )
                edge_page.set(edge_row, len(edge_page.history(edge_row)), (
                    "argument",
                    str(caller_graph.graph.get("function_name")),
                    int(caller_id),
                ))"""

NEW_EDGE = """                # One row per EDGE.  A formal legitimately has many callers;
                # putting them on one row made a normal fan-in look like a
                # value changing its mind.
                edge_row = (
                    str(callee_graph.graph.get("function_name")),
                    int(callee_id),
                    str(caller_graph.graph.get("function_name")),
                    int(caller_id),
                )
                edge_page.set(edge_row, 0, "argument")"""

OLD_MERGE = """                    if key in {"shape", "rank"} and (
                        marker in polymorphic_formals
                    ):"""

NEW_MERGE = """                    if key == "shape":
                        # A shape arriving on a formal that already had a
                        # contract is the event worth seeing; only the first
                        # install was being recorded.
                        note_shape(callee_graph, int(callee_id), (
                            f"merged from "
                            f"{caller_graph.graph.get('function_name')}"
                            f" value {int(caller_id)}",
                            tuple(value or ()),
                            str(source.get("dtype") or ""),
                            str(source.get("storage") or ""),
                        ))
                    if key in {"shape", "rank"} and (
                        marker in polymorphic_formals
                    ):"""

path = Path(__file__).resolve().parents[2] / "src/compiler/fortran_c_shell.py"
text = path.read_text(encoding="utf-8")
for anchor, replacement in (
    (OLD_NOTE, NEW_NOTE),
    (OLD_EDGE, NEW_EDGE),
    (OLD_MERGE, NEW_MERGE),
):
    assert text.count(anchor) == 1, (text.count(anchor), anchor[:50])
    text = text.replace(anchor, replacement)
path.write_text(text, encoding="utf-8")
print("pages record facts rather than restatements")
