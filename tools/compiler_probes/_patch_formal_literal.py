"""A formal whose callers proved a literal IS a literal.

`_source_static_value` asked whether THIS graph copy had already been told a
value, never whether the value is known:

    if data.get("type") == "Input":
        name = (data.get("attributes") or {}).get("binding_name")
        return name in (graph.G.graph.get("planner_specializations") or {})

So the same parameter, in the same program, carrying the same `2`, read as a
literal in the copy that was told and as runtime data in every other copy --
and `_source_static_literal` then raised `ValueError`, which the caller
swallowed with `except ValueError: pass`.  A callee built from an untold copy
can never learn the count, which is why `arange` had none.

Every specialization a callsite proves is now published to the concordance
under (function, parameter), and a copy with no local binding reads it there.
Two callsites proving DIFFERENT literals for one parameter mark the row
conflicting, and it stops answering -- a parameter that genuinely varies is
runtime data, and the page says which callsites disagreed.
"""

from pathlib import Path

PUBLISH_ANCHOR = """        if static_argument:
            try:
                specializations[parameter] = _source_static_literal(
                    caller, int(parent)
                )
            except ValueError:
                pass
    for parameter, default in (
        original.G.graph.get("parameter_defaults") or {}"""

PUBLISH = '''        if static_argument:
            try:
                specializations[parameter] = _source_static_literal(
                    caller, int(parent)
                )
            except ValueError:
                pass
            else:
                _publish_formal_literal(
                    str(original.G.graph.get("function_name")),
                    parameter,
                    specializations[parameter],
                    str(caller.G.graph.get("function_name")),
                )
    for parameter, default in (
        original.G.graph.get("parameter_defaults") or {}'''

HELPER_ANCHOR = "def _source_static_value(graph: Any, node_id: int, visiting=None) -> bool:"

HELPER = '''_FORMAL_LITERAL_CONFLICT = object()


def _publish_formal_literal(
    function: str, parameter: str, value: Any, caller: str,
) -> None:
    """Record that a callsite proved one literal for one formal."""

    try:
        from .identity_concordance import current_identity_book

        page = current_identity_book().page("formal_literal")
        row = (str(function), str(parameter))
        previous = page.latest(row)
        if previous is None:
            page.set(row, 0, ("proven", value, caller))
            return
        if previous[0] == "conflicting":
            return
        try:
            agrees = bool(previous[1] == value)
        except Exception:
            agrees = False
        if not agrees:
            # Genuinely parametric: two callsites, two values.
            page.set(
                row, len(page.history(row)),
                ("conflicting", (previous[1], value), caller),
            )
    except Exception:
        pass


def _proven_formal_literal(graph: Any, name: Any) -> Any:
    """The literal every callsite agrees this formal carries, if any."""

    if not name:
        return _FORMAL_LITERAL_CONFLICT
    try:
        from .identity_concordance import current_identity_book

        page = current_identity_book().page("formal_literal")
        row = (str(graph.G.graph.get("function_name")), str(name))
        proven = page.latest(row)
    except Exception:
        return _FORMAL_LITERAL_CONFLICT
    if proven is None or proven[0] != "proven":
        return _FORMAL_LITERAL_CONFLICT
    return proven[1]


''' + HELPER_ANCHOR

VALUE_ANCHOR = """    if data.get("type") == "Input":
        name = (data.get("attributes") or {}).get("binding_name")
        return name in (graph.G.graph.get("planner_specializations") or {})"""

VALUE = """    if data.get("type") == "Input":
        name = (data.get("attributes") or {}).get("binding_name")
        if name in (graph.G.graph.get("planner_specializations") or {}):
            return True
        # Not told HERE is not the same as unknown.  Copies of one function
        # share its formals; if every callsite that proved this one agrees,
        # the value is a literal in all of them.
        return _proven_formal_literal(
            graph, name
        ) is not _FORMAL_LITERAL_CONFLICT"""

LITERAL_ANCHOR = """    if data.get("type") == "Input":
        name = (data.get("attributes") or {}).get("binding_name")
        specializations = graph.G.graph.get("planner_specializations") or {}
        if name in specializations:
            return specializations[name]
        raise ValueError("input has no source-static planner binding")"""

LITERAL = """    if data.get("type") == "Input":
        name = (data.get("attributes") or {}).get("binding_name")
        specializations = graph.G.graph.get("planner_specializations") or {}
        if name in specializations:
            return specializations[name]
        proven = _proven_formal_literal(graph, name)
        if proven is not _FORMAL_LITERAL_CONFLICT:
            return proven
        raise ValueError("input has no source-static planner binding")"""

path = Path(__file__).resolve().parents[2] / "src/compiler/glsl_deployment_strategy.py"
text = path.read_text(encoding="utf-8")
for anchor, replacement in (
    (HELPER_ANCHOR, HELPER),
    (VALUE_ANCHOR, VALUE),
    (LITERAL_ANCHOR, LITERAL),
    (PUBLISH_ANCHOR, PUBLISH),
):
    assert text.count(anchor) == 1, (text.count(anchor), anchor[:60])
    text = text.replace(anchor, replacement)
path.write_text(text, encoding="utf-8")
print("a formal's literal is a fact about the formal, not about the copy")
