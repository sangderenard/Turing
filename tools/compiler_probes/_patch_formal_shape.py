"""A formal whose callsites all prove one shape HAS that shape.

The literal fix said a formal's value is a fact about the formal, not about
whichever copy was told.  Its shape is the same kind of fact, and the same
copies were going without: `_first_occurrence` is lowered from a copy with no
caller, so its mask formal has no extents and `cumsum` cannot state the axis
it scans.

Published at shell construction alongside the literals, read by the shape
query when a copy has nothing local.  Two callsites proving different shapes
mark the row conflicting and it stops answering -- a genuinely polymorphic
formal is not a fact.
"""

from pathlib import Path

PUBLISH_ANCHOR = """            else:
                _publish_formal_literal(
                    str(original.G.graph.get("function_name")),
                    parameter,
                    specializations[parameter],
                    str(caller.G.graph.get("function_name")),
                )"""

PUBLISH = """            else:
                _publish_formal_literal(
                    str(original.G.graph.get("function_name")),
                    parameter,
                    specializations[parameter],
                    str(caller.G.graph.get("function_name")),
                )
        proven_descriptor = tensor_descriptors.get(parameter)
        if proven_descriptor is not None and tuple(
            proven_descriptor.get("shape") or ()
        ):
            _publish_formal_shape(
                str(original.G.graph.get("function_name")),
                parameter,
                proven_descriptor,
                str(caller.G.graph.get("function_name")),
            )"""

HELPER_ANCHOR = "def _proven_formal_literal(graph: Any, name: Any) -> Any:"

HELPER = '''def _publish_formal_shape(
    function: str, parameter: str, descriptor: Any, caller: str,
) -> None:
    """Record that a callsite proved one shape for one formal."""

    try:
        from .identity_concordance import current_identity_book

        extents = tuple(int(e) for e in (descriptor.get("shape") or ()))
        dtype = str(descriptor.get("dtype") or "float64")
        page = current_identity_book().page("formal_shape")
        row = (str(function), str(parameter))
        previous = page.latest(row)
        if previous is None:
            page.set(row, 0, ("proven", extents, dtype, caller))
        elif previous[0] == "proven" and tuple(previous[1]) != extents:
            page.set(
                row, len(page.history(row)),
                ("conflicting", extents, dtype, caller),
            )
    except Exception:
        pass


def _proven_formal_shape(graph: Any, name: Any) -> Any:
    """The shape every callsite agrees this formal carries, if any."""

    if not name:
        return None
    try:
        from .identity_concordance import current_identity_book

        page = current_identity_book().page("formal_shape")
        row = (str(graph.G.graph.get("function_name")), str(name))
        proven = page.latest(row)
    except Exception:
        return None
    if proven is None or proven[0] != "proven":
        return None
    return {
        "shape": tuple(proven[1]),
        "dtype": str(proven[2]),
        "rank": len(tuple(proven[1])),
    }


''' + HELPER_ANCHOR

INPUT_ANCHOR = """    if "shape" not in tensor and data.get("type") == "Input":
        binding_name = (data.get("attributes") or {}).get("binding_name")"""

INPUT = """    if "shape" not in tensor and data.get("type") == "Input":
        binding_name = (data.get("attributes") or {}).get("binding_name")
        # A copy with no caller was told nothing; the formal still has the
        # shape its callsites proved.
        agreed = _proven_formal_shape(graph, binding_name)
        if agreed is not None:
            return agreed"""

path = Path(__file__).resolve().parents[2] / "src/compiler/glsl_deployment_strategy.py"
text = path.read_text(encoding="utf-8")
for anchor, replacement in (
    (HELPER_ANCHOR, HELPER),
    (INPUT_ANCHOR, INPUT),
    (PUBLISH_ANCHOR, PUBLISH),
):
    assert text.count(anchor) == 1, (text.count(anchor), anchor[:60])
    text = text.replace(anchor, replacement)
path.write_text(text, encoding="utf-8")
print("a formal's shape is a fact about the formal, not about the copy")
