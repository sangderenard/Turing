"""Teach the shape query the axis-changing operations.

`_tensor_descriptor` knew `reshape` and `transpose` but had no rule at all for
`unsqueeze`, `squeeze`, `expand` or `broadcast_to`.  Source that indexes and
slices rarely needs them; source written as whole-axis arithmetic uses them
everywhere -- `index.unsqueeze(-1) == index.unsqueeze(-2)` is how it states an
identity matrix -- so the very first step of every mask had no shape, and
nothing downstream of it could have one either.
"""

from pathlib import Path

ANCHOR = """        if operation in _ELEMENTWISE_BINARY_OPERATIONS:"""

RULE = '''        if operation in {"unsqueeze", "squeeze", "expand", "broadcast_to"}:
            parents = data.get("parents") or ()
            source_id = next((
                int(parent)
                for parent, role in parents
                if str(role).casefold() in {
                    "operand", "value", "base", "input", "self", "receiver",
                }
                and int(parent) in graph.G
            ), None)

            def _axis_literal(candidate: int) -> Any:
                node = graph.G.nodes[int(candidate)]
                literal = node.get("constant")
                if literal is None:
                    literal = (node.get("attributes") or {}).get("value")
                return literal

            arguments = [
                _axis_literal(int(parent))
                for parent, role in parents
                if str(role).casefold().startswith("arg:")
                and int(parent) in graph.G
            ]
            declared_axis = _attribute_axis(data)
            if declared_axis is not None:
                arguments = [declared_axis, *arguments]
            if source_id is not None:
                base = _tensor_descriptor(graph, source_id, seen)
                if base is not None and descriptor_states_a_shape(base):
                    extents = tuple(int(e) for e in (base.get("shape") or ()))
                    dtype = str(base.get("dtype") or "float64")
                    settled = None
                    first = arguments[0] if arguments else None
                    if operation == "unsqueeze" and isinstance(first, int):
                        # A new axis of length one; a negative position counts
                        # from the END OF THE RESULT, which is one longer.
                        position = (
                            first if first >= 0 else first + len(extents) + 1
                        )
                        if 0 <= position <= len(extents):
                            settled = (
                                extents[:position] + (1,) + extents[position:]
                            )
                    elif operation == "squeeze":
                        if isinstance(first, int):
                            position = (
                                first if first >= 0 else first + len(extents)
                            )
                            if (
                                0 <= position < len(extents)
                                and extents[position] == 1
                            ):
                                settled = (
                                    extents[:position]
                                    + extents[position + 1:]
                                )
                        elif not arguments:
                            settled = tuple(e for e in extents if e != 1)
                    else:
                        requested = (
                            first
                            if isinstance(first, (tuple, list))
                            else tuple(arguments)
                        )
                        if requested and all(
                            isinstance(extent, int) and not isinstance(
                                extent, bool
                            )
                            for extent in requested
                        ):
                            settled = tuple(int(e) for e in requested)
                    if settled is not None:
                        return {
                            "shape": settled,
                            "dtype": dtype,
                            "rank": len(settled),
                        }
''' + ANCHOR

HELPER_ANCHOR = "_ELEMENTWISE_BINARY_OPERATIONS = frozenset({"

HELPER = '''def _attribute_axis(data: Any) -> int | None:
    """The axis an operation names in its own attributes, when it names one."""

    attributes = data.get("attributes") or {}
    for key in ("dim", "axis", "dims", "axes"):
        value = attributes.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            return int(value)
    return None


''' + HELPER_ANCHOR

path = Path(__file__).resolve().parents[2] / "src/compiler/glsl_deployment_strategy.py"
text = path.read_text(encoding="utf-8")
assert text.count(ANCHOR) == 1, text.count(ANCHOR)
text = text.replace(ANCHOR, RULE)
assert text.count(HELPER_ANCHOR) == 1, text.count(HELPER_ANCHOR)
text = text.replace(HELPER_ANCHOR, HELPER)
path.write_text(text, encoding="utf-8")
print("unsqueeze / squeeze / expand / broadcast_to now have shape rules")
