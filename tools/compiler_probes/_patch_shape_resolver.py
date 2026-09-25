"""One resolver for a value's shape; every store defers to it.

Measured on the solve compile, four stores held a value's shape and none
deferred to another:

    store node:    65 rows   the graph node's own `tensor` dict
    store linked:  42 rows   the settled `linked_value_abi` contracts
    store ssa:      2 rows   the SSAValue field the lowering reads
    store proven: 132 rows   the descriptor query's answer

Continuity survived only where the copies happened to agree.  The answer
exists for every identity on the `proven_shape` page; nothing was obliged to
ask it.  This gives the concordance the resolver -- `proven_shape_of` /
`record_proven_shape` -- so a consumer reads the fact instead of trusting
whichever copy was nearest, and the column stays the dependency level so a
deeper derivation supersedes a shallower one.
"""

from pathlib import Path

CONCORDANCE_ANCHOR = "def row_value_id(row: Any) -> int | None:"

RESOLVER = '''def authored_function_name(name: Any) -> str:
    """The authored name behind a lowered symbol.

    Stores are keyed by the function as written -- ``solve`` -- while a
    lowered symbol carries the module prefix, the callsite specialization
    hash and the region index.  Without stripping those, rows never line up
    and every value appears to have exactly one source.
    """

    text = str(name)
    for separator in ("__specialized_", "__planned_region"):
        if separator in text:
            text = text.split(separator)[0]
    return text.rsplit("__", 1)[-1] if "__" in text else text


def record_proven_shape(
    function: Any, value_id: int, extents: Any, dtype: Any, level: int = 0,
) -> None:
    """Record extents proven for one value identity at one causal level.

    Only EXTENTS are recorded.  An empty shape is both a rank-0 scalar and
    what a query returns when recovery stops, so storing it would let an
    unknown win a race against a real shape.  A deeper level supersedes a
    shallower one because it was derived from more of the program; the same
    answer arriving deeper is recorded at its own level so the page shows how
    far it has been confirmed.
    """

    extents = tuple(int(extent) for extent in (extents or ()))
    if not extents:
        return
    page = current_identity_book().page("proven_shape")
    row = (authored_function_name(function), int(value_id))
    recorded = page.history(row)
    fact = ("proven", extents, str(dtype or "float64"))
    if not recorded:
        page.set(row, int(level), fact)
        return
    deepest = max(recorded, key=lambda entry: int(entry[0]))
    if tuple(deepest[1][1]) == extents or int(level) > int(deepest[0]):
        page.set(row, int(level), fact)
        return
    page.set(row, int(level), ("conflicting", extents, str(dtype or "")))


def proven_shape_of(function: Any, value_id: int) -> tuple[int, ...] | None:
    """The extents proven for this identity, deepest first, or None.

    This is the question every store was answering separately.  A row that
    two derivations contradict at the same causal level answers nothing --
    concurrent and genuinely in conflict is not a fact.
    """

    page = current_identity_book().page("proven_shape")
    row = (authored_function_name(function), int(value_id))
    recorded = page.history(row)
    if not recorded:
        return None
    deepest = max(recorded, key=lambda entry: int(entry[0]))[1]
    if not isinstance(deepest, tuple) or deepest[0] != "proven":
        return None
    return tuple(int(extent) for extent in deepest[1])


def shape_store_report(book: Any, stores: Any = None) -> str:
    """Where the stores of one shape disagree, as a report.

    Kept because it turned a day of inference into three named rows: a value
    whose stores differ is a row, not a hunt.
    """

    names = tuple(stores or ("node", "linked", "ssa"))
    pages = {name: book.page(f"shape.{name}") for name in names}
    pages["proven"] = book.page("proven_shape")
    rows: set = set()
    for page in pages.values():
        rows.update(page.rows())

    def extents(name: str, row: Any):
        page = pages[name]
        if row not in set(page.rows()):
            return None
        recorded = page.history(row)
        if not recorded:
            return None
        fact = max(recorded, key=lambda entry: int(entry[0]))[1]
        if name == "proven":
            return tuple(fact[1]) if fact[0] == "proven" else None
        return tuple(fact)

    lines = []
    disagreeing = []
    for row in sorted(rows, key=str):
        present = {}
        for name in pages:
            value = extents(name, row)
            if value is not None:
                present[name] = value
        if len({tuple(value) for value in present.values()}) > 1:
            disagreeing.append((row, present))
    lines.append(
        f"shape stores: {len(disagreeing)} disagreeing of {len(rows)} "
        "value identit(ies)"
    )
    for row, present in disagreeing[:10]:
        lines.append(f"  {render_row(row)} {present}")
    for name, page in pages.items():
        lines.append(f"  store {name}: {len(page.rows())} row(s)")
    return "\\n".join(lines)


''' + CONCORDANCE_ANCHOR

path = Path(__file__).resolve().parents[2] / "src/compiler/identity_concordance.py"
text = path.read_text(encoding="utf-8")
assert text.count(CONCORDANCE_ANCHOR) == 1, text.count(CONCORDANCE_ANCHOR)
path.write_text(
    text.replace(CONCORDANCE_ANCHOR, RESOLVER), encoding="utf-8",
)
print("the concordance owns the resolver")
