"""Make the concordance the authority for a value's shape, not a log of it.

Evidence that forced this: the descriptor query answered differently for the
same value at different moments --

    ('solve', 0, 'input')  [None, ((),'unknown'), ((2,2),'float64'),
                            ((),'unknown'), ((2,2),'float64')]
    ('_lu_decompose_inplace', 3, 'clone')
                           [None, ((),'unknown'), ..., ((2,2),'float64'),
                            ((),'unknown')]

`solve`'s A has a DECLARED contract and still flips.  `U = A.clone()` proves
(2,2) and then regresses, and the regressed answer is what its function's
return contract was written from.  Nothing owns "the shape of value V in
function F"; every query re-derives it from whatever is in the graph at that
instant, so the last caller to ask wins and every repair to an individual
derivation rule is invisible.

The invariants this installs:

* monotone -- extents, once proven, are never replaced by an unknown;
* single owner -- the query reads the page before deriving, and publishes
  what it derives, so no pass re-derives what another already proved;
* an extent-less answer is NEVER stored, so `shape=() with a known dtype`
  (which is both a real rank-0 scalar and what the query returns when
  recovery stops) can never win a race against a real shape;
* a genuine conflict -- two callsites proving two different shapes for one
  formal -- is recorded and the row stops answering, falling back to live
  derivation rather than letting one callsite cement its answer for the
  other.
"""

from pathlib import Path

OLD = '''    answer = _tensor_descriptor_rule(graph, node_id, _seen)
    try:
        from .identity_concordance import current_identity_book

        data = graph.G.nodes[int(node_id)] if int(node_id) in graph.G else {}
        page = current_identity_book().page("tensor_descriptor")
        row = (
            str(graph.G.graph.get("function_name")),
            int(data.get("value_id", node_id)),
            str(data.get("op") or data.get("type") or ""),
        )
        fact = (
            None if answer is None
            else (tuple(answer.get("shape") or ()),
                  str(answer.get("dtype") or ""),
                  str(answer.get("metadata_state") or "")),
        )
        if page.latest(row) != fact:
            page.set(row, len(page.history(row)), fact)
    except Exception:
        # A diagnostic must never change whether a compile succeeds.
        pass
    return answer'''

NEW = '''    row = None
    page = None
    try:
        from .identity_concordance import current_identity_book

        data = graph.G.nodes[int(node_id)] if int(node_id) in graph.G else {}
        page = current_identity_book().page("proven_shape")
        row = (
            str(graph.G.graph.get("function_name")),
            int(data.get("value_id", node_id)),
        )
        proven = page.latest(row)
        if proven is not None and proven[0] != "conflicting":
            # Already proven.  Answering from the page is what makes the
            # query a function of the value rather than of the moment.
            return {
                "shape": tuple(proven[1]),
                "dtype": str(proven[2]),
                "rank": len(tuple(proven[1])),
            }
    except Exception:
        row = None

    answer = _tensor_descriptor_rule(graph, node_id, _seen)

    if row is not None and page is not None:
        try:
            extents = tuple(answer.get("shape") or ()) if answer else ()
            dynamic = bool(
                answer and str(answer.get("metadata_state") or "") == "dynamic"
            )
            # Only an answer with real extents is a proof.  An empty shape is
            # never cemented: it is indistinguishable from "recovery stopped".
            if extents and not dynamic:
                dtype = str((answer or {}).get("dtype") or "float64")
                previous = page.latest(row)
                if previous is None:
                    page.set(row, 0, ("proven", extents, dtype))
                elif previous[0] == "proven" and tuple(previous[1]) != extents:
                    # Two derivations prove two different shapes for one
                    # identity.  Record it and stop answering from the page
                    # rather than let whichever asked first speak for both.
                    page.set(
                        row, len(page.history(row)),
                        ("conflicting", extents, dtype),
                    )
        except Exception:
            pass
    return answer'''

path = Path(__file__).resolve().parents[2] / "src/compiler/glsl_deployment_strategy.py"
text = path.read_text(encoding="utf-8")
assert text.count(OLD) == 1, text.count(OLD)
path.write_text(text.replace(OLD, NEW), encoding="utf-8")
print("the concordance now owns proven shapes")
