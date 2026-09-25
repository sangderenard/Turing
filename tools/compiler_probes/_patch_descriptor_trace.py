"""Record what the descriptor query answers, for every value it is asked.

Three rounds in a row I inferred which rule produced a wrong shape from the
symptom downstream, and was wrong each time.  The query is the single place
every shape comes from, so its answers belong on a page: one row per value
identity, carrying the operation that produced it and the shape it was given.
A chain of wrong shapes is then read off in one run instead of guessed at one
link per run.

Deduplicated on the answer, so the fixed points that call this thousands of
times record a row only when the answer actually changes.
"""

from pathlib import Path

RENAME = "def _tensor_descriptor(\n    graph: Any, node_id: int, _seen: set[int] | None = None,\n) -> dict[str, Any] | None:"

WRAPPER = '''def _tensor_descriptor(
    graph: Any, node_id: int, _seen: set[int] | None = None,
) -> dict[str, Any] | None:
    """The compiler-owned shape query, with its answer recorded."""

    answer = _tensor_descriptor_rule(graph, node_id, _seen)
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
    return answer


def _tensor_descriptor_rule(
    graph: Any, node_id: int, _seen: set[int] | None = None,
) -> dict[str, Any] | None:'''

path = Path(__file__).resolve().parents[2] / "src/compiler/glsl_deployment_strategy.py"
text = path.read_text(encoding="utf-8")
assert text.count(RENAME) == 1, text.count(RENAME)
text = text.replace(RENAME, WRAPPER)
path.write_text(text, encoding="utf-8")
print("descriptor answers are recorded on the tensor_descriptor page")
