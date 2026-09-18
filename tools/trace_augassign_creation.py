"""One-shot diagnostic: monkeypatch networkx.DiGraph.add_node to print the
real Python call stack the moment a graph node is created for the specific
`ctrl.clamp_events += 1` AugAssign at source line 391 of the composited
repro source (tools/repro_step_with_dt_control_used.py) -- to find, by
direct observation instead of more grepping, which module actually builds
the ProcessGraph's reaching-definition edges for an attribute mutation
inside conditional branches.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import networkx as nx  # noqa: E402

_seen = set()
_original_add_node = nx.DiGraph.add_node


def _patched_add_node(self, node_for_adding, **attrs):
    nested_attrs = attrs.get("attributes")
    source_type = (
        attrs.get("source_type")
        or (nested_attrs.get("source_type") if isinstance(nested_attrs, dict) else None)
    )
    if source_type == "AugAssign":
        caller = traceback.extract_stack(limit=6)[-3]
        key = (caller.filename, caller.lineno)
        if key not in _seen:
            _seen.add(key)
            print(f"\n=== add_node AugAssign node_id={node_for_adding} "
                  f"attrs={attrs} ===", file=sys.stderr)
            traceback.print_stack(file=sys.stderr, limit=15)
    return _original_add_node(self, node_for_adding, **attrs)


nx.DiGraph.add_node = _patched_add_node

import runpy  # noqa: E402

runpy.run_path(
    str(Path(__file__).resolve().parent / "repro_step_with_dt_control_used.py"),
    run_name="__main__",
)
