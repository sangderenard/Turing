"""A launchable HTML shell for a process-graph schedule -- modeled after
``wasm_html_shell.py``, the same way that file is the browser's launch
environment for one compiled WebAssembly artifact.

That file's own words apply here almost unchanged: nothing in this page is
specific to any one program. It is generated from a schedule descriptor
(``schedule_table``), so the table it draws is whatever that schedule
contains -- compile something else and the page reshapes itself.

**What "schedule" means here**, concretely: ``wasm_class_modules.py``
segments a compiled program into class modules and can build a real
``transmogrifier.graph.graph_express2.ProcessGraph`` over them, with a real
``ILPScheduler`` attached (``build_module_process_graph``,
``schedule_module_levels``) -- not a hand-rolled ordering. ``schedule_table``
in this file turns that into the one shape this page draws: every module is
a node; its **row** is the ILP scheduler's level (a module is always at a
strictly later level than everything it calls -- level is "how many calls
deep from the earliest point this could run", the same ASAP/ALAP concept any
instruction scheduler uses); its **column** is which weakly-connected
component of the dependency graph it belongs to (two modules with no path
between them, even indirectly, never share a column, regardless of level).

``schedule_table`` is called fresh from a live list of ``ClassModuleSpec``
each time -- it is not a static picture baked in advance. A caller with a
different compiled program, or a different ``size_threshold`` choice for
``plan_class_modules``, gets a different table without touching this file.

This module intentionally does **not** decide how data moves between the
nodes it draws (which is ``derive_process_graph``'s concern, or a runner's)
or how any node actually executes. It draws the schedule shape; nothing
about the visualization holds an opinion about what runs the graph.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

_CSS = """
:root {
  color-scheme: light dark;
  --line: color-mix(in srgb, currentColor 18%, transparent);
  --soft: color-mix(in srgb, currentColor 6%, transparent);
  --accent: #3b82f6;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  padding: 1.5rem;
  font: 14px/1.5 ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
  max-width: 60rem;
}
.title { font-size: 1.25rem; font-weight: 600; }
.sub { opacity: .7; font-size: .85rem; margin-top: .15rem; }
table.schedule { border-collapse: collapse; margin-top: 1.25rem; width: 100%; }
table.schedule th, table.schedule td {
  border: 1px solid var(--line);
  padding: .5rem .6rem;
  vertical-align: top;
  min-width: 8rem;
}
table.schedule th { text-align: left; font-size: .75rem; opacity: .65;
  text-transform: uppercase; letter-spacing: .04em; }
table.schedule td.level-label { font-weight: 600; opacity: .8; background: var(--soft); }
.node {
  display: inline-block;
  padding: .25rem .55rem;
  border-radius: .35rem;
  background: var(--soft);
  border: 1px solid var(--line);
  font-family: ui-monospace, monospace;
  font-size: .8rem;
  margin: .15rem;
}
.node.is-root { border-color: var(--accent); color: var(--accent); }
.node.lit { background: var(--accent); color: #fff; border-color: var(--accent); }
.empty-cell { opacity: .35; }
.note { font-size: .85rem; padding: .6rem .75rem; border-radius: .35rem;
  background: var(--soft); margin-top: 1rem; }
"""

_JS = r"""
function renderSchedule(schedule) {
  const table = document.getElementById("schedule");
  const byCell = new Map(); // "level:group" -> [node, ...]
  for (const node of schedule.nodes) {
    const key = node.level + ":" + node.group;
    if (!byCell.has(key)) byCell.set(key, []);
    byCell.get(key).push(node);
  }

  const thead = document.createElement("thead");
  const headRow = document.createElement("tr");
  headRow.appendChild(document.createElement("th")).textContent = "Level \\ Group";
  for (let g = 0; g < schedule.groups; g++) {
    const th = document.createElement("th");
    th.textContent = "Group " + g;
    headRow.appendChild(th);
  }
  thead.appendChild(headRow);
  table.appendChild(thead);

  const tbody = document.createElement("tbody");
  // level_min is typically negative: prerequisite work the root (the
  // entrypoint's own module) depends on sits behind it, at level_min..-1,
  // and the root itself is always exactly level 0.
  for (let level = schedule.level_min; level <= schedule.level_max; level++) {
    const tr = document.createElement("tr");
    const label = document.createElement("td");
    label.className = "level-label";
    label.textContent = level;
    tr.appendChild(label);
    for (let g = 0; g < schedule.groups; g++) {
      const td = document.createElement("td");
      const nodes = byCell.get(level + ":" + g) || [];
      if (nodes.length === 0) {
        td.className = "empty-cell";
        td.textContent = "·";
      } else {
        for (const node of nodes) {
          const span = document.createElement("span");
          span.className = "node" + (node.is_root ? " is-root" : "");
          span.dataset.nodeId = node.id;
          span.textContent = node.id;
          td.appendChild(span);
        }
      }
      tr.appendChild(td);
    }
    tbody.appendChild(tr);
  }
  table.appendChild(tbody);
}

// Lights up one node -- the hook a runner (process_graph_runner.js or
// anything else driving the actual modules) calls as each one executes, so
// the table reflects the real schedule being walked, not just its static
// shape. Left as a free function rather than wired to any particular
// runner: this page draws the schedule, it does not run it.
function markScheduleNodeLit(nodeId, lit = true) {
  const span = document.querySelector('.node[data-node-id="' + CSS.escape(nodeId) + '"]');
  if (span) span.classList.toggle("lit", lit);
}

renderSchedule(SCHEDULE);
"""


def schedule_table(
    specs: Sequence[Any], *, method: str = "asap"
) -> dict:
    """Build the JSON-able schedule this shell draws, fresh from a live list
    of ``ClassModuleSpec`` -- pulled from the real ``ProcessGraph``/
    ``ILPScheduler`` (``wasm_class_modules.build_module_process_graph``/
    ``schedule_module_levels``), not authored by hand and not cached from a
    previous compile.

    ``nodes``: one entry per module -- ``id`` (``ClassModuleSpec.module_name``),
    ``level`` (the ILP scheduler's level: strictly greater than everything
    this module calls), ``group`` (which weakly-connected component of the
    dependency graph it belongs to -- unrelated branches never share a
    column even if they land on the same level), and ``is_root``.

    ``edges``: the same call-dependency edges ``build_module_process_graph``
    drew (callee -> caller).

    ``level_min``/``level_max`` are inclusive row bounds, not a count:
    ``schedule_module_levels`` puts the root (the process graph's owner,
    the entrypoint) at level 0 and shifts everything it depends on negative,
    so a schedule with real prerequisite work has ``level_min < 0``.
    ``groups`` remains a plain count -- columns have no root to anchor to.
    """

    import networkx as nx

    from .wasm_class_modules import (
        build_module_process_graph, schedule_module_levels,
    )

    graph = build_module_process_graph(specs)
    levels = schedule_module_levels(specs, method=method)

    components = list(nx.weakly_connected_components(graph.G))
    group_of: dict[str, int] = {}
    for group_index, component in enumerate(components):
        for node_id in component:
            group_of[node_id] = group_index

    nodes = [
        {
            "id": spec.module_name,
            "level": levels[spec.module_name],
            "group": group_of[spec.module_name],
            "is_root": spec.is_root,
        }
        for spec in specs
    ]
    edges = [{"from": src, "to": dst} for src, dst in graph.G.edges]

    return {
        "nodes": nodes,
        "edges": edges,
        "level_min": min((n["level"] for n in nodes), default=0),
        "level_max": max((n["level"] for n in nodes), default=0),
        "groups": (max((n["group"] for n in nodes), default=-1) + 1),
    }


@dataclass(frozen=True)
class ProcessGraphShell:
    """A self-contained HTML page: one ``<table>``, rows are schedule
    levels, columns are schedule groups, cells are the modules the real
    scheduler placed there. See the module docstring."""

    html: str

    def write(self, path: str) -> str:
        from pathlib import Path

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.html, encoding="utf-8")
        return str(target)


def emit_process_graph_shell(
    schedule: Mapping[str, Any], *, title: str = "process graph"
) -> ProcessGraphShell:
    """Render ``schedule`` (from ``schedule_table``) as a standalone HTML
    page. Rendering is entirely client-side, from the embedded JSON, so the
    same page shape works for any schedule this file is handed -- the table
    is drawn by ``renderSchedule`` reading ``SCHEDULE`` at load time, not
    written out as static rows here."""

    payload = json.dumps(schedule)
    level_min = schedule.get("level_min", 0)
    level_max = schedule.get("level_max", 0)
    level_count = level_max - level_min + 1
    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>{_CSS}</style>
</head>
<body>
<div class="title">{title}</div>
<div class="sub">
  {len(schedule.get("nodes", ()))} modules
  &middot; {level_count} levels ({level_min}..{level_max})
  &middot; {schedule.get("groups", 0)} groups
</div>
<table class="schedule" id="schedule"></table>
<div class="note">
  Rows are ILPScheduler levels, shifted so the entrypoint's own module sits
  at level 0 and everything it depends on sits behind it at negative
  levels; columns are weakly-connected components
  of the dependency graph. Built fresh from the compiled program's own
  ClassModuleSpec list -- see wasm_class_modules.py and this file's
  schedule_table().
</div>
<script>
const SCHEDULE = {payload};
{_JS}
</script>
</body>
</html>
"""
    return ProcessGraphShell(html=html)


__all__ = ["ProcessGraphShell", "emit_process_graph_shell", "schedule_table"]
