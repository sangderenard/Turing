"""Planned shell tree: walk of the planned shell hierarchy and the deployment program table lines."""

from __future__ import annotations

from typing import Any


def _walk_planned_shells(
    shell: Any,
    *,
    include_function_registry: bool = True,
):
    """Yield planner-created shell instances exactly once.

    ``function_shells`` is the definition catalogue;
    ``callsite_function_shells`` is one selected program's activation tree.
    Compilation/capture of one entrypoint passes
    ``include_function_registry=False`` so catalogue entries are not mistaken
    for executed children.  Administrative callers retain the broad default.
    """
    pending = [shell]
    seen = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        yield current
        if include_function_registry:
            pending.extend(
                getattr(current, "function_shells", {}).values()
            )
        elif (
            getattr(current, "runtime_closure_only", False)
            and getattr(current, "_owns_function_shells", False)
        ):
            # In runtime-root mode this shell is an administrative owner of
            # the definition catalogue.  Its executable children are exactly
            # the submitted activation roots, not every catalogued definition
            # and not zero children.  Each selected root then exposes its full
            # explicit callsite tree below.
            function_shells = getattr(current, "function_shells", {})
            pending.extend(
                function_shells[int(reference)]
                for reference in reversed(tuple(
                    getattr(current, "activation_root_references", ())
                ))
                if int(reference) in function_shells
            )
        pending.extend(
            getattr(current, "callsite_function_shells", {}).values()
        )


def _deployment_program_table_lines(shell: Any) -> tuple[str, ...]:
    """Render the planner-owned shell hierarchy and region compartments."""

    def shell_name(current):
        return str(
            current.process_graph.G.graph.get("function_name")
            or type(current).__name__
        )

    def render_table(headers, rows):
        rows = [tuple(str(cell) for cell in row) for row in rows]
        widths = [
            max(len(str(header)), *(len(row[index]) for row in rows))
            for index, header in enumerate(headers)
        ]
        yield " | ".join(
            str(header).ljust(widths[index])
            for index, header in enumerate(headers)
        )
        yield "-+-".join("-" * width for width in widths)
        for row in rows:
            yield " | ".join(
                cell.ljust(widths[index])
                for index, cell in enumerate(row)
            )

    hierarchy = []
    ordered_shells = []
    seen = set()

    def visit(current, depth, compartment):
        if id(current) in seen:
            return
        seen.add(id(current))
        shell_index = len(ordered_shells)
        ordered_shells.append(current)
        compiled = {
            key[-2] if isinstance(key, tuple) else key
            for key in current.captured_region_programs
        }
        hierarchy.append((
            shell_index,
            depth,
            compartment,
            ("  " * depth) + shell_name(current),
            current.source_node_count,
            current.primitive_count,
            current.dispatch_count,
            len(compiled),
            len(current.coordinator_region_indices),
            current.planned_invocation_slots,
        ))
        for node_id, child in sorted(
            current.callsite_function_shells.items(),
            key=lambda item: item[0],
        ):
            visit(child, depth + 1, f"callsite-{node_id}")

    visit(shell, 0, "root")
    lines = [
        f"runtime selection: {shell.control_runtime}",
        "compiled program shell hierarchy",
    ]
    lines.extend(render_table(
        (
            "id", "depth", "compartment", "shell", "graph", "scheduled",
            "regions", "shaders", "coord", "slots",
        ),
        hierarchy,
    ))

    regions = []
    for shell_index, current in enumerate(ordered_shells):
        compiled = {
            key[-2] if isinstance(key, tuple) else key
            for key in current.captured_region_programs
        }
        for region_index, subgraph in enumerate(current.dispatch_subgraphs):
            if region_index in current.coordinator_region_indices:
                kind = "coordinator"
            elif region_index in compiled:
                kind = "shader"
            else:
                kind = "uncaptured"
            operations = " -> ".join(
                str(
                    subgraph.G.nodes[node_id].get("op")
                    or subgraph.G.nodes[node_id].get("type")
                )
                for node_id in subgraph.G.graph.get("deployment_nodes", ())
            )
            regions.append((
                shell_index,
                region_index,
                kind,
                len(subgraph.G.graph.get("deployment_inputs", ())),
                len(subgraph.G.graph.get("deployment_outputs", ())),
                len(subgraph.G.graph.get("compartment_schedule", ())),
                ",".join(subgraph.G.graph.get("rewrite_history", ())) or "-",
                operations or "-",
            ))
    lines.extend(("", "compiled program region compartments"))
    lines.extend(render_table(
        (
            "shell", "region", "kind", "in", "out", "waves", "identities",
            "operations",
        ),
        regions,
    ))
    loop_reductions = []
    for shell_index, current in enumerate(ordered_shells):
        for reduction in current.loop_shader_reductions:
            loop_reductions.append((
                shell_index,
                reduction.loop_node_id,
                ",".join(map(str, reduction.region_indices)) or "-",
                "collapse" if reduction.collapsible else "coordinator",
                (
                    "dynamic"
                    if reduction.estimated_dispatches_removed is None
                    else reduction.estimated_dispatches_removed
                ),
                ",".join(reduction.blockers) or "-",
                ",".join(
                    name for name, _initial, _updated
                    in reduction.carried_bindings
                ) or "-",
            ))
    if loop_reductions:
        lines.extend(("", "planner loop-to-shader reduction analysis"))
        lines.extend(render_table(
            (
                "shell", "loop", "regions", "verdict",
                "dispatches removed", "blockers", "carried",
            ),
            loop_reductions,
        ))
    return tuple(lines)
