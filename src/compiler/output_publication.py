"""Backend-neutral semantic contract for compiled program outputs."""

from __future__ import annotations

from typing import Any, Iterable, Mapping


SCHEMA = "turing.semantic-output-publications.v1"
SURFACE_SCHEMA = "turing.semantic-output-surfaces.v1"


def function_output_publications(function: Any) -> tuple[Mapping[str, Any], ...]:
    """Validate and normalize semantic publications authored on one SSA function."""

    metadata = dict(getattr(function, "metadata", {}) or {})
    output_names = set(map(str, metadata.get("output_names", ())))
    output_names.update(
        str(name) for name, _value_id in metadata.get("named_outputs", ())
    )
    normalized = []
    for row in metadata.get("publications", ()):
        item = dict(row)
        output = str(item.get("output", ""))
        if not output:
            raise ValueError(f"{function.name} has a publication without an output")
        if output not in output_names:
            raise ValueError(
                f"{function.name} publishes unknown output {output!r}"
            )
        semantic = str(item.get("semantic", ""))
        presentation = str(item.get("presentation", ""))
        if not semantic or not presentation:
            raise ValueError(
                f"{function.name}.{output} publication needs semantic and presentation"
            )
        normalized.append({
            "entry_point": str(function.name),
            "output": output,
            "semantic": semantic,
            "presentation": presentation,
            "unit": None if item.get("unit") is None else str(item["unit"]),
        })
    return tuple(normalized)


def module_output_publications(
    functions: Mapping[str, Any] | Iterable[Any],
) -> tuple[Mapping[str, Any], ...]:
    """Collect the same publication rows for any emitted target lane."""

    values = functions.values() if isinstance(functions, Mapping) else functions
    return tuple(
        publication
        for function in values
        for publication in function_output_publications(function)
    )


def publication_metadata(
    functions: Mapping[str, Any] | Iterable[Any],
    *,
    target: str = "native",
) -> Mapping[str, Any]:
    """CompiledProgramAPI metadata fragment consumed by host shells."""

    publications = module_output_publications(functions)
    return {
        "semantic_output_schema": SCHEMA,
        "semantic_outputs": publications,
        "semantic_output_surfaces": publication_surface_plan(
            publications, target=target,
        ),
    }


def publication_surface_plan(
    publications: Iterable[Mapping[str, Any]],
    *,
    target: str,
) -> Mapping[str, Any]:
    """Bind semantic outputs to a target-family shell surface.

    The simulation authors only semantic publications. This adapter chooses a
    shell presentation without inserting rendering calls into its ProcessGraph
    or repository SSA. Vector components sharing a semantic stem are grouped
    into one surface; scalar fields and metrics remain independently named.
    """

    target = str(target).casefold()
    aliases = {
        "c": "native", "llvm": "native", "fortran": "native",
        "wasm": "web", "webassembly": "web", "browser": "web",
        "json": "headless", "report": "headless",
    }
    family = aliases.get(target, target)
    if family not in {"native", "web", "headless"}:
        raise ValueError(f"unknown semantic-output target family {target!r}")
    rows = tuple(dict(row) for row in publications)
    vector_groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    singles = []
    for row in rows:
        if row.get("presentation") != "vector-component":
            singles.append(row)
            continue
        semantic = str(row["semantic"])
        stem, separator, component = semantic.rpartition(".")
        if not separator or component not in {"x", "y", "z", "u", "v", "w"}:
            singles.append(row)
            continue
        vector_groups.setdefault(
            (str(row["entry_point"]), stem), []
        ).append({**row, "component": component})

    adapters = {
        "native": {
            "field": "display.scalar-field",
            "vector": "display.vector-field",
            "metric": "console.live-metric",
        },
        "web": {
            "field": "webgl.scalar-field",
            "vector": "canvas.vector-field",
            "metric": "dom.live-metric",
        },
        "headless": {
            "field": "report.field-statistics",
            "vector": "report.vector-statistics",
            "metric": "report.metric",
        },
    }[family]
    surfaces = []
    for row in singles:
        presentation = str(row["presentation"])
        role = "metric" if presentation == "metric" else "field"
        surfaces.append({
            "id": f"{row['entry_point']}:{row['semantic']}",
            "entry_point": str(row["entry_point"]),
            "role": role,
            "adapter": adapters[role],
            "capability": (
                "diagnostic_stream" if role == "metric"
                else "display_double_buffer"
            ),
            "outputs": (str(row["output"]),),
            "semantics": (str(row["semantic"]),),
            "unit": row.get("unit"),
        })
    for (entry_point, stem), components in vector_groups.items():
        ordered = sorted(
            components,
            key=lambda row: "xyzuvw".index(str(row["component"])),
        )
        surfaces.append({
            "id": f"{entry_point}:{stem}",
            "entry_point": entry_point,
            "role": "vector",
            "adapter": adapters["vector"],
            "capability": "display_double_buffer",
            "outputs": tuple(str(row["output"]) for row in ordered),
            "semantics": tuple(str(row["semantic"]) for row in ordered),
            "components": tuple(str(row["component"]) for row in ordered),
            "unit": next((row.get("unit") for row in ordered), None),
        })
    surfaces.sort(key=lambda row: (row["entry_point"], row["id"]))
    return {
        "schema": SURFACE_SCHEMA,
        "target_family": family,
        "surfaces": tuple(surfaces),
    }


__all__ = [
    "SCHEMA",
    "SURFACE_SCHEMA",
    "function_output_publications",
    "module_output_publications",
    "publication_metadata",
    "publication_surface_plan",
]
