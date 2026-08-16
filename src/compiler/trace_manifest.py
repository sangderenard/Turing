"""Companion data describing how a trace resolves, at every level.

A running artifact can only afford to write down the cheapest true thing it
knows: an integer, stamped into the launch site when the backend was built.
That integer is not meaningful on its own, and it must not have to be -- the
alternative is either making the runtime carry names it would have to format,
or hoping two independently assigned numberings happen to agree, which fails
silently and animates the wrong part of the program.

So the correspondence is recorded here instead, at compile time, beside the
Dual IR it describes. A consumer holding a trace record joins through this
table to whichever representation it wants to talk about:

* ``dependency`` -- ProcessGraph value ids, where the program is still a
  dependency structure and nothing has been scheduled;
* ``dual_ir`` -- the region index the control shell dispatches through;
* ``ssa`` -- the value ids the region's steps produce once lowered.

None of those is more correct than the others; they are the same computation
described three ways, and which one a viewer wants is a question about the
viewer. Recording all three costs a table built once and answers the question
without the runtime knowing any of them.

The table is built only when diagnostics are asked for. It is companion data,
not part of the artifact: nothing in the compiled program reads it, and its
absence changes nothing about what runs.
"""

from __future__ import annotations

from typing import Any, Mapping

SCHEMA = "turing-trace-manifest-v1"


def build_trace_manifest(
    *,
    region_programs: Mapping[int, Any],
    control_program: Any = None,
    identity_table: Mapping[str, Any] | None = None,
    entrypoint: str = "",
) -> dict[str, Any]:
    """Describe every launch site and how its identity resolves.

    One site per region: a region is what the control shell dispatches, and a
    dispatch is what the launch boundary times, so they are the same event seen
    from two sides. ``site`` is the integer the artifact writes; it is assigned
    here so the manifest defines the numbering rather than discovering it.
    """

    sites: list[dict[str, Any]] = []
    dependency: dict[int, tuple[int, ...]] = {}
    dual_ir: dict[int, int] = {}
    ssa: dict[int, tuple[int, ...]] = {}

    for site, index in enumerate(sorted(region_programs)):
        captured = region_programs[index]
        program = getattr(captured, "program", captured)
        steps = tuple(getattr(program, "steps", ()) or ())
        feeds = tuple(sorted(int(value) for value in
                             (getattr(program, "feeds", ()) or ())))
        produced = tuple(int(step.result_id) for step in steps)

        sites.append({
            "site": site,
            "region": int(index),
            "steps": len(steps),
            "feeds": len(feeds),
        })
        # A region's feeds are the values it consumes from the surrounding
        # dependency structure, so they are what a trace points at when the
        # question is "which part of the source graph just ran".
        dependency[site] = feeds
        dual_ir[site] = int(index)
        # Lowering preserves value ids, so a region's results name the same
        # values in SSA that they do here.
        ssa[site] = produced

    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "entrypoint": str(entrypoint),
        "sites": tuple(sites),
        "levels": {
            "dependency": dependency,
            "dual_ir": dual_ir,
            "ssa": ssa,
        },
    }

    if control_program is not None:
        # Recorded so a consumer can tell whether the numbering it holds came
        # from the same control shell, rather than assuming it did.
        manifest["region_indices"] = tuple(
            int(value) for value in
            (getattr(control_program, "region_indices", ()) or ())
        )
    if identity_table:
        # Source names for the values a trace can land on, so a viewer can say
        # "loss" instead of "value 12" without the runtime carrying strings.
        manifest["names"] = {
            str(name): tuple(int(value) for value in ids)
            for name, ids in identity_table.items()
        }
    return manifest


def resolve(manifest: Mapping[str, Any], site: int, level: str = "dual_ir"):
    """Look up what a traced site corresponds to at one level."""

    levels = manifest.get("levels") or {}
    if level not in levels:
        raise KeyError(
            f"unknown trace level {level!r}; one of {sorted(levels)}"
        )
    return levels[level].get(int(site))


__all__ = ["SCHEMA", "build_trace_manifest", "resolve"]
