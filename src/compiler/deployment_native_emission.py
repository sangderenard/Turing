"""Emit pooled C control shells: parallel waves become turing_pool deploys.

The native answer to the browser's tile workers.  ``render_control_block``
flattens a ``ParallelDeployment`` into consecutive region calls -- correct,
serial.  This module renders the same control program with each provable
wave handed to the persistent native pool (``turing_pool.c``) instead:

    static void turing_deploy_lane_0(void*, long lane, long, long) {
        turing_deploy_lane_table_0[lane]();
    }
    ...
    if (turing_pool_deploy(turing_deploy_lane_0, 0, 2, 1) != 0) {
        turing_region_1();      /* serial fallback, inline */
        turing_region_2();
    }

Because every region function is a nullary ``void turing_region_N(void)``
operating on slot state, lane-level parallelism needs no new calling
convention -- a function-pointer table is the whole trampoline.  Chunk
splitting *within* a region does need a context argument and is the
declared next step, not smuggled in here.

Provability gate: a wave is pooled only when every lane is purely scheduled
region calls (the exact shape ``partition_threaded_wasm_program`` and loop
evaporation produce).  Any richer lane -- loops, validation, publishes --
sends the whole render to the serial path with a note, mirroring the
conservative projection ``wasm_class_coordinator`` applies for the browser.
The emitted text always carries its own serial fallback: if the pool fails
to start or a deploy is refused, the wave runs inline in recorded order,
so linking ``turing_pool.c`` is an optimization, never an obligation the
program's correctness hangs on.

Adoption point: ``fortran_c_shell``/``profiled_c_shell`` swap their
``reduce_control_ir(..., C)`` call for ``render_pooled_control_c`` when the
deployment plan chooses the pool for the wasm... for the *c* backend; until
then this module stands alone with its compile-and-run proof in
``tests/test_deployment_native_emission.py``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .control_source import (
    ControlProgram,
    ParallelDeployment,
    SequenceBlock,
    StatementBlock,
)

_REGION_MARKER = re.compile(r"__scheduled_region_(\d+)__")

_POOL_DECLARATIONS = (
    "typedef void (*turing_region_fn)(void);",
    "extern int turing_pool_start(int workers);",
    "extern int turing_pool_deploy(void (*fn)(void*, long, long, long), "
    "void* context, long lane_count, long chunks_per_lane);",
)

DEFAULT_POOL_WORKERS = 7  # min(8, cores-1) spirit; calibration overrides.


@dataclass(frozen=True)
class PooledControlSource:
    """Rendered C control shell, with the pooling evidence attached."""

    function_name: str
    source: str
    region_indices: tuple[int, ...]
    pooled_waves: int
    serial_only: bool
    notes: tuple[str, ...]


def _lane_regions(block) -> tuple[int, ...] | None:
    """Region indices a lane calls, or ``None`` if the lane is richer."""

    if isinstance(block, StatementBlock):
        indices = []
        for line in block.lines:
            match = _REGION_MARKER.fullmatch(str(line).strip())
            if match is None:
                return None
            indices.append(int(match.group(1)))
        return tuple(indices)
    if isinstance(block, SequenceBlock):
        collected: list[int] = []
        for child in block.blocks:
            inner = _lane_regions(child)
            if inner is None:
                return None
            collected.extend(inner)
        return tuple(collected)
    return None


def render_pooled_control_c(
    program: ControlProgram,
    *,
    function_name: str = "turing_control",
    workers: int | None = None,
) -> PooledControlSource:
    """Render the control program as C with provable waves pooled.

    ``workers`` is the pool size to start (a calibrated best when a
    verdict exists); ``None`` uses ``DEFAULT_POOL_WORKERS``.
    """

    if not function_name.isidentifier():
        raise ValueError(f"invalid control function name {function_name!r}")
    worker_count = DEFAULT_POOL_WORKERS if workers is None else int(workers)
    notes: list[str] = []
    body: list[str] = []
    tables: list[str] = []
    declared_regions: set[int] = set()
    pooled_waves = 0

    def serial_call_lines(indices: tuple[int, ...]) -> list[str]:
        declared_regions.update(indices)
        return [f"turing_region_{index}();" for index in indices]

    def emit_block(block) -> bool:
        """Emit one top-level block; False if the shape is unsupported."""

        nonlocal pooled_waves
        if isinstance(block, StatementBlock):
            indices = _lane_regions(block)
            if indices is None:
                return False
            body.extend(serial_call_lines(indices))
            return True
        if isinstance(block, SequenceBlock):
            return all(emit_block(child) for child in block.blocks)
        if isinstance(block, ParallelDeployment):
            lanes = [_lane_regions(lane) for lane in block.lanes]
            if any(lane is None for lane in lanes) or len(lanes) < 2:
                return False
            wave = pooled_waves
            pooled_waves += 1
            flattened = [index for lane in lanes for index in lane]
            declared_regions.update(flattened)
            lane_bodies = []
            for lane in lanes:
                calls = " ".join(
                    f"turing_region_{index}();" for index in lane
                )
                lane_bodies.append(calls)
            tables.extend((
                f"static void turing_deploy_wave_{wave}_lane(void* context, "
                "long lane, long chunk, long chunks_per_lane) {",
                "    (void)context; (void)chunk; (void)chunks_per_lane;",
                "    switch (lane) {",
                *(
                    f"    case {position}: {calls} break;"
                    for position, calls in enumerate(lane_bodies)
                ),
                "    }",
                "}",
            ))
            body.extend((
                f"if (turing_pool_deploy(turing_deploy_wave_{wave}_lane, 0, "
                f"{len(lanes)}, 1) != 0) {{",
                *(
                    f"    turing_region_{index}();" for index in flattened
                ),
                "}",
            ))
            return True
        return False

    supported = emit_block(program.root)
    if not supported or pooled_waves == 0:
        # Serial path: either an unsupported shape (note why) or nothing
        # parallel to pool.  Both render through the established renderer
        # so the serial text stays byte-identical to today's.
        from .control_source import ControlTarget, render_control_block

        if not supported:
            notes.append(
                "unsupported control shape for pooling (a lane is richer "
                "than scheduled region calls); rendered serial"
            )
        else:
            notes.append("no multi-lane waves; rendered serial")
        lines = render_control_block(program.root, ControlTarget.C)
        source = "\n".join((
            f"void {function_name}(void) {{",
            *(f"    {line}" if line else "" for line in lines),
            "}",
            "",
        ))
        return PooledControlSource(
            function_name=function_name,
            source=source,
            region_indices=tuple(program.region_indices),
            pooled_waves=0,
            serial_only=True,
            notes=tuple(notes),
        )

    region_declarations = [
        f"extern void turing_region_{index}(void);"
        for index in sorted(declared_regions)
    ]
    source = "\n".join((
        *_POOL_DECLARATIONS,
        *region_declarations,
        "",
        *tables,
        "",
        f"void {function_name}(void) {{",
        "    static int turing_pool_ready = 0;",
        "    if (!turing_pool_ready) {",
        f"        turing_pool_start({worker_count});",
        "        turing_pool_ready = 1;",
        "    }",
        *(f"    {line}" for line in body),
        "}",
        "",
    ))
    return PooledControlSource(
        function_name=function_name,
        source=source,
        region_indices=tuple(program.region_indices),
        pooled_waves=pooled_waves,
        serial_only=False,
        notes=tuple(notes),
    )


__all__ = [
    "DEFAULT_POOL_WORKERS",
    "PooledControlSource",
    "render_pooled_control_c",
]
