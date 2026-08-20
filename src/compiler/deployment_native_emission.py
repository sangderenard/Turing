"""Emit pooled C control shells: parallel waves become native span deploys.

The native answer to the browser's tile workers.  ``render_control_block``
flattens a ``ParallelDeployment`` into consecutive region calls -- correct,
serial.  This module renders the same control program with each provable
wave handed to the persistent native pool (``turing_pool.c``) instead:

    static void turing_deploy_wave_0_span(void*, long start, long stop) {
        for (long lane = start; lane < stop; ++lane) { ... }
    }
    ...
    if (turing_pool_deploy_span(turing_deploy_wave_0_span, 0, 2, 1) != 0) {
        turing_region_1();      /* serial fallback, inline */
        turing_region_2();
    }

Because every region function is a nullary ``void turing_region_N(void)``
operating on slot state, a span trampoline can dispatch consecutive planned
lanes without changing region ABIs. Workers and chunk size come from the
frame-level deployment decision and are literal facts in the emitted source.

Provability gate: a wave is pooled only when every lane is purely scheduled
region calls (the exact shape ``partition_threaded_wasm_program`` and loop
evaporation produce).  Any richer lane -- loops, validation, publishes --
sends the whole render to the serial path with a note, mirroring the
conservative projection ``wasm_class_coordinator`` applies for the browser.
The emitted text always carries its own serial fallback: if the pool fails
to start or a deploy is refused, the wave runs inline in recorded order,
so linking ``turing_pool.c`` is an optimization, never an obligation the
program's correctness hangs on.

The renderer and ``turing_pool_deploy_span`` have compile-and-run proofs in
``tests/test_deployment_native_emission.py``. Product shells still need to
select this renderer at their control-source adoption point; until that seam
is connected, manifests carry the decision but existing products remain on
their established serial control renderer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .control_source import (
    ControlProgram,
    ParallelDeployment,
    SequenceBlock,
    StatementBlock,
)

_REGION_MARKER = re.compile(r"__scheduled_region_(\d+)__")

_POOL_DECLARATIONS = (
    "typedef void (*turing_region_fn)(void);",
    "typedef void (*turing_span_fn)(void*, long, long);",
    "extern int turing_pool_start(int workers);",
    "extern int turing_pool_deploy(void (*fn)(void*, long, long, long), "
    "void* context, long lane_count, long chunks_per_lane);",
    "extern int turing_pool_deploy_span(turing_span_fn fn, void* context, "
    "long item_count, long chunk_size);",
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
    deployment_record: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class NativeWaveChoice:
    """One control wave after plan compatibility and ABI checks."""

    wave: int
    region_indices: tuple[int, ...]
    pooled: bool
    workers: int | None
    chunk_size: int
    planned_chunks: tuple[int | None, ...]
    reasons: tuple[str, ...]

    def as_record(self) -> dict[str, Any]:
        return {
            "wave": self.wave,
            "region_indices": list(self.region_indices),
            "pooled": self.pooled,
            "workers": self.workers,
            "chunk_size": self.chunk_size,
            "planned_chunks": list(self.planned_chunks),
            "reasons": list(self.reasons),
        }


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


def _parallel_waves(block) -> tuple[ParallelDeployment, ...]:
    if isinstance(block, ParallelDeployment):
        return (block,)
    if isinstance(block, SequenceBlock):
        return tuple(
            wave for child in block.blocks for wave in _parallel_waves(child)
        )
    return ()


def adapt_region_deployment_plan(
    program: ControlProgram,
    plan: Any,
    *,
    backend: str = "c",
) -> tuple[NativeWaveChoice, ...]:
    """Project region choices onto the nullary native control-wave ABI.

    Frame choices are authoritative here: their work items are independent
    lanes and ``chunk`` is lanes per native span claim.  Per-region choices
    describe how each body is served and are intentionally not reused as a
    frame schedule.
    """

    choices: list[NativeWaveChoice] = []
    for wave_index, wave in enumerate(_parallel_waves(program.root)):
        lanes = [_lane_regions(lane) for lane in wave.lanes]
        flattened = tuple(
            region for lane in lanes if lane is not None for region in lane
        )
        reasons: list[str] = []
        wave_decision = None
        wave_choice = None
        if any(lane is None for lane in lanes) or len(lanes) < 2:
            reasons.append(
                "wave is not two or more lanes of nullary scheduled regions"
            )
        else:
            wave_decision = plan.wave_for_lanes(lanes)
            wave_choice = (
                wave_decision.choice_for(backend)
                if wave_decision is not None else None
            )
        if wave_decision is None:
            reasons.append("control wave has no matching frame decision")
        if (
            wave_choice is None
            or wave_choice.strategy != "pool"
            or wave_choice.join_mode != "barrier"
        ):
            reasons.append(
                "the matching frame must choose a barrier C pool"
            )
        worker_value = (
            None if wave_choice is None or wave_choice.workers is None
            else int(wave_choice.workers)
        )
        chunk_value = (
            1 if wave_choice is None or wave_choice.chunk is None
            else max(1, int(wave_choice.chunk))
        )
        choices.append(NativeWaveChoice(
            wave=wave_index,
            region_indices=flattened,
            pooled=not reasons,
            workers=worker_value,
            chunk_size=chunk_value,
            planned_chunks=(
                None if wave_choice is None else wave_choice.chunk,
            ),
            reasons=tuple(reasons),
        ))

    # turing_pool_start grows one process-global pool and cannot cap workers
    # per frame.  Different wave budgets therefore cannot both be honored in
    # one product; retain the first and serialize incompatible waves.
    global_workers = {
        choice.workers for choice in choices
        if choice.pooled and choice.workers is not None
    }
    if len(global_workers) > 1:
        expected = min(global_workers)
        choices = [
            choice if not choice.pooled or choice.workers == expected
            else NativeWaveChoice(
                wave=choice.wave,
                region_indices=choice.region_indices,
                pooled=False,
                workers=choice.workers,
                chunk_size=choice.chunk_size,
                planned_chunks=choice.planned_chunks,
                reasons=(*choice.reasons,
                         "process-global pool cannot honor a different "
                         f"wave budget than {expected}"),
            )
            for choice in choices
        ]
    return tuple(choices)


def render_pooled_control_c(
    program: ControlProgram,
    *,
    function_name: str = "turing_control",
    workers: int | None = None,
    deployment_plan: Any | None = None,
    backend: str = "c",
) -> PooledControlSource:
    """Render the control program as C with provable waves pooled.

    ``workers`` is the pool size to start (a calibrated best when a
    verdict exists); ``None`` uses ``DEFAULT_POOL_WORKERS``.
    """

    if not function_name.isidentifier():
        raise ValueError(f"invalid control function name {function_name!r}")
    native_choices = (
        adapt_region_deployment_plan(program, deployment_plan, backend=backend)
        if deployment_plan is not None else ()
    )
    planned_workers = {
        choice.workers for choice in native_choices
        if choice.pooled and choice.workers is not None
    }
    if workers is not None and planned_workers and planned_workers != {int(workers)}:
        raise ValueError(
            "manual workers contradict the consumed deployment plan: "
            f"{workers} versus {sorted(planned_workers)}"
        )
    worker_count = (
        int(workers) if workers is not None
        else next(iter(planned_workers)) if planned_workers
        else DEFAULT_POOL_WORKERS
    )
    notes: list[str] = []
    body: list[str] = []
    tables: list[str] = []
    declared_regions: set[int] = set()
    pooled_waves = 0
    wave_cursor = 0

    def serial_call_lines(indices: tuple[int, ...]) -> list[str]:
        declared_regions.update(indices)
        return [f"turing_region_{index}();" for index in indices]

    def emit_block(block) -> bool:
        """Emit one top-level block; False if the shape is unsupported."""

        nonlocal pooled_waves, wave_cursor
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
            wave = wave_cursor
            wave_cursor += 1
            flattened = [index for lane in lanes for index in lane]
            declared_regions.update(flattened)
            native_choice = (
                native_choices[wave] if deployment_plan is not None else None
            )
            if native_choice is not None and not native_choice.pooled:
                notes.extend(native_choice.reasons)
                body.extend(serial_call_lines(tuple(flattened)))
                return True
            pooled_waves += 1
            lane_bodies = []
            for lane in lanes:
                calls = " ".join(
                    f"turing_region_{index}();" for index in lane
                )
                lane_bodies.append(calls)
            tables.extend((
                f"static void turing_deploy_wave_{wave}_span(void* context, "
                "long start, long stop) {",
                "    (void)context;",
                "    for (long position = start; position < stop; ++position) {",
                "    switch (position) {",
                *(
                    f"    case {position}: {calls} break;"
                    for position, calls in enumerate(lane_bodies)
                ),
                "    }",
                "    }",
                "}",
            ))
            body.extend((
                f"if (turing_pool_deploy_span(turing_deploy_wave_{wave}_span, 0, "
                f"{len(lanes)}, "
                f"{native_choice.chunk_size if native_choice else 1}) "
                "!= 0) {",
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
            deployment_record=tuple(
                choice.as_record() for choice in native_choices
            ),
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
        deployment_record=tuple(
            choice.as_record() for choice in native_choices
        ),
    )


__all__ = [
    "DEFAULT_POOL_WORKERS",
    "NativeWaveChoice",
    "PooledControlSource",
    "adapt_region_deployment_plan",
    "render_pooled_control_c",
]
