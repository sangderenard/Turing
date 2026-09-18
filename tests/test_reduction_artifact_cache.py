"""Incremental final reductions: each planner region is lowered to its own
WebAssembly kernel once, then reloaded from a durable content-addressed store
instead of being re-emitted.

This is the resumable-bake substrate. The whole-program checkpoint saves one
monolithic plan object all-or-nothing; the reductions are many small
independent kernels, and an interrupted bake used to discard every one it had
already produced. ``ReductionArtifactStore`` persists each lowered region under
a key derived from its own content, so a rerun reuses finished regions and only
recomputes what actually changed.
"""
from __future__ import annotations

import tempfile

from src.common.tensors.fused_ir import FusedProgram, OpStep
from src.compiler.control_source import ControlProgram, LoopBlock, StatementBlock
from src.compiler.wasm_class_modules import emit_control_region_modules
from src.common.tensors.accelerator_backends.aot_checkpoint import (
    ReductionArtifactStore,
    callable_digest,
)


def _fixture():
    program = FusedProgram(
        version=1,
        feeds={1},
        steps=[OpStep(0, "mul", [1], {"right_scalar": 2.0}, 100)],
        outputs={"result": 100},
        extras={"capture_feed_origins": {1: {"binding_name": "x"}}},
    )
    control = ControlProgram(
        LoopBlock(
            "iteration_0", "0", "3", "1",
            StatementBlock(("__scheduled_region_0__",)),
        ),
        region_indices=(0,),
    )
    return control, {0: program}


def test_region_reduction_is_lowered_once_then_reused():
    control, region_programs = _fixture()
    root = tempfile.mkdtemp(prefix="reduxtest-")
    store = ReductionArtifactStore(
        callable_digest(emit_control_region_modules), root=root
    )
    seen: list[tuple[int, bool]] = []

    first, _ = emit_control_region_modules(
        control, region_programs, owner_name="loop_kernel", module_dir=".",
        reduction_cache=store, progress=lambda r, c: seen.append((r, c)),
    )
    second, _ = emit_control_region_modules(
        control, region_programs, owner_name="loop_kernel", module_dir=".",
        reduction_cache=store, progress=lambda r, c: seen.append((r, c)),
    )

    # First pass lowers the region (miss); second pass reuses it (hit).
    assert seen == [(0, False), (0, True)]
    assert (store.misses, store.hits) == (1, 1)
    # The reused kernel is byte-for-byte the one first lowered.
    assert first[0].binary == second[0].binary


def test_no_cache_argument_preserves_existing_behavior():
    """Callers that pass no ``reduction_cache`` still emit exactly as before."""

    control, region_programs = _fixture()
    modules, manifest = emit_control_region_modules(
        control, region_programs, owner_name="loop_kernel", module_dir=".",
    )
    assert set(modules) == {0}
    assert modules[0].complete
