"""Pooled C control emission: rendering rules and a compile-and-run proof.

The end-to-end test concatenates the emitted control shell, the
turing_pool runtime, and synthetic region functions into one translation
unit, builds it with the repository's own toolchain discovery, and runs the
control function from Python -- proving the emitted text really deploys
lanes through the pool and really falls back serially when the pool is
absent-shaped.
"""

from __future__ import annotations

import ctypes
import sys
from pathlib import Path

import pytest

from src.compiler.control_source import (
    ControlDeploymentLane,
    ControlDeploymentRegion,
    ControlProgram,
    LoopBlock,
    ParallelDeployment,
    SequenceBlock,
    StatementBlock,
)
from src.common.tensors.fused_ir import FusedProgram, OpStep
from src.compiler.deployment_native_emission import render_pooled_control_c
from src.compiler.deployment_stage import plan_region_deployments

_BACKEND_DIR = (
    Path(__file__).resolve().parents[1]
    / "src" / "common" / "tensors" / "accelerator_backends" / "c_backend"
)


def _program() -> ControlProgram:
    return ControlProgram(
        root=SequenceBlock((
            StatementBlock(("__scheduled_region_0__",)),
            ParallelDeployment((
                StatementBlock(("__scheduled_region_1__",)),
                StatementBlock(("__scheduled_region_2__",)),
            )),
        )),
        region_indices=(0, 1, 2),
    )


def test_pooled_render_declares_everything_it_uses():
    rendered = render_pooled_control_c(_program(), workers=2)
    assert not rendered.serial_only
    assert rendered.pooled_waves == 1
    for index in (0, 1, 2):
        assert f"extern void turing_region_{index}(void);" in rendered.source
    assert "turing_pool_start(2)" in rendered.source
    assert "turing_pool_deploy_span(turing_deploy_wave_0_span" in rendered.source
    # The serial fallback for the wave is inline, in recorded order.
    assert "turing_region_1();" in rendered.source
    assert "turing_region_2();" in rendered.source


def test_rich_lane_falls_back_to_the_established_serial_renderer():
    program = ControlProgram(
        root=ParallelDeployment((
            StatementBlock(("__scheduled_region_0__",)),
            LoopBlock(
                "i", 0, 4, 1, StatementBlock(("__scheduled_region_1__",)),
            ),
        )),
        region_indices=(0, 1),
    )
    rendered = render_pooled_control_c(program)
    assert rendered.serial_only
    assert rendered.pooled_waves == 0
    assert any("unsupported control shape" in note for note in rendered.notes)
    assert "turing_pool_deploy" not in rendered.source


def test_purely_serial_program_renders_without_pool_plumbing():
    program = ControlProgram(
        root=StatementBlock(("__scheduled_region_0__",)),
        region_indices=(0,),
    )
    rendered = render_pooled_control_c(program)
    assert rendered.serial_only
    assert "turing_pool" not in rendered.source


_REGION_HARNESS = """
#include <stddef.h>
static long region_hits[3];
static void* region_threads[3];
static long span_hits[17];
/* No __declspec(dllexport) anywhere: with none present, mingw exports every
   non-static symbol, including the pool API the teardown calls. */
long turing_test_region_hits(int index) { return region_hits[index]; }
int turing_test_distinct_threads(void) {
    return region_threads[1] != region_threads[2];
}
static void turing_test_span_fn(void* context, long start, long stop) {
    (void)context;
    for (long index = start; index < stop; ++index) span_hits[index]++;
}
int turing_test_span(int chunk) {
    for (long index = 0; index < 17; ++index) span_hits[index] = 0;
    return turing_pool_deploy_span(turing_test_span_fn, 0, 17, chunk);
}
long turing_test_span_hit(int index) { return span_hits[index]; }
static void* current_thread(void);
void turing_region_0(void) { region_hits[0]++; region_threads[0] = current_thread(); }
void turing_region_1(void) { region_hits[1]++; region_threads[1] = current_thread(); }
void turing_region_2(void) { region_hits[2]++; region_threads[2] = current_thread(); }
#ifdef _WIN32
#include <windows.h>
static void* current_thread(void) { return (void*)(size_t)GetCurrentThreadId(); }
#else
#include <pthread.h>
static void* current_thread(void) { return (void*)pthread_self(); }
#endif
"""


@pytest.fixture(scope="module")
def compiled_control(tmp_path_factory):
    from src.common.tensors.accelerator_backends.native_library import (
        compile_shared_library,
        preferred_toolchain,
    )

    if preferred_toolchain() is None:
        pytest.skip(
            "no native toolchain found (native_library.detect_toolchains)"
        )
    rendered = render_pooled_control_c(_program(), workers=2)
    header = (_BACKEND_DIR / "turing_pool.h").read_text(encoding="utf-8")
    pool = (_BACKEND_DIR / "turing_pool.c").read_text(encoding="utf-8")
    pool = pool.replace('#include "turing_pool.h"', header)
    unit = "\n".join((pool, _REGION_HARNESS, rendered.source))
    extra_flags = () if sys.platform == "win32" else ("-pthread",)
    library_path, _toolchain = compile_shared_library(
        unit,
        name="pooled_control_test",
        directory=tmp_path_factory.mktemp("pooled_control"),
        extra_flags=extra_flags,
    )
    library = ctypes.CDLL(str(library_path))
    library.turing_test_region_hits.restype = ctypes.c_long
    library.turing_test_region_hits.argtypes = [ctypes.c_int]
    yield library
    library.turing_pool_stop()


def test_emitted_control_runs_every_region_exactly_once_per_call(
    compiled_control,
):
    compiled_control.turing_control()
    assert [
        compiled_control.turing_test_region_hits(index) for index in (0, 1, 2)
    ] == [1, 1, 1]
    compiled_control.turing_control()  # pool persists across calls
    assert [
        compiled_control.turing_test_region_hits(index) for index in (0, 1, 2)
    ] == [2, 2, 2]


def test_native_span_chunk_visits_every_item_exactly_once(compiled_control):
    assert compiled_control.turing_test_span(4) == 0
    assert [compiled_control.turing_test_span_hit(i) for i in range(17)] == [
        1
    ] * 17


def _region(index: int) -> FusedProgram:
    return FusedProgram(
        version=1,
        feeds={index * 2 + 1},
        steps=[OpStep(
            step_id=0, op_name="add",
            input_ids=[index * 2 + 1], result_id=index * 2 + 2,
        )],
        outputs={"out": index * 2 + 2},
    )


def _planned_wave(count: int, *, cores: int = 2):
    deployment = ControlDeploymentRegion(
        region_id=7, kind="parallel_candidate",
        schedule="independent_lanes",
        lanes=tuple(
            ControlDeploymentLane(index=i, region_indices=(i,))
            for i in range(count)
        ),
    )
    plan = plan_region_deployments(
        {i: _region(i) for i in range(count)},
        deployment_regions=(deployment,), cores=cores,
    )
    program = ControlProgram(
        root=ParallelDeployment(tuple(
            StatementBlock((f"__scheduled_region_{i}__",))
            for i in range(count)
        )),
        region_indices=tuple(range(count)),
        deployment_regions=(deployment,),
    )
    return program, plan


def test_frame_plan_workers_and_chunk_are_literal_in_native_source():
    program, plan = _planned_wave(40, cores=2)
    rendered = render_pooled_control_c(program, deployment_plan=plan)

    # 40 lanes / (2 workers * 4 claims each) => five lanes per claim.
    assert "turing_pool_start(2)" in rendered.source
    assert "turing_deploy_wave_0_span, 0, 40, 5" in rendered.source
    assert rendered.deployment_record[0]["workers"] == 2
    assert rendered.deployment_record[0]["chunk_size"] == 5
