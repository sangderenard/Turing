from __future__ import annotations

import numpy as np
import pytest

from src.common.tensors.accelerator_backends.aot_compile import compile_ast_aot
from src.common.tensors.fused_ir import ordered_feed_ids
from src.compiler.backend_sources import collect_backend_sources
from src.compiler.fortran_fidelity import verify_fortran_module
from src.compiler.ssa_fortran_backend import FortranModule, fortran_compiler


SOURCE = """
def kernel(x, gain):
    return x * gain + 1.0
"""


def _fortran_artifact():
    named_feeds = {
        "x": np.asarray([-2.0, -0.25, 0.5, 3.0]),
        "gain": np.asarray([0.5, 2.0, -1.5, 4.0]),
    }
    aot = compile_ast_aot(
        SOURCE, "kernel", named_feeds, precompile_only=True
    )
    program = getattr(
        aot.compiled_shell_program, "program", aot.compiled_shell_program
    )
    sources = collect_backend_sources(
        aot,
        numerical_name="kernel",
        control_name="kernel_control",
        program=program,
    )
    artifact = next(
        source.artifact
        for source in sources.sources
        if source.language == "fortran"
    )
    origins = program.extras["capture_feed_origins"]
    feeds = {
        feed_id: named_feeds[origins[feed_id]["binding_name"]]
        for feed_id in ordered_feed_ids(program)
    }
    return artifact, program, feeds


@pytest.mark.skipif(
    fortran_compiler() is None, reason="no Fortran compiler installed"
)
def test_ast_generated_fortran_matches_reference_across_recorded_cases(tmp_path):
    artifact, program, feeds = _fortran_artifact()

    proof = verify_fortran_module(
        artifact, program, feeds, tmp_path, entrypoint="kernel"
    )

    assert proof["passed"] is True
    assert proof["case_count"] == 3
    assert all(case["passed"] for case in proof["cases"])
    assert max(
        output["max_absolute_error"]
        for case in proof["cases"]
        for output in case["outputs"]
    ) == 0.0


@pytest.mark.skipif(
    fortran_compiler() is None, reason="no Fortran compiler installed"
)
def test_fidelity_check_rejects_a_compilable_wrong_fortran_program(tmp_path):
    artifact, program, feeds = _fortran_artifact()
    corrupted_source = artifact.source.replace(
        "+ 1.0_c_double", "+ 2.0_c_double", 1
    )
    assert corrupted_source != artifact.source
    corrupted = FortranModule(
        artifact.name,
        corrupted_source,
        artifact.subroutines,
        api=artifact.api,
    )

    with pytest.raises(AssertionError, match="disagrees with the reference"):
        verify_fortran_module(
            corrupted, program, feeds, tmp_path, entrypoint="kernel"
        )
