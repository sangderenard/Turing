"""A load must not be re-read after an in-place store overwrites its source.

Found while compiling a Jacobi ``eigh`` kernel to LLVM.  The kernel emitted
with zero shortfalls, ran natively, and produced a genuinely orthogonal
similarity transform (its trace was preserved exactly), but its eigenvalues
were wrong and matched no sweep count of the same kernel run in Python.

The cause reduces to six lines.  A rotation reads two elements into locals and
then writes both back::

    x = a[k]
    y = a[n + k]
    a[k]     = 2.0 * x - 3.0 * y
    a[n + k] = x + 4.0 * y

The first store is correct.  The second is not: the native result is
``2x + y``, which is ``(2x - 3y) + 4y`` -- the *stored* value of ``a[k]``
substituted for ``x``.  So ``x`` was re-loaded from memory after ``a[k]`` had
been overwritten, instead of keeping the value it was bound to beforehand.

This matters well beyond ``eigh``: reading a pair, combining them, and writing
both back is the shape of every plane rotation, every in-place swap, and most
in-place linear algebra.  Any of them compiles cleanly and computes the wrong
thing.

Aliased in-place storage is deliberate in this tree, not accidental -- it is
where the speed comes from -- so the fix is to regulate the load, not to stop
aliasing.  This test pins the defect at its smallest size so the fix has
something exact to satisfy.
"""
from __future__ import annotations

import pathlib
import tempfile
import warnings

import numpy as np
import pytest

from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.ssa_llvm_backend import (
    compile_artifact,
    emit_ssa_function_to_llvm,
    prepare_artifact_execution,
)


PAIR_UPDATE = """
def pair_update(a, n):
    for k in range(n):
        x = a[k]
        y = a[n + k]
        a[k] = 2.0 * x - 3.0 * y
        a[n + k] = x + 4.0 * y
    return a
"""


def _run_native(source: str, entrypoint: str, name: str, arrays, scalars, read):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        module, _outputs, _exports = lower_ast_source_to_ssa(
            source, entrypoint, name=name
        )
    qualified = f"{name}__{entrypoint}"
    function = module.functions[qualified]
    parameters = dict(function.metadata["parameter_names"])

    artifact = emit_ssa_function_to_llvm(module, qualified)
    assert artifact.shortfalls == (), (
        "this kernel is expected to emit cleanly; the defect is in what it "
        f"computes, not in what it refuses: {artifact.shortfalls}"
    )
    directory = pathlib.Path(tempfile.mkdtemp())
    native = compile_artifact(artifact, directory=directory / name)

    feed = {parameters[key]: value.copy() for key, value in arrays.items()}
    feed.update(
        {parameters[key]: np.array([value]) for key, value in scalars.items()}
    )
    # Formals the lowering added beyond the authored signature (a dependent
    # inner-loop bound leaks its outer induction variable, for one). Feeding
    # them zero is what the working cases in this suite do.
    for formal in function.args:
        identifier = int(formal.id)
        if identifier not in feed:
            feed[identifier] = (
                np.array([0]) if formal.dtype == "int" else np.array([0.0])
            )

    execution = prepare_artifact_execution(native, feed)
    execution.run()
    return np.asarray(execution.buffers[parameters[read]]).copy()


def _pair_update_reference(a: np.ndarray, n: int) -> np.ndarray:
    out = a.copy()
    for k in range(n):
        x = out[k]
        y = out[n + k]
        out[k] = 2.0 * x - 3.0 * y
        out[n + k] = x + 4.0 * y
    return out


def test_a_simple_indexed_loop_still_executes_correctly():
    """The control case: without an aliasing store, native agrees exactly."""

    source = """
def total(a, n):
    acc = 0.0
    for i in range(n):
        acc = acc + a[i]
    return acc
"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        module, outputs, _ = lower_ast_source_to_ssa(source, "total", name="ctl")
    function = module.functions["ctl__total"]
    parameters = dict(function.metadata["parameter_names"])
    artifact = emit_ssa_function_to_llvm(module, "ctl__total")
    native = compile_artifact(
        artifact, directory=pathlib.Path(tempfile.mkdtemp()) / "ctl"
    )
    values = np.arange(1.0, 6.0)
    feed = {parameters["a"]: values.copy(), parameters["n"]: np.array([5])}
    for formal in function.args:
        identifier = int(formal.id)
        if identifier not in feed:
            feed[identifier] = (
                np.array([0]) if formal.dtype == "int" else np.array([0.0])
            )
    execution = prepare_artifact_execution(native, feed)
    execution.run()
    produced = float(
        np.asarray(execution.buffers[int(outputs["ctl__total"][0].id)]).reshape(-1)[0]
    )
    assert produced == pytest.approx(values.sum())


N = 4
VALUES = np.arange(1.0, 2 * N + 1.0)


@pytest.fixture(scope="module")
def pair_update_native():
    """Compile the kernel once.

    Deliberately module-scoped: each compile spawns zig, and compiling this
    same artifact twice in quick succession inside one session raced its
    compiler_rt cache ("failed to check cache ... file_open Unexpected").
    That is a toolchain contention hazard this repository already records, not
    a property of the defect, so it is avoided rather than measured.
    """

    return _run_native(
        PAIR_UPDATE, "pair_update", "alias",
        {"a": VALUES}, {"n": N}, "a",
    )


def test_a_value_read_before_an_inplace_store_survives_it(pair_update_native):
    assert np.allclose(
        pair_update_native, _pair_update_reference(VALUES, N)
    )


def test_the_second_store_uses_the_operand_it_was_written_with(pair_update_native):
    """Not merely 'right': right in the one specific way that was wrong.

    The defect produced ``2x + y``, which is ``(2x - 3y) + 4y`` -- the
    already-stored ``a[k]`` substituted for ``x``. Asserting against that
    exact value, rather than only against the correct one, is what keeps this
    a regression test for THIS defect instead of a general smoke test that
    some future unrelated change could satisfy by accident.
    """

    x = VALUES[:N]
    y = VALUES[N:]
    assert np.allclose(pair_update_native[:N], 2.0 * x - 3.0 * y)
    assert np.allclose(pair_update_native[N:], x + 4.0 * y)
    # The old wrong answer, named so it cannot come back unnoticed.
    assert not np.allclose(pair_update_native[N:], 2.0 * x + y)


def test_the_loaded_value_is_pinned_rather_than_re_read(pair_update_native):
    """The emitted IR must load each element once, not once per use.

    The behavioural tests above would also pass if a later pass happened to
    hide the duplicate load; this asserts the actual shape of the emission, so
    the fix is checked where it was made.
    """

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        module, _outputs, _exports = lower_ast_source_to_ssa(
            PAIR_UPDATE, "pair_update", name="ir"
        )
    ir = emit_ssa_function_to_llvm(module, "ir__pair_update").llvm_ir
    region = ir.split("define void @__ssa_ir__pair_update__planned_region_0")[1]
    region = region.split("\n}")[0]

    # Two elements are read (a[k] and a[n+k]); each is loaded from its array
    # address exactly once, and both loads precede the first store.
    array_loads = [
        line for line in region.splitlines()
        if "load double, ptr %address." in line
    ]
    assert len(array_loads) == 2, (
        "expected one load per element read; a third means a use re-read the "
        f"element instead of using the pinned value:\n{region}"
    )
