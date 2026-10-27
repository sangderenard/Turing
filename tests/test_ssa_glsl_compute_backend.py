"""Desktop-GLSL compute lane for precision cores: emission and GPU execution.

The GPU tests require a working OpenGL 4.3+ context and skip (never silently
pass) when one is unavailable, following test_glsl_backend.py's fixture idiom.
The behavioural oracle is the C module lane run on the SAME lowered module:
both lanes claim to preserve the precision sections exactly, so their float64
outputs should agree bit for bit when the GPU driver honours ``precise``.
When they do not, the test reports the maximum ULP separation loudly and only
then falls back to an rtol=1e-12 closeness assertion -- knowing WHICH of the
two held is the point of running this on real hardware at all.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.common.tensors.accelerator_backends.gl_context import (
    GLContextUnavailable,
    require_gl_context,
)
from src.compiler.ssa_c_backend import emit_ssa_module_to_c
from src.compiler.ssa_glsl_compute_backend import (
    emit_ssa_module_to_glsl_compute,
)

import tools.benchmark_microprecision_matrix as m


WIDTH = 2
COUNT = 64


@pytest.fixture(scope="session")
def gl():
    try:
        return require_gl_context()
    except GLContextUnavailable as exc:
        pytest.skip(f"no OpenGL 4.3+ compute context: {exc}")


# Building a core lowers authored source through the whole identity pipeline
# and takes on the order of ten seconds, so each flavour is built once per
# session and shared by the emitter test and the GPU test.
@pytest.fixture(scope="session")
def fma_core():
    core = m._prepare_core("sin", WIDTH)
    return core, m._wrapper_name("sin", WIDTH, core)


@pytest.fixture(scope="session")
def split_core():
    core = m._prepare_core("sin", WIDTH, "split")
    return core, m._wrapper_name("sin", WIDTH, core)


def _sample_x(width: int, count: int) -> np.ndarray:
    """Leading limbs uniform in the sin core's radius, low limbs exact zero."""

    x = np.zeros(count * width, dtype=np.float64)
    x[::width] = np.random.default_rng(0xBEEF).uniform(-0.7, 0.7, count)
    return x


def _run_glsl(core, wrapper: str, x: np.ndarray) -> np.ndarray:
    artifact = emit_ssa_module_to_glsl_compute(core.module, wrapper)
    assert not artifact.shortfalls, artifact.shortfalls
    y = np.zeros(COUNT * WIDTH, dtype=np.float64)
    feeds = m._feeds(core, WIDTH, x.copy(), y, COUNT)
    execution = artifact.prepare_execution(feeds)
    try:
        execution.run()
        output_id = int(core.roles["output"].id)
        return np.asarray(execution.buffers[output_id]).ravel().copy()
    finally:
        execution.close()


def _run_c(core, wrapper: str, x: np.ndarray, directory) -> np.ndarray:
    artifact = emit_ssa_module_to_c(core.module, wrapper)
    assert not artifact.shortfalls, artifact.shortfalls
    # A wrapper formal the C lane does not publish as a buffer is passed to
    # the native entry unallocated, and running it corrupts this process's
    # heap. Refuse here, by name, instead of aborting the whole session.
    missing = [
        int(value.id) for value in core.module.functions[wrapper].args
        if int(value.id) not in set(artifact.buffer_order)
    ]
    assert not missing, (
        f"the C lane publishes no buffer for wrapper formals {missing}; "
        "running it would write through unallocated pointers"
    )
    native = artifact.compile(directory)
    y = np.zeros(COUNT * WIDTH, dtype=np.float64)
    execution = native.prepare_execution(
        m._feeds(core, WIDTH, x.copy(), y, COUNT)
    )
    execution.run()
    output_id = int(core.roles["output"].id)
    return np.asarray(execution.buffers[output_id]).ravel().copy()


def _max_ulp_distance(a: np.ndarray, b: np.ndarray) -> int:
    """Largest per-element separation in binary64 representation steps.

    The uint64 bit patterns are remapped so that the total order of doubles
    (negative values reversed, -0.0 meeting +0.0) becomes the natural order
    of the keys; the ULP distance is then a plain key difference.
    """

    # Positive floats map to [2^63, 2^64) in bit order, negatives map
    # downward from 2^63, and both zeros land exactly on 2^63, so the keys
    # are monotone in the numeric order of the doubles they encode.
    def keys(values: np.ndarray) -> np.ndarray:
        bits = np.ascontiguousarray(values, dtype=np.float64).view(np.uint64)
        half = np.uint64(1) << np.uint64(63)
        return np.where(bits < half, bits + half, half - (bits - half))

    ka, kb = keys(a), keys(b)
    difference = np.where(ka > kb, ka - kb, kb - ka)
    return int(difference.max()) if difference.size else 0


def _assert_matches_c(name: str, gpu: np.ndarray, cpu: np.ndarray):
    """Bit-identity preferred; a measured ULP report otherwise, never silence."""

    if np.array_equal(gpu, cpu):
        print(f"[{name}] GPU output is BIT-IDENTICAL to the C lane "
              f"({gpu.size} float64 values)")
        return
    worst = _max_ulp_distance(gpu, cpu)
    print(f"[{name}] GPU output is NOT bit-identical to the C lane; "
          f"max separation {worst} ulp over {gpu.size} values")
    np.testing.assert_allclose(
        gpu, cpu, rtol=1e-12, atol=0.0,
        err_msg=(
            f"{name}: GPU/CPU disagreement beyond 1e-12 relative; "
            f"max separation {worst} ulp -- the driver is not honouring "
            "precise/fma semantics"
        ),
    )


# ---------------------------------------------------------------------------
# emission -- no GPU required
# ---------------------------------------------------------------------------

def test_fma_flavour_emits_precise_fp64_compute_shader(fma_core):
    core, wrapper = fma_core
    artifact = emit_ssa_module_to_glsl_compute(core.module, wrapper)
    assert not artifact.shortfalls, artifact.shortfalls
    assert artifact.precision_sections
    assert artifact.source.startswith("#version 430")
    assert "GL_ARB_gpu_shader_fp64" in artifact.source
    assert "precise " in artifact.source
    assert "fma(" in artifact.source
    # The ABI must name the two arrays (input x, output y) and put the count
    # first among the scalars, because that is the documented SSBO layout.
    assert len(artifact.buffer_order) == 2
    assert int(core.roles["output"].id) in artifact.written_buffers
    assert artifact.scalar_order[0] == int(core.roles["count"].id)


def test_split_flavour_emits_no_fma_at_all(split_core):
    core, wrapper = split_core
    artifact = emit_ssa_module_to_glsl_compute(core.module, wrapper)
    assert not artifact.shortfalls, artifact.shortfalls
    assert artifact.precision_sections
    assert "precise " in artifact.source
    assert "fma(" not in artifact.source


# ---------------------------------------------------------------------------
# execution -- real GPU against the C lane on the same module
# ---------------------------------------------------------------------------

def test_gpu_matches_c_lane_fma_flavour(gl, fma_core, tmp_path):
    core, wrapper = fma_core
    x = _sample_x(WIDTH, COUNT)
    gpu = _run_glsl(core, wrapper, x)
    cpu = _run_c(core, wrapper, x, tmp_path)
    # A kernel that ran but wrote nothing would compare zeros against zeros
    # in the low limbs and could still slip through; demand real signal.
    assert np.count_nonzero(gpu[::WIDTH]) > COUNT // 2
    _assert_matches_c("sin w2 fma", gpu, cpu)


def test_gpu_matches_c_lane_split_flavour(gl, split_core, tmp_path):
    core, wrapper = split_core
    x = _sample_x(WIDTH, COUNT)
    gpu = _run_glsl(core, wrapper, x)
    cpu = _run_c(core, wrapper, x, tmp_path)
    assert np.count_nonzero(gpu[::WIDTH]) > COUNT // 2
    _assert_matches_c("sin w2 split", gpu, cpu)


# ---------------------------------------------------------------------------
# reference evaluation -- the lowered module itself, interpreted in Python
# ---------------------------------------------------------------------------

def _evaluate_reference(core, wrapper: str, feeds, *, fma_rounds_once=True):
    """Run the wrapper's counted loop over the module's own SSA in Python.

    Independent of the shader emitter: each region call is evaluated as a
    call (its result is the list of its Ret operands, projected by
    GetElementPtr/Load), every float operation is one IEEE binary64
    operation, and Fma is ``math.fma`` (one rounding) or, with
    ``fma_rounds_once=False``, a multiply then an add (two roundings).
    """

    import math

    module = core.module
    function = module.functions[wrapper]
    arrays = {key: np.array(value, dtype=np.float64, copy=True)
              for key, value in feeds.items() if np.ndim(value)}
    scalars = {key: value for key, value in feeds.items() if not np.ndim(value)}
    binary = {
        "Add": lambda a, b: a + b, "Sub": lambda a, b: a - b,
        "Mul": lambda a, b: a * b, "Div": lambda a, b: a / b,
    }

    def run_block(instructions, env):
        returned = None
        for instruction in instructions:
            op = str(instruction.op)
            args = [env.get(int(value.id), None) for value in instruction.args]
            result = None
            if op == "Const":
                result = instruction.attributes.get(
                    "constant", instruction.attributes.get("value")
                )
            elif op == "Cast":
                result = args[0]
            elif op == "GetElementPtr":
                result = (args[0], int(args[1]))
            elif op == "Load":
                base, index = args[0]
                result = base[index]
            elif op == "Store":
                base, index = args[1]
                base[index] = args[0]
            elif op in binary:
                result = np.float64(binary[op](np.float64(args[0]), np.float64(args[1])))
            elif op == "Neg":
                result = np.float64(-np.float64(args[0]))
            elif op == "Fma":
                a, b, c = (float(value) for value in args)
                result = np.float64(
                    math.fma(a, b, c) if fma_rounds_once
                    else np.float64(np.float64(a) * np.float64(b)) + np.float64(c)
                )
            elif op == "Call":
                region = module.functions[str(instruction.attributes["callee"])]
                inner = {int(formal.id): value for formal, value in zip(region.args, args)}
                result = run_block(region.blocks["entry"].instrs, inner)
            elif op == "Ret":
                returned = args
            elif op in ("Br", "Phi", "Lt", "CondBr"):
                continue
            else:
                raise AssertionError(f"reference evaluation has no rule for {op}")
            if instruction.res is not None:
                env[int(instruction.res.id)] = result
        return returned

    phi = next(
        instruction for block in function.blocks.values()
        for instruction in block.instrs if str(instruction.op) == "Phi"
    )
    env = {}
    for value in function.args:
        identity = int(value.id)
        env[identity] = arrays.get(identity, scalars.get(identity))
    run_block(function.blocks["entry"].instrs, env)
    count = int(np.asarray(feeds[int(core.roles["count"].id)]))
    for index in range(count):
        env[int(phi.res.id)] = index
        run_block(function.blocks["loop_body"].instrs, env)
    return arrays[int(core.roles["output"].id)]


@pytest.mark.parametrize("flavour", ["fma", "split"])
def test_gpu_matches_reference_evaluation_of_the_lowered_core(gl, flavour, fma_core, split_core):
    """The current lowering: a Cast bound and a load/compute/store body."""

    core, wrapper = fma_core if flavour == "fma" else split_core
    wrapper_function = core.module.functions[wrapper]
    body_calls = [
        instruction for instruction in wrapper_function.blocks["loop_body"].instrs
        if instruction.op == "Call"
    ]
    assert len(body_calls) > 1  # the shape this lane used to refuse
    x = _sample_x(WIDTH, COUNT)
    gpu = _run_glsl(core, wrapper, x)
    feeds = m._feeds(core, WIDTH, x.copy(), np.zeros(COUNT * WIDTH), COUNT)
    once = _evaluate_reference(core, wrapper, feeds)
    assert np.count_nonzero(gpu[:COUNT]) > COUNT // 4
    # It is a sine core: element i reads x[i] (the lowering's indexing).
    np.testing.assert_allclose(gpu[:COUNT], np.sin(x[:COUNT]), rtol=1e-13, atol=1e-300)
    if flavour == "split":
        # No fma anywhere: every operation is one IEEE operation in a
        # precise variable, so the GPU must agree bit for bit.
        assert gpu.tobytes() == once.tobytes()
        return
    twice = _evaluate_reference(core, wrapper, feeds, fma_rounds_once=False)
    # GLSL lets fma() round twice; each element must be one rule or the
    # other, never anything else.
    assert np.all((gpu == once) | (gpu == twice))

