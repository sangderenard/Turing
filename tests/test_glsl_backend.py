"""GLSL compute-shader backend: real GPU execution checked against numpy.

These tests require a working OpenGL 4.3+ context and skip (never silently pass)
when one is unavailable. The emitter tests need no GPU at all and always run.

numpy is the behavioural oracle, per docs/c_backend_status.md's rule for the C
backend. Tolerances are float32-appropriate: this backend narrows to float32 on
the way to the GPU, which is a documented contract, not an accident.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.common.tensors.accelerator_backends.glsl_backend import (
    _broadcast_index_source,
    GLSL_OPS,
    GLChunk,
    GLContextUnavailable,
    GLComputeLimits,
    GLSLUnsupportedOp,
    arange_chunk,
    canonical_op,
    cat_chunks,
    cumsum_chunk,
    dispatch_batch,
    dispatch_stats,
    emit_cat_source,
    emit_arange_source,
    emit_cumsum_source,
    emit_expand_source,
    emit_matmul_source,
    emit_reduce_source,
    emit_repeat_source,
    emit_op_source,
    emit_permute_source,
    emit_program_source,
    emit_stack_source,
    emit_topk_offsets_source,
    execute_captured_fused_program,
    execute_program,
    fuse_elementwise,
    gl_context_info,
    expand_chunk,
    matmul_chunks,
    reduce_chunk,
    repeat_chunk,
    permute_chunk,
    plan_launch,
    require_gl_context,
    reshape_chunk,
    run_op,
    shader_cache_stats,
    slice_axis_chunk,
    stack_chunks,
    topk_chunks,
)
from src.common.tensors.fused_ir import FusedProgram, Meta, OpStep
from src.common.tensors.accelerator_backends.c_primitive_program import (
    CapturedFusedProgram,
)

from src.common.tensors.accelerator_backends import gl_context as glctx

RTOL, ATOL = 1e-5, 1e-5


def test_equal_extent_rank_change_uses_the_recorded_linear_index():
    lines, index = _broadcast_index_source("value", (1,), ())

    assert lines == []
    assert index == "gid"


def _program(feeds, specs, output):
    return FusedProgram(
        1,
        set(feeds),
        [
            OpStep(index, op, inputs, attrs, result_id)
            for index, (op, result_id, inputs, attrs) in enumerate(specs)
        ],
        {"result": output},
    )


@pytest.fixture(scope="session")
def gl():
    try:
        return require_gl_context()
    except GLContextUnavailable as exc:
        pytest.skip(f"no OpenGL 4.3+ compute context: {exc}")


# ---------------------------------------------------------------------------
# context acquisition -- borrow before you build
# ---------------------------------------------------------------------------

def test_provider_registry_orders_by_priority():
    try:
        glctx.register_context_provider("t_low", lambda: None, priority=1)
        glctx.register_context_provider("t_high", lambda: None, priority=999)
        names = glctx.registered_providers()
        assert names.index("t_high") < names.index("t_low")
    finally:
        glctx.unregister_context_provider("t_low")
        glctx.unregister_context_provider("t_high")


def test_named_provider_that_does_not_exist_is_a_hard_error(monkeypatch):
    """A misconfigured TURING_GL_CONTEXT must not quietly fall through."""
    monkeypatch.setattr(glctx, "_lease", None)
    monkeypatch.setenv("TURING_GL_CONTEXT", "no_such_provider")
    with pytest.raises(GLContextUnavailable, match="no such provider"):
        glctx.require_gl_context()


def test_no_owned_context_flag_forbids_creating_one(monkeypatch):
    """A host that must supply its own context gets an error, not a stray window."""
    monkeypatch.setattr(glctx, "_lease", None)
    monkeypatch.setattr(glctx, "_providers", {})
    monkeypatch.setattr(glctx, "_probe_current", lambda *a, **k: None)
    monkeypatch.setenv("TURING_GL_NO_OWNED_CONTEXT", "1")
    with pytest.raises(GLContextUnavailable, match="TURING_GL_NO_OWNED_CONTEXT"):
        glctx.require_gl_context()


def test_registered_provider_is_preferred_over_creating_one(monkeypatch):
    """The seam a nodus/pluck frontend uses to donate its context."""
    monkeypatch.setattr(glctx, "_lease", None)
    monkeypatch.setattr(glctx, "_providers", {})
    monkeypatch.setattr(glctx, "_probe_current", lambda *a, **k: None)

    calls = []

    def donor():
        calls.append("used")
        # Pretend the host made a context current; probe reports it afterwards.
        monkeypatch.setattr(
            glctx, "_probe_current",
            lambda *a, **k: {"major": 4, "minor": 6, "vendor": "v",
                             "renderer": "r", "version": "4.6", "glsl": "4.60"},
        )
        return object()

    glctx.register_context_provider("donor", donor, priority=500)
    try:
        info = glctx.require_gl_context()
    finally:
        glctx.unregister_context_provider("donor")
    assert calls == ["used"]
    assert info["source"] == "provider" and info["provider"] == "donor"


def test_context_reports_its_source_and_nodus_handle(gl):
    assert gl["source"] in {"host_current", "provider", "pluck", "owned_sdl"}
    # nodus registers a frontend-owned context via gp_mem_backend_register_gl_context;
    # a context we own must be able to supply that native pointer.
    handle = glctx.nodus_registration_handle()
    assert isinstance(handle, int)
    if gl["source"] == "owned_sdl":
        assert handle != 0, "an owned context must expose a registerable handle"


# ---------------------------------------------------------------------------
# emitter -- no GPU required
# ---------------------------------------------------------------------------

def test_canonical_op_resolves_aliases_and_reverse():
    assert canonical_op("add") == ("add", False)
    assert canonical_op("radd") == ("add", True)
    assert canonical_op("iadd") == ("add", False)
    assert canonical_op("rsub") == ("sub", True)
    assert canonical_op("div") == ("truediv", False)
    assert canonical_op("less_equal") == ("less_equal", False)


@pytest.mark.parametrize("op", ["isnan", "isinf", "isfinite", "round"])
def test_prefix_stripping_does_not_mangle_ops_starting_with_i_or_r(op):
    """Blind ``op[1:]`` would turn these into snan/sinf/sfinite/ound."""
    assert canonical_op(op) == (op, False)


def test_unknown_op_raises_rather_than_defaulting():
    with pytest.raises(GLSLUnsupportedOp):
        canonical_op("definitely_not_an_op")


def test_emitted_program_fuses_intermediates_into_locals():
    # (a + b) then exp(): two instructions, one dispatch, zero intermediate buffers.
    program = _program(
        [0, 1],
        [("add", 2, [0, 1], {}), ("exp", 3, [2], {})],
        3,
    )
    src = emit_program_source(program)
    assert src.count("buffer") == 1          # one shared arena, no temporary SSBOs
    assert "float s2 = s0 + s1;" in src
    assert "float s3 = exp(s2);" in src
    assert "arena[u_slot[2] + (gid)] = floatBitsToUint(s3);" in src


def test_typed_fused_program_keeps_integer_intermediates_and_bool_output(gl):
    program = FusedProgram(
        version=1,
        feeds={0},
        steps=[
            OpStep(0, "add", [0], {"right_scalar": 3}, 1),
            OpStep(1, "floordiv", [1], {"right_scalar": 2}, 2),
            OpStep(2, "greater", [2], {"right_scalar": 4}, 3),
        ],
        outputs={"result": 3},
        meta={
            0: Meta(shape=(9,), dtype="int32"),
            1: Meta(shape=(9,), dtype="int32"),
            2: Meta(shape=(9,), dtype="int32"),
            3: Meta(shape=(9,), dtype="bool"),
        },
    )
    values = np.arange(-4, 5, dtype=np.int32)
    result = execute_program(program, {0: GLChunk.from_numpy(values)})
    assert result.dtype == np.dtype(np.bool_)
    np.testing.assert_array_equal(result.numpy(), ((values + 3) // 2) > 4)


def test_cat_emitter_maps_arbitrary_rank_output_to_source_buffers():
    src = emit_cat_source(((2, 3, 4), (2, 5, 4)), dim=1)
    assert src.count("buffer Arena") == 1
    assert "uint inner = gid % uint(4);" in src
    assert "uint axis_index = block % uint(8);" in src
    assert "arena[u_slot[0] + (source_index)]" in src
    assert "arena[u_slot[1] + (source_index)]" in src


def test_stack_emitter_preserves_integer_storage_and_inserted_axis_mapping():
    src = emit_stack_source((2, 3, 4), 3, dim=1, dtype=np.int32)
    assert src.count("buffer Arena") == 1
    assert "arena[u_slot[0] + (source_index)]" in src
    assert "arena[u_slot[3] + (gid)]" in src
    assert "uint inner = gid % uint(12);" in src
    assert "uint source_number = block % uint(3);" in src


def test_permute_emitter_maps_output_coordinates_to_input_strides():
    src = emit_permute_source((2, 3, 4), (2, 0, 1), dtype=np.uint32)
    assert "arena[u_slot[0] + (source_index)]" in src
    assert "arena[u_slot[1] + (gid)]" in src
    assert "coord0 * uint(1)" in src
    assert "coord1 * uint(12)" in src
    assert "coord2 * uint(4)" in src


def test_arange_emitter_uses_gid_without_an_input_buffer():
    src = emit_arange_source(3, 2, dtype=np.int32)
    assert src.count("buffer Arena") == 1
    assert "int(gid)" in src
    assert "int(3) + int(gid) * int(2)" in src


def test_primitive_emitter_indexes_broadcast_inputs_without_expanding_buffers():
    from src.common.tensors.accelerator_backends.glsl_backend import (
        _emit_primitive_source,
    )

    src = _emit_primitive_source(
        "mul",
        left_dtype=np.float32,
        right_dtype=np.float32,
        out_dtype=np.float32,
        left_shape=(8, 1),
        right_shape=(1, 8),
        out_shape=(8, 8),
    )
    assert src.count("buffer Arena") == 1
    assert "lhs_index" in src
    assert "rhs_index" in src
    assert "arena[u_slot[0] + (lhs_index)]" in src
    assert "arena[u_slot[1] + (rhs_index)]" in src


def test_expand_emitter_uses_direct_broadcast_indexing():
    src = emit_expand_source((1, 8), (4, 3, 8), dtype=np.float32)
    assert src.count("buffer Arena") == 1
    assert "source_index" in src
    assert "arena[u_slot[1] + (gid)]" in src
    assert "arena[u_slot[0] + (source_index)]" in src


def test_matmul_emitter_contains_one_batched_accumulation_kernel():
    src = emit_matmul_source(
        (8, 8),
        (2, 3, 8, 8),
        left_dtype=np.float32,
        right_dtype=np.float32,
    )
    assert src.count("buffer Arena") == 1
    assert "batch_coord0" in src
    assert "shared float left_tile[16][16]" in src
    assert "shared float right_tile[16][16]" in src
    assert "barrier();" in src
    assert "arena[u_slot[2] + (output_index)] = floatBitsToUint(total)" in src


def test_topk_emitter_selects_offsets_without_host_sorting():
    src = emit_topk_offsets_source(
        (2, 5, 3), 3, 1, dtype=np.float32, local_size=64
    )
    assert "uint chosen[3]" in src
    assert "candidate > best" in src
    assert "arena[u_slot[1] + (gid)]" in src
    assert "isnan(candidate)" in src


def test_repeat_and_reduction_emitters_are_single_dispatch_kernels():
    repeat_src = emit_repeat_source((1, 3), (4, 2), dtype=np.int32)
    assert "coord0 % uint(1)" in repeat_src
    assert "coord1 % uint(3)" in repeat_src
    reduce_src = emit_reduce_source(
        "sum", (2, 3, 4), dim=1, dtype=np.float32
    )
    assert "for (uint k = uint(0); k < uint(3); ++k)" in reduce_src
    assert "arena[u_slot[1] + (gid)] = floatBitsToUint(total)" in reduce_src


def test_cumsum_emitter_assigns_one_bounded_output_per_invocation():
    src = emit_cumsum_source((2, 3, 4), dim=1, dtype=np.int32)
    assert "uint axis_index = block % uint(3)" in src
    assert "k <= axis_index" in src
    assert "arena[u_slot[1] + (gid)] = uint(total)" in src
    assert "u_slot[1] + (index)" not in src


def test_emitter_rejects_reading_an_unwritten_slot():
    bad = _program([0], [("add", 1, [0, 5], {})], 1)
    with pytest.raises(ValueError, match="before it is written"):
        emit_program_source(bad)


def test_emitter_rejects_both_operand_kinds():
    bad = _program(
        [0], [("add", 1, [0, 0], {"right_scalar": 1.0})], 1
    )
    with pytest.raises(ValueError, match="invalid operand layout"):
        emit_program_source(bad)


def test_mod_is_floored_not_glsl_mod():
    """C and numpy use floored modulo; GLSL's mod() is not guaranteed to match."""
    float_src = emit_op_source("mod")
    int_src = emit_op_source(
        "mod",
        left_dtype=np.int32,
        right_dtype=np.int32,
        output_dtype=np.int32,
    )
    assert "floor(" in float_src and "mod(" not in float_src.split("void main")[1]
    assert "floor_div_i(" in int_src


def test_round_is_half_away_from_zero():
    """GLSL round() may break ties either way; C round() does not."""
    src = emit_op_source("round")
    assert "sign(" in src and "floor(abs(" in src


def test_every_advertised_op_emits():
    integer_ops = {"invert", "bitand", "bitor", "bitxor", "shl", "shr"}
    for op in sorted(GLSL_OPS):
        dtype = np.int32 if op in integer_ops else np.float32
        src = emit_op_source(op, left_dtype=dtype, right_dtype=dtype)
        assert src.startswith("#version 430")
        assert "arena[u_slot[" in src
        assert " + (gid)]" in src


def test_glsl_ops_are_exactly_the_abstract_tensor_primitive_vocabulary():
    catalog = (
        Path(__file__).resolve().parents[2]
        / "nodus"
        / "ops"
        / "canonical_ops.json"
    )
    operations = json.loads(catalog.read_text(encoding="utf-8"))["ops"]
    lowerable = {op["name"] for op in operations if op["lowerable"]}
    assert len(lowerable) == 56
    assert GLSL_OPS == lowerable


# ---------------------------------------------------------------------------
# execution -- needs a real GPU
# ---------------------------------------------------------------------------

def test_context_reports_a_compute_capable_gl(gl):
    assert (gl["major"], gl["minor"]) >= (4, 3)
    assert gl_context_info() is not None


def test_chunk_residency_is_explicit_and_roundtrips(gl):
    src = np.arange(12, dtype=np.float32).reshape(3, 4)
    chunk = GLChunk.from_numpy(src)
    assert chunk.on_cpu and not chunk.on_gpu

    chunk.to_gpu()
    assert chunk.on_gpu and chunk.buffer_id is not None

    np.testing.assert_allclose(chunk.to_cpu(), src, rtol=RTOL, atol=ATOL)
    assert chunk.shape == (3, 4) and chunk.count == 12 and chunk.nbytes == 48
    chunk.release()
    assert not chunk.on_gpu


def test_wrapped_buffer_is_not_deleted_by_release(gl):
    """Interop contract: a host-owned buffer stays the host's to free."""
    from OpenGL import GL

    values = np.ones(8, dtype=np.float32)
    buffer_id = GL.glGenBuffers(1)
    try:
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, buffer_id)
        GL.glBufferData(
            GL.GL_SHADER_STORAGE_BUFFER,
            values.nbytes,
            values,
            GL.GL_STATIC_DRAW,
        )
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, 0)
        wrapper = GLChunk.wrap(buffer_id, (8,))
        assert wrapper.on_gpu
        np.testing.assert_allclose(
            wrapper.to_cpu(), values, rtol=RTOL, atol=ATOL
        )
        wrapper.release()
        # Re-wrapping proves release did not delete the host-owned buffer.
        survivor = GLChunk.wrap(buffer_id, (8,))
        np.testing.assert_allclose(
            survivor.to_cpu(), values, rtol=RTOL, atol=ATOL
        )
        survivor.release()
    finally:
        GL.glDeleteBuffers(1, [buffer_id])


BINARY_CASES = {
    "add": lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "mul": lambda a, b: a * b,
    "truediv": lambda a, b: a / b,
    "pow": lambda a, b: np.power(a, b),
    "mod": lambda a, b: np.mod(a, b),
    "floordiv": lambda a, b: np.floor(a / b),
    "maximum": np.maximum,
    "minimum": np.minimum,
    "less": lambda a, b: (a < b).astype(np.float32),
    "less_equal": lambda a, b: (a <= b).astype(np.float32),
    "greater": lambda a, b: (a > b).astype(np.float32),
    "greater_equal": lambda a, b: (a >= b).astype(np.float32),
    "equal": lambda a, b: (a == b).astype(np.float32),
    "not_equal": lambda a, b: (a != b).astype(np.float32),
}


@pytest.mark.parametrize("op", sorted(BINARY_CASES))
def test_binary_ops_match_numpy(gl, op):
    rng = np.random.default_rng(0xC0FFEE)
    a = rng.uniform(0.5, 4.0, size=257).astype(np.float32)   # not a multiple of 256
    b = rng.uniform(0.5, 4.0, size=257).astype(np.float32)
    got = run_op(op, a, b).numpy()
    np.testing.assert_allclose(got, BINARY_CASES[op](a, b), rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("op", ["floordiv", "mod"])
def test_signed_integer_division_matches_numpy_for_negative_values(gl, op):
    """PCM byte extraction depends on Python/NumPy floor semantics for negatives."""
    a = np.array(
        [-65535, -32768, -16384, -300, -257, -256, -255, -44, -1,
         0, 1, 255, 256, 257],
        dtype=np.int32,
    )
    for divisor in (256, 65536):
        got = run_op(op, a, divisor).numpy()
        expected = (
            np.floor_divide(a, divisor)
            if op == "floordiv"
            else np.mod(a, divisor)
        )
        np.testing.assert_array_equal(got, expected)


UNARY_CASES = {
    "sqrt": np.sqrt, "exp": np.exp, "log": np.log, "neg": np.negative,
    "abs": np.abs, "trunc": np.trunc, "floor": np.floor, "ceil": np.ceil,
    "tanh": np.tanh, "sin": np.sin, "cos": np.cos, "tan": np.tan,
    "asin": np.arcsin, "acos": np.arccos, "atan": np.arctan,
    "sinh": np.sinh, "cosh": np.cosh, "asinh": np.arcsinh,
    "acosh": np.arccosh, "atanh": np.arctanh, "sign": np.sign,
}


@pytest.mark.parametrize("op", sorted(UNARY_CASES))
def test_unary_ops_match_numpy(gl, op):
    rng = np.random.default_rng(7)
    if op in {"asin", "acos", "atanh"}:
        a = rng.uniform(-0.9, 0.9, size=300).astype(np.float32)
    elif op == "acosh":
        a = rng.uniform(1.0, 6.0, size=300).astype(np.float32)
    elif op == "tan":
        a = rng.uniform(-1.0, 1.0, size=300).astype(np.float32)
    else:
        a = rng.uniform(0.25, 6.0, size=300).astype(np.float32)
    got = run_op(op, a).numpy()
    np.testing.assert_allclose(
        got, UNARY_CASES[op](a), rtol=1e-4, atol=1e-4
    )


@pytest.mark.parametrize(
    ("op", "expected"),
    [
        ("invert", lambda a, b: np.invert(a)),
        ("bitand", np.bitwise_and),
        ("bitor", np.bitwise_or),
        ("bitxor", np.bitwise_xor),
        ("shl", np.left_shift),
        ("shr", np.right_shift),
    ],
)
def test_integer_primitives_match_numpy(gl, op, expected):
    a = np.asarray([1, 2, 7, -8], dtype=np.int32)
    b = np.asarray([1, 2, 1, 2], dtype=np.int32)
    got = run_op(op, a) if op == "invert" else run_op(op, a, b)
    want = expected(a, b) if op != "invert" else expected(a, b)
    assert got.dtype == np.dtype(np.int32)
    np.testing.assert_array_equal(got.numpy(), want)


@pytest.mark.parametrize(
    ("op", "source", "dtype", "expected"),
    [
        ("int_trunc", [1.9, -2.1], np.float32, np.asarray([1, -2], np.int32)),
        ("zext", [0, 2], np.int32, np.asarray([0, 2], np.uint32)),
        ("sext", [0, 2], np.uint32, np.asarray([0, 2], np.int32)),
        ("fptosi", [1.9, -2.1], np.float32, np.asarray([1, -2], np.int32)),
        ("fptoui", [1.9, 2.1], np.float32, np.asarray([1, 2], np.uint32)),
        ("sitofp", [1, -2], np.int32, np.asarray([1.0, -2.0], np.float32)),
        ("uitofp", [1, 2], np.uint32, np.asarray([1.0, 2.0], np.float32)),
    ],
)
def test_cast_primitives_have_canonical_output_dtype(
    gl, op, source, dtype, expected
):
    got = run_op(op, np.asarray(source, dtype=dtype))
    assert got.dtype == expected.dtype
    np.testing.assert_array_equal(got.numpy(), expected)


@pytest.mark.parametrize(
    ("op", "expected"),
    [
        ("logical_and", [False, False, True]),
        ("logical_or", [True, True, True]),
    ],
)
def test_logical_binary_primitives_return_bool_storage(gl, op, expected):
    a = np.asarray([0, 1, 2], dtype=np.int32)
    b = np.asarray([1, 0, 3], dtype=np.int32)
    got = run_op(op, a, b)
    assert got.dtype == np.dtype(np.bool_)
    np.testing.assert_array_equal(got.numpy(), expected)


def test_round_ties_go_away_from_zero_like_c(gl):
    a = np.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5], dtype=np.float32)
    got = run_op("round", a).numpy()
    # C round() / the away-from-zero convention -- NOT numpy's round-half-to-even,
    # which would give [-2, -2, -0, 0, 2, 2].
    np.testing.assert_allclose(got, [-3, -2, -1, 1, 2, 3], rtol=RTOL, atol=ATOL)


def test_predicates_on_nonfinite_values(gl):
    a = np.array([1.0, np.nan, np.inf, -np.inf, 0.0], dtype=np.float32)
    np.testing.assert_allclose(run_op("isnan", a).numpy(), [0, 1, 0, 0, 0])
    np.testing.assert_allclose(run_op("isinf", a).numpy(), [0, 0, 1, 1, 0])
    np.testing.assert_allclose(run_op("isfinite", a).numpy(), [1, 0, 0, 0, 1])
    np.testing.assert_allclose(run_op("logical_not", a).numpy(), [0, 0, 0, 0, 1])


def test_scalar_operand_and_reverse_order(gl):
    a = np.array([1.0, 2.0, 4.0], dtype=np.float32)
    np.testing.assert_allclose(run_op("sub", a, 1.0).numpy(), a - 1.0,
                               rtol=RTOL, atol=ATOL)
    # rsub must compute (scalar - a), not (a - scalar).
    np.testing.assert_allclose(run_op("rsub", a, 10.0).numpy(), 10.0 - a,
                               rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(run_op("rtruediv", a, 8.0).numpy(), 8.0 / a,
                               rtol=RTOL, atol=ATOL)


def test_fused_program_matches_numpy(gl):
    """The payoff: a 4-instruction program in a single dispatch."""
    rng = np.random.default_rng(99)
    x = rng.uniform(0.5, 3.0, size=1024).astype(np.float32)
    y = rng.uniform(0.5, 3.0, size=1024).astype(np.float32)

    # t = sqrt((x * y) + 2.0);  out = t - x
    program = _program(
        [0, 1],
        [
            ("mul", 2, [0, 1], {}),
            ("add", 3, [2], {"right_scalar": 2.0}),
            ("sqrt", 4, [3], {}),
            ("sub", 5, [4, 0], {}),
        ],
        5,
    )
    got = execute_program(program, [x, y]).numpy()
    np.testing.assert_allclose(got, np.sqrt((x * y) + 2.0) - x, rtol=1e-4, atol=1e-5)


def test_opt_in_fusion_defers_expression_to_one_dispatch(
    gl, monkeypatch
):
    from src.common.tensors.accelerator_backends import glsl_backend

    values = np.linspace(0.25, 2.0, 1024, dtype=np.float32)
    source = GLChunk.from_numpy(values).to_gpu()
    source.discard_host()
    dispatches = []
    original_dispatch = glsl_backend._dispatch

    def counted_dispatch(*args, **kwargs):
        dispatches.append(args[0])
        return original_dispatch(*args, **kwargs)

    monkeypatch.setattr(glsl_backend, "_dispatch", counted_dispatch)
    with fuse_elementwise():
        result = run_op(
            "sin",
            run_op("add", run_op("mul", source, 2.0), 1.0),
        )
        assert not result.on_gpu and not result.on_cpu
        assert dispatches == []

    np.testing.assert_allclose(
        result.numpy(), np.sin(values * 2.0 + 1.0), rtol=RTOL, atol=ATOL
    )
    assert len(dispatches) == 1
    result.release()
    source.release()


def test_opt_in_fusion_keeps_typed_glsl_primitives_and_reverse_scalars(
    gl, monkeypatch
):
    from src.common.tensors.accelerator_backends import glsl_backend

    values = np.arange(8, dtype=np.int32)
    source = GLChunk.from_numpy(values).to_gpu()
    source.discard_host()
    dispatch_count = 0
    original_dispatch = glsl_backend._dispatch

    def counted_dispatch(*args, **kwargs):
        nonlocal dispatch_count
        dispatch_count += 1
        return original_dispatch(*args, **kwargs)

    monkeypatch.setattr(glsl_backend, "_dispatch", counted_dispatch)
    with fuse_elementwise():
        shifted = run_op("shl", source, 1)
        as_float = run_op("sitofp", shifted)
        reversed_subtract = run_op("sub", 20.0, as_float)
        result = run_op("greater", reversed_subtract, 9.0)

    np.testing.assert_array_equal(
        result.numpy(), (20.0 - (values << 1).astype(np.float32)) > 9.0
    )
    assert result.dtype == np.dtype(np.bool_)
    assert dispatch_count == 1
    result.release()
    source.release()


def test_opt_in_fusion_carries_linear_reshape_inside_region(gl, monkeypatch):
    from src.common.tensors.accelerator_backends import glsl_backend

    values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    source = GLChunk.from_numpy(values).to_gpu()
    source.discard_host()
    dispatch_count = 0
    original_dispatch = glsl_backend._dispatch

    def counted_dispatch(*args, **kwargs):
        nonlocal dispatch_count
        dispatch_count += 1
        return original_dispatch(*args, **kwargs)

    monkeypatch.setattr(glsl_backend, "_dispatch", counted_dispatch)
    with fuse_elementwise():
        first = run_op("mul", source, 2.0)
        viewed = reshape_chunk(first, (4, 6))
        result = run_op("add", viewed, 3.0)
        assert not result.on_gpu and dispatch_count == 0

    assert result.shape == (4, 6)
    np.testing.assert_array_equal(result.numpy(), values.reshape(4, 6) * 2 + 3)
    assert dispatch_count == 1
    result.release()
    source.release()


def test_opt_in_fusion_keeps_broadcast_branches_in_one_dispatch(gl, monkeypatch):
    from src.common.tensors.accelerator_backends import glsl_backend

    left_values = np.arange(8, dtype=np.float32).reshape(8, 1)
    right_values = np.arange(8, dtype=np.float32).reshape(1, 8)
    left = GLChunk.from_numpy(left_values).to_gpu()
    right = GLChunk.from_numpy(right_values).to_gpu()
    dispatch_count = 0
    original_dispatch = glsl_backend._dispatch

    def counted_dispatch(*args, **kwargs):
        nonlocal dispatch_count
        dispatch_count += 1
        return original_dispatch(*args, **kwargs)

    monkeypatch.setattr(glsl_backend, "_dispatch", counted_dispatch)
    with fuse_elementwise():
        left_branch = run_op("mul", left, 2.0)
        right_branch = run_op("add", right, 3.0)
        result = run_op("sub", left_branch, right_branch)
        assert dispatch_count == 0

    np.testing.assert_allclose(
        result.numpy(),
        left_values * 2.0 - (right_values + 3.0),
        rtol=RTOL,
        atol=ATOL,
    )
    assert dispatch_count == 1
    result.release()
    left.release()
    right.release()


def test_program_runs_on_gpu_resident_feeds_without_readback(gl):
    """Feeds already on the GPU are used in place -- the interop path."""
    a = GLChunk.from_numpy(np.full(64, 3.0, dtype=np.float32)).to_gpu()
    b = GLChunk.from_numpy(np.full(64, 4.0, dtype=np.float32)).to_gpu()
    program = _program([0, 1], [("add", 2, [0, 1], {})], 2)
    out = execute_program(program, [a, b])
    assert out.on_gpu
    np.testing.assert_allclose(out.numpy(), np.full(64, 7.0), rtol=RTOL, atol=ATOL)


def test_program_reuses_caller_owned_output_buffer(gl):
    """Repeated render/simulation steps need no output allocation or upload."""
    a = GLChunk.from_numpy(np.full(64, 3.0, dtype=np.float32)).to_gpu()
    out = GLChunk((64,)).to_gpu()
    buffer_id = out.buffer_id
    program = _program([0], [("mul", 1, [0], {"right_scalar": 2.0})], 1)

    assert execute_program(program, [a], out=out) is out
    assert execute_program(program, [a], out=out) is out
    assert out.buffer_id == buffer_id
    np.testing.assert_allclose(out.numpy(), np.full(64, 6.0), rtol=RTOL, atol=ATOL)

    out.release()
    a.release()


def test_program_rejects_mismatched_output_shape(gl):
    a = GLChunk.from_numpy(np.ones(8, dtype=np.float32)).to_gpu()
    out = GLChunk((4,)).to_gpu()
    program = _program([0], [("neg", 1, [0], {})], 1)
    with pytest.raises(ValueError, match="output must share"):
        execute_program(program, [a], out=out)
    out.release()
    a.release()


def test_shader_cache_reuses_compiled_programs(gl):
    a = np.ones(32, dtype=np.float32)
    run_op("add", a, a)
    before = shader_cache_stats()
    for _ in range(5):
        run_op("add", a, a)
    after = shader_cache_stats()
    assert after["hits"] >= before["hits"] + 5
    assert after["size"] == before["size"], "identical sources must not recompile"


def test_shape_mismatch_is_an_error_not_a_broadcast(gl):
    with pytest.raises(ValueError, match="not broadcastable"):
        run_op("add", np.ones(4, dtype=np.float32), np.ones(8, dtype=np.float32))


@pytest.mark.parametrize(
    "left_shape,right_shape",
    [
        ((8, 1), (1, 8)),
        ((2, 1, 4), (3, 1)),
        ((), (2, 3, 4)),
        ((1,), (2, 3, 4)),
    ],
)
def test_run_op_broadcasts_in_shader_without_expanded_inputs(
    gl, left_shape, right_shape
):
    left = np.arange(
        max(1, int(np.prod(left_shape))), dtype=np.float32
    ).reshape(left_shape)
    right = (
        np.arange(max(1, int(np.prod(right_shape))), dtype=np.float32)
        .reshape(right_shape)
        + 1.0
    )
    out = run_op("mul", GLChunk.from_numpy(left), GLChunk.from_numpy(right))
    assert out.shape == np.broadcast_shapes(left_shape, right_shape)
    assert out.on_gpu and not out.on_cpu
    np.testing.assert_allclose(
        out.numpy(), left * right, rtol=RTOL, atol=ATOL
    )


def test_glsl_builds_encoder_dct_basis_through_broadcasting(gl):
    from src.common.tensors.abstraction import AbstractTensor
    from src.common.tensors.compression.block_transform import (
        orthonormal_dct_basis,
    )

    with AbstractTensor.use_backend("glsl"):
        like = AbstractTensor.tensor(np.zeros((8, 8), dtype=np.float32))
        basis = orthonormal_dct_basis(8, like=like)
    with AbstractTensor.use_backend("numpy"):
        reference_like = AbstractTensor.zeros((8, 8))
        reference = orthonormal_dct_basis(8, like=reference_like)

    assert basis.shape == (8, 8)
    assert basis.data.on_gpu and not basis.data.on_cpu
    np.testing.assert_allclose(
        basis.numpy(), reference.numpy(), rtol=RTOL, atol=ATOL
    )


@pytest.mark.parametrize(
    "source_shape,target_shape",
    [
        ((1, 8), (4, 3, 8)),
        ((2, 1, 4), (2, 5, 4)),
        ((4,), (3, -1)),
    ],
)
def test_expand_chunk_materializes_broadcast_on_gpu(
    gl, source_shape, target_shape
):
    values = np.arange(np.prod(source_shape), dtype=np.float32).reshape(
        source_shape
    )
    source = GLChunk.from_numpy(values).to_gpu()
    source.discard_host()
    out = expand_chunk(source, target_shape)
    resolved = tuple(
        source_shape[axis - (len(target_shape) - len(source_shape))]
        if size == -1
        else size
        for axis, size in enumerate(target_shape)
    )
    try:
        assert out.shape == resolved
        assert out.on_gpu and not out.on_cpu
        np.testing.assert_array_equal(
            out.numpy(), np.broadcast_to(values, resolved)
        )
    finally:
        out.release()
        source.release()


def test_expand_chunk_rejects_invalid_shapes():
    source = GLChunk.from_numpy(np.ones((2, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="fewer dimensions"):
        expand_chunk(source, (6,))
    with pytest.raises(ValueError, match="cannot expand dimension"):
        expand_chunk(source, (2, 4))


@pytest.mark.parametrize(
    "left_shape,right_shape",
    [
        ((5, 7), (7, 3)),
        ((8, 8), (2, 3, 8, 8)),
        ((2, 1, 4, 6), (1, 3, 6, 5)),
    ],
)
def test_matmul_chunks_runs_native_broadcasted_batches(
    gl, left_shape, right_shape
):
    left_values = (
        np.arange(np.prod(left_shape), dtype=np.float32).reshape(left_shape)
        / 17.0
    )
    right_values = (
        np.arange(np.prod(right_shape), dtype=np.float32).reshape(right_shape)
        / 13.0
    )
    left = GLChunk.from_numpy(left_values).to_gpu()
    right = GLChunk.from_numpy(right_values).to_gpu()
    left.discard_host()
    right.discard_host()
    out = matmul_chunks(left, right)
    try:
        assert out.shape == np.matmul(left_values, right_values).shape
        assert out.on_gpu and not out.on_cpu
        np.testing.assert_allclose(
            out.numpy(),
            np.matmul(left_values, right_values),
            rtol=2e-5,
            atol=2e-5,
        )
    finally:
        out.release()
        left.release()
        right.release()


@pytest.mark.parametrize("dim", [0, 1, -1])
def test_topk_chunks_matches_sorted_numpy_values_and_indices(gl, dim):
    values = np.asarray(
        [
            [[2.0, 9.0, 1.0], [7.0, 3.0, 8.0], [4.0, 6.0, 5.0]],
            [[8.0, 1.0, 7.0], [3.0, 9.0, 2.0], [6.0, 4.0, 5.0]],
        ],
        dtype=np.float32,
    )
    source = GLChunk.from_numpy(values).to_gpu()
    top_values, indices = topk_chunks(source, 2, dim)
    normalized_dim = dim % values.ndim
    order = np.argsort(-values, axis=normalized_dim, kind="stable")
    expected_indices = np.take(order, np.arange(2), axis=normalized_dim)
    expected_values = np.take_along_axis(
        values, expected_indices, axis=normalized_dim
    )
    try:
        np.testing.assert_array_equal(indices.numpy(), expected_indices)
        np.testing.assert_array_equal(top_values.numpy(), expected_values)
    finally:
        top_values.release()
        indices.release()
        source.release()


def test_abstract_tensor_glsl_batched_matmul_uses_native_capability(gl):
    from src.common.tensors.abstraction import AbstractTensor

    left_values = np.arange(64, dtype=np.float32).reshape(8, 8) / 31.0
    right_values = (
        np.arange(2 * 3 * 8 * 8, dtype=np.float32).reshape(2, 3, 8, 8)
        / 29.0
    )
    with AbstractTensor.use_backend("glsl"):
        left = AbstractTensor.tensor(left_values)
        right = AbstractTensor.tensor(right_values)
        result = left @ right

    assert result.shape == (2, 3, 8, 8)
    assert result.data.on_gpu and not result.data.on_cpu
    np.testing.assert_allclose(
        result.numpy(),
        np.matmul(left_values, right_values),
        rtol=2e-5,
        atol=2e-5,
    )


def test_repeat_chunk_tiles_on_gpu(gl):
    values = np.arange(6, dtype=np.int32).reshape(1, 2, 3)
    source = GLChunk.from_numpy(values).to_gpu()
    out = repeat_chunk(source, (4, 2, 1))
    try:
        assert out.shape == (4, 4, 3)
        assert out.on_gpu and not out.on_cpu
        np.testing.assert_array_equal(
            out.numpy(), np.tile(values, (4, 2, 1))
        )
    finally:
        out.release()
        source.release()


@pytest.mark.parametrize("op", ["sum", "mean", "min", "max", "any", "all"])
@pytest.mark.parametrize("dim,keepdim", [(None, False), (1, False), (-1, True)])
def test_reduce_chunk_matches_numpy(gl, op, dim, keepdim):
    values = np.arange(24, dtype=np.float32).reshape(2, 3, 4) - 7.0
    source_values = values != 0 if op in {"any", "all"} else values
    source = GLChunk.from_numpy(source_values).to_gpu()
    out = reduce_chunk(source, op, dim, keepdim)
    expected = getattr(np, op)(
        source_values,
        axis=dim,
        keepdims=keepdim,
    )
    try:
        assert out.on_gpu and not out.on_cpu
        np.testing.assert_allclose(
            out.numpy(), expected, rtol=RTOL, atol=ATOL
        )
    finally:
        out.release()
        source.release()


@pytest.mark.parametrize("dim", [0, 1, -1])
def test_cumsum_chunk_matches_numpy_axis_scan(gl, dim):
    values = np.arange(24, dtype=np.int32).reshape(2, 3, 4) - 5
    source = GLChunk.from_numpy(values).to_gpu()
    out = cumsum_chunk(source, dim)
    try:
        assert out.shape == values.shape
        assert out.on_gpu and not out.on_cpu
        np.testing.assert_array_equal(
            out.numpy(), np.cumsum(values, axis=dim, dtype=np.int32)
        )
    finally:
        out.release()
        source.release()


def test_cumsum_bool_promotes_to_integer_counts(gl):
    values = np.asarray([[True, False, True], [False, True, True]])
    source = GLChunk.from_numpy(values).to_gpu()
    out = cumsum_chunk(source, 1)
    try:
        assert out.dtype == np.dtype(np.int32)
        np.testing.assert_array_equal(
            out.numpy(), np.cumsum(values, axis=1, dtype=np.int32)
        )
    finally:
        out.release()
        source.release()


def test_launch_planner_auto_sizes_and_folds_flat_work(monkeypatch):
    from src.common.tensors.accelerator_backends import glsl_backend

    limits = GLComputeLimits(
        max_group_count=(4, 3, 2),
        max_group_size=(128, 8, 4),
        max_invocations=128,
        max_ssbo_bindings=8,
        max_compute_ssbo_blocks=8,
    )
    monkeypatch.setattr(glsl_backend, "_compute_limits", lambda: limits)

    small = plan_launch(17, preferred_local_size=256, binding_count=2)
    assert small.local_size == 32
    assert small.groups == (1, 1, 1)

    folded = plan_launch(1000, preferred_local_size=128, binding_count=2)
    assert folded.local_size == 128
    assert folded.groups == (4, 2, 1)
    assert np.prod(folded.groups) * folded.local_size >= folded.count
    assert folded.deployment.backend == "glsl"
    assert folded.deployment.compute.groups == folded.groups

    empty = plan_launch(0, binding_count=2)
    assert empty.skipped and empty.groups == (0, 0, 0)
    with pytest.raises(ValueError, match="SSBO bindings"):
        plan_launch(1, binding_count=9)
    with pytest.raises(ValueError, match="uint u_count"):
        plan_launch(0x100000000)


def test_glsl_reads_the_prebaked_matrix_without_rewriting_it(monkeypatch):
    from src.common.tensors.accelerator_backends import glsl_backend
    from src.compiler.tiling_strategy import (
        build_gemm_tile_plan,
        prebake_gemm_launch_matrix,
    )

    limits = GLComputeLimits(
        max_group_count=(65535, 65535, 65535),
        max_group_size=(256, 256, 64), max_invocations=256,
        max_ssbo_bindings=8, max_compute_ssbo_blocks=8,
    )
    monkeypatch.setattr(glsl_backend, "_compute_limits", lambda: limits)
    matrix = prebake_gemm_launch_matrix(
        build_gemm_tile_plan(192, 128, 64, 64, worker_budget=7),
        variant_key="one-universal-gemm", parameter_ids={},
        total_layout={}, core_layout={}, chunk_size=1,
    )
    interpreted = glsl_backend.plan_gemm_matrix_deployment(matrix)
    assert interpreted.module_key == "one-universal-gemm"
    assert interpreted.lane_count == 6
    assert interpreted.calls_per_lane == (1,) * 6
    assert interpreted.choice.compute.count == 6


def test_captured_matmul_dispatches_through_named_glsl_blas_intrinsic(
    monkeypatch,
):
    from src.common.tensors.accelerator_backends import glsl_backend

    program = FusedProgram(
        version=1,
        feeds={10, 11},
        steps=[OpStep(0, "matmul", [10, 11], {}, 12)],
        outputs={"result": 12},
        meta={
            10: Meta(shape=(2, 3), dtype="float32", device="glsl"),
            11: Meta(shape=(3, 4), dtype="float32", device="glsl"),
            12: Meta(shape=(2, 4), dtype="float32", device="glsl"),
        },
        extras={"kernel_kind": "matmul"},
    )
    left, right, result = object(), object(), object()
    observed = []
    monkeypatch.setattr(
        glsl_backend,
        "glslblas_gemm",
        lambda first, second: observed.append((first, second)) or result,
    )

    outputs = execute_captured_fused_program(
        CapturedFusedProgram(program, {}),
        {10: left, 11: right},
    )

    assert outputs == {"result": result}
    assert observed == [(left, right)]


@pytest.mark.parametrize(
    "start,end,step,dtype",
    [
        (0, 11, 2, np.int32),
        (5, -4, -2, np.int32),
        (0, 6, 1, np.uint32),
        (0.5, 2.0, 0.3, np.float32),
    ],
)
def test_arange_chunk_is_device_native_and_resident(
    gl, start, end, step, dtype
):
    out = arange_chunk(start, end, step, dtype=dtype)
    try:
        assert out.dtype == np.dtype(dtype)
        assert out.on_gpu and not out.on_cpu
        np.testing.assert_allclose(
            out.numpy(),
            np.arange(start, end, step, dtype=dtype),
            rtol=RTOL,
            atol=ATOL,
        )
    finally:
        out.release()


def test_arange_chunk_validates_empty_direction_step_and_dtype(gl):
    empty = arange_chunk(5, 1, 1)
    assert empty.shape == (0,)
    np.testing.assert_array_equal(empty.numpy(), np.arange(5, 1, 1))
    with pytest.raises(ValueError, match="nonzero"):
        arange_chunk(0, 5, 0)
    with pytest.raises(TypeError, match="boolean"):
        arange_chunk(0, 5, 1, dtype=np.bool_)
    with pytest.raises(ValueError, match="unsigned"):
        arange_chunk(5, 0, -1, dtype=np.uint32)


@pytest.mark.parametrize("dim", [0, 1, -1])
def test_cat_chunks_is_one_resident_structural_operation(gl, dim):
    left = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    right_shape = list(left.shape)
    right_shape[dim] = 2
    right = (
        np.arange(np.prod(right_shape), dtype=np.float32).reshape(right_shape)
        + 100
    )
    left_chunk = GLChunk.from_numpy(left).to_gpu()
    right_chunk = GLChunk.from_numpy(right).to_gpu()
    out = cat_chunks((left_chunk, right_chunk), dim=dim)
    try:
        assert out.on_gpu and not out.on_cpu
        np.testing.assert_array_equal(
            out.numpy(), np.concatenate((left, right), axis=dim)
        )
    finally:
        out.release()
        left_chunk.release()
        right_chunk.release()


@pytest.mark.parametrize("dim", [0, 1, -1])
def test_stack_chunks_is_one_resident_structural_operation(gl, dim):
    first = np.arange(2 * 3, dtype=np.int32).reshape(2, 3)
    second = first + 100
    first_chunk = GLChunk.from_numpy(first).to_gpu()
    second_chunk = GLChunk.from_numpy(second).to_gpu()
    out = stack_chunks((first_chunk, second_chunk), dim=dim)
    try:
        assert out.on_gpu and not out.on_cpu
        assert out.dtype == np.dtype(np.int32)
        np.testing.assert_array_equal(
            out.numpy(), np.stack((first, second), axis=dim)
        )
    finally:
        out.release()
        first_chunk.release()
        second_chunk.release()


def test_stack_and_cat_chunk_around_compute_stage_ssbo_limit(gl):
    values = [
        GLChunk.from_numpy(np.asarray([index], dtype=np.int32)).to_gpu()
        for index in range(20)
    ]
    stacked = stack_chunks(values, dim=0)
    concatenated = cat_chunks(values, dim=0)
    try:
        expected = np.arange(20, dtype=np.int32).reshape(20, 1)
        np.testing.assert_array_equal(stacked.numpy(), expected)
        np.testing.assert_array_equal(
            concatenated.numpy(), expected.reshape(20)
        )
    finally:
        stacked.release()
        concatenated.release()
        for value in values:
            value.release()


def test_stack_and_cat_promote_mixed_input_dtypes_in_one_dispatch(gl):
    integers = GLChunk.from_numpy(np.asarray([[1, 2]], dtype=np.int32)).to_gpu()
    floats = GLChunk.from_numpy(
        np.asarray([[3.5, 4.5]], dtype=np.float32)
    ).to_gpu()
    stacked = stack_chunks((integers, floats), dim=0)
    concatenated = cat_chunks((integers, floats), dim=0)
    try:
        assert stacked.dtype == np.dtype(np.float32)
        assert concatenated.dtype == np.dtype(np.float32)
        np.testing.assert_array_equal(
            stacked.numpy(),
            np.stack(
                (
                    np.asarray([[1, 2]], dtype=np.int32),
                    np.asarray([[3.5, 4.5]], dtype=np.float32),
                ),
                axis=0,
            ),
        )
        np.testing.assert_array_equal(
            concatenated.numpy(),
            np.concatenate(
                (
                    np.asarray([[1, 2]], dtype=np.int32),
                    np.asarray([[3.5, 4.5]], dtype=np.float32),
                ),
                axis=0,
            ),
        )
    finally:
        stacked.release()
        concatenated.release()
        integers.release()
        floats.release()


def test_reshape_chunk_stays_on_gpu_and_preserves_dtype(gl):
    values = np.arange(2 * 3 * 4, dtype=np.int32).reshape(2, 3, 4)
    source = GLChunk.from_numpy(values).to_gpu()
    source.discard_host()
    out = reshape_chunk(source, (4, -1))
    try:
        assert source.shape == (2, 3, 4)
        assert out.shape == (4, 6)
        assert out.dtype == np.dtype(np.int32)
        assert out.on_gpu and not out.on_cpu
        np.testing.assert_array_equal(out.numpy(), values.reshape(4, 6))
    finally:
        out.release()
        source.release()


def test_reshape_view_keeps_shared_storage_alive_after_source_release(gl):
    values = np.arange(12, dtype=np.float32).reshape(3, 4)
    source = GLChunk.from_numpy(values).to_gpu()
    source.discard_host()
    view = reshape_chunk(source, (2, 6))
    buffer_id = source.buffer_id
    assert view.buffer_id == buffer_id

    source.release()
    assert view.buffer_id == buffer_id and view.on_gpu
    np.testing.assert_array_equal(view.numpy(), values.reshape(2, 6))
    view.release()


def test_first_axis_prefix_slice_is_a_zero_dispatch_shared_view(gl):
    values = np.arange(30, dtype=np.float32).reshape(5, 6)
    source = GLChunk.from_numpy(values).to_gpu()
    source.discard_host()
    dispatch_stats(reset=True)

    prefix = slice_axis_chunk(source, 0, 0, 1, 3)
    try:
        assert prefix.shape == (3, 6)
        assert prefix.buffer_id == source.buffer_id
        assert dispatch_stats()["calls"] == 0
        np.testing.assert_array_equal(prefix.numpy(), values[:3])
        # A partial readback must not replace the parent storage's full cache.
        np.testing.assert_array_equal(source.numpy(), values)
    finally:
        prefix.release()
        source.release()


def test_partial_glsl_view_cannot_replace_shared_storage(gl):
    source = GLChunk.from_numpy(np.arange(12, dtype=np.float32)).to_gpu()
    prefix = source.prefix_view((6,))
    try:
        with pytest.raises(RuntimeError, match="partial GLChunk view"):
            prefix.update_numpy(np.zeros(6, dtype=np.float32))
    finally:
        prefix.release()
        source.release()


def test_zero_copy_view_retains_arena_allocation_after_source_release(gl):
    values = np.arange(12, dtype=np.float32)
    source = GLChunk.from_numpy(values).to_gpu()
    viewed = source.view((3, 4))

    source.release()
    replacement = GLChunk.from_numpy(
        np.full(12, -99.0, dtype=np.float32)
    ).to_gpu()
    try:
        np.testing.assert_array_equal(viewed.numpy(), values.reshape(3, 4))
    finally:
        replacement.release()
        viewed.release()


def test_temporary_glsl_tensor_survives_unsqueeze_view(gl):
    from src.common.tensors.abstraction import AbstractTensor

    with AbstractTensor.use_backend("glsl"):
        # The arange result has no Python name and dies immediately after
        # reshape returns.  The view must itself retain the resident slot.
        column = AbstractTensor.arange(8).to_dtype("float32").unsqueeze(1)
        row = AbstractTensor.arange(8).to_dtype("float32").unsqueeze(0)
        outer = column * (row + 0.5)

    expected = (
        np.arange(8, dtype=np.float32)[:, None]
        * (np.arange(8, dtype=np.float32)[None, :] + 0.5)
    )
    np.testing.assert_array_equal(outer.numpy(), expected)


def test_aligned_first_axis_range_slice_is_zero_dispatch(gl):
    from src.common.tensors.accelerator_backends import glsl_backend

    elements_per_alignment = (
        glsl_backend._compute_limits().ssbo_offset_alignment // 4
    )
    values = np.arange(
        4 * elements_per_alignment,
        dtype=np.float32,
    ).reshape(4, elements_per_alignment)
    source = GLChunk.from_numpy(values).to_gpu()
    source.discard_host()
    dispatch_stats(reset=True)

    middle = slice_axis_chunk(source, 0, 1, 1, 2)
    try:
        assert middle.shape == (2, elements_per_alignment)
        assert middle.buffer_id == source.buffer_id
        assert dispatch_stats()["calls"] == 0
        np.testing.assert_array_equal(middle.numpy(), values[1:3])
        np.testing.assert_array_equal(source.numpy(), values)
    finally:
        middle.release()
        source.release()


def test_dispatch_batch_preserves_dependent_kernel_results(gl):
    source = GLChunk.from_numpy(np.arange(64, dtype=np.float32)).to_gpu()
    dispatch_stats(reset=True)
    with dispatch_batch():
        doubled = run_op("mul", source, 2.0)
        shifted = run_op("add", doubled, 3.0)
        shifted.to_gpu()
    try:
        assert dispatch_stats()["calls"] == 2
        np.testing.assert_allclose(
            shifted.numpy(),
            np.arange(64, dtype=np.float32) * 2.0 + 3.0,
        )
    finally:
        shifted.release()
        doubled.release()
        source.release()


def test_stack_coalesces_deferred_fanout_into_one_producer_dispatch(gl):
    source = GLChunk.from_numpy(np.arange(64, dtype=np.float32)).to_gpu()
    dispatch_stats(reset=True)
    with fuse_elementwise():
        first = run_op("add", source, 1.0)
        second = run_op("mul", source, 2.0)
        third = run_op("sub", source, 3.0)
        stacked = stack_chunks((first, second, third), dim=0)
    try:
        assert dispatch_stats()["calls"] == 2
        np.testing.assert_allclose(
            stacked.numpy(),
            np.stack(
                (
                    np.arange(64, dtype=np.float32) + 1.0,
                    np.arange(64, dtype=np.float32) * 2.0,
                    np.arange(64, dtype=np.float32) - 3.0,
                ),
                axis=0,
            ),
        )
    finally:
        stacked.release()
        third.release()
        second.release()
        first.release()
        source.release()


def test_same_dtype_glsl_cast_is_a_zero_dispatch_view(gl):
    from src.common.tensors.accelerator_backends.glsl_tensor_backend import (
        GLSLTensorOperations,
    )

    source = GLChunk.from_numpy(np.arange(32, dtype=np.int32)).to_gpu()
    operations = GLSLTensorOperations()
    operations.data = source
    dispatch_stats(reset=True)
    cast = operations.to_dtype_("int64")
    try:
        assert cast.dtype == np.dtype(np.int32)
        assert cast.buffer_id == source.buffer_id
        assert dispatch_stats()["calls"] == 0
        np.testing.assert_array_equal(cast.numpy(), np.arange(32, dtype=np.int32))
    finally:
        cast.release()
        source.release()


@pytest.mark.parametrize(
    "shape, message",
    [
        ((-1, -1), "only one inferred"),
        ((5, 5), "incompatible"),
        ((2, -2, 3), "non-negative"),
    ],
)
def test_reshape_chunk_rejects_invalid_shapes_without_dispatch(gl, shape, message):
    source = GLChunk.from_numpy(np.arange(24, dtype=np.float32))
    try:
        with pytest.raises(ValueError, match=message):
            reshape_chunk(source, shape)
    finally:
        source.release()


@pytest.mark.parametrize(
    "dims",
    [
        (2, 0, 1),
        (1, 2, 0),
        (-1, -3, -2),
    ],
)
def test_permute_chunk_is_one_resident_planned_dispatch(gl, dims):
    values = np.arange(2 * 3 * 4, dtype=np.uint32).reshape(2, 3, 4)
    source = GLChunk.from_numpy(values).to_gpu()
    source.discard_host()
    out = permute_chunk(source, dims)
    try:
        normalized = tuple(axis % values.ndim for axis in dims)
        assert out.shape == tuple(values.shape[axis] for axis in normalized)
        assert out.dtype == np.dtype(np.uint32)
        assert out.on_gpu and not out.on_cpu
        np.testing.assert_array_equal(
            out.numpy(), np.transpose(values, normalized)
        )
    finally:
        out.release()
        source.release()


@pytest.mark.parametrize(
    "dims, message",
    [
        ((0, 1), "match tensor dimensions"),
        ((0, 0, 2), "permutation"),
        ((0, 1, 3), "out of range"),
    ],
)
def test_permute_chunk_rejects_invalid_axis_orders(dims, message):
    source = GLChunk.from_numpy(np.arange(24).reshape(2, 3, 4))
    with pytest.raises(ValueError, match=message):
        permute_chunk(source, dims)


def test_abstract_tensor_glsl_cat_and_stack_use_structural_hooks(gl):
    from src.common.tensors.abstraction import AbstractTensor
    from src.common.tensors.accelerator_backends.glsl_tensor_backend import (
        GLSLTensorOperations,
    )

    first_values = np.arange(6, dtype=np.float32).reshape(2, 3)
    second_values = first_values + 10
    first = GLSLTensorOperations._tensor_from_list(first_values)
    second = GLSLTensorOperations._tensor_from_list(second_values)

    with AbstractTensor.use_backend("glsl"):
        concatenated = GLSLTensorOperations.cat((first, second), dim=1)
        stacked = GLSLTensorOperations.stack((first, second), dim=-1)

    assert isinstance(concatenated.data, GLChunk) and concatenated.data.on_gpu
    assert isinstance(stacked.data, GLChunk) and stacked.data.on_gpu
    np.testing.assert_array_equal(
        concatenated.numpy(),
        np.concatenate((first_values, second_values), axis=1),
    )
    np.testing.assert_array_equal(
        stacked.numpy(),
        np.stack((first_values, second_values), axis=-1),
    )


def test_abstract_tensor_glsl_reshape_view_and_flatten_use_backend_hook(gl):
    from src.common.tensors.abstraction import AbstractTensor
    from src.common.tensors.accelerator_backends.glsl_tensor_backend import (
        GLSLTensorOperations,
    )

    values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    tensor = GLSLTensorOperations._tensor_from_list(values)
    tensor.data.to_gpu().discard_host()
    with AbstractTensor.use_backend("glsl"):
        reshaped = tensor.reshape(4, -1)
        viewed = reshaped.view((2, 12))
        flattened = viewed.flatten()

    for result, expected_shape in (
        (reshaped, (4, 6)),
        (viewed, (2, 12)),
        (flattened, (24,)),
    ):
        assert isinstance(result.data, GLChunk)
        assert result.shape == expected_shape
        assert result.data.on_gpu and not result.data.on_cpu

    assert {
        tensor.data.buffer_id,
        reshaped.data.buffer_id,
        viewed.data.buffer_id,
        flattened.data.buffer_id,
    } == {tensor.data.buffer_id}

    for result, expected_shape in (
        (reshaped, (4, 6)),
        (viewed, (2, 12)),
        (flattened, (24,)),
    ):
        np.testing.assert_array_equal(
            result.numpy(), values.reshape(expected_shape)
        )


@pytest.mark.parametrize(
    "dim, expected_shape",
    [
        (0, (1, 2, 3, 4)),
        (2, (2, 3, 1, 4)),
        (-1, (2, 3, 4, 1)),
        (-4, (1, 2, 3, 4)),
    ],
)
def test_abstract_tensor_glsl_unsqueeze_uses_universal_reshape_fallback(
    gl, dim, expected_shape
):
    from src.common.tensors.abstraction import AbstractTensor
    from src.common.tensors.accelerator_backends.glsl_tensor_backend import (
        GLSLTensorOperations,
    )

    values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    tensor = GLSLTensorOperations._tensor_from_list(values)
    tensor.data.to_gpu().discard_host()
    with AbstractTensor.use_backend("glsl"):
        result = tensor.unsqueeze(dim)

    assert isinstance(result.data, GLChunk)
    assert result.shape == expected_shape
    assert result.data.on_gpu and not result.data.on_cpu
    np.testing.assert_array_equal(result.numpy(), np.expand_dims(values, dim))


def test_abstract_tensor_glsl_unsqueeze_rejects_out_of_range_dimension(gl):
    from src.common.tensors.abstraction import AbstractTensor
    from src.common.tensors.accelerator_backends.glsl_tensor_backend import (
        GLSLTensorOperations,
    )

    tensor = GLSLTensorOperations._tensor_from_list(
        np.arange(6, dtype=np.float32).reshape(2, 3)
    )
    with AbstractTensor.use_backend("glsl"):
        with pytest.raises(ValueError, match="out of range"):
            tensor.unsqueeze(3)
        with pytest.raises(ValueError, match="out of range"):
            tensor.unsqueeze(-4)


def test_abstract_tensor_glsl_permute_transpose_and_swapaxes(gl):
    from src.common.tensors.abstraction import AbstractTensor
    from src.common.tensors.accelerator_backends.glsl_tensor_backend import (
        GLSLTensorOperations,
    )

    values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    tensor = GLSLTensorOperations._tensor_from_list(values)
    tensor.data.to_gpu().discard_host()
    with AbstractTensor.use_backend("glsl"):
        permuted = tensor.permute(2, 0, 1)
        transposed = tensor.transpose(0, 2)
        swapped = tensor.swapaxes(-1, 0)

    for result, expected in (
        (permuted, np.transpose(values, (2, 0, 1))),
        (transposed, np.swapaxes(values, 0, 2)),
        (swapped, np.swapaxes(values, -1, 0)),
    ):
        assert isinstance(result.data, GLChunk)
        assert result.data.on_gpu and not result.data.on_cpu
        np.testing.assert_array_equal(result.numpy(), expected)


def test_abstract_tensor_glsl_arange_creation_hook(gl):
    from src.common.tensors.abstraction import AbstractTensor

    with AbstractTensor.use_backend("glsl"):
        integer = AbstractTensor.arange(7)
        descending = AbstractTensor.arange(4, -3, -2, dtype="float")

    assert isinstance(integer.data, GLChunk)
    assert integer.data.on_gpu and integer.dtype == np.dtype(np.int32)
    np.testing.assert_array_equal(
        integer.numpy(), np.arange(7, dtype=np.int32)
    )
    assert isinstance(descending.data, GLChunk)
    assert descending.data.on_gpu and descending.dtype == np.dtype(np.float32)
    np.testing.assert_array_equal(
        descending.numpy(), np.arange(4, -3, -2, dtype=np.float32)
    )


def test_abstract_tensor_glsl_chain_stays_resident_until_numpy(gl):
    from src.common.tensors.accelerator_backends.glsl_tensor_backend import (
        GLSLTensorOperations,
    )

    x = GLSLTensorOperations._tensor_from_list(
        np.linspace(0.1, 1.0, 32, dtype=np.float32)
    )
    x.data.to_gpu().discard_host()
    result = ((x * 2.0) + 1.0).sin()
    assert isinstance(result.data, GLChunk)
    assert result.data.on_gpu and not result.data.on_cpu
    np.testing.assert_allclose(
        result.numpy(),
        np.sin(np.linspace(0.1, 1.0, 32, dtype=np.float32) * 2.0 + 1.0),
        rtol=RTOL,
        atol=ATOL,
    )


def test_abstract_tensor_glsl_getitem_supports_native_index_forms():
    from src.common.tensors.accelerator_backends.glsl_tensor_backend import (
        GLSLTensorOperations,
    )

    values = np.arange(3 * 4 * 5, dtype=np.int32).reshape(3, 4, 5)
    tensor = GLSLTensorOperations._tensor_from_list(values)

    selections = (
        1,
        (1, 2, 3),
        (slice(None), 2, slice(None, None, -1)),
        (Ellipsis, slice(1, 4, 2)),
        (np.asarray([2, 0]), slice(None), 3),
        (None, 1, slice(1, None), Ellipsis),
    )
    for index in selections:
        result = tensor[index]
        assert isinstance(result.data, GLChunk)
        np.testing.assert_array_equal(result.numpy(), values[index])

    scalar = tensor[1, 2, 3]
    assert scalar.shape == ()
    assert scalar.item() == values[1, 2, 3]

    index_tensor = GLSLTensorOperations._tensor_from_list(
        np.asarray([2, 0], dtype=np.int32)
    )
    result = tensor[index_tensor]
    assert isinstance(result.data, GLChunk)
    np.testing.assert_array_equal(result.numpy(), values[[2, 0]])


def test_abstract_tensor_glsl_getitem_reads_gpu_resident_chunk(gl):
    from src.common.tensors.accelerator_backends.glsl_tensor_backend import (
        GLSLTensorOperations,
    )

    values = np.arange(6 * 7, dtype=np.float32).reshape(6, 7)
    tensor = GLSLTensorOperations._tensor_from_list(values)
    tensor.data.to_gpu().discard_host()

    result = tensor[::-2, 1:6:2]
    assert isinstance(result.data, GLChunk)
    np.testing.assert_array_equal(result.numpy(), values[::-2, 1:6:2])


def test_abstract_tensor_glsl_setitem_supports_native_index_forms():
    from src.common.tensors.accelerator_backends.glsl_tensor_backend import (
        GLSLTensorOperations,
    )

    expected = np.arange(3 * 4, dtype=np.int32).reshape(3, 4)
    tensor = GLSLTensorOperations._tensor_from_list(expected.copy())

    tensor[1, 2] = -7
    expected[1, 2] = -7

    replacement = GLSLTensorOperations._tensor_from_list(
        np.asarray([40, 41], dtype=np.int32)
    )
    tensor[[2, 0], 1] = replacement
    expected[[2, 0], 1] = np.asarray([40, 41], dtype=np.int32)

    tensor[:, ::2] = 9
    expected[:, ::2] = 9
    np.testing.assert_array_equal(tensor.numpy(), expected)


def test_abstract_tensor_glsl_setitem_updates_gpu_resident_chunk(gl):
    from src.common.tensors.accelerator_backends.glsl_tensor_backend import (
        GLSLTensorOperations,
    )

    expected = np.arange(6 * 7, dtype=np.float32).reshape(6, 7)
    tensor = GLSLTensorOperations._tensor_from_list(expected.copy())
    tensor.data.to_gpu().discard_host()

    tensor[::-2, 1:6:2] = -3.5
    expected[::-2, 1:6:2] = -3.5
    assert tensor.data.on_gpu and not tensor.data.on_cpu
    np.testing.assert_array_equal(tensor.numpy(), expected)

    tensor.data.to_gpu().discard_host()
    np.testing.assert_array_equal(tensor.numpy(), expected)


def test_base_clamp_composes_glsl_primitives_when_hook_is_missing(gl):
    from src.common.tensors.accelerator_backends.glsl_tensor_backend import (
        GLSLTensorOperations,
    )

    values = np.asarray([-2.0, -0.25, 0.5, 3.0], dtype=np.float32)
    tensor = GLSLTensorOperations._tensor_from_list(values)
    result = tensor.clamp(-0.5, 1.0)

    assert isinstance(result.data, GLChunk)
    np.testing.assert_allclose(
        result.numpy(),
        np.clip(values, -0.5, 1.0),
        rtol=RTOL,
        atol=ATOL,
    )


def test_fused_program_broadcasts_scalar_feeds(gl):
    program = _program(
        [0, 1],
        [("mul", 2, [0, 1], {})],
        2,
    )
    values = np.arange(32, dtype=np.float32)
    got = execute_program(
        program, [values, np.asarray([2.5], dtype=np.float32)]
    ).numpy()
    np.testing.assert_allclose(got, values * 2.5, rtol=RTOL, atol=ATOL)


def test_glchunk_updates_existing_parameter_buffer(gl):
    chunk = GLChunk.from_numpy(np.asarray([1.0], dtype=np.float32)).to_gpu()
    buffer_id = chunk.buffer_id
    chunk.update_numpy(np.asarray([3.0], dtype=np.float32)).to_gpu()
    assert chunk.buffer_id == buffer_id
    np.testing.assert_allclose(chunk.numpy(), [3.0], rtol=RTOL, atol=ATOL)
    chunk.release()


def test_feed_count_mismatch_is_an_error(gl):
    program = _program([0, 1], [("add", 2, [0, 1], {})], 2)
    with pytest.raises(ValueError, match="expected 2 feeds"):
        execute_program(program, [np.ones(4, dtype=np.float32)])


def test_large_input_covers_many_workgroups(gl):
    n = 1_000_003  # prime: exercises the bounds guard on a ragged final group
    a = np.linspace(1.0, 2.0, n, dtype=np.float32)
    got = run_op("mul", a, a).numpy()
    np.testing.assert_allclose(got, a * a, rtol=1e-4, atol=1e-5)
