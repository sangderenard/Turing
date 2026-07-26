"""GLSL compute-shader backend: real GPU execution checked against numpy.

These tests require a working OpenGL 4.3+ context and skip (never silently pass)
when one is unavailable. The emitter tests need no GPU at all and always run.

numpy is the behavioural oracle, per docs/c_backend_status.md's rule for the C
backend. Tolerances are float32-appropriate: this backend narrows to float32 on
the way to the GPU, which is a documented contract, not an accident.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.common.tensors.accelerator_backends.glsl_backend import (
    GLSL_OPS,
    GLChunk,
    GLContextUnavailable,
    GLSLUnsupportedOp,
    canonical_op,
    emit_op_source,
    emit_program_source,
    execute_program,
    gl_context_info,
    require_gl_context,
    run_op,
    shader_cache_stats,
)
from src.common.tensors.fused_ir import FusedProgram, OpStep

from src.common.tensors.accelerator_backends import gl_context as glctx

RTOL, ATOL = 1e-5, 1e-5


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
    assert canonical_op("less_equal") == ("le", False)


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
    assert src.count("buffer") == 3          # 2 feeds + 1 output, no temporaries
    assert "float s2 = s0 + s1;" in src
    assert "float s3 = exp(s2);" in src
    assert "outbuf[gid] = s3;" in src


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
    src = emit_op_source("mod")
    assert "floor(" in src and "mod(" not in src.split("void main")[1]


def test_round_is_half_away_from_zero():
    """GLSL round() may break ties either way; C round() does not."""
    src = emit_op_source("round")
    assert "sign(" in src and "floor(abs(" in src


def test_every_advertised_op_emits():
    for op in sorted(GLSL_OPS):
        src = emit_op_source(op)
        assert src.startswith("#version 430")
        assert "outbuf[gid]" in src


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
    owned = GLChunk.from_numpy(np.ones(8, dtype=np.float32)).to_gpu()
    wrapper = GLChunk.wrap(owned.buffer_id, (8,))
    assert wrapper.on_gpu
    np.testing.assert_allclose(wrapper.to_cpu(), np.ones(8), rtol=RTOL, atol=ATOL)

    wrapper.release()
    # The original still works because release() did not touch a wrapped buffer.
    np.testing.assert_allclose(owned.to_cpu(), np.ones(8), rtol=RTOL, atol=ATOL)
    owned.release()


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
    "lt": lambda a, b: (a < b).astype(np.float32),
    "le": lambda a, b: (a <= b).astype(np.float32),
    "gt": lambda a, b: (a > b).astype(np.float32),
    "ge": lambda a, b: (a >= b).astype(np.float32),
    "eq": lambda a, b: (a == b).astype(np.float32),
    "ne": lambda a, b: (a != b).astype(np.float32),
}


@pytest.mark.parametrize("op", sorted(BINARY_CASES))
def test_binary_ops_match_numpy(gl, op):
    rng = np.random.default_rng(0xC0FFEE)
    a = rng.uniform(0.5, 4.0, size=257).astype(np.float32)   # not a multiple of 256
    b = rng.uniform(0.5, 4.0, size=257).astype(np.float32)
    got = run_op(op, a, b).numpy()
    np.testing.assert_allclose(got, BINARY_CASES[op](a, b), rtol=RTOL, atol=ATOL)


UNARY_CASES = {
    "sqrt": np.sqrt, "exp": np.exp, "log": np.log, "neg": np.negative,
    "abs": np.abs, "trunc": np.trunc, "floor": np.floor, "ceil": np.ceil,
}


@pytest.mark.parametrize("op", sorted(UNARY_CASES))
def test_unary_ops_match_numpy(gl, op):
    rng = np.random.default_rng(7)
    a = rng.uniform(0.25, 6.0, size=300).astype(np.float32)
    got = run_op(op, a).numpy()
    np.testing.assert_allclose(got, UNARY_CASES[op](a), rtol=RTOL, atol=ATOL)


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
    with pytest.raises(ValueError, match="share one shape"):
        run_op("add", np.ones(4, dtype=np.float32), np.ones(8, dtype=np.float32))


def test_feed_count_mismatch_is_an_error(gl):
    program = _program([0, 1], [("add", 2, [0, 1], {})], 2)
    with pytest.raises(ValueError, match="expected 2 feeds"):
        execute_program(program, [np.ones(4, dtype=np.float32)])


def test_large_input_covers_many_workgroups(gl):
    n = 1_000_003  # prime: exercises the bounds guard on a ragged final group
    a = np.linspace(1.0, 2.0, n, dtype=np.float32)
    got = run_op("mul", a, a).numpy()
    np.testing.assert_allclose(got, a * a, rtol=1e-4, atol=1e-5)
