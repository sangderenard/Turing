import ctypes

import numpy as np
import pytest

from src.common.tensors.accelerator_backends.native_library import (
    ToolchainError,
    compile_and_load,
    detect_toolchains,
    preferred_toolchain,
)

AXPY = r"""
int axpy(const double *a, const double *b, double *out, int n) {
    for (int i = 0; i < n; ++i) {
        out[i] = a[i] * 2.5 + b[i];
    }
    return 1;
}

int turing_shell_entry(void *context, unsigned long long *device_ns) {
    void **slots = (void **)context;
    if (device_ns) {
        *device_ns = 0;
    }
    return axpy(
        (const double *)slots[0],
        (const double *)slots[1],
        (double *)slots[2],
        *(int *)slots[3]
    );
}
"""

needs_toolchain = pytest.mark.skipif(
    not detect_toolchains(), reason="no native toolchain available"
)


def test_toolchains_are_ordered_with_gnu_first():
    """A GNU toolchain builds from a bare environment; MSVC does not."""

    toolchains = detect_toolchains()
    if not toolchains:
        pytest.skip("no native toolchain available")
    assert preferred_toolchain() is toolchains[0]
    # MSVC may only appear when its developer environment is initialised.
    for toolchain in toolchains:
        if toolchain.kind == "msvc":
            import os

            assert os.environ.get("INCLUDE") and os.environ.get("LIB")


@needs_toolchain
def test_compiles_to_a_plain_shared_library_not_an_extension(tmp_path):
    library = compile_and_load(AXPY, name="probe_axpy", directory=tmp_path)

    assert library.path.exists()
    # A CPython extension would carry the interpreter ABI tag in its name.
    assert "cpython" not in library.path.name
    assert library.path.suffix in (".dll", ".so", ".dylib")


@needs_toolchain
def test_exposes_a_raw_callable_address_for_the_launch_shell(tmp_path):
    library = compile_and_load(AXPY, name="probe_addr", directory=tmp_path)

    address = library.address("turing_shell_entry")
    assert isinstance(address, int)
    assert address != 0


@needs_toolchain
def test_missing_symbol_is_reported_not_silently_zero(tmp_path):
    library = compile_and_load(AXPY, name="probe_missing", directory=tmp_path)

    with pytest.raises(ToolchainError):
        library.address("no_such_symbol")


@needs_toolchain
def test_compiled_kernel_is_numerically_correct(tmp_path):
    library = compile_and_load(
        AXPY,
        name="probe_numeric",
        directory=tmp_path,
        extra_flags=("-march=native",),
    )
    pointer = ctypes.POINTER(ctypes.c_double)
    axpy = library.function(
        "axpy",
        restype=ctypes.c_int,
        argtypes=[pointer, pointer, pointer, ctypes.c_int],
    )

    rng = np.random.default_rng(0)
    a = rng.standard_normal(4096)
    b = rng.standard_normal(4096)
    out = np.zeros(4096)
    axpy(
        a.ctypes.data_as(pointer),
        b.ctypes.data_as(pointer),
        out.ctypes.data_as(pointer),
        a.size,
    )

    # Not bit-exact against NumPy: -march=native contracts a*2.5+b into a
    # single FMA, which rounds once instead of twice.  That is more accurate
    # than the separate multiply and add, so the tolerance is one ulp of the
    # data scale rather than zero.
    expected = a * 2.5 + b
    tolerance = 2 * np.spacing(float(np.max(np.abs(expected))))
    assert np.max(np.abs(out - expected)) <= tolerance


@needs_toolchain
def test_compilation_failure_surfaces_the_compiler_diagnostic(tmp_path):
    with pytest.raises(ToolchainError) as raised:
        compile_and_load(
            "int broken(void) { return undefined_symbol; }",
            name="probe_broken",
            directory=tmp_path,
        )
    assert "undefined_symbol" in str(raised.value)
