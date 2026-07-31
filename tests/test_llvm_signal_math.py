import ctypes
import math

import numpy as np
import pytest
from llvmlite import binding as llvm

from src.common.tensors.accelerator_backends.llvm_signal_math import (
    LLVMTrigSolver,
    build_llvm_signal_math,
    link_llvm_trig_solver,
)


def _engine(source):
    llvm.initialize_native_target()
    llvm.initialize_native_asmprinter()
    module = llvm.parse_assembly(source)
    module.verify()
    engine = llvm.create_mcjit_compiler(
        module,
        llvm.Target.from_default_triple().create_target_machine(),
    )
    engine.finalize_object()
    return engine


def _scalar_function(engine, name):
    address = engine.get_function_address(name)
    assert address
    return ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double)(address)


def test_epsilon_selects_lut_resolution_and_continuous_degree():
    loose = build_llvm_signal_math(1.0e-3)
    tight = build_llvm_signal_math(1.0e-6)

    assert loose.lut_error_bound <= loose.epsilon
    assert tight.lut_error_bound <= tight.epsilon
    assert tight.lut_intervals > loose.lut_intervals
    assert tight.continuous_degree > loose.continuous_degree
    assert loose.continuous_error_bound <= loose.epsilon
    assert tight.continuous_error_bound <= tight.epsilon
    assert "selected_epsilon=9.9999999999999995e-07" in tight.llvm_ir


@pytest.mark.parametrize("epsilon", [1.0e-3, 1.0e-6])
def test_lut_and_continuous_sine_cosine_respect_selected_epsilon(epsilon):
    signal = build_llvm_signal_math(epsilon)
    engine = _engine(signal.llvm_ir)
    functions = {
        name: _scalar_function(engine, name)
        for name in (
            "turing_lut_sin_f64",
            "turing_lut_cos_f64",
            "turing_continuous_sin_f64",
            "turing_continuous_cos_f64",
        )
    }
    samples = np.linspace(-4.0 * math.pi, 4.0 * math.pi, 257)

    for name, function in functions.items():
        actual = np.asarray([function(float(value)) for value in samples])
        expected = (
            np.sin(samples) if "sin_f64" in name else np.cos(samples)
        )
        np.testing.assert_allclose(
            actual,
            expected,
            atol=max(epsilon * 1.1, 2.0e-15),
            rtol=0.0,
        )


def test_lut_and_continuous_tangent_share_the_selected_trig_solver():
    signal = build_llvm_signal_math(1.0e-6)
    engine = _engine(signal.llvm_ir)
    lut = _scalar_function(engine, "turing_lut_tan_f64")
    continuous = _scalar_function(engine, "turing_continuous_tan_f64")
    samples = np.linspace(-1.2, 1.2, 121)
    expected = np.tan(samples)

    np.testing.assert_allclose(
        [lut(float(value)) for value in samples],
        expected,
        atol=3.0e-6,
        rtol=3.0e-6,
    )
    np.testing.assert_allclose(
        [continuous(float(value)) for value in samples],
        expected,
        atol=3.0e-6,
        rtol=3.0e-6,
    )


@pytest.mark.parametrize("epsilon", [0.0, -1.0, float("nan"), 1.0e-16, 0.1])
def test_invalid_trig_epsilon_is_rejected(epsilon):
    with pytest.raises(ValueError, match="trig epsilon"):
        build_llvm_signal_math(epsilon)


@pytest.mark.parametrize(
    ("solver", "selected_symbol"),
    [
        (LLVMTrigSolver.LUT, "turing_lut_sin_f64"),
        (LLVMTrigSolver.CONTINUOUS, "turing_continuous_sin_f64"),
    ],
)
def test_selected_solver_replaces_libm_calls_and_executes(solver, selected_symbol):
    source = """
        source_filename = "signal-consumer"
        declare double @sin(double)
        define double @consumer(double %x) {
        entry:
          %result = call double @sin(double %x)
          ret double %result
        }
    """
    linked, metadata = link_llvm_trig_solver(
        source,
        solver,
        epsilon=1.0e-6,
    )

    assert metadata is not None
    assert f"call double @{selected_symbol}" in linked
    engine = _engine(linked)
    consumer = _scalar_function(engine, "consumer")
    assert consumer(0.75) == pytest.approx(math.sin(0.75), abs=1.1e-6)
