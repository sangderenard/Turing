from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from src.common.tensors.extended_precision import ComplexPrecision, Precision
from src.common.tensors.numpy_backend import NumPyTensorOperations


def _tensor(value, dtype=np.float64):
    tensor = NumPyTensorOperations()
    tensor.data = np.asarray(value, dtype=dtype)
    return tensor


def _fraction_precision(text: str, width: int = 2):
    exact = Fraction(text)
    remainder = exact
    terms = []
    for _index in range(width):
        term = float(remainder)
        terms.append(_tensor([term]))
        remainder -= Fraction.from_float(term)
    return Precision(terms, width), exact


def _represented_fraction(value: Precision) -> Fraction:
    return sum(
        (
            Fraction.from_float(float(term.tolist()[0]))
            for term in value.terms()
        ),
        Fraction(),
    )


def test_complex_precision_keeps_algebra_components_separate_from_limbs():
    real, _ = _fraction_precision("1.0000000000000000000000000000001")
    imag, _ = _fraction_precision("2.0000000000000000000000000000003")
    value = ComplexPrecision(real, imag, 2)

    assert value.shape == (1,)
    assert value.limbs == 2
    assert value.components() == (value.real, value.imag)
    assert value.real.to_float_lists()[1] != [0.0]
    assert value.imag.to_float_lists()[1] != [0.0]

    native = _tensor([1.0 + 2.0j], np.complex128)
    with pytest.raises(TypeError, match="ComplexPrecision"):
        Precision.of(native, 2)
    with pytest.raises(TypeError, match="ComplexPrecision"):
        Precision([1.0 + 2.0j, 0.0j], 2)


def test_complex_precision_multiply_retains_information_native_complex_loses():
    ar, ar_exact = _fraction_precision("1.0000000000000000000000000000001")
    ai, ai_exact = _fraction_precision("2.0000000000000000000000000000003")
    br, br_exact = _fraction_precision("3.0000000000000000000000000000007")
    bi, bi_exact = _fraction_precision("-4.0000000000000000000000000000009")

    result = ComplexPrecision(ar, ai, 2) * ComplexPrecision(br, bi, 2)
    exact_real = ar_exact * br_exact - ai_exact * bi_exact
    exact_imag = ar_exact * bi_exact + ai_exact * br_exact
    wide_error = (
        abs(_represented_fraction(result.real) - exact_real)
        + abs(_represented_fraction(result.imag) - exact_imag)
    )

    native = complex(float(ar_exact), float(ai_exact)) * complex(
        float(br_exact), float(bi_exact)
    )
    native_error = (
        abs(Fraction.from_float(native.real) - exact_real)
        + abs(Fraction.from_float(native.imag) - exact_imag)
    )
    assert wide_error < native_error
    assert result.conjugate().collapse().tolist() == [11.0 - 2.0j]

    quotient = ComplexPrecision(_tensor([3.0]), _tensor([4.0]), 2) / (
        ComplexPrecision(_tensor([1.0]), _tensor([-2.0]), 2)
    )
    assert quotient.collapse().tolist() == [-1.0 + 2.0j]


def test_torch_abstract_tensor_preserves_complex_dtype_and_components():
    torch = pytest.importorskip("torch")
    from src.common.tensors import AbstractTensor
    from src.common.tensors.torch_backend import PyTorchTensorOperations

    real = PyTorchTensorOperations()
    real.data = torch.tensor([1.0, 2.0], dtype=torch.float64)
    imag = PyTorchTensorOperations()
    imag.data = torch.tensor([3.0, 4.0], dtype=torch.float64)

    value = AbstractTensor.complex(real, imag)
    assert value.dtype == torch.complex128
    assert AbstractTensor.real(value).tolist() == [1.0, 2.0]
    assert AbstractTensor.imag(value).tolist() == [3.0, 4.0]
    assert real.to_dtype("complex128").dtype == torch.complex128
    with pytest.raises(ValueError, match="unrecognised dtype"):
        real.to_dtype("not-a-dtype")


def test_source_compiler_lowers_complex_precision_as_two_real_expansions(
    tmp_path,
):
    from src.compiler.extraction_contract import ExtractionContract
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
    from src.compiler.ir_identities import PRECISION_PIPELINE_METADATA
    from src.compiler.ssa_llvm_backend import (
        compile_artifact,
        emit_ssa_function_to_llvm,
        prepare_artifact_execution,
    )

    source = """
def complex_mul(
    ar: Precision[2], ai: Precision[2],
    br: Precision[2], bi: Precision[2],
):
    product_real = ar * br - ai * bi
    product_imag = ar * bi + ai * br
    denominator = br * br + bi * bi
    quotient_real = (ar * br + ai * bi) / denominator
    quotient_imag = (ai * br - ar * bi) / denominator
    return product_real, product_imag, quotient_real, quotient_imag
"""
    values = [
        {
            "function": "complex_mul",
            "parameter": parameter,
            "storage": "scalar",
            "dtype": "float64",
            "rank": 0,
            "python_type": "src.common.tensors.extended_precision.Precision",
        }
        for parameter in ("ar", "ai", "br", "bi")
    ]
    policy = ExtractionContract(
        Path("extraction_contracts/program_extraction.yaml")
    ).with_program_abi({"records": {}, "bindings": [], "values": values})

    module, outputs, _exports = lower_ast_source_to_ssa(
        source, "complex_mul", name="complex_precision",
        extraction_contract=policy,
    )
    receipt = module.metadata[PRECISION_PIPELINE_METADATA]
    root = module.functions["complex_precision__complex_mul"]
    region = module.functions[
        "complex_precision__complex_mul__planned_region_0"
    ]

    assert receipt["status"] == "lowered"
    assert len(root.args) == len(region.args) == 8
    assert len(outputs[root.name]) == 4
    assert all(len(limbs) == 2 for _, limbs in root.metadata[
        "precision_lowered_values"
    ])

    artifact = emit_ssa_function_to_llvm(module, root.name)
    assert artifact.shortfalls == ()
    assert "llvm.fma.f64" in artifact.llvm_ir
    assert artifact.buffer_dtypes == ("double",) * 12

    parameter_ids = dict(root.metadata["parameter_names"])
    lowered_ids = dict(root.metadata["precision_lowered_values"])
    components = {
        "ar": (1.0, 1.0e-30),
        "ai": (2.0, 3.0e-30),
        "br": (3.0, 7.0e-30),
        "bi": (-4.0, -9.0e-30),
    }
    feeds = {
        limb_id: np.asarray([component], dtype=np.float64)
        for name, values_by_limb in components.items()
        for limb_id, component in zip(
            lowered_ids[parameter_ids[name]], values_by_limb,
        )
    }
    native = compile_artifact(
        artifact, directory=tmp_path / "complex-precision",
        optimization="O0",
    )
    execution = prepare_artifact_execution(native, feeds).run()
    real_id, imag_id, quotient_real_id, quotient_imag_id = outputs[root.name]
    real = float(np.asarray(execution.buffers[int(real_id.id)]).reshape(-1)[0])
    imag = float(np.asarray(execution.buffers[int(imag_id.id)]).reshape(-1)[0])
    quotient_real = float(np.asarray(
        execution.buffers[int(quotient_real_id.id)]
    ).reshape(-1)[0])
    quotient_imag = float(np.asarray(
        execution.buffers[int(quotient_imag_id.id)]
    ).reshape(-1)[0])
    assert real == pytest.approx(11.0, rel=0, abs=2.0e-15)
    assert imag == pytest.approx(2.0, rel=0, abs=2.0e-15)
    assert quotient_real == pytest.approx(-0.2, rel=0, abs=2.0e-15)
    assert quotient_imag == pytest.approx(0.4, rel=0, abs=2.0e-15)
