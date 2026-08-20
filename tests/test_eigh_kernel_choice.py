"""``eigh`` has two rotation kernels; they must be the same decomposition.

The jacobi kernel is the definition (pure AbstractTensor, differentiable,
batch-safe). The blas kernel runs the identical sweep with every plane
rotation issued as one admission-verified ``rot`` launch through the
kernel bank. These tests pin the seam between them: that the choice is
made once and stated, that a refused choice raises instead of silently
degrading, and that the fast kernel computes what the definitional one
computes.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.tensors.abstraction import AbstractTensor
from src.common.tensors.abstraction_methods.eigen import (
    EIGH_KERNELS, eigh, select_eigh_kernel,
)


def _symmetric(n: int, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    matrix = rng.uniform(-1.0, 1.0, size=(n, n))
    return (matrix + matrix.T) / 2.0


@pytest.fixture(scope="module")
def sample():
    # One shared matrix so the bank is opened (and rot compiled) once for
    # the module rather than per test -- the cold start is ~3s.
    return _symmetric(6)


def test_the_named_kernels_are_the_ones_eigh_offers():
    assert EIGH_KERNELS == ("jacobi", "blas")


def test_a_plain_matrix_auto_selects_the_compiled_kernel(sample):
    A = AbstractTensor.get_tensor(sample.tolist())
    assert select_eigh_kernel(A, "auto") == "blas"
    # An explicit request is honoured in both directions: 'auto' choosing
    # blas must not make 'jacobi' unreachable.
    assert select_eigh_kernel(A, "jacobi") == "jacobi"
    assert select_eigh_kernel(A, "blas") == "blas"


def test_a_batched_input_falls_to_jacobi_under_auto_and_refuses_under_blas():
    batched = np.stack([_symmetric(4, seed=1), _symmetric(4, seed=2)])
    A = AbstractTensor.get_tensor(batched.tolist())
    assert select_eigh_kernel(A, "auto") == "jacobi"
    # The point of the seam: asking for blas and getting jacobi would make
    # any measurement of "the compiled path" a measurement of the other one.
    with pytest.raises(ValueError, match="rank 3"):
        select_eigh_kernel(A, "blas")


def test_an_unknown_method_is_refused():
    A = AbstractTensor.get_tensor(_symmetric(3).tolist())
    with pytest.raises(ValueError, match="method must be one of"):
        select_eigh_kernel(A, "lapack")


def test_both_kernels_decompose_the_same_matrix(sample):
    A = AbstractTensor.get_tensor(sample.tolist())
    w_jacobi, v_jacobi = eigh(A, method="jacobi")
    w_blas, v_blas = eigh(A, method="blas")

    w_jacobi = np.asarray(w_jacobi, dtype=float)
    w_blas = np.asarray(w_blas, dtype=float)
    v_jacobi = np.asarray(v_jacobi, dtype=float)
    v_blas = np.asarray(v_blas, dtype=float)

    # Tolerance, not equality: the kernels are permitted to reassociate
    # (rot folds the sign into s), even though they happen to agree
    # bit-exactly today.
    assert w_blas == pytest.approx(w_jacobi, abs=1e-12)
    # Eigenvectors are defined up to sign, so compare the alignment.
    alignment = np.abs((v_jacobi * v_blas).sum(axis=0))
    assert alignment == pytest.approx(np.ones(len(alignment)), abs=1e-9)


def test_the_compiled_kernel_agrees_with_numpy(sample):
    A = AbstractTensor.get_tensor(sample.tolist())
    w_blas, v_blas = eigh(A, method="blas")
    w_blas = np.asarray(w_blas, dtype=float)
    v_blas = np.asarray(v_blas, dtype=float)

    assert w_blas == pytest.approx(np.linalg.eigvalsh(sample), abs=1e-9)
    identity = v_blas.T @ v_blas
    assert identity == pytest.approx(np.eye(len(w_blas)), abs=1e-9)


def test_the_default_kernel_is_still_the_definitional_one(sample):
    """eigh's default must not move: it is the differentiable path, and
    the blas kernel materializes to NumPy and records nothing on the tape.
    """

    A = AbstractTensor.get_tensor(sample.tolist())
    default_w, _ = eigh(A)
    jacobi_w, _ = eigh(A, method="jacobi")
    assert np.asarray(default_w, dtype=float) == pytest.approx(
        np.asarray(jacobi_w, dtype=float), abs=0.0
    )
