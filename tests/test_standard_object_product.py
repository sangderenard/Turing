from __future__ import annotations

import json

import numpy as np

from src.common.tensors.accelerator_backends.ssa_backend import (
    SSATensorOperations,
    SSATensorProgram,
)
from src.compiler.kernel_bank import KernelSpec
from src.compiler.standard_object_product import (
    MethodGraphCapture,
    STANDARD_OBJECT_SCHEMA,
    StandardMethod,
    StandardObject,
    StandardProperty,
    cook_standard_object,
)
from src.compiler.standard_object_blas import blas_standard_object
from src.compiler.process_graph_autograd import (
    abstract_tensor_program_to_process_graph,
    compile_process_graph_backward,
)


SOURCE = """
def scale(n, x, alpha, y):
    for i in range(n):
        y[i] = x[i] * alpha
    return y
"""


def _inputs(sizes, rng):
    n = int(sizes["n"])
    return {"n": n, "x": rng.normal(size=n), "alpha": 2.0, "y": np.zeros(n)}


def _reference(n, x, alpha, y):
    y[:] = x * alpha
    return y


def _capture():
    program = SSATensorProgram("standard_scale")
    value = SSATensorOperations.input(program, (4,))
    alpha = SSATensorOperations.input(program, (4,))
    output = value * alpha
    return MethodGraphCapture(output, {"value": value, "alpha": alpha}, (0, 1))


def test_standard_object_publishes_only_compiled_forward_and_reverse(tmp_path):
    kernel = KernelSpec(
        "scale", SOURCE, "scale", _reference,
        ("n", "x", "alpha", "y"), ("n",), _inputs,
        extents={"x": ("n",), "alpha": (), "y": ("n",)},
    )
    spec = StandardObject(
        "Scale object", "test.scale",
        (StandardMethod("scale", kernel, _capture),),
        (StandardProperty("dtype", "string"),),
    )

    product = cook_standard_object(spec, directory=tmp_path)
    manifest = json.loads(product.manifest_path.read_text(encoding="utf-8"))

    assert manifest["schema"] == STANDARD_OBJECT_SCHEMA
    assert manifest["publication_invariant"] == {
        "parametric_forward_required": True,
        "parametric_graph_reverse_required": True,
        "reverse_must_be_backend_compiled": True,
        "specializations_are_optional_overlays": True,
    }
    method = product.methods["scale"]
    assert method.parametric_forward.specialized == {}
    assert method.parametric_reverse.artifact.library_path.is_file()
    assert method.parametric_reverse.seed_value_ids
    assert method.specialized_forwards == ()


def test_blas_is_a_complete_graph_invertible_standard_object():
    spec = blas_standard_object()

    assert tuple(method.name for method in spec.methods) == (
        "scal", "axpy", "dot", "gemv", "gemm", "rot",
    )
    for method in spec.methods:
        capture = method.capture_graph()
        forward = abstract_tensor_program_to_process_graph(
            capture.output, bindings=capture.bindings,
        )
        reverse = compile_process_graph_backward(
            forward,
            wrt=capture.wrt_value_ids,
            packaging="independent",
            unit_loss_seed=False,
        )
        assert reverse.adjoint.output_value_ids == tuple(forward.roots)
        assert set(reverse.adjoint.gradient_value_ids) == set(
            capture.wrt_value_ids
        )
