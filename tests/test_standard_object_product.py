from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess

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
    standard_object_from_source,
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

SURFACE_SOURCE = """
class ScaleSurface:
    dtype: str = "float64"

    @property
    def methods(self):
        return ("scale",)

    def scale(self, value, alpha):
        return value * alpha

def run(value, alpha):
    return ScaleSurface().scale(value, alpha)
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
    spec = standard_object_from_source(
        name="Scale object",
        identity="test.scale",
        source=SURFACE_SOURCE,
        entrypoint="run",
        feeds={"value": 2.0, "alpha": 3.0},
        kernels={"scale": kernel},
        graph_captures={"scale": _capture},
    )

    product = cook_standard_object(spec, directory=tmp_path)
    manifest = json.loads(product.manifest_path.read_text(encoding="utf-8"))

    assert manifest["schema"] == STANDARD_OBJECT_SCHEMA
    assert manifest["publication_invariant"] == {
        "parametric_forward_required": True,
        "parametric_graph_reverse_required": True,
        "reverse_must_be_backend_compiled": True,
        "browser_reverse_must_be_compiled": True,
        "specializations_are_optional_overlays": True,
    }
    assert [item["name"] for item in manifest["object"]["properties"]] == [
        "dtype", "methods",
    ]
    surface = manifest["object"]["surface"]
    assert surface["unresolved_call_count"] == 0
    assert surface["feed_names"] == ["alpha", "value"]
    assert (product.directory / surface["artifact"]).is_file()
    method = product.methods["scale"]
    assert method.parametric_forward.specialized == {}
    assert method.parametric_reverse.artifact.library_path.is_file()
    assert method.browser_reverse.wasm_path.is_file()
    assert method.parametric_reverse.seed_value_ids
    assert method.specialized_forwards == ()
    browser_loader = product.directory / manifest["browser_loader"]
    assert browser_loader.is_file()
    assert hashlib.sha256(browser_loader.read_bytes()).hexdigest() == (
        manifest["browser_loader_source_sha256"]
    )
    node = shutil.which("node")
    if node is not None:
        script = f'''import {{readFile}} from "node:fs/promises";
import {{fileURLToPath,pathToFileURL}} from "node:url";
globalThis.fetch=async url=>new Response(await readFile(fileURLToPath(url)));
const module=await import(pathToFileURL({json.dumps(str(browser_loader))}).href);
const reverse=await module.CompiledObjectReverse.load(pathToFileURL({json.dumps(str(product.directory) + str(Path('/')))}));
const gradients=await reverse.vjp("scale",new Float64Array([1,2,3,4]),{{value:new Float64Array([2,3,5,7]),alpha:new Float64Array([11,13,17,19])}});
console.log(JSON.stringify({{value:[...gradients.value],alpha:[...gradients.alpha]}}));'''
        completed = subprocess.run(
            [node, "--input-type=module", "--eval", script],
            capture_output=True, text=True, check=True,
        )
        result = json.loads(completed.stdout)
        assert result == {
            "value": [11.0, 26.0, 51.0, 76.0],
            "alpha": [2.0, 6.0, 15.0, 28.0],
        }


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
