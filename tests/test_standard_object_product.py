from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np

from src.common.tensors.mathematical_library import (
    LINALG_LIBRARY,
    TRIGONOMETRY_LIBRARY,
)
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
    mathematical_sublibrary_object,
    standard_object_from_source,
)
from src.common.tensors.source_realization import authored_source_realization
from src.compiler.mathematical_library_product import _python_loader
from src.compiler.standard_object_blas import blas_standard_object
from src.compiler.standard_object_trigonometry import (
    trigonometry_standard_object,
)
from src.compiler.standard_object_linalg import linalg_standard_object
from src.compiler.ssa_llvm_backend import prepare_artifact_execution
from src.compiler.llvm_training_runtime import (
    compile_native_graph_forward,
    compile_native_graph_reverse,
)
from src.compiler.process_graph_autograd import (
    abstract_tensor_program_to_process_graph,
    compile_process_graph_backward,
    obtain_graph_reverse,
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


def _capture(parameters=None):
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
        "compiled_reverse_backends": ["native", "wasm"],
        "browser_reverse_must_be_compiled": True,
        "specializations_are_optional_overlays": True,
        "realization": "recursive_authored_source",
        "interior_compiled_artifact_calls": False,
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
    assert manifest["methods"][0]["source_revision"] == hashlib.sha256(
        SOURCE.encode("utf-8")
    ).hexdigest()
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


def test_installed_objects_recursively_reveal_authored_source_during_bake():
    namespace = {}
    exec(_python_loader(), namespace)
    CompiledStandardObject = namespace["CompiledStandardObject"]

    class Host:
        def __init__(self):
            self.data = "host-storage"

        def ensure_tensor(self, value):
            return value

        def operation(self, value):
            return ("authored", value)

    record = {
        "methods": [{
            "name": "operation",
            "source": "def operation(self, value): return ('authored', value)",
            "installation": "instance_operator",
        }],
        "deployment_matrix": [{
            "method": "operation", "parameters": {}, "key": "parametric",
        }],
        "artifacts": {
            "operation": {"parametric_forward": {
                "kind": "captured_graph",
                "input_value_ids": {"template": 0, "value": 1},
            }},
        },
    }
    first = CompiledStandardObject(Path("."), record)
    second = CompiledStandardObject(Path("."), record)
    first._call = lambda *args, **kwargs: "first-binary"
    second._call = lambda *args, **kwargs: "second-binary"

    original = Host.operation
    try:
        first.install(Host)
        second.install(Host)
        assert Host().operation(3) == "second-binary"
        with authored_source_realization():
            assert Host().operation(3) == ("authored", 3)
    finally:
        second.uninstall(Host)
        first.uninstall(Host)
    assert Host.operation is original


def test_namespace_object_installation_replaces_and_restores_linalg_surface():
    namespace = {}
    exec(_python_loader(), namespace)
    CompiledStandardObject = namespace["CompiledStandardObject"]

    class Linalg:
        @staticmethod
        def dot(left, right):
            return ("authored-dot", left, right)

    class Host:
        linalg = Linalg()

    record = {
        "methods": [{
            "name": "dot",
            "source": "def dot(left, right): return left * right",
            "installation": "namespace_operator:linalg",
        }],
        "deployment_matrix": [{
            "method": "dot", "parameters": {}, "key": "parametric",
        }],
        "artifacts": {"dot": {"parametric_forward": {
            "kind": "captured_graph", "input_value_ids": {"left": 0, "right": 1},
        }}},
    }
    pack = CompiledStandardObject(Path("."), record)
    pack._call = lambda *args, **kwargs: "compiled-dot"
    original = Host.linalg.dot
    try:
        pack.install(Host)
        assert Host.linalg.dot(2, 3) == "compiled-dot"
        with authored_source_realization():
            assert Host.linalg.dot(2, 3) == ("authored-dot", 2, 3)
    finally:
        pack.uninstall(Host)
    assert Host.linalg.dot is original


def test_generic_object_maker_expands_parameter_domains_cartesianly():
    unused_captures = {
        method.name: (lambda parameters: None)
        for method in LINALG_LIBRARY.methods
    }
    spec = mathematical_sublibrary_object(
        LINALG_LIBRARY,
        kernels=None,
        graph_captures=unused_captures,
        parameter_domains={
            "eigh": {"sweeps": (8, 16), "sort": (False, True)},
        },
        baseline_parameters={"eigh": {"sweeps": 8, "sort": False}},
    )

    assert next(
        method for method in spec.methods if method.name == "eigh"
    ).specializations == (
        {"sweeps": 8, "sort": True},
        {"sweeps": 16, "sort": False},
        {"sweeps": 16, "sort": True},
    )
    assert next(
        method for method in spec.methods if method.name == "eigh"
    ).baseline_parameters == {"sweeps": 8, "sort": False}


def test_linalg_is_one_complete_source_captured_standard_object():
    spec = linalg_standard_object()

    assert tuple(method.name for method in spec.methods) == tuple(
        method.name for method in LINALG_LIBRARY.methods
    )
    for method in spec.methods:
        capture = method.capture_graph({})
        reverse = obtain_graph_reverse(
            capture.output,
            bindings=capture.bindings,
            wrt=capture.wrt_value_ids,
            unit_output_seed=False,
        )
        assert reverse.adjoint.output_value_ids
        assert set(reverse.adjoint.gradient_value_ids) == set(
            capture.wrt_value_ids
        )


def test_native_graph_forward_preserves_shape_only_buffer_aliases(tmp_path):
    program = SSATensorProgram("shape_alias_forward")
    value = SSATensorOperations.input(program, (2, 2))
    output = value.reshape((4,))
    forward = compile_native_graph_forward(
        output,
        bindings={"value": value},
        source="def flatten(value): return value.reshape((4,))",
        name="shape_alias_forward",
        directory=tmp_path,
    )

    execution = prepare_artifact_execution(
        forward.artifact,
        {forward.input_value_ids["value"]: np.asarray([[1.0, 2.0], [3.0, 4.0]])},
    ).run()
    np.testing.assert_array_equal(
        execution.buffers[forward.output_value_ids[0]],
        np.asarray([1.0, 2.0, 3.0, 4.0]),
    )


def test_constant_only_method_has_a_compiled_zero_gradient_reverse(tmp_path):
    eye = next(
        method for method in linalg_standard_object().methods
        if method.name == "eye"
    )
    capture = eye.capture_graph({})
    reverse = compile_native_graph_reverse(
        capture.output,
        bindings=capture.bindings,
        wrt_value_ids=capture.wrt_value_ids,
        name="standard_eye_zero_gradient_vjp",
        directory=tmp_path,
        unit_output_seed=False,
    )

    assert reverse.input_value_ids == {}
    assert reverse.gradient_value_ids == {}
    assert reverse.seed_value_ids
    assert reverse.artifact.library_path.is_file()


def test_blas_is_a_complete_graph_invertible_standard_object():
    spec = blas_standard_object()

    assert tuple(method.name for method in spec.methods) == (
        "scal", "axpy", "dot", "gemv", "gemm", "rot",
    )
    for method in spec.methods:
        capture = method.capture_graph({})
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


def test_trigonometry_is_a_complete_second_graph_invertible_standard_object():
    spec = trigonometry_standard_object()

    assert tuple(method.name for method in spec.methods) == tuple(
        method.name for method in TRIGONOMETRY_LIBRARY.methods
    )
    for method in spec.methods:
        capture = method.capture_graph({})
        reverse = obtain_graph_reverse(
            capture.output,
            bindings=capture.bindings,
            wrt=capture.wrt_value_ids,
            unit_output_seed=False,
        )
        assert reverse.adjoint.output_value_ids
        assert set(reverse.adjoint.gradient_value_ids) == set(
            capture.wrt_value_ids
        )


def test_trigonometry_can_publish_native_reverse_without_forcing_wasm(tmp_path):
    product = cook_standard_object(
        trigonometry_standard_object(),
        directory=tmp_path,
        contract="fast",
        reverse_backends=("native",),
    )

    assert product.manifest["publication_invariant"][
        "compiled_reverse_backends"
    ] == ["native"]
    assert "browser_loader" not in product.manifest
    assert tuple(product.methods) == tuple(
        method.name for method in TRIGONOMETRY_LIBRARY.methods
    )
    assert {
        row["method"] for row in product.manifest["deployment_matrix"]
    } == set(product.methods)
    assert all(
        row["parameters"] == {} and row["kind"] == "captured_graph"
        for row in product.manifest["deployment_matrix"]
    )
    assert {
        method["name"]: method["source"]
        for method in product.manifest["methods"]
    } == {
        method.name: method.source
        for method in TRIGONOMETRY_LIBRARY.methods
    }
    for method in product.methods.values():
        assert method.parametric_reverse.artifact.library_path.is_file()
        assert method.browser_reverse is None

    samples = {
        "acosh": np.asarray([1.25, 1.5, 2.0, 3.0]),
        "asin": np.asarray([-0.6, -0.2, 0.25, 0.7]),
        "acos": np.asarray([-0.6, -0.2, 0.25, 0.7]),
        "atanh": np.asarray([-0.6, -0.2, 0.25, 0.7]),
    }
    default = np.asarray([0.25, 0.5, 0.75, 1.0])
    references = {
        "sin": np.sin, "cos": np.cos, "tan": np.tan,
        "asin": np.arcsin, "acos": np.arccos, "atan": np.arctan,
        "sinh": np.sinh, "cosh": np.cosh, "tanh": np.tanh,
        "asinh": np.arcsinh, "acosh": np.arccosh, "atanh": np.arctanh,
        "sec": lambda x: 1 / np.cos(x),
        "csc": lambda x: 1 / np.sin(x),
        "cot": lambda x: np.cos(x) / np.sin(x),
        "sech": lambda x: 1 / np.cosh(x),
        "csch": lambda x: 1 / np.sinh(x),
        "coth": lambda x: np.cosh(x) / np.sinh(x),
        "sinc": lambda x: np.sin(x) / x,
    }
    for name, method in product.methods.items():
        values = samples.get(name, default)
        forward = method.parametric_forward
        execution = prepare_artifact_execution(
            forward.artifact,
            {forward.input_value_ids["value"]: values},
        ).run()
        np.testing.assert_allclose(
            execution.buffers[forward.output_value_ids[0]],
            references[name](values),
            rtol=1.0e-12,
            atol=1.0e-12,
        )
        reverse = method.parametric_reverse
        seed_id = reverse.seed_value_ids[reverse.output_value_ids[0]]
        reverse_execution = prepare_artifact_execution(
            reverse.artifact,
            {
                reverse.input_value_ids["value"]: values,
                seed_id: np.ones_like(values),
            },
        ).run()
        gradient = reverse_execution.buffers[
            reverse.gradient_value_ids[reverse.input_value_ids["value"]]
        ]
        step = 1.0e-6
        finite_difference = (
            references[name](values + step) - references[name](values - step)
        ) / (2 * step)
        np.testing.assert_allclose(
            gradient, finite_difference, rtol=2.0e-5, atol=2.0e-6,
        )
