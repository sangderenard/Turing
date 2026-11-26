"""A training graph linked into the sanctioned source compiler and run natively.

``training_function_recipe`` builds the forward/loss/backward graph inside
the compile's identity book; ``lower_ast_source_to_ssa`` links it into an
authored host whose tensor inputs are declared through the extraction
contract's ProgramABI.  Every ``unbroadcast`` callsite in the backward rules
supplies a literal shape, so each is a specialized copy and no generic copy
of the helper is lowered.  The LLVM artifact is compared with the
hand-written MSE gradients at every step of a short descent.
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from src.common.tensors.abstraction import AbstractTensor
from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
    c_backend_repository_ssa_reference,
)
from src.common.tensors.accelerator_backends.ssa_backend import (
    SSATensorOperations, SSATensorProgram,
)
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import concordance_report
from src.compiler.process_graph_autograd import training_function_recipe


SHAPES = {"x": (2, 1), "weight": (1, 1), "bias": (1, 1), "target": (2, 1)}


def _linear_mse_module():
    program = SSATensorProgram("linear_mse")
    inputs = {
        name: SSATensorOperations.input(program, shape)
        for name, shape in SHAPES.items()
    }
    prediction = inputs["x"] @ inputs["weight"] + inputs["bias"]
    residual = prediction - inputs["target"]
    loss = (residual * residual).mean()
    recipe = training_function_recipe(
        loss, bindings=inputs,
        wrt=(
            int(inputs["weight"].data.value.id),
            int(inputs["bias"].data.value.id),
        ),
        function_name="linear_mse_motion",
        observed_outputs={"prediction": int(prediction.data.value.id)},
    )
    contract = ExtractionContract(
        Path(__file__).resolve().parents[1]
        / "extraction_contracts/program_extraction.yaml"
    ).with_program_abi({
        "values": [
            {"function": "train_step", "parameter": name, "storage": "span",
             "dtype": "float64", "rank": 2, "shape": list(shape),
             "python_type": "src.common.tensors.abstraction.AbstractTensor"}
            for name, shape in SHAPES.items()
        ],
        "records": {}, "bindings": [],
    })
    return lower_ast_source_to_ssa(
        "def train_step(x: AbstractTensor, weight: AbstractTensor, "
        "bias: AbstractTensor, target: AbstractTensor):\n"
        "    return linear_mse_motion(x, weight, bias, target)\n",
        "train_step", name="linear_mse_host",
        linked_process_graphs={"linear_mse_motion": recipe},
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
        extraction_contract=contract,
        python_bindings={"AbstractTensor": AbstractTensor},
        progress=lambda _message: None,
    )


def test_training_graph_descends_natively_through_the_source_compiler(tmp_path):
    module, outputs, exports = _linear_mse_module()
    report = concordance_report(module)
    assert report.splitlines()[0].endswith("0 finding(s)"), report
    # The training graph's tensor descriptors and classes are on the book:
    # every descriptor row reaches the recorded program's values, and every
    # graph-native class stands on a descriptor (or a copy of such a class).
    from src.compiler.concordance_declarations import (
        ABSTRACT_TENSOR_PROGRAM_VALUE, GRAPH_TENSOR_DESCRIPTOR,
        SOURCE_VALUE_CLASS,
    )
    from src.compiler.identity_concordance import identity_book

    book = identity_book(module)
    descriptor_rows = tuple(book.page(GRAPH_TENSOR_DESCRIPTOR).rows())
    assert descriptor_rows
    for row in descriptor_rows:
        causes = {
            source.page for source, _stage in book.edges_into(
                book.latest_ref(GRAPH_TENSOR_DESCRIPTOR, row),
            )
        }
        assert ABSTRACT_TENSOR_PROGRAM_VALUE in causes, row
    class_page = book.page(SOURCE_VALUE_CLASS)
    graph_classes = [
        row for row in class_page.rows()
        if tuple(class_page.latest(row)[2:3]) == ("graph-native tensor value",)
    ]
    assert graph_classes
    for row in graph_classes:
        causes = {
            source.page for source, _stage in book.edges_into(
                book.latest_ref(SOURCE_VALUE_CLASS, row),
            )
        }
        assert causes & {GRAPH_TENSOR_DESCRIPTOR, SOURCE_VALUE_CLASS}, row
    # Every unbroadcast callsite is a specialized copy; the helper's
    # function-table original is never lowered.
    unbroadcast_symbols = [
        name for name in module.functions if "__unbroadcast" in name
    ]
    assert unbroadcast_symbols
    assert all("__unbroadcast__specialized_" in name for name in unbroadcast_symbols)

    from src.compiler.ssa_llvm_backend import (
        compile_artifact, emit_ssa_function_to_llvm, prepare_artifact_execution,
    )
    from src.compiler.vehicle_python_compilation import _managed_native_feeds_by_id

    root = module.functions[exports[0]]
    artifact = emit_ssa_function_to_llvm(module, root.name)
    assert artifact.complete, artifact.shortfalls
    artifact = compile_artifact(artifact, directory=tmp_path, optimization="O3")

    x = np.array([[1.0], [2.0]])
    target = np.array([[3.0], [5.0]])
    weight = np.array([[0.5]])
    bias = np.array([[0.25]])
    losses = []
    for _step in range(4):
        feeds = _managed_native_feeds_by_id(
            SimpleNamespace(module=module, root_name=root.name),
            {"x": x, "weight": weight, "bias": bias, "target": target},
        )
        execution = prepare_artifact_execution(artifact, feeds)
        execution.run()
        prediction, loss, grad_weight, grad_bias = (
            np.array(execution.buffers[value.id], dtype=float).reshape(value.shape or ())
            for value in outputs[root.name]
        )
        residual = x @ weight + bias - target
        assert np.array_equal(prediction, x @ weight + bias)
        assert float(loss) == float((residual * residual).mean())
        assert np.array_equal(grad_weight, (2.0 / residual.size) * (x.T @ residual))
        assert np.array_equal(
            grad_bias, (2.0 / residual.size) * residual.sum(axis=0, keepdims=True),
        )
        losses.append(float(loss))
        weight = weight - 0.1 * grad_weight
        bias = bias - 0.1 * grad_bias
    assert losses == sorted(losses, reverse=True) and losses[-1] < losses[0] / 100


MASK_SHAPES = {"x": (4, 1), "mask": (4, 1), "weight": (1, 2), "bias": (1, 2), "target": (4, 2)}


def test_masked_rows_contribute_no_gradient_through_the_source_compiler(tmp_path):
    """``(r * r * mask).sum()``: the loss rule's call result is classed.

    ``bw_sum`` returns ``expand_reduction(...)``, which returns ``expand_to``;
    their callsites are planned after the motion's call reads its result
    class, so the reader waits on that row and is re-run when it is posted.
    Without the class the outer ``mul``'s ``g * x`` had none, its
    ``unbroadcast`` copy to the mask's (4, 1) had no ``value_type_abi``, and
    its ``sum`` was refused.  Padded rows (mask 0) give no gradient."""

    program = SSATensorProgram("masked_linear")
    inputs = {
        name: SSATensorOperations.input(program, shape)
        for name, shape in MASK_SHAPES.items()
    }
    residual = inputs["x"] @ inputs["weight"] + inputs["bias"] - inputs["target"]
    loss = (residual * residual * inputs["mask"]).sum()
    recipe = training_function_recipe(
        loss, bindings=inputs,
        wrt=(int(inputs["weight"].data.value.id), int(inputs["bias"].data.value.id)),
        function_name="masked_motion",
    )
    contract = ExtractionContract(
        Path(__file__).resolve().parents[1]
        / "extraction_contracts/program_extraction.yaml"
    ).with_program_abi({
        "values": [
            {"function": "train_step", "parameter": name, "storage": "span",
             "dtype": "float64", "rank": 2, "shape": list(shape),
             "python_type": "src.common.tensors.abstraction.AbstractTensor"}
            for name, shape in MASK_SHAPES.items()
        ],
        "records": {}, "bindings": [],
    })
    module, outputs, exports = lower_ast_source_to_ssa(
        "def train_step(x: AbstractTensor, mask: AbstractTensor, weight: AbstractTensor,\n"
        "               bias: AbstractTensor, target: AbstractTensor):\n"
        "    return masked_motion(x, mask, weight, bias, target)\n",
        "train_step", name="masked_host",
        linked_process_graphs={"masked_motion": recipe},
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
        extraction_contract=contract,
        python_bindings={"AbstractTensor": AbstractTensor},
        progress=lambda _message: None,
    )
    report = concordance_report(module)
    assert report.splitlines()[0].endswith("0 finding(s)"), report

    from src.compiler.ssa_llvm_backend import (
        compile_artifact, emit_ssa_function_to_llvm, prepare_artifact_execution,
    )
    from src.compiler.vehicle_python_compilation import _managed_native_feeds_by_id

    root = module.functions[exports[0]]
    artifact = emit_ssa_function_to_llvm(module, root.name)
    assert artifact.complete, artifact.shortfalls
    artifact = compile_artifact(artifact, directory=tmp_path, optimization="O3")

    x = np.array([[1.0], [2.0], [7.0], [-3.0]])
    target = np.array([[3.0, 1.0], [5.0, -1.0], [40.0, 9.0], [-8.0, 2.0]])
    mask = np.array([[1.0], [1.0], [0.0], [0.0]])     # the last two rows are padding
    weight = np.array([[0.5, -0.25]])
    bias = np.array([[0.25, 0.5]])
    feeds = _managed_native_feeds_by_id(
        SimpleNamespace(module=module, root_name=root.name),
        {"x": x, "mask": mask, "weight": weight, "bias": bias, "target": target},
    )
    execution = prepare_artifact_execution(artifact, feeds)
    execution.run()
    loss_value, grad_weight, grad_bias = (
        np.array(execution.buffers[value.id], dtype=float).reshape(value.shape or ())
        for value in outputs[root.name]
    )
    live = slice(0, 2)
    residual = x[live] @ weight + bias - target[live]
    assert float(loss_value) == float((residual * residual).sum())
    assert np.array_equal(grad_weight, x[live].T @ (2.0 * residual))
    assert np.array_equal(grad_bias, (2.0 * residual).sum(axis=0, keepdims=True))


TANH_SHAPES = {
    "features": (8, 3), "targets": (8, 2), "input_weight": (3, 4),
    "hidden_bias": (1, 4), "output_weight": (4, 2), "output_bias": (1, 2),
}
TANH_PARAMETERS = ("input_weight", "hidden_bias", "output_weight", "output_bias")


def test_two_layer_tanh_step_with_its_update_compiles_through_the_source_compiler(tmp_path):
    """Forward, loss, graph reverse and the SGD update in one compiled program.

    The host calls the training function in the recorded program's argument
    order (the motion's formals follow it, not topological order), and the
    ``tanh`` rule's ``x.detach()`` keeps its operand's descriptor, so both
    bias gradients are unbroadcast to their (1, n) shapes."""

    program = SSATensorProgram("tanh_mse")
    inputs = {
        name: SSATensorOperations.input(program, shape)
        for name, shape in TANH_SHAPES.items()
    }
    prediction = ((inputs["features"] @ inputs["input_weight"] + inputs["hidden_bias"]).tanh()
                  @ inputs["output_weight"] + inputs["output_bias"]).tanh()
    error = prediction - inputs["targets"]
    loss = (error * error).mean()
    recipe = training_function_recipe(
        loss, bindings=inputs,
        wrt=tuple(int(inputs[name].data.value.id) for name in TANH_PARAMETERS),
        function_name="tanh_motion",
        observed_outputs={"prediction": int(prediction.data.value.id)},
    )
    contract = ExtractionContract(
        Path(__file__).resolve().parents[1]
        / "extraction_contracts/program_extraction.yaml"
    ).with_program_abi({
        "values": [
            *({"function": "train_step", "parameter": name, "storage": "span",
               "dtype": "float64", "rank": 2, "shape": list(shape),
               "python_type": "src.common.tensors.abstraction.AbstractTensor"}
              for name, shape in TANH_SHAPES.items()),
            {"function": "train_step", "parameter": "learning_rate",
             "storage": "scalar", "dtype": "float64", "mutable": False,
             "python_type": "builtins.float"},
        ],
        "records": {}, "bindings": [],
    })
    module, outputs, exports = lower_ast_source_to_ssa(
        "def train_step(features: AbstractTensor, targets: AbstractTensor,\n"
        "               input_weight: AbstractTensor, hidden_bias: AbstractTensor,\n"
        "               output_weight: AbstractTensor, output_bias: AbstractTensor,\n"
        "               learning_rate: float):\n"
        "    prediction, loss, g1, gb1, g2, gb2 = tanh_motion(\n"
        "        features, targets, input_weight, hidden_bias, output_weight, output_bias)\n"
        "    return (prediction, loss, input_weight - g1 * learning_rate,\n"
        "            hidden_bias - gb1 * learning_rate, output_weight - g2 * learning_rate,\n"
        "            output_bias - gb2 * learning_rate)\n",
        "train_step", name="tanh_host",
        linked_process_graphs={"tanh_motion": recipe},
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
        extraction_contract=contract,
        python_bindings={"AbstractTensor": AbstractTensor},
        progress=lambda _message: None,
    )
    report = concordance_report(module)
    assert report.splitlines()[0].endswith("0 finding(s)"), report

    from src.compiler.ssa_llvm_backend import (
        compile_artifact, emit_ssa_function_to_llvm, prepare_artifact_execution,
    )
    from src.compiler.vehicle_python_compilation import _managed_native_feeds_by_id

    root = module.functions[exports[0]]
    assert [tuple(value.shape or ()) for value in outputs[root.name]] == [
        (8, 2), (), (3, 4), (1, 4), (4, 2), (1, 2),
    ]
    artifact = emit_ssa_function_to_llvm(module, root.name)
    assert artifact.complete, artifact.shortfalls
    artifact = compile_artifact(artifact, directory=tmp_path, optimization="O3")

    rng = np.random.default_rng(3)
    x = rng.normal(0.0, 1.0, TANH_SHAPES["features"])
    target = rng.uniform(-1.0, 1.0, TANH_SHAPES["targets"])
    weights = {
        "input_weight": rng.normal(0.0, 0.3, (3, 4)), "hidden_bias": np.zeros((1, 4)),
        "output_weight": rng.normal(0.0, 0.3, (4, 2)), "output_bias": np.zeros((1, 2)),
    }
    learning_rate = 0.05
    for _step in range(3):
        w1, b1, w2, b2 = (weights[name] for name in TANH_PARAMETERS)
        hidden = np.tanh(x @ w1 + b1)
        expected_prediction = np.tanh(hidden @ w2 + b2)
        residual = expected_prediction - target
        d_out = (2.0 / residual.size) * residual * (1.0 - expected_prediction ** 2)
        d_hidden = (d_out @ w2.T) * (1.0 - hidden ** 2)
        expected = {
            "input_weight": w1 - (x.T @ d_hidden) * learning_rate,
            "hidden_bias": b1 - d_hidden.sum(axis=0, keepdims=True) * learning_rate,
            "output_weight": w2 - (hidden.T @ d_out) * learning_rate,
            "output_bias": b2 - d_out.sum(axis=0, keepdims=True) * learning_rate,
        }
        feeds = _managed_native_feeds_by_id(
            SimpleNamespace(module=module, root_name=root.name),
            {"features": x, "targets": target, "learning_rate": learning_rate, **weights},
        )
        execution = prepare_artifact_execution(artifact, feeds)
        execution.run()
        result = [
            np.array(execution.buffers[value.id], dtype=float).reshape(value.shape or ())
            for value in outputs[root.name]
        ]
        assert np.allclose(result[0], expected_prediction, rtol=0, atol=1e-15)
        assert abs(float(result[1]) - float((residual * residual).mean())) <= 1e-15
        weights = dict(zip(TANH_PARAMETERS, result[2:]))
        for name in TANH_PARAMETERS:
            assert np.allclose(weights[name], expected[name], rtol=0, atol=1e-14), name
