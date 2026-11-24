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
