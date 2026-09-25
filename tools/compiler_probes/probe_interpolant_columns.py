"""Compile the real post-limb_terms interpolant evaluator boundary."""

from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.ssa_llvm_backend import (
    compile_artifact,
    emit_ssa_function_to_llvm,
    prepare_artifact_execution,
)
from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
    c_backend_repository_ssa_reference,
)
from src.common.tensors.interpolants import Interpolant


ENTRY = "compile_interpolant"
COEFFICIENTS = tuple(
    f"{side}{coefficient}{limb}"
    for coefficient in range(4)
    for side in ("left", "right")
    for limb in range(2)
)
PARAMETERS = ("x", "points", *COEFFICIENTS)


def contract():
    shapes = {"x": (3,), "points": (2,)}
    shapes.update({name: (2,) for name in COEFFICIENTS})
    values = [{
        "function": ENTRY,
        "parameter": name,
        "storage": "span",
        "dtype": "float64",
        "rank": 1,
        "shape": list(shapes[name]),
        "python_type": "src.common.tensors.abstraction.AbstractTensor",
    } for name in PARAMETERS]
    return ExtractionContract(
        Path("extraction_contracts/program_extraction.yaml")
    ).with_program_abi({"records": {}, "bindings": [], "values": values})


def source():
    arguments = ", ".join(PARAMETERS)
    left = ", ".join(
        f"(left{coefficient}0, left{coefficient}1)"
        for coefficient in range(4)
    )
    right = ", ".join(
        f"(right{coefficient}0, right{coefficient}1)"
        for coefficient in range(4)
    )
    return f"""
from src.common.tensors.interpolants import _wide_from_columns

def {ENTRY}({arguments}):
    left_columns = ({left})
    right_columns = ({right})
    return _wide_from_columns(
        x, 2, points, "value", 0,
        left_columns, right_columns, (), (),
    ).collapse()
"""


if __name__ == "__main__":
    module, outputs, exports = lower_ast_source_to_ssa(
        source(),
        ENTRY,
        name="actual_interpolant_columns",
        extraction_contract=contract(),
        runtime_closure_only=True,
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
    )
    root = module.functions[f"actual_interpolant_columns__{ENTRY}"]

    artifact = emit_ssa_function_to_llvm(module, root.name)
    if not artifact.complete:
        for shortfall in artifact.shortfalls:
            function = module.functions[shortfall.function]
            print("llvm_shortfall", shortfall)
            for block_name, block in function.blocks.items():
                for instruction in block.instrs:
                    if str(instruction.op) == str(shortfall.operation):
                        print(" ", block_name, instruction)
        raise RuntimeError(tuple(
            (item.function, item.operation, item.reason)
            for item in artifact.shortfalls
        ))
    output_directory = ROOT / "build" / "interpolant_llvm_probe"
    output_directory.mkdir(parents=True, exist_ok=True)
    compile_artifact(artifact, directory=output_directory)
    x = np.asarray([0.0, 1.0, 2.0], dtype=np.float64)
    points = np.asarray([0.25, 1.5], dtype=np.float64)
    curve = Interpolant(x, [0.0, 1.0, 4.0], method="linear", limbs=2)
    columns = {"x": x, "points": points}
    for coefficient in range(4):
        for side, values in (
            ("left", curve._left[coefficient]),
            ("right", curve._right[coefficient]),
        ):
            for limb, value in enumerate(values):
                columns[f"{side}{coefficient}{limb}"] = np.asarray(
                    value.tolist(), dtype=np.float64
                )
    execution = prepare_artifact_execution(artifact, {
        int(argument.id): columns[name]
        for argument, name in zip(root.args, PARAMETERS)
    })
    execution.run()
    result_id = int(outputs[root.name][0].id)
    actual = np.asarray(execution.buffers[result_id])
    expected = np.asarray(curve.evaluate(points).tolist())
    if not np.array_equal(actual, expected):
        raise AssertionError((actual, expected))
    print("LLVM interpolant probe passed:", actual.tolist())
