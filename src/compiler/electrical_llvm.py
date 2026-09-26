"""Native complex nodal solves through a canonical retained-loop solve law.

The electrical graph owns topology and admittance assembly. This module owns
only the numerical publication ``Y V = I``. Complex values cross the compiler
ABI as paired real tensors and use the exact real block transformation::

    [ Re(Y) -Im(Y) ] [ Re(V) ] = [ Re(I) ]
    [ Im(Y)  Re(Y) ] [ Im(V) ]   [ Im(I) ]

The real system is solved by partial-pivot Gauss-Jordan elimination against
the same ``numpy.linalg.solve`` oracle as the repository tensor solve. Its
loops remain loops in LLVM; matrix extent does not expand into one SSA
arithmetic cell per scalar operation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..common.tensors.linalg_kernels import SOLVE_SOURCE
from .extraction_contract import ExtractionContract
from .fortran_c_shell import lower_ast_source_to_ssa
from .ssa_llvm_backend import (
    LLVMFunctionArtifact,
    compile_artifact,
    emit_ssa_function_to_llvm,
    prepare_artifact_execution,
)


def _solve_contract(lanes: int, real_nodes: int) -> ExtractionContract:
    root = Path(__file__).resolve().parents[2]
    values = [
        {
            "function": "solve", "parameter": "matrix", "storage": "span",
            "dtype": "float64", "rank": 1,
            "shape": [lanes * real_nodes * real_nodes],
            "python_type": "src.common.tensors.abstraction.AbstractTensor",
        },
        {
            "function": "solve", "parameter": "rhs", "storage": "span",
            "dtype": "float64", "rank": 1,
            "shape": [lanes * real_nodes],
            "python_type": "src.common.tensors.abstraction.AbstractTensor",
        },
        *(
            {
                "function": "solve", "parameter": name, "storage": "span",
                "dtype": "float64", "rank": 1,
                "shape": [lanes * real_nodes],
                "python_type": "src.common.tensors.abstraction.AbstractTensor",
            }
            for name in (
                "current_row", "pivot_row_values", "current_rhs", "pivot_rhs",
            )
        ),
        {
            "function": "solve", "parameter": "pivot_rows", "storage": "span",
            "dtype": "float64", "rank": 1,
            "shape": [lanes * real_nodes * real_nodes],
            "python_type": "src.common.tensors.abstraction.AbstractTensor",
        },
        {
            "function": "solve", "parameter": "pivot_magnitudes", "storage": "span",
            "dtype": "float64", "rank": 1,
            "shape": [lanes * real_nodes * real_nodes],
            "python_type": "src.common.tensors.abstraction.AbstractTensor",
        },
    ]
    return ExtractionContract(
        root / "extraction_contracts" / "program_extraction.yaml"
    ).with_program_abi({"records": {}, "bindings": [], "values": values})


@dataclass
class NativeComplexNodalSolver:
    """One fixed-buffer LLVM artifact for batched complex nodal equations."""

    lanes: int
    nodes: int
    artifact: LLVMFunctionArtifact
    input_value_ids: dict[str, int]
    _execution: Any = field(default=None, init=False, repr=False)

    def solve(self, admittance: Any, current: Any) -> np.ndarray:
        y = np.asarray(admittance, dtype=np.complex128)
        i = np.asarray(current, dtype=np.complex128)
        expected_y = (self.lanes, self.nodes, self.nodes)
        expected_i = (self.lanes, self.nodes)
        if y.shape != expected_y:
            raise ValueError(f"expected admittance shape {expected_y}, got {y.shape}")
        if i.shape != expected_i:
            raise ValueError(f"expected current shape {expected_i}, got {i.shape}")

        g = y.real
        b = y.imag
        matrix = np.concatenate((
            np.concatenate((g, -b), axis=2),
            np.concatenate((b, g), axis=2),
        ), axis=1)
        rhs = np.concatenate((i.real, i.imag), axis=1)
        real_nodes = self.nodes * 2
        ids = self.input_value_ids
        feeds = {
            ids["matrix"]: np.ascontiguousarray(matrix.reshape(-1)),
            ids["rhs"]: np.ascontiguousarray(rhs.reshape(-1)),
            ids["current_row"]: np.zeros(self.lanes * real_nodes),
            ids["pivot_row_values"]: np.zeros(self.lanes * real_nodes),
            ids["current_rhs"]: np.zeros(self.lanes * real_nodes),
            ids["pivot_rhs"]: np.zeros(self.lanes * real_nodes),
            ids["pivot_rows"]: np.zeros(
                self.lanes * real_nodes * real_nodes,
            ),
            ids["pivot_magnitudes"]: np.zeros(
                self.lanes * real_nodes * real_nodes,
            ),
        }
        if self._execution is None:
            self._execution = prepare_artifact_execution(self.artifact, feeds)
        else:
            for value_id, value in feeds.items():
                self._execution.buffers[value_id][...] = value
        execution = self._execution.run()
        solution = np.asarray(
            execution.buffers[ids["rhs"]]
        ).reshape(self.lanes, real_nodes)
        return solution[:, :self.nodes] + 1j * solution[:, self.nodes:]


def compile_complex_nodal_solver(
    directory: str | Path,
    *,
    lanes: int,
    nodes: int,
    name: str = "complex_nodal_solve",
) -> NativeComplexNodalSolver:
    """Compile a retained-loop partial-pivot solve for one electrical ABI."""
    lanes = int(lanes)
    nodes = int(nodes)
    if lanes < 1 or nodes < 1:
        raise ValueError("complex nodal solve requires positive lane and node counts")
    real_nodes = nodes * 2
    module_name = f"{name}__l{lanes}__n{nodes}"
    module, _outputs, _exports = lower_ast_source_to_ssa(
        SOLVE_SOURCE.replace("__BATCH__", str(lanes)).replace(
            "__N__", str(real_nodes)
        ),
        "solve",
        name=module_name,
        extraction_contract=_solve_contract(lanes, real_nodes),
    )
    qualified = f"{module_name}__solve"
    function = module.functions[qualified]
    ids = {
        str(parameter): int(value_id)
        for parameter, value_id in function.metadata["parameter_names"]
    }
    artifact = emit_ssa_function_to_llvm(
        module, qualified, entry_name=module_name,
    )
    if artifact.shortfalls:
        raise RuntimeError(
            f"{module_name} LLVM shortfalls: {artifact.shortfalls!r}")
    native = compile_artifact(artifact, directory=Path(directory))
    return NativeComplexNodalSolver(lanes, nodes, native, ids)


__all__ = ["NativeComplexNodalSolver", "compile_complex_nodal_solver"]
