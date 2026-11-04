"""The authored broadcast extent law as an accepted repository-SSA component."""

from __future__ import annotations

import hashlib
import pickle
from pathlib import Path
from typing import Mapping, Any

from ..transmogrifier.tensor_ssa_reference import (
    SSATensorCodeReference,
    SSATensorOperationReference,
)


def with_authored_broadcast_extents(
    reference: SSATensorCodeReference,
    *, module: Any = None,
) -> SSATensorCodeReference:
    """Add the existing authored integer extent law to one tensor reference.

    This is called only for a mixed dynamic binary tensor operation. The
    helper's own completed module and identity book stay together; the outer
    caller will cite its cells through the accepted-component import path.
    """

    canonical_op = "broadcast_shape_extents"
    if module is not None:
        recorded = (getattr(module, "metadata", {}) or {}).get(
            "tensor_ssa_reference"
        )
        if isinstance(recorded, SSATensorCodeReference) and recorded.operation(
            canonical_op
        ) is not None:
            if any(recorded.operation(key) != operation
                   for key, operation in reference.operations.items()):
                raise ValueError(
                    "authored broadcast reference conflicts with the compile's "
                    "existing code reference"
                )
            return recorded
    if reference.operation(canonical_op) is not None:
        return reference

    from .extraction_contract import ExtractionContract
    from .fortran_c_shell import lower_ast_source_to_ssa
    from .identity_concordance import CorrelationTable, identity_book

    repository = Path(__file__).resolve().parents[2]
    source = Path(__file__).with_name("graph_refinement_kernels.py").read_text()
    parameters = ("rank", "start", "left", "right", "output")
    contract = ExtractionContract(
        repository / "extraction_contracts/program_extraction.yaml"
    ).with_program_abi({
        "records": {}, "bindings": [],
        "values": [
            {
                "function": canonical_op,
                "parameter": parameter,
                "storage": "scalar" if parameter in {"rank", "start"}
                           else "span",
                "dtype": "int32",
                "rank": 0 if parameter in {"rank", "start"} else 1,
                "python_type": "builtins.int" if parameter in {"rank", "start"}
                               else "src.common.tensors.abstraction.AbstractTensor",
                "mutable": parameter == "output",
            }
            for parameter in parameters
        ],
    })
    helper_module, outputs, exports = lower_ast_source_to_ssa(
        source, canonical_op, name="compiler_broadcast_extents",
        extraction_contract=contract, runtime_closure_only=True,
        python_bindings={}, tensor_ssa_reference=reference,
        progress=lambda _: None, progress_bars=False,
    )
    findings = CorrelationTable.build(helper_module).findings(helper_module)
    if findings:
        raise ValueError(
            "authored broadcast extent helper has incomplete concordance: "
            f"{findings[:3]!r}"
        )
    if len(exports) != 1 or exports[0] not in helper_module.functions:
        raise ValueError("authored broadcast extent helper has no unique export")
    root = str(exports[0])
    helper = helper_module.functions[root]
    parameter_names = dict(helper.metadata.get("parameter_names") or ())
    if set(parameter_names) != set(parameters) or {
        int(parameter_names[name]) for name in parameters
    } != {int(argument.id) for argument in helper.args}:
        raise ValueError("authored broadcast extent helper lost formal order")
    artifact_bytes = pickle.dumps(
        (helper_module, outputs, exports), protocol=pickle.HIGHEST_PROTOCOL,
    )
    digest = hashlib.sha256(artifact_bytes).hexdigest()
    owner_book = identity_book(helper_module)
    for function in helper_module.functions.values():
        function.metadata["accepted_ssa_component"] = {
            "schema": "turing.accepted-ssa-component.v1",
            "artifact_sha256": digest,
            "identity_book": owner_book,
        }
    for name, function in reference.module.functions.items():
        old = helper_module.functions.get(name)
        if old is not None and old is not function:
            raise ValueError(f"authored broadcast reference collision: {name}")
        helper_module.functions[name] = function
    for name, table in reference.module.tensor_tables.items():
        old = helper_module.tensor_tables.get(name)
        if old is not None and old != table:
            raise ValueError(f"authored broadcast tensor table collision: {name}")
        helper_module.tensor_tables[name] = table
    augmented = SSATensorCodeReference(
        f"{reference.name}+authored-broadcast-extents", helper_module,
        {
            **reference.operations,
            canonical_op: SSATensorOperationReference(canonical_op, (root,)),
        },
        f"{reference.source_identity}+sha256:{digest}",
        reference.external_primitives,
    )
    if module is not None:
        module.metadata["tensor_ssa_reference"] = augmented
    return augmented


def authored_broadcast_call_actuals(
    reference: SSATensorCodeReference, values: Mapping[str, Any],
) -> tuple[Any, tuple[Any, ...], tuple[Any, ...]]:
    """Return the exact helper formals' actuals and imported formal cells."""

    from .frame_identity_book import _frame_value_cell

    operation = reference.operation("broadcast_shape_extents")
    if operation is None or len(operation.entrypoints) != 1:
        raise ValueError("authored broadcast extent helper has no unique entry")
    helper = reference.module.functions[operation.entrypoints[0]]
    parameter_names = dict(helper.metadata.get("parameter_names") or ())
    names = ("rank", "start", "left", "right", "output")
    if set(values) != set(names) or set(parameter_names) != set(names):
        raise ValueError("authored broadcast extent helper parameter mismatch")
    by_formal_id = {
        int(parameter_names[name]): values[name] for name in names
    }
    if len(by_formal_id) != len(helper.args) or set(by_formal_id) != {
        int(formal.id) for formal in helper.args
    }:
        raise ValueError("authored broadcast extent helper formal mismatch")
    actuals = tuple(by_formal_id[int(formal.id)] for formal in helper.args)
    cells = tuple(
        _frame_value_cell(helper, int(formal.id)) for formal in helper.args
    )
    if any(cell is None for cell in cells):
        raise ValueError("authored broadcast extent helper lost formal proof")
    return helper, actuals, cells


__all__ = ["with_authored_broadcast_extents", "authored_broadcast_call_actuals"]
