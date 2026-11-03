"""The imported top-k kernel's two output formals share one aggregate call."""

import numpy as np
import pytest

from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
    c_backend_repository_ssa_reference,
)
from src.compiler.concordance_declarations import (
    ADDRESS, CONTROL_CONST, CONTROL_EXPRESSION, CONTROL_SSA,
    CONTROL_SSA_ENTRY, LOAD, REGION_CALL_RESULT, SSA_BLOCK, SSABlockKind,
)
from src.compiler.identity_concordance import (
    CorrelationTable, Derived, Mode, begin_identity_book, end_identity_book,
    identity_book,
)
from src.compiler.precompile_to_ssa import _function_root_cell
from src.compiler.ssa_aggregate_abi import analyze_aggregate_abi
from src.compiler.ssa_c_backend import emit_ssa_module_to_c
from src.compiler.ssa_record_return_state import mint_ssa_value
from src.transmogrifier.ssa import BasicBlock, Function, Instr, IRModule, SSAValue


@pytest.mark.parametrize("binding", ("all_formals", "input_formals", "duplicate_formal"))
def test_imported_topk_two_named_outputs_use_one_booked_aggregate_call(
    tmp_path, binding,
):
    # The output record is values, indices, although the C formal order is
    # indices, values. The imported named_outputs cells own that distinction.
    book, token = begin_identity_book()
    try:
        reference = c_backend_repository_ssa_reference()
        module = IRModule(dict(reference.dependency_closure("topk_int64_dim")))
        kernel = module.functions["topk_int64_dim"]
        ports = dict(kernel.metadata["named_outputs"])
        assert tuple(ports) == ("values", "indices")
        assert ports["values"] == kernel.args[6].id
        assert ports["indices"] == kernel.args[5].id

        root = Function("probe_topk", [], {"entry": BasicBlock("entry", [])})

        def fresh(dtype, shape=(), transform=CONTROL_EXPRESSION, operands=()):
            value_id = mint_ssa_value(
                root, transform, operands, dtype=dtype, shape=shape,
                stage=CONTROL_SSA,
            )
            return SSAValue(value_id, dtype, shape)

        source = fresh("int64", (3,))
        shape = fresh("int32", (1,))
        indices = fresh("int64", (2,))
        values = fresh("int64", (2,))
        root.args[:] = [source, shape, indices, values]
        ndim = fresh("int32", transform=CONTROL_CONST)
        dim = fresh("int32", transform=CONTROL_CONST)
        k = fresh("int32", transform=CONTROL_CONST)
        call_inputs = [source, shape, ndim, dim, k]
        call_actuals = (
            [*call_inputs, indices, values]
            if binding == "all_formals" else call_inputs
        )
        input_ids = (
            tuple(int(formal.id) for formal in kernel.args[:5])
            if binding != "duplicate_formal"
            else tuple(int(formal.id) for formal in (
                kernel.args[0], kernel.args[0], *kernel.args[2:5]
            ))
        )
        aggregate = fresh(
            "ssa.aggregate", transform=REGION_CALL_RESULT,
            operands=tuple(call_actuals),
        )
        index0 = fresh("int32", transform=CONTROL_CONST)
        index1 = fresh("int32", transform=CONTROL_CONST)
        address0 = fresh("ptr", transform=ADDRESS, operands=(aggregate, index0))
        address1 = fresh("ptr", transform=ADDRESS, operands=(aggregate, index1))
        projected_values = fresh("int64", (2,), LOAD, (address0,))
        projected_indices = fresh("int64", (2,), LOAD, (address1,))
        if binding != "all_formals":
            root.args[:] = [source, shape]

        call = Instr(
            "Call", call_actuals,
            aggregate, attributes={
                "callee": kernel.name,
                "result_convention": "ssa.aggregate",
                "output_ids": (projected_values.id, projected_indices.id),
                "callee_output_ids": (ports["values"], ports["indices"]),
                "output_positions": (0, 1),
                **({"callee_input_ids": input_ids}
                   if binding != "all_formals" else {}),
            },
        )
        root.blocks["entry"].instrs[:] = [
            Instr("Const", [], ndim, attributes={"value": 1}),
            Instr("Const", [], dim, attributes={"value": 0}),
            Instr("Const", [], k, attributes={"value": 2}),
            call,
            Instr("Const", [], index0, attributes={"value": 0}),
            Instr("GetElementPtr", [aggregate, index0], address0,
                  attributes={"aggregate_index": 0}),
            Instr("Load", [address0], projected_values),
            Instr("Const", [], index1, attributes={"value": 1}),
            Instr("GetElementPtr", [aggregate, index1], address1,
                  attributes={"aggregate_index": 1}),
            Instr("Load", [address1], projected_indices),
            Instr("Ret", [] if binding == "all_formals" else [
                projected_values, projected_indices,
            ], None),
        ]
        module.functions[root.name] = root
        identity_book(module)
        root_cell = _function_root_cell(book, root.name)
        book.post(
            SSA_BLOCK, (root.name, root.name, "entry"), SSABlockKind.ENTRY,
            stage=CONTROL_SSA_ENTRY, provenance=Derived((root_cell,)),
            mode=Mode.CONCORD,
        )

        analysis = analyze_aggregate_abi(module, (root.name, kernel.name))
        assert len(analysis.calls) == 1
        assert analysis.calls[0].output_ids == (
            ports["values"], ports["indices"],
        )
        assert tuple(
            (projection.position, projection.output_id)
            for projection in analysis.calls[0].projections
        ) == ((0, ports["values"]), (1, ports["indices"]))

        artifact = emit_ssa_module_to_c(module, root.name)
        if binding == "duplicate_formal":
            assert not artifact.complete
            assert any(
                "incomplete exact callee_input_ids/output projection binding"
                in item.reason for item in artifact.shortfalls
            )
            return
        assert artifact.complete, artifact.shortfalls
        artifact.compile(tmp_path / "topk_aggregate")
        abi = artifact.runtime_abi()
        output_values = (
            (indices, values) if binding == "all_formals"
            else (projected_values, projected_indices)
        )
        assert tuple(item["value_id"] for item in abi["buffers"]) == (
            source.id, shape.id, *(value.id for value in output_values),
        )
        execution = artifact.prepare_execution({
            source.id: np.array([2**53, 2**53 + 1, 2**53 + 2], dtype=np.int64),
            shape.id: np.array([3], dtype=np.int32),
            **{
                value.id: np.zeros(2, dtype=np.int64)
                for value in output_values
            },
        }).run()
        value_id = (
            values.id if binding == "all_formals"
            else projected_values.id
        )
        index_id = (
            indices.id if binding == "all_formals"
            else projected_indices.id
        )
        assert execution.buffers[value_id].tolist() == [2**53 + 2, 2**53 + 1]
        assert execution.buffers[index_id].tolist() == [2, 1]
        assert CorrelationTable.build(module).findings(module) == []
    finally:
        end_identity_book(token)
