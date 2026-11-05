"""An untyped runtime slice cannot allocate a typed axis-sum output."""

from src.common.tensors.abstraction import AbstractTensor
from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
    c_backend_repository_ssa_reference,
)
from src.compiler.concordance_declarations import (
    TENSOR_OWNED_HEAP_BINDING,
    TENSOR_REDUCTION_DOMAIN,
)
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.ssa_record_return_state import (
    function_scope_of,
    posted_ssa_value_fact,
)
import src.compiler.tensor_ssa_lowering as tensor_lowering


def test_untyped_axis_source_keeps_authored_sum_before_allocation(monkeypatch):
    class CapturedFirstVisit(Exception):
        pass

    original = tensor_lowering.lower_tensor_calls_to_repository_ssa
    observed = []

    def capture_first_visit(module, reference, **kwargs):
        book = module.metadata["identity_book"]
        for function in module.functions.values():
            for block in function.blocks.values():
                for instruction in block.instrs:
                    if (instruction.op != "sum" or not instruction.args
                            or instruction.res is None):
                        continue
                    source, result = instruction.args[0], instruction.res
                    source_fact = posted_ssa_value_fact(
                        function, int(source.id), book=book,
                    )
                    if (source_fact is None or source_fact.rank != 2
                            or source_fact.physical_dtype is not None):
                        continue
                    scope = function_scope_of(function)
                    owner_row = (scope, function.name, int(result.id))
                    domain_row = (function.name, int(result.id))
                    before = (
                        book.latest_ref(TENSOR_OWNED_HEAP_BINDING, owner_row),
                        book.latest_ref(TENSOR_REDUCTION_DOMAIN, domain_row),
                    )
                    assert before == (None, None)
                    table = module.tensor_tables.get(function.name)
                    # Run this actual authored function before its producer
                    # enters the closure. The shared book and tensor table
                    # are the same ones the public lowering would read.
                    module.functions = {function.name: function}
                    module.tensor_tables = (
                        {} if table is None else {function.name: table}
                    )
                    local = original(
                        module, reference, finalize_private_owners=False,
                    )
                    after = (
                        book.latest_ref(TENSOR_OWNED_HEAP_BINDING, owner_row),
                        book.latest_ref(TENSOR_REDUCTION_DOMAIN, domain_row),
                    )
                    raw_sum = any(
                        candidate.op == "sum" and candidate.res is not None
                        and int(candidate.res.id) == int(result.id)
                        for body in function.blocks.values()
                        for candidate in body.instrs
                    )
                    observed.append((
                        source_fact.dtype, source_fact.storage,
                        source_fact.rank, source_fact.physical_dtype,
                        [(item.operation, item.reason) for item in local],
                        after, raw_sum,
                    ))
                    raise CapturedFirstVisit
        return original(module, reference, **kwargs)

    monkeypatch.setattr(
        tensor_lowering, "lower_tensor_calls_to_repository_ssa",
        capture_first_visit,
    )
    contract = ExtractionContract(
        "extraction_contracts/program_extraction.yaml"
    ).with_program_abi({
        "values": [
            {
                "function": "root", "parameter": name, "storage": "span",
                "dtype": "int64", "rank": rank,
                "python_type":
                    "src.common.tensors.abstraction.AbstractTensor",
            }
            for name, rank in (("membership", 2), ("selected", 1))
        ],
        "records": {}, "bindings": [],
    })
    source = """def root(membership: AbstractTensor, selected: AbstractTensor):
    gathered = membership[selected]
    merged = gathered.sum(dim=0)
    return merged.sum().item()
"""
    try:
        lower_ast_source_to_ssa(
            source, "root", name="runtime_axis_untyped_source",
            extraction_contract=contract,
            python_bindings={"AbstractTensor": AbstractTensor},
            tensor_ssa_reference=c_backend_repository_ssa_reference(),
            progress=lambda _: None, progress_bars=False,
        )
    except CapturedFirstVisit:
        pass
    assert observed == [(
        "int64", "span", 2, None,
        [("sum", "dynamic reduction lacks a typed source span")],
        (None, None), True,
    )]
