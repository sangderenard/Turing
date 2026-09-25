import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
import numpy as np
from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import c_backend_repository_ssa_reference
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.ssa_reference_evaluator import SSAReferenceEvaluator, bind_program_abi_arguments

SOURCE = """
import torch

def solve_two(matrix: torch.Tensor, rhs: torch.Tensor):
    return torch.linalg.solve(matrix, rhs)
"""
MATRIX = np.asarray([[4.0, 1.0], [2.0, 3.0]], dtype=np.float64)
RHS = np.asarray([1.0, 2.0], dtype=np.float64)
root = Path.cwd()
contract = ExtractionContract(root / "extraction_contracts" / "program_extraction.yaml").with_program_abi({
    "records": {}, "bindings": [],
    "values": [
        {"function": "solve_two", "parameter": "matrix", "storage": "span", "dtype": "float64", "rank": 2, "shape": list(MATRIX.shape), "python_type": "AbstractTensor"},
        {"function": "solve_two", "parameter": "rhs", "storage": "span", "dtype": "float64", "rank": 1, "shape": list(RHS.shape), "python_type": "AbstractTensor"},
    ],
})
module, outputs, _exports = lower_ast_source_to_ssa(
    SOURCE, "solve_two", name="abstract_solve_numeric",
    extraction_contract=contract,
    tensor_ssa_reference=c_backend_repository_ssa_reference(),
    progress=lambda m: None,
)
import os
if (os.environ.get("TURING_DUMP_FORWARD_SSA") or os.environ.get("TURING_DUMP_LU_SSA")
        or os.environ.get("TURING_DUMP_MASKED_SSA")):
    from src.compiler.identity_concordance import identity_book, loop_scope_declarations
    for dump_name, dump_function in module.functions.items():
        wanted = (
            "___forward_substitute__specialized_" in dump_name
            if os.environ.get("TURING_DUMP_FORWARD_SSA") else
            "___lu_decompose_inplace__specialized_" in dump_name
            if os.environ.get("TURING_DUMP_LU_SSA") else
            "___masked_pivot_rows__specialized_" in dump_name
        )
        if not wanted or ("__planned_region" in dump_name and not os.environ.get("TURING_DUMP_MASKED_SSA")):
            continue
        print("FUNCTION", dump_name)
        print("ARGS", [
            (int(value.id), str(value.dtype), tuple(value.shape or ()), dict(value.accounting or {}))
            for value in dump_function.args
        ])
        print("SCOPES", loop_scope_declarations(identity_book(module), dump_name))
        print("CALL TABLE", [record for record in module.call_table.get(dump_name, ()) if int(record.callsite_id) in {40, 42, 44, 75, 76, 80}])
        print("CARRIED PORTS", {
            int(key): int(value.id)
            for key, value in (dump_function.metadata.get("carried_port_values") or {}).items()
        })
        for dump_block_name, dump_block in dump_function.blocks.items():
            print("BLOCK", dump_block_name, "SUCCESSORS", tuple(dump_block.successors))
            for dump_index, dump_instruction in enumerate(dump_block.instrs):
                print(
                    dump_index, dump_instruction.op,
                    tuple(int(value.id) for value in dump_instruction.args),
                    None if dump_instruction.res is None else int(dump_instruction.res.id),
                    dict(dump_instruction.attributes),
                )
    raise SystemExit(0)
from src.compiler.identity_concordance import identity_book
book = identity_book(module)
if book is not None:
    for page_name in ("ssa_call_shape", "ssa_call_shape_evidence", "formal_shape", "callsite_return_specialization", "call_metadata_mutation", "proven_shape"):
        page = book.pages.get(page_name)
        if page is None:
            continue
        print("PAGE", page_name)
        for row in page.rows():
            if (("forward_substitute" in repr(row) and ("29" in repr(row) or page_name in {"ssa_call_shape", "callsite_return_specialization"})) or (page_name == "formal_shape" and "_row" in repr(row)) or (page_name == "callsite_return_specialization" and "solve" in repr(row)) or (page_name in {"ssa_call_shape", "ssa_call_shape_evidence"} and "___row__specialized" in repr(row))):
                print(" ", row, page.history(row))
qualified = "abstract_solve_numeric__solve_two"
fn = module.functions[qualified]
arguments, unbound = bind_program_abi_arguments(
    fn, named={"matrix": MATRIX, "rhs": RHS}, functions=module.functions, scratch=True,
)

evaluator = SSAReferenceEvaluator(module)
original_execute = evaluator._execute

def wrapped_execute(function, values):
    returned = original_execute(function, values)
    if "___lu_decompose_inplace__" in str(function.name) and not str(function.name).endswith("__planned_region_3"):
        print("LU EXEC", function.name, {
            value_id: np.asarray(values[value_id]).tolist()
            for value_id in (11, 22, 38, 40, 75, 100, 103, 107, 109,
                             111, 113, 114, 115, 116,
                             2305843010213694219, 2305843010213694220,
                             2305843010213694221, 2305843010213695051,
                             2305843010213695052, 2305843010213695053)
            if value_id in values
        }, "RETURNED", [np.asarray(value).tolist() for value in returned])
    if "___forward_substitute__" in str(function.name):
        print("FORWARD EXEC", function.name, {
            value_id: np.asarray(values[value_id]).tolist()
            for value_id in (1, 12, 14, 17, 18, 28, 29, 37, 47,
                             2305843010213694111, 2305843010213695050, 48)
            if value_id in values
        }, "RETURNED", [np.asarray(value).tolist() for value in returned])
    if str(function.name).endswith("__planned_region_3") and "___lu_decompose_inplace__" in str(function.name):
        print("LU REGION VALUES", {
            value_id: np.asarray(values[value_id]).tolist()
            for value_id in (76, 75, 77, 79)
            if value_id in values
        }, "RETURNED", [np.asarray(value).tolist() for value in returned],
        "METADATA", dict(function.metadata))
    return returned

evaluator._execute = wrapped_execute
stack = []
snapshot = []
first_nonfinite = []
original_call = evaluator._call

def wrapped_call(instruction, values):
    callee_name = str(instruction.attributes.get("callee") or "")
    arg_ids = [int(getattr(a, "id", -1)) for a in instruction.args]
    arg_vals = []
    for aid in arg_ids:
        v = values.get(aid)
        try:
            arg_vals.append(np.asarray(v).tolist() if v is not None else None)
        except Exception:
            arg_vals.append(repr(v))
    stack.append((callee_name, arg_ids, arg_vals, dict(instruction.attributes)))
    try:
        result = original_call(instruction, values)
        if "___masked_pivot_rows__" in callee_name:
            print("MASKED CALL", arg_vals[:3], "OUTPUTS", {
                int(output_id): np.asarray(values[int(output_id)]).tolist()
                for output_id in instruction.attributes.get("output_ids", ())
                if int(output_id) in values
            })
        if not first_nonfinite:
            produced_ids = []
            if instruction.res is not None:
                produced_ids.append(int(instruction.res.id))
            produced_ids.extend(map(int, instruction.attributes.get("output_ids", ())))
            output_position = instruction.attributes.get("ssa_output_argument")
            if output_position is not None and int(output_position) < len(instruction.args):
                produced_ids.append(int(instruction.args[int(output_position)].id))
            for produced_id in dict.fromkeys(produced_ids):
                if produced_id not in values:
                    continue
                try:
                    produced = np.asarray(values[produced_id], dtype=np.float64)
                except (TypeError, ValueError):
                    continue
                if produced.size and not np.all(np.isfinite(produced)):
                    first_nonfinite.append((
                        callee_name, arg_ids, arg_vals, produced_id,
                        produced.tolist(), dict(instruction.attributes),
                        list(stack),
                    ))
                    break
        stack.pop()
        return result
    except Exception:
        snapshot[:] = list(stack)
        raise

evaluator._call = wrapped_call

try:
    result = evaluator.run(qualified, arguments)
    print("NO ERROR")
    expected = np.linalg.solve(MATRIX, RHS)
    published = [int(v.id) for v in outputs[qualified]]
    produced = [np.asarray(result.values[vid]).reshape(-1) for vid in published if vid in result.values]
    print("EXPECTED", expected)
    print("PRODUCED", produced)
    print("FIRST NONFINITE", first_nonfinite)
    for name, function in module.functions.items():
        if "___lu_decompose_inplace__specialized_" not in name or not name.endswith("__planned_region_3"):
            continue
        print("LU REGION", name, "FORMALS", [(int(value.id), tuple(value.shape or ())) for value in function.args])
        for block_name, block in function.blocks.items():
            for index, instruction in enumerate(block.instrs):
                print("LU", block_name, index, instruction.op,
                      [(int(value.id), tuple(value.shape or ())) for value in instruction.args],
                      None if instruction.res is None else (int(instruction.res.id), tuple(instruction.res.shape or ())),
                      dict(instruction.attributes))
except Exception as exc:
    print("ERROR:", type(exc).__name__, str(exc)[:200])
    print(f"call stack depth at crash: {len(snapshot)}")
    for depth, (callee, arg_ids, arg_vals, attrs) in enumerate(snapshot):
        print(f"  [{depth}] callee={callee} args={arg_vals}")
    crashed_region = snapshot[-2][0] if len(snapshot) >= 2 else ""
    if crashed_region in module.functions:
        function = module.functions[crashed_region]
        print("CRASHED REGION", crashed_region, "FORMALS", [(int(value.id), tuple(value.shape or ())) for value in function.args])
        for block_name, block in function.blocks.items():
            for index, instruction in enumerate(block.instrs):
                print("CRASHED", block_name, index, instruction.op,
                      [(int(value.id), tuple(value.shape or ())) for value in instruction.args],
                      None if instruction.res is None else (int(instruction.res.id), tuple(instruction.res.shape or ())),
                      dict(instruction.attributes))
    for name, function in module.functions.items():
        if "___forward_substitute__specialized_" not in name or not name.endswith("__planned_region_4"):
            continue
        print("REGION", name, "FORMALS", [(int(value.id), tuple(value.shape or ())) for value in function.args])
        for block_name, block in function.blocks.items():
            for index, instruction in enumerate(block.instrs):
                print(block_name, index, instruction.op,
                      [(int(value.id), tuple(value.shape or ())) for value in instruction.args],
                      None if instruction.res is None else (int(instruction.res.id), tuple(instruction.res.shape or ())),
                      dict(instruction.attributes))
    for name, function in module.functions.items():
        if "___forward_substitute__specialized_" not in name or "__planned_region" in name:
            continue
        for block_name, block in function.blocks.items():
            for index, instruction in enumerate(block.instrs):
                if (
                    instruction.res is not None and int(instruction.res.id) in {14, 28, 29, 37, 41}
                    or any(int(value.id) in {28, 29} for value in instruction.args)
                ):
                    print("FORWARD VALUE", name, block_name, index, instruction.op,
                          [(int(value.id), tuple(value.shape or ())) for value in instruction.args],
                          None if instruction.res is None else (int(instruction.res.id), tuple(instruction.res.shape or ())),
                          dict(instruction.attributes))
                if str(instruction.attributes.get("callee") or "").endswith("__planned_region_4"):
                    print("REGION CALL", name, block_name, index,
                          [(int(value.id), tuple(value.shape or ())) for value in instruction.args],
                          dict(instruction.attributes))
