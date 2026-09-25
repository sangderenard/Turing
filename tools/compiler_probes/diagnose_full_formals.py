import json
import pickle
import runpy
from pathlib import Path

from src.compiler import fortran_c_shell as shell
from src.compiler.project_compilation_product import _dump_resolved_process_graph
from src.compiler.ssa_self_check import check_formal_parity


original = shell._full_native_link_failures
original_lower = shell.lower_ast_source_to_ssa
output = Path("build/full_formal_diagnostic")
output.mkdir(parents=True, exist_ok=True)


def capture_graph(*args, **kwargs):
    def sink(graph):
        with (output / "resolved-process-graph.pkl").open("wb") as handle:
            _dump_resolved_process_graph(graph, handle)

    kwargs["resolved_process_graph_sink"] = sink
    return original_lower(*args, **kwargs)


def inspect(*args, **kwargs):
    failures = original(*args, **kwargs)
    module = kwargs.get("module")
    reports = []
    if module is not None:
        with (output / "repository-ssa.pkl").open("wb") as handle:
            pickle.dump(module, handle)
        for finding in check_formal_parity(module):
            function = module.functions[finding.function]
            metadata = dict(function.metadata or {})
            named = {
                int(value) for _name, value in metadata.get("parameter_names") or ()
            }
            accounted = set(named)
            accounted.update(
                int(entry["value_id"])
                for entry in metadata.get("storage_formals") or ()
            )
            accounted.update(
                int(entry["value_id"])
                for entry in metadata.get("parameter_member_formals") or ()
                if entry.get("parameter") in metadata.get("authored_parameters", ())
                and entry.get("path")
            )
            accounted.update(
                int(argument.id)
                for argument in function.args
                if (argument.accounting or {}).get("program_abi_parameter") is not None
            )
            for formal in function.args:
                value_id = int(formal.id)
                if value_id in accounted:
                    continue
                consumers = []
                definitions = []
                region_outputs = []
                for block_name, block in function.blocks.items():
                    for index, instruction in enumerate(block.instrs):
                        if instruction.res is not None and int(instruction.res.id) == value_id:
                            definitions.append((block_name, index, instruction.op))
                        if any(int(argument.id) == value_id for argument in instruction.args):
                            consumers.append((block_name, index, instruction.op))
                        if value_id in tuple(map(
                            int, (instruction.attributes or {}).get("output_ids") or ()
                        )):
                            region_outputs.append((
                                block_name, index, instruction.op,
                                (instruction.attributes or {}).get("callee"),
                            ))
                reports.append({
                    "function": finding.function,
                    "value_id": value_id,
                    "dtype": formal.dtype,
                    "shape": tuple(formal.shape),
                    "accounting": dict(formal.accounting or {}),
                    "definitions": definitions,
                    "consumers": consumers,
                    "region_outputs": region_outputs,
                    "authored_parameters": metadata.get("authored_parameters"),
                    "parameter_names": metadata.get("parameter_names"),
                })
    (output / "formals.json").write_text(
        json.dumps(reports, indent=2, default=str), encoding="utf-8"
    )
    print("FORMAL-DIAGNOSTIC", len(reports), output, flush=True)
    return failures


shell._full_native_link_failures = inspect
shell.lower_ast_source_to_ssa = capture_graph
runpy.run_module("tools.scan_managed_duplicates", run_name="__main__")
