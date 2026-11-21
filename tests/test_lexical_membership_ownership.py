"""Membership must read each loop target inside its actual branch and loop."""
from pathlib import Path
import numpy as np
import pytest
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import CorrelationTable

@pytest.mark.parametrize("scope", ["plain", "outer_guard", "inner_guard"])
@pytest.mark.parametrize("operator", ["in", "not in"])
def test_membership_owns_its_loop_target_natively(tmp_path, scope, operator):
    from src.compiler.ssa_llvm_backend import emit_ssa_function_to_llvm, compile_artifact, prepare_artifact_execution
    header = "def root(keys: list[int], values: list[int], enabled: bool):\n"
    if scope == "inner_guard":
        body = ("    allowed = {v for v in values}\n    total = 0\n"
                "    for key in keys:\n        if enabled:\n"
                f"            if key {operator} allowed:\n                total += key\n"
                "    return total\n")
    else:
        indent = "        " if scope == "outer_guard" else "    "
        body = "    if enabled:\n" if scope == "outer_guard" else ""
        body += indent + "allowed = {v for v in values}\n"
        body += indent + f"selected = [key for key in keys if key {operator} allowed]\n"
        body += indent + "total = 0\n" + indent + "for key in selected:\n" + indent + "    total += key\n"
        body += indent + "return total\n"
        if scope == "outer_guard": body += "    return 0\n"
    source = header + body
    contract = ExtractionContract(Path(__file__).resolve().parents[1]/"extraction_contracts/program_extraction.yaml")
    module, _, exports = lower_ast_source_to_ssa(source, "root", extraction_contract=contract, progress=lambda _:None, progress_bars=False)
    assert CorrelationTable.build(module).findings(module) == []
    fn = module.functions[exports[0]]
    names = dict(fn.metadata["parameter_names"])
    table = module.sequence_tables[exports[0]]
    emitted = emit_ssa_function_to_llvm(module, exports[0])
    assert emitted.shortfalls == ()
    artifact = compile_artifact(emitted, directory=tmp_path/"llvm")
    output = next(i.args[0].id for block in fn.blocks.values() for i in block.instrs if i.op=="Ret" and not i.attributes.get("fault_exit"))
    for keys, values in [([2,9,-3,2,0], [2,-3,7]), ([], [1]), ([5,6], []), ([1,1], [1,1])]:
        for enabled in [False, True]:
            feeds = {}
            for name, data in [("keys",keys),("values",values)]:
                seq=table.sequences[names[name]]
                feeds.update({seq.column_value_ids[0]:np.array(data,dtype=np.int64),
                              seq.length_address_id:np.array([len(data)],dtype=np.int64),
                              seq.capacity_value_id:np.array([len(data)],dtype=np.int64)})
                if seq.status_address_id is not None:feeds[seq.status_address_id]=np.array([0],dtype=np.int64)
            if names.get("enabled") in {a.id for a in fn.args}:
                feeds[names["enabled"]]=np.array([enabled],dtype=np.bool_)
            originals={k:v.copy() for k,v in feeds.items()}
            expected=sum(k for k in keys if ((k in set(values)) if operator=="in" else (k not in set(values)))) if enabled or scope=="plain" else 0
            for _ in range(2):
                result=prepare_artifact_execution(artifact, feeds).run()
                assert int(result.buffers[output].reshape(-1)[0])==expected
                for key in feeds:np.testing.assert_array_equal(feeds[key], originals[key])
