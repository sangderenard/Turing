"""One accepted specialization keeps the graph which owns its return Phi."""
from pathlib import Path

import numpy as np
import pytest

from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import concordance_report

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def reused_return_module():
    return lower_ast_source_to_ssa(
        (ROOT / "tests/record_view_corpus/c637_reused_return_specialization.py").read_text(),
        "root", name="reused_return",
        extraction_contract=ROOT / "extraction_contracts/program_extraction.yaml",
        progress=lambda _: None, progress_bars=False,
    )


def test_reused_specialization_source_owner_preserves_control_owned_return(reused_return_module):
    module, _outputs, _exports = reused_return_module
    names = [name for name in module.functions
             if "___atom_token__specialized" in name
             and "__planned_region_" not in name]
    assert len(names) == 1
    function = module.functions[names[0]]
    terminator = next(instruction for block in function.blocks.values()
                      for instruction in block.instrs if instruction.op == "Ret")
    result, = terminator.args
    phi = next(instruction for block in function.blocks.values()
               for instruction in block.instrs
               if instruction.op == "Phi" and instruction.res.id == result.id)
    # Both source return arms are retained as distinct incoming values.
    assert len({value.id for value in phi.args}) == 2
    assert dict(function.metadata["named_outputs"])["result_0"] == result.id
    assert function.metadata["semantic_output_ids"] == (result.id,)
    aliases = dict(function.metadata["output_identity_aliases"])
    assert all(aliases[value.id] == result.id for value in phi.args)
    report = concordance_report(module)
    assert "0 finding(s)" in report, report


@pytest.mark.parametrize("backend", ["c", "llvm"])
def test_reused_return_specialization_executes_each_return_arm(backend, reused_return_module, tmp_path):
    module, outputs, exports = reused_return_module
    root_name = next(name for name in exports if name.endswith("__root"))
    root = module.functions[root_name]
    names = dict(root.metadata["parameter_names"])
    if backend == "c":
        from src.compiler.ssa_c_backend import emit_ssa_to_c
        artifact = emit_ssa_to_c(module, root_name)
        assert artifact.complete, artifact.shortfalls
        artifact.compile(tmp_path / "native")
        prepare = artifact.prepare_execution
    else:
        from src.compiler.ssa_llvm_backend import emit_ssa_function_to_llvm, compile_artifact, prepare_artifact_execution
        artifact = emit_ssa_function_to_llvm(module, root_name)
        assert artifact.complete, artifact.shortfalls
        native = compile_artifact(artifact, directory=tmp_path / "native")
        prepare = lambda feeds: prepare_artifact_execution(native, feeds)
    result_id = outputs[root_name][0].id
    for domain, atom, expected in [(2, 65, 2 * (1112067 + 65)),
                                    (1, 65, 2 * (3 + 65)),
                                    (1, 57344, 2 * (3 + 57344 - 2048))]:
        execution = prepare({names["domain"]: np.array([domain], dtype="int64"),
                             names["atom"]: np.array([atom], dtype="int64")}).run()
        assert execution.buffers[result_id].item() == expected
